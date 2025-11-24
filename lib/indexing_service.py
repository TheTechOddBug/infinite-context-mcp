#!/usr/bin/env python3
"""
Indexing Service for Infinite Context
Handles indexing of repositories, documentation, websites, and local filesystems
"""
import os
import json
import hashlib
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urlparse, urljoin
from pathlib import Path
import re

import base64
import httpx
from bs4 import BeautifulSoup
import trafilatura
from github import Github
import aiofiles
from openai import OpenAI
# Pinecone Index type - we'll use Any for type hints since Index is accessed via Pinecone instance
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pinecone import Index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Base exception for indexing errors"""
    pass


class APIError(IndexingError):
    """API-related errors"""
    def __init__(self, message: str, status_code: int = None, detail: str = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class IndexingService:
    """Service for indexing various content sources"""
    
    def __init__(self, pinecone_index, openai_client: OpenAI):
        # pinecone_index is a Pinecone Index object from pc.Index()
        self.index = pinecone_index
        self.openai = openai_client
        self.github_token = os.getenv("GITHUB_TOKEN")
        # Initialize GitHub client only if token is available
        try:
            self.github = Github(self.github_token) if self.github_token else None
        except Exception as e:
            logger.warning(f"Failed to initialize GitHub client: {e}")
            self.github = None
        
        # In-memory status tracking (could be moved to Redis/DB for production)
        self.indexing_status: Dict[str, Dict] = {}
        
    def _generate_source_id(self, source_type: str, identifier: str) -> str:
        """Generate a unique source ID"""
        combined = f"{source_type}:{identifier}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.openai.embeddings.create(
            input=text[:30000],  # Truncate to be safe
            model="text-embedding-3-large",
            dimensions=1024
        )
        return response.data[0].embedding
    
    async def _store_chunks(
        self,
        source_id: str,
        source_type: str,
        source_url: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ):
        """Store chunks in Pinecone"""
        vectors = []
        
        for i, chunk_data in enumerate(chunks):
            chunk_id = f"{source_id}_chunk_{i}"
            embedding = await self._generate_embedding(chunk_data["content"])
            
            vector_metadata = {
                "source_type": source_type,
                "source_id": source_id,
                "source_url": source_url,
                "file_path": chunk_data.get("file_path", ""),
                "content": chunk_data["content"][:25000],  # Pinecone metadata limit
                "chunk_index": i,
                "indexed_at": datetime.now().isoformat(),
                **metadata,
                **chunk_data.get("metadata", {})
            }
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": vector_metadata
            })
        
        # Batch upsert to Pinecone
        if vectors:
            # Pinecone supports batches of up to 100 vectors
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                self.index.upsert(vectors=batch)
                logger.info(f"Stored {len(batch)} chunks for source {source_id}")
    
    async def index_repository(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Index a GitHub repository"""
        if not self.github:
            raise APIError("GitHub token not configured. Set GITHUB_TOKEN environment variable.")
        
        try:
            # Parse repository URL
            if "github.com" not in repo_url:
                raise IndexingError(f"Invalid GitHub URL: {repo_url}")
            
            # Extract owner/repo from URL
            parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
            if len(parts) < 2:
                raise IndexingError(f"Invalid repository URL format: {repo_url}")
            
            owner = parts[0]
            repo_name = parts[1].replace(".git", "")
            repo_path = f"{owner}/{repo_name}"
            
            # Get repository
            repo = self.github.get_repo(repo_path)
            branch = branch or repo.default_branch
            
            source_id = self._generate_source_id("repository", f"{repo_path}:{branch}")
            
            # Update status
            self.indexing_status[source_id] = {
                "status": "indexing",
                "source_type": "repository",
                "source_url": repo_url,
                "repository": repo_path,
                "branch": branch,
                "progress": 0,
                "started_at": datetime.now().isoformat()
            }
            
            # Get all files from repository
            contents = repo.get_contents("", ref=branch)
            files_to_index = []
            
            def get_files_recursive(contents_item, path=""):
                if contents_item.type == "file":
                    file_path = f"{path}/{contents_item.name}" if path else contents_item.name
                    # Check file patterns if provided
                    if file_patterns:
                        if not any(re.match(pattern, file_path) for pattern in file_patterns):
                            return
                    files_to_index.append((file_path, contents_item))
                elif contents_item.type == "dir":
                    try:
                        dir_contents = repo.get_contents(contents_item.path, ref=branch)
                        for item in dir_contents:
                            get_files_recursive(item, contents_item.path)
                    except Exception as e:
                        logger.warning(f"Error accessing directory {contents_item.path}: {e}")
            
            for item in contents:
                get_files_recursive(item)
            
            total_files = len(files_to_index)
            logger.info(f"Found {total_files} files to index in {repo_path}")
            
            chunks = []
            indexed_count = 0
            
            for file_path, file_content in files_to_index:
                try:
                    # Get file content
                    if file_content.encoding == "base64":
                        content = base64.b64decode(file_content.content).decode('utf-8', errors='ignore')
                    else:
                        content = file_content.content
                    
                    # Chunk the content
                    text_chunks = self._chunk_text(content)
                    
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        chunks.append({
                            "content": chunk_text,
                            "file_path": file_path,
                            "metadata": {
                                "file_size": len(content),
                                "file_sha": file_content.sha
                            }
                        })
                    
                    indexed_count += 1
                    progress = int((indexed_count / total_files) * 100)
                    self.indexing_status[source_id]["progress"] = progress
                    
                except Exception as e:
                    logger.warning(f"Error indexing file {file_path}: {e}")
                    continue
            
            # Store chunks
            await self._store_chunks(
                source_id=source_id,
                source_type="repository",
                source_url=repo_url,
                chunks=chunks,
                metadata={
                    "repository": repo_path,
                    "branch": branch,
                    "file_count": indexed_count
                }
            )
            
            # Update status to completed
            self.indexing_status[source_id].update({
                "status": "completed",
                "progress": 100,
                "completed_at": datetime.now().isoformat(),
                "page_count": indexed_count,
                "chunk_count": len(chunks)
            })
            
            return {
                "source_id": source_id,
                "status": "completed",
                "repository": repo_path,
                "branch": branch,
                "files_indexed": indexed_count,
                "chunks_created": len(chunks)
            }
            
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'status') and e.status == 404:
                raise APIError(f"Repository not found: {repo_url}", status_code=404)
            elif hasattr(e, 'status') and e.status == 403:
                raise APIError("GitHub API rate limit exceeded or insufficient permissions", status_code=403)
            else:
                raise IndexingError(f"Error indexing repository: {error_msg}")
    
    async def index_website(
        self,
        url: str,
        max_depth: int = 3,
        max_pages: int = 100,
        url_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        only_main_content: bool = True,
        wait_for: Optional[int] = None,
        include_screenshot: bool = False
    ) -> Dict[str, Any]:
        """Index a full website by crawling"""
        source_id = self._generate_source_id("website", url)
        
        # Update status
        self.indexing_status[source_id] = {
            "status": "indexing",
            "source_type": "website",
            "source_url": url,
            "progress": 0,
            "started_at": datetime.now().isoformat()
        }
        
        visited_urls: Set[str] = set()
        urls_to_visit: List[tuple] = [(url, 0)]  # (url, depth)
        chunks = []
        page_count = 0
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            while urls_to_visit and page_count < max_pages:
                current_url, depth = urls_to_visit.pop(0)
                
                if current_url in visited_urls or depth > max_depth:
                    continue
                
                # Check URL patterns
                if url_patterns:
                    if not any(re.match(pattern, current_url) for pattern in url_patterns):
                        continue
                
                if exclude_patterns:
                    if any(re.match(pattern, current_url) for pattern in exclude_patterns):
                        continue
                
                visited_urls.add(current_url)
                
                try:
                    # Fetch page
                    if wait_for:
                        await asyncio.sleep(wait_for / 1000.0)
                    
                    response = await client.get(current_url)
                    response.raise_for_status()
                    
                    # Extract content
                    if only_main_content:
                        # Use trafilatura for main content extraction
                        extracted = trafilatura.extract(response.text, url=current_url)
                        if extracted:
                            content = extracted
                        else:
                            # Fallback to BeautifulSoup
                            soup = BeautifulSoup(response.text, 'lxml')
                            # Remove script and style elements
                            for script in soup(["script", "style", "nav", "header", "footer"]):
                                script.decompose()
                            content = soup.get_text(separator='\n', strip=True)
                    else:
                        soup = BeautifulSoup(response.text, 'lxml')
                        content = soup.get_text(separator='\n', strip=True)
                    
                    if not content or len(content.strip()) < 100:
                        logger.warning(f"Skipping {current_url}: insufficient content")
                        continue
                    
                    # Get page title
                    soup = BeautifulSoup(response.text, 'lxml')
                    title = soup.title.string if soup.title else ""
                    
                    # Chunk the content
                    text_chunks = self._chunk_text(content)
                    
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        chunks.append({
                            "content": chunk_text,
                            "file_path": current_url,
                            "metadata": {
                                "page_title": title[:500],
                                "depth": depth,
                                "url": current_url
                            }
                        })
                    
                    page_count += 1
                    progress = int((page_count / min(max_pages, 100)) * 100)
                    self.indexing_status[source_id]["progress"] = progress
                    
                    # Extract links for crawling (if not at max depth)
                    if depth < max_depth:
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            absolute_url = urljoin(current_url, href)
                            
                            # Only follow links from same domain
                            parsed_base = urlparse(url)
                            parsed_link = urlparse(absolute_url)
                            
                            if parsed_link.netloc == parsed_base.netloc:
                                if absolute_url not in visited_urls:
                                    urls_to_visit.append((absolute_url, depth + 1))
                    
                except Exception as e:
                    logger.warning(f"Error indexing page {current_url}: {e}")
                    continue
        
        # Store chunks
        await self._store_chunks(
            source_id=source_id,
            source_type="website",
            source_url=url,
            chunks=chunks,
            metadata={
                "max_depth": max_depth,
                "max_pages": max_pages
            }
        )
        
        # Update status
        self.indexing_status[source_id].update({
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now().isoformat(),
            "page_count": page_count,
            "chunk_count": len(chunks)
        })
        
        return {
            "source_id": source_id,
            "status": "completed",
            "url": url,
            "pages_indexed": page_count,
            "chunks_created": len(chunks)
        }
    
    async def index_documentation(
        self,
        url: str,
        url_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        only_main_content: bool = True
    ) -> Dict[str, Any]:
        """Index documentation site (similar to website but optimized for docs)"""
        # Documentation indexing is essentially website indexing with defaults
        return await self.index_website(
            url=url,
            max_depth=5,  # Deeper for docs
            max_pages=500,  # More pages for docs
            url_patterns=url_patterns or ["/docs/*", "/documentation/*", "/*.html", "/*.md"],
            exclude_patterns=exclude_patterns or ["/blog/*", "/changelog/*", "/api/*"],
            only_main_content=only_main_content
        )
    
    async def index_local_filesystem(
        self,
        directory_path: str,
        inclusion_patterns: Optional[List[str]] = None,
        exclusion_patterns: Optional[List[str]] = None,
        max_file_size_mb: int = 50
    ) -> Dict[str, Any]:
        """Index a local filesystem directory"""
        if not os.path.isabs(directory_path):
            raise IndexingError(f"Directory path must be absolute: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise IndexingError(f"Directory does not exist: {directory_path}")
        
        source_id = self._generate_source_id("local_filesystem", directory_path)
        
        # Update status
        self.indexing_status[source_id] = {
            "status": "indexing",
            "source_type": "local_filesystem",
            "source_url": directory_path,
            "progress": 0,
            "started_at": datetime.now().isoformat()
        }
        
        # Find all files
        files_to_index = []
        max_size_bytes = max_file_size_mb * 1024 * 1024
        
        for root, dirs, files in os.walk(directory_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory_path)
                
                # Check exclusion patterns
                if exclusion_patterns:
                    if any(re.match(pattern, rel_path) for pattern in exclusion_patterns):
                        continue
                
                # Check inclusion patterns
                if inclusion_patterns:
                    if not any(re.match(pattern, rel_path) for pattern in inclusion_patterns):
                        continue
                
                # Check file size
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > max_size_bytes:
                        logger.warning(f"Skipping {rel_path}: file too large ({file_size / 1024 / 1024:.2f} MB)")
                        continue
                    
                    files_to_index.append((file_path, rel_path, file_size))
                except Exception as e:
                    logger.warning(f"Error checking file {file_path}: {e}")
                    continue
        
        total_files = len(files_to_index)
        logger.info(f"Found {total_files} files to index in {directory_path}")
        
        chunks = []
        indexed_count = 0
        
        for file_path, rel_path, file_size in files_to_index:
            try:
                # Read file content
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                
                # Chunk the content
                text_chunks = self._chunk_text(content)
                
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunks.append({
                        "content": chunk_text,
                        "file_path": rel_path,
                        "metadata": {
                            "file_size": file_size,
                            "absolute_path": file_path
                        }
                    })
                
                indexed_count += 1
                progress = int((indexed_count / total_files) * 100) if total_files > 0 else 100
                self.indexing_status[source_id]["progress"] = progress
                
            except Exception as e:
                logger.warning(f"Error indexing file {rel_path}: {e}")
                continue
        
        # Store chunks
        await self._store_chunks(
            source_id=source_id,
            source_type="local_filesystem",
            source_url=directory_path,
            chunks=chunks,
            metadata={
                "file_count": indexed_count
            }
        )
        
        # Update status
        self.indexing_status[source_id].update({
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now().isoformat(),
            "page_count": indexed_count,
            "chunk_count": len(chunks)
        })
        
        return {
            "source_id": source_id,
            "status": "completed",
            "directory_path": directory_path,
            "files_indexed": indexed_count,
            "chunks_created": len(chunks)
        }
    
    def get_indexing_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get indexing status for a source"""
        return self.indexing_status.get(source_id)
    
    def list_indexed_sources(self) -> List[Dict[str, Any]]:
        """List all indexed sources"""
        return list(self.indexing_status.values())
    
    async def index_url(
        self,
        url: str,
        only_main_content: bool = True,
        wait_for: Optional[int] = None
    ) -> Dict[str, Any]:
        """Index a single URL (any type - ChatGPT, Twitter, blog post, etc.)"""
        source_id = self._generate_source_id("url", url)
        
        # Update status
        self.indexing_status[source_id] = {
            "status": "indexing",
            "source_type": "url",
            "source_url": url,
            "progress": 0,
            "started_at": datetime.now().isoformat()
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                # Fetch page
                if wait_for:
                    await asyncio.sleep(wait_for / 1000.0)
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                # Determine content type
                content_type = response.headers.get("content-type", "").lower()
                
                # Extract content based on URL type or content
                content = ""
                title = ""
                
                if "text/html" in content_type:
                    # HTML content
                    soup = BeautifulSoup(response.text, 'lxml')
                    title = soup.title.string if soup.title else ""
                    
                    if only_main_content:
                        # Try trafilatura first for better content extraction
                        extracted = trafilatura.extract(response.text, url=url)
                        if extracted:
                            content = extracted
                        else:
                            # Fallback: remove common non-content elements
                            for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                                element.decompose()
                            content = soup.get_text(separator='\n', strip=True)
                    else:
                        content = soup.get_text(separator='\n', strip=True)
                
                elif "application/json" in content_type:
                    # JSON content (e.g., API responses)
                    try:
                        json_data = response.json()
                        content = json.dumps(json_data, indent=2)
                        title = url.split("/")[-1]  # Use last part of URL as title
                    except:
                        content = response.text
                
                elif "text/plain" in content_type or "text/markdown" in content_type:
                    # Plain text or markdown
                    content = response.text
                    title = url.split("/")[-1]
                
                else:
                    # Try to extract text anyway
                    content = response.text
                    if not content:
                        # Try parsing as HTML even if content-type says otherwise
                        soup = BeautifulSoup(response.text, 'lxml')
                        title = soup.title.string if soup.title else ""
                        if only_main_content:
                            extracted = trafilatura.extract(response.text, url=url)
                            content = extracted or soup.get_text(separator='\n', strip=True)
                        else:
                            content = soup.get_text(separator='\n', strip=True)
                
                if not content or len(content.strip()) < 50:
                    raise IndexingError(f"Insufficient content extracted from URL: {url}")
                
                # Chunk the content
                text_chunks = self._chunk_text(content)
                
                chunks = []
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunks.append({
                        "content": chunk_text,
                        "file_path": url,
                        "metadata": {
                            "page_title": title[:500] if title else "",
                            "url": url,
                            "content_type": content_type
                        }
                    })
                
                # Store chunks
                await self._store_chunks(
                    source_id=source_id,
                    source_type="url",
                    source_url=url,
                    chunks=chunks,
                    metadata={
                        "title": title[:500] if title else "",
                        "content_type": content_type
                    }
                )
                
                # Update status
                self.indexing_status[source_id].update({
                    "status": "completed",
                    "progress": 100,
                    "completed_at": datetime.now().isoformat(),
                    "page_count": 1,
                    "chunk_count": len(chunks)
                })
                
                return {
                    "source_id": source_id,
                    "status": "completed",
                    "url": url,
                    "title": title,
                    "chunks_created": len(chunks),
                    "content_length": len(content)
                }
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} when fetching {url}"
            self.indexing_status[source_id].update({
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            })
            raise APIError(error_msg, status_code=e.response.status_code)
        except Exception as e:
            error_msg = f"Error indexing URL {url}: {str(e)}"
            self.indexing_status[source_id].update({
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            })
            raise IndexingError(error_msg)
    
    async def delete_indexed_source(self, source_id: str) -> bool:
        """Delete all chunks for an indexed source"""
        try:
            # Query Pinecone to find all chunks for this source
            # Note: Pinecone doesn't support querying by metadata directly in all cases
            # We'll need to use a workaround or track chunk IDs
            
            # For now, mark as deleted in status
            if source_id in self.indexing_status:
                self.indexing_status[source_id]["status"] = "deleted"
                self.indexing_status[source_id]["deleted_at"] = datetime.now().isoformat()
            
            # In production, you'd want to:
            # 1. Query Pinecone with metadata filter to find all chunks
            # 2. Delete them using delete(ids=[...])
            # For now, we'll just mark as deleted
            
            return True
        except Exception as e:
            logger.error(f"Error deleting source {source_id}: {e}")
            return False

