#!/usr/bin/env python3
"""
Test script for Query Understanding features
Demonstrates the Instacart-inspired QU capabilities
"""
import asyncio
from main import InfiniteContextMCP

async def test_query_classification():
    """Test query classification"""
    print("🧪 Testing Query Classification...\n")
    
    mcp = InfiniteContextMCP()
    
    test_queries = [
        "find conversations about MCP server setup",
        "how do I save context to Pinecone?",
        "save this conversation about query understanding",
        "what is the difference between search and enhanced search?"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        result = await mcp.classify_query({"query": query})
        print(result.text)
        print("-" * 60)
        print()

async def test_query_rewrites():
    """Test query rewrites"""
    print("🧪 Testing Query Rewrites...\n")
    
    mcp = InfiniteContextMCP()
    
    test_queries = [
        "MCP configuration",
        "Pinecone vector storage",
        "context compression"
    ]
    
    for query in test_queries:
        print(f"Original Query: '{query}'")
        result = await mcp.rewrite_query({
            "query": query,
            "rewrite_types": ["synonym", "broader", "expansion"]
        })
        print(result.text)
        print("-" * 60)
        print()

async def test_enhanced_search():
    """Test enhanced search with all QU features"""
    print("🧪 Testing Enhanced Search...\n")
    
    mcp = InfiniteContextMCP()
    
    # First, save some test context
    print("📝 Saving test context...")
    await mcp.save_context({
        "summary": "MCP server setup with Pinecone integration for vector storage",
        "topics": ["MCP", "Pinecone", "Vector Storage", "Python"],
        "key_findings": [
            "Used OpenAI text-embedding-3-large for embeddings",
            "Implemented session-based chunking",
            "Created secure environment variable configuration"
        ],
        "data": {
            "embedding_model": "text-embedding-3-large",
            "vector_dimension": 1024
        }
    })
    print("✅ Test context saved\n")
    
    # Now test enhanced search
    test_queries = [
        "MCP server configuration",
        "vector embeddings storage",
        "context management system"
    ]
    
    for query in test_queries:
        print(f"Enhanced Search: '{query}'")
        result = await mcp.enhanced_search({
            "query": query,
            "top_k": 3,
            "use_rewrites": True,
            "min_relevance": 0.6
        })
        print(result.text)
        print("-" * 60)
        print()

async def test_cache_stats():
    """Test cache statistics"""
    print("🧪 Testing Cache Statistics...\n")
    
    mcp = InfiniteContextMCP()
    
    # Run some queries to populate cache
    print("Running queries to populate cache...")
    await mcp.classify_query({"query": "test query 1"})
    await mcp.classify_query({"query": "test query 2"})
    await mcp.classify_query({"query": "test query 1"})  # Should hit cache
    
    # Get stats
    stats = await mcp.get_memory_stats()
    print(stats.text)
    print()

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Query Understanding Integration Tests")
    print("Inspired by Instacart's Intent Engine")
    print("=" * 60)
    print()
    
    try:
        await test_query_classification()
        await test_query_rewrites()
        await test_enhanced_search()
        await test_cache_stats()
        
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

