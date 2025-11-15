#!/usr/bin/env python3
"""
Helper script to manually capture context from conversations or notes.
This can be used to save context outside of the MCP tool interface.
"""
import asyncio
import json
import sys
from main import InfiniteContextMCP

async def capture_context_interactive():
    """Interactive context capture"""
    print("📝 Infinite Context MCP - Context Capture\n")
    print("Enter the information to save (press Ctrl+D or Ctrl+C when done):\n")
    
    try:
        # Get summary
        print("Summary (required):")
        summary = input("> ").strip()
        if not summary:
            print("❌ Summary is required!")
            return False
        
        # Get topics
        print("\nTopics (comma-separated, optional):")
        topics_input = input("> ").strip()
        topics = [t.strip() for t in topics_input.split(",") if t.strip()] if topics_input else []
        
        # Get key findings
        print("\nKey Findings (one per line, empty line to finish, optional):")
        findings = []
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    break
                findings.append(line)
            except (EOFError, KeyboardInterrupt):
                break
        
        # Get structured data
        print("\nStructured Data (JSON format, optional, empty to skip):")
        data_input = input("> ").strip()
        data = {}
        if data_input:
            try:
                data = json.loads(data_input)
            except json.JSONDecodeError:
                print("⚠️  Invalid JSON, skipping structured data")
        
        # Initialize MCP and save
        print("\n💾 Saving context...")
        mcp = InfiniteContextMCP()
        
        args = {
            "summary": summary,
            "topics": topics,
            "key_findings": findings,
            "data": data
        }
        
        result = await mcp.save_context(args)
        print(f"\n{result.text}")
        print("\n✅ Context saved successfully!")
        
        return True
        
    except (EOFError, KeyboardInterrupt):
        print("\n\n❌ Cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

async def capture_context_from_args(summary, topics=None, findings=None, data=None):
    """Capture context from command line arguments"""
    mcp = InfiniteContextMCP()
    
    args = {
        "summary": summary,
        "topics": topics or [],
        "key_findings": findings or [],
        "data": data or {}
    }
    
    result = await mcp.save_context(args)
    print(result.text)
    return result

async def search_context(query, top_k=3):
    """Search for context"""
    mcp = InfiniteContextMCP()
    
    args = {
        "query": query,
        "top_k": top_k
    }
    
    result = await mcp.search_context(args)
    print(result.text)
    return result

async def show_stats():
    """Show memory statistics"""
    mcp = InfiniteContextMCP()
    result = await mcp.get_memory_stats()
    print(result.text)
    return result

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        # Interactive mode
        success = asyncio.run(capture_context_interactive())
        sys.exit(0 if success else 1)
    
    command = sys.argv[1]
    
    if command == "save":
        if len(sys.argv) < 3:
            print("Usage: python capture_context.py save 'Summary text' [topics] [findings]")
            sys.exit(1)
        
        summary = sys.argv[2]
        topics = sys.argv[3].split(",") if len(sys.argv) > 3 else []
        findings = sys.argv[4].split("|") if len(sys.argv) > 4 else []
        
        asyncio.run(capture_context_from_args(summary, topics, findings))
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python capture_context.py search 'query' [top_k]")
            sys.exit(1)
        
        query = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        asyncio.run(search_context(query, top_k))
    
    elif command == "stats":
        asyncio.run(show_stats())
    
    else:
        print("Usage:")
        print("  python capture_context.py              # Interactive mode")
        print("  python capture_context.py save 'Summary' [topics] [findings]")
        print("  python capture_context.py search 'query' [top_k]")
        print("  python capture_context.py stats")
        sys.exit(1)

if __name__ == "__main__":
    main()

