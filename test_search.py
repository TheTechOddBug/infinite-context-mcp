#!/usr/bin/env python3
"""
Test script to verify the search_context tool works correctly
"""
import asyncio
from main import InfiniteContextMCP

async def test_search_context():
    """Test the search_context tool"""
    print("🧪 Testing search_context tool...")
    
    try:
        # Initialize the MCP server
        mcp = InfiniteContextMCP()
        print("✅ MCP server initialized successfully")
        
        # Test search_context
        print("\n🔍 Testing search_context...")
        test_args = {
            "query": "MCP server setup",
            "top_k": 5
        }
        
        result = await mcp.search_context(test_args)
        print(f"✅ Search result:\n{result.text}")
        
        print("\n✅ Search test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing search_context: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_search_context())
    exit(0 if success else 1)






