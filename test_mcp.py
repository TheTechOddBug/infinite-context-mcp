#!/usr/bin/env python3
"""
Test script to verify the MCP server is working correctly
"""
import asyncio
import json
import sys
from main import InfiniteContextMCP

async def test_mcp():
    """Test the MCP server functionality"""
    print("🧪 Testing MCP Server...")
    
    try:
        # Initialize the MCP server
        mcp = InfiniteContextMCP()
        print("✅ MCP server initialized successfully")
        
        # Test memory stats (should work without API keys)
        print("\n📊 Testing memory stats...")
        stats_result = await mcp.get_memory_stats()
        print(f"Memory stats: {stats_result.text}")
        
        print("\n✅ MCP server test completed successfully!")
        print("\nTo use this MCP server:")
        print("1. Make sure you have a .env file with your API keys:")
        print("   - OPENAI_API_KEY=your_key_here")
        print("   - PINECONE_API_KEY=your_key_here")
        print("2. Use the mcp_config.json file to configure your MCP client")
        print("3. The server provides these tools:")
        print("   - save_context: Save conversation context to Pinecone")
        print("   - search_context: Search past conversation context")
        print("   - auto_compress: Compress and save conversations")
        print("   - get_memory_stats: Get statistics about stored memory")
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp())
    sys.exit(0 if success else 1)















