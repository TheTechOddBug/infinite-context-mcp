#!/usr/bin/env python3
"""
Test script to verify the save_context tool works correctly
"""
import asyncio
import os
from main import InfiniteContextMCP

async def test_save_context():
    """Test the save_context tool"""
    print("🧪 Testing save_context tool...")
    
    try:
        # Initialize the MCP server
        mcp = InfiniteContextMCP()
        print("✅ MCP server initialized successfully")
        
        # Test save_context
        print("\n💾 Testing save_context...")
        test_args = {
            "summary": "Test conversation about MCP server setup",
            "topics": ["MCP", "Python", "Testing"],
            "key_findings": ["Server works correctly", "Dependencies resolved"],
            "data": {"test": True, "version": "1.0"}
        }
        
        result = await mcp.save_context(test_args)
        print(f"✅ Save context result: {result.text}")
        
        # Test memory stats
        print("\n📊 Testing memory stats...")
        stats_result = await mcp.get_memory_stats()
        print(f"✅ Memory stats: {stats_result.text}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing save_context: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_save_context())
    exit(0 if success else 1)















