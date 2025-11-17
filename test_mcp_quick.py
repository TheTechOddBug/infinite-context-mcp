#!/usr/bin/env python3
"""
Quick test script to verify MCP server functionality
"""
import asyncio
import os
from main import InfiniteContextMCP

async def quick_test():
    """Quick test of MCP functionality"""
    print("🧪 Testing MCP Server...\n")
    
    try:
        # Initialize MCP
        print("1️⃣ Initializing MCP server...")
        mcp = InfiniteContextMCP()
        print("   ✅ MCP server initialized\n")
        
        # Test 1: Get memory stats
        print("2️⃣ Testing get_memory_stats...")
        stats_result = await mcp.get_memory_stats()
        print(f"   ✅ Stats retrieved:\n{stats_result.text[:200]}...\n")
        
        # Test 2: Classify a query
        print("3️⃣ Testing classify_query...")
        classify_result = await mcp.classify_query({
            "query": "how do I save context?"
        })
        print(f"   ✅ Query classified:\n{classify_result.text[:200]}...\n")
        
        # Test 3: Generate query rewrites
        print("4️⃣ Testing rewrite_query...")
        rewrite_result = await mcp.rewrite_query({
            "query": "MCP server",
            "rewrite_types": ["synonym", "broader"]
        })
        print(f"   ✅ Rewrites generated:\n{rewrite_result.text[:200]}...\n")
        
        # Test 4: Smart action (save)
        print("5️⃣ Testing smart_action (save)...")
        save_result = await mcp.smart_action({
            "request": "save this test conversation about testing the MCP server",
            "conversation_context": "We are testing the MCP server functionality including smart_action orchestration."
        })
        print(f"   ✅ Smart action completed:\n{save_result.text[:300]}...\n")
        
        # Test 5: Smart action (search)
        print("6️⃣ Testing smart_action (search)...")
        search_result = await mcp.smart_action({
            "request": "find past discussions about MCP"
        })
        print(f"   ✅ Smart search completed:\n{search_result.text[:300]}...\n")
        
        print("=" * 60)
        print("✅ All tests passed! MCP server is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("   Please set them in your .env file")
        exit(1)
    
    success = asyncio.run(quick_test())
    exit(0 if success else 1)

