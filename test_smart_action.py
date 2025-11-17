#!/usr/bin/env python3
"""
Test script for Smart Action orchestration
Demonstrates frictionless context management
"""
import asyncio
from main import InfiniteContextMCP

async def test_smart_save():
    """Test smart save action"""
    print("🧪 Testing Smart Save Action...\n")
    
    mcp = InfiniteContextMCP()
    
    result = await mcp.smart_action({
        "request": "save this conversation about integrating smart action orchestration into the MCP server",
        "conversation_context": "We discussed creating an intelligent orchestration layer that automatically routes requests and combines tools for frictionless context management."
    })
    
    print(result.text)
    print("\n" + "=" * 60 + "\n")

async def test_smart_search():
    """Test smart search action"""
    print("🧪 Testing Smart Search Action...\n")
    
    mcp = InfiniteContextMCP()
    
    result = await mcp.smart_action({
        "request": "find past discussions about query understanding"
    })
    
    print(result.text)
    print("\n" + "=" * 60 + "\n")

async def test_smart_stats():
    """Test smart stats action"""
    print("🧪 Testing Smart Stats Action...\n")
    
    mcp = InfiniteContextMCP()
    
    result = await mcp.smart_action({
        "request": "show me memory statistics"
    })
    
    print(result.text)
    print("\n" + "=" * 60 + "\n")

async def test_multi_action():
    """Test multi-action orchestration"""
    print("🧪 Testing Multi-Action Orchestration...\n")
    
    mcp = InfiniteContextMCP()
    
    result = await mcp.smart_action({
        "request": "search for context about MCP and show me the stats"
    })
    
    print(result.text)
    print("\n" + "=" * 60 + "\n")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Smart Action Orchestration Tests")
    print("Frictionless Context Management")
    print("=" * 60)
    print()
    
    try:
        await test_smart_save()
        await test_smart_search()
        await test_smart_stats()
        await test_multi_action()
        
        print("✅ All smart action tests completed!")
        print("\n💡 Tip: Try using smart_action with natural language requests!")
        print("   Examples:")
        print("   - 'save this conversation about X'")
        print("   - 'find discussions about Y'")
        print("   - 'show me memory stats'")
        print("   - 'search for Z and save the top 3'")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

