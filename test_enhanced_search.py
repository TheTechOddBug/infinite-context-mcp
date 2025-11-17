#!/usr/bin/env python3
"""
Test script for enhanced_search with auto-refine and recommendations
"""
import asyncio
from main import InfiniteContextMCP

async def test_enhanced_search():
    """Test enhanced_search with auto-refine and recommendations"""
    print("🧪 Testing Enhanced Search with Auto-Refine & Recommendations\n")
    print("=" * 60)
    
    mcp = InfiniteContextMCP()
    
    # First, save some test context
    print("\n📝 Step 1: Saving test context...")
    await mcp.save_context({
        "summary": "Enhanced search implementation with auto-refine and follow-up recommendations",
        "topics": ["Enhanced Search", "Auto-Refine", "Query Understanding", "Recommendations"],
        "key_findings": [
            "Auto-refine automatically tries best rewrite if results are poor",
            "Follow-up recommendations suggest better queries",
            "Uses LLM to generate intelligent suggestions"
        ],
        "data": {"feature": "enhanced_search", "version": "2.0"}
    })
    print("✅ Test context saved\n")
    
    # Test 1: Good query (should find results)
    print("=" * 60)
    print("🔍 Test 1: Good Query (should find results)")
    print("=" * 60)
    result1 = await mcp.enhanced_search({
        "query": "enhanced search",
        "top_k": 3
    })
    print(result1.text)
    
    # Test 2: Poor query (should auto-refine)
    print("\n" + "=" * 60)
    print("🔍 Test 2: Poor Query (should auto-refine)")
    print("=" * 60)
    result2 = await mcp.enhanced_search({
        "query": "xyz123badquery",
        "top_k": 3
    })
    print(result2.text)
    
    # Test 3: Query with recommendations
    print("\n" + "=" * 60)
    print("🔍 Test 3: Query to see recommendations")
    print("=" * 60)
    result3 = await mcp.enhanced_search({
        "query": "query understanding",
        "top_k": 5
    })
    print(result3.text)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_enhanced_search())

