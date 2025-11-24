#!/usr/bin/env python3
"""
Debug script to test the save operation
"""
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def debug_save():
    print("🔍 Debugging save operation...")
    
    # Check environment variables
    print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    print(f"PINECONE_API_KEY: {'✅ Set' if os.getenv('PINECONE_API_KEY') else '❌ Missing'}")
    
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('PINECONE_API_KEY'):
        print("❌ Missing API keys. Please check your .env file.")
        return
    
    try:
        from main import InfiniteContextMCP
        mcp = InfiniteContextMCP()
        print("✅ MCP server initialized")
        
        # Test save with minimal data
        test_args = {
            "summary": "Debug test save operation",
            "topics": ["debug", "test"],
            "key_findings": ["Testing save functionality"],
            "data": {"test": True}
        }
        
        print("💾 Attempting to save context...")
        result = await mcp.save_context(test_args)
        print(f"✅ Save result: {result.text}")
        
        # Check index stats
        stats = mcp.index.describe_index_stats()
        print(f"📊 Index stats after save: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_save())















