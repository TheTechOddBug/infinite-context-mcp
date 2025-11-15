# How to Capture Context with Infinite Context MCP

## Overview

The Infinite Context MCP provides tools that allow you to save, search, and manage conversation context. Here's how to use them effectively.

## Available Tools

1. **`save_context`** - Save important conversation context
2. **`search_context`** - Search past conversations
3. **`auto_compress`** - Automatically compress long conversations
4. **`get_memory_stats`** - View memory statistics

## How to Capture Context

### Method 1: Using MCP Tools in Claude/Cursor

When the MCP server is running and connected, you can call these tools directly in your conversation:

#### Example: Saving Context

```
I want to save this conversation context. Use the save_context tool with:
- summary: "We discussed setting up the Infinite Context MCP server and securing it for GitHub"
- topics: ["MCP", "GitHub", "Security", "Python"]
- key_findings: ["Created .gitignore to protect API keys", "Made Pinecone index name configurable", "Repository pushed to GitHub successfully"]
- data: {"repository_url": "https://github.com/kayacancode/infinite-context-mcp", "status": "public"}
```

#### Example: Searching Context

```
Search for past conversations about "GitHub repository setup" using the search_context tool
```

### Method 2: Manual Context Capture Script

You can also use the Python script directly (see `capture_context.py` below).

### Method 3: Automatic Context Capture

The `auto_compress` tool can automatically compress and save conversations when they get too long.

## Best Practices

### When to Save Context

1. **After important decisions** - Save when you make key architectural or design decisions
2. **After solving complex problems** - Capture the solution and approach
3. **At conversation milestones** - Save at natural break points
4. **Before switching topics** - Preserve context before major topic changes
5. **After research sessions** - Save findings and insights

### What to Include

- **Summary**: Clear, concise summary of what was discussed
- **Topics**: Relevant tags for categorization
- **Key Findings**: Important insights, solutions, or discoveries
- **Data**: Structured information (URLs, configurations, metrics, etc.)

### Example Context Capture

```python
{
    "summary": "Built Infinite Context MCP server with Pinecone integration",
    "topics": ["MCP", "Pinecone", "Vector Storage", "Python"],
    "key_findings": [
        "Used OpenAI text-embedding-3-large for embeddings",
        "Implemented session-based chunking",
        "Created secure environment variable configuration"
    ],
    "data": {
        "embedding_model": "text-embedding-3-large",
        "vector_dimension": 1024,
        "index_metric": "cosine"
    }
}
```

## Workflow Example

1. **Start a conversation** about a topic
2. **Work through the problem** with Claude
3. **When you reach a milestone**, ask Claude to save the context:
   ```
   "Please save this conversation context using save_context. 
   Summary: [what we discussed]
   Topics: [relevant topics]
   Key findings: [important points]"
   ```
4. **Later, search for context** when you need to reference past work:
   ```
   "Search for conversations about [topic] using search_context"
   ```
5. **Review memory stats** periodically:
   ```
   "Show me memory statistics using get_memory_stats"
   ```

## Tips

- **Be specific in summaries** - Clear summaries make searching easier
- **Use consistent topics** - Similar topics help with semantic search
- **Save regularly** - Don't wait until conversations are too long
- **Include structured data** - JSON data is preserved and searchable
- **Use meaningful queries** - Natural language queries work best for search

