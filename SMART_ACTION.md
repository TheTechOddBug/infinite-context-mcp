# Smart Action Orchestration

## Overview

The `smart_action` tool is an intelligent orchestration layer that makes context management completely frictionless. Instead of manually calling individual tools, you simply describe what you want in natural language, and the system automatically:

1. **Understands your intent** using query understanding
2. **Routes to the right tools** automatically
3. **Extracts parameters** intelligently
4. **Orchestrates multiple actions** when needed
5. **Returns unified results**

## How It Works

The `smart_action` tool uses an LLM to:
- Parse your natural language request
- Classify the intent (save, search, stats, etc.)
- Extract relevant parameters
- Determine which tool(s) to call
- Execute actions in the right sequence
- Combine results intelligently

## Usage Examples

### Simple Requests

```python
# Save context - just describe what to save
smart_action({
    "request": "save this conversation about MCP server setup"
})

# Search - natural language query
smart_action({
    "request": "find past discussions about Pinecone vector storage"
})

# Get stats - simple request
smart_action({
    "request": "show me memory statistics"
})

# Classify query
smart_action({
    "request": "what type of query is 'find MCP configuration'?"
})
```

### Complex Requests

```python
# Multi-step actions
smart_action({
    "request": "search for context about query understanding and then save the results"
})

# With conversation context
smart_action({
    "request": "save this conversation",
    "conversation_context": "We discussed integrating Instacart's QU system..."
})
```

## Supported Actions

The orchestration system automatically routes to:

1. **save_context** - When you want to save something
   - Keywords: "save", "store", "remember", "capture"
   - Auto-extracts: summary, topics (via classification), key findings

2. **enhanced_search** - When you want to find something
   - Keywords: "find", "search", "look for", "retrieve", "get"
   - Uses query understanding, rewrites, and guardrails automatically

3. **get_memory_stats** - When you want statistics
   - Keywords: "stats", "statistics", "memory", "how many", "count"

4. **classify_query** - When you want to understand a query
   - Keywords: "classify", "what type", "intent", "category"

5. **rewrite_query** - When you want query alternatives
   - Keywords: "rewrite", "alternatives", "synonyms", "expand"

6. **auto_compress** - When you want to compress
   - Keywords: "compress", "summarize", "condense"

7. **multi_action** - When you want multiple things
   - Automatically detected for complex requests

## Intelligent Parameter Extraction

The system automatically extracts parameters even when not explicitly provided:

### For Save Actions
- **Summary**: Extracted from request or conversation context
- **Topics**: Auto-classified using query understanding
- **Key Findings**: Can be inferred from context
- **Data**: Structured data extracted if present

### For Search Actions
- **Query**: Extracted from natural language request
- **top_k**: Defaults to 5, can be inferred ("show me 10 results")
- **use_rewrites**: Automatically enabled for better recall
- **min_relevance**: Defaults to 0.7

## Benefits

1. **Zero Friction**: Just describe what you want, no need to know tool names
2. **Intelligent Routing**: Automatically picks the best tool for the job
3. **Parameter Inference**: Extracts parameters from natural language
4. **Multi-Action Support**: Handles complex requests with multiple steps
5. **Fallback Safety**: Falls back to enhanced_search if parsing fails
6. **Unified Results**: Returns everything in one cohesive response

## Comparison: Before vs After

### Before (Manual Tool Calls)
```python
# User needs to know tool names and parameters
await mcp.classify_query({"query": "MCP setup"})
await mcp.enhanced_search({"query": "MCP setup", "top_k": 5})
await mcp.save_context({
    "summary": "...",
    "topics": ["MCP"],
    "key_findings": []
})
```

### After (Smart Action)
```python
# Just describe what you want
await mcp.smart_action({
    "request": "find and save context about MCP setup"
})
```

## Advanced Usage

### Multi-Step Workflows

The system can handle complex workflows:

```python
smart_action({
    "request": "search for discussions about query understanding, classify the results, and save the top 3"
})
```

This automatically:
1. Searches using enhanced_search
2. Classifies the query
3. Saves the top results

### With Context

Provide conversation context for better understanding:

```python
smart_action({
    "request": "save this important discussion",
    "conversation_context": """
    We discussed integrating Instacart's Query Understanding system.
    Key points:
    - Context-engineering with RAG
    - Query classification
    - Post-processing guardrails
    """
})
```

## Error Handling

The system has robust error handling:

1. **Primary**: Uses LLM to parse and route
2. **Fallback**: If parsing fails, falls back to enhanced_search
3. **Safety**: Always returns a result, even if it's an error message

## Performance

- Uses GPT-4o-mini for fast, cost-effective orchestration
- Leverages existing query understanding cache
- Minimal overhead - just one additional LLM call for orchestration

## Best Practices

1. **Be Specific**: More specific requests lead to better routing
   - ✅ "save this conversation about MCP server configuration"
   - ❌ "save stuff"

2. **Use Natural Language**: Write as you would speak
   - ✅ "find past discussions about Pinecone"
   - ❌ "search_context query=Pinecone top_k=5"

3. **Provide Context**: Include conversation_context when available
   - Helps with parameter extraction
   - Improves understanding of intent

4. **Combine Actions**: Use multi-action for complex workflows
   - ✅ "search for X and save the results"
   - ✅ "find Y, classify it, and show stats"

## Examples in Practice

### Example 1: Simple Save
```
Request: "save this conversation about integrating query understanding"
→ Routes to: save_context
→ Auto-extracts: summary, topics (via classification)
→ Result: Context saved with intelligent topic extraction
```

### Example 2: Smart Search
```
Request: "find discussions about vector storage"
→ Routes to: enhanced_search
→ Uses: query understanding, rewrites, guardrails
→ Result: Best matching results with filtering
```

### Example 3: Multi-Action
```
Request: "search for MCP setup guides and save the top 3"
→ Routes to: multi_action
→ Executes: enhanced_search → save_context (x3)
→ Result: Searched, filtered, and saved top results
```

## Integration with Other Tools

The `smart_action` tool orchestrates all other tools:

- ✅ Uses `classify_query` for intent understanding
- ✅ Uses `enhanced_search` for intelligent search
- ✅ Uses `save_context` for storage
- ✅ Uses `get_memory_stats` for statistics
- ✅ Can chain multiple tools together

## Future Enhancements

Potential improvements:

1. **Learning**: Remember user preferences over time
2. **Suggestions**: Proactive suggestions based on context
3. **Batch Operations**: Handle multiple requests at once
4. **Custom Workflows**: User-defined action sequences
5. **Voice Commands**: Natural language voice input support

