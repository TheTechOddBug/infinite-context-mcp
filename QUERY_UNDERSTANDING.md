# Query Understanding Integration

This document explains how concepts from Instacart's Query Understanding (QU) system have been integrated into the Infinite Context MCP.

## Overview

Inspired by [Instacart's Intent Engine article](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac), we've enhanced the MCP with advanced query understanding capabilities that improve search quality and recall.

## Key Concepts Integrated

### 1. Context-Engineering (RAG)
**From Instacart**: "We build data pipelines that retrieve and inject Instacart-specific context, such as conversion history and catalog data, directly into the prompt."

**Our Implementation**: 
- The `QueryUnderstandingEngine` retrieves domain-specific context before processing queries
- Context includes common topics, query patterns, and relevant domain knowledge
- This context is injected into LLM prompts to ground the model in our specific use case

### 2. Query Classification
**From Instacart**: "Accurately classifying queries into our product taxonomy is essential. It directly powers recall and ranking."

**Our Implementation**:
- `classify_query` tool classifies queries into types (search, question, command, etc.)
- Determines intent and relevant categories
- Uses RAG-enhanced prompts with domain context
- Results are cached for performance (hybrid approach)

### 3. Query Rewrites
**From Instacart**: "Query rewrites are critical for improving recall, especially when the original query does not return sufficient results."

**Our Implementation**:
- `rewrite_query` tool generates multiple types of rewrites:
  - **Synonyms**: Alternative phrasings with the same meaning
  - **Broader**: More general queries that encompass the original
  - **Expansion**: Queries with related terms to improve recall
  - **Substitutes**: Alternative queries that could substitute for the original
- Uses specialized prompts with chain-of-thought reasoning
- Results are cached for frequently used queries

### 4. Post-Processing Guardrails
**From Instacart**: "We refine LLM outputs through validation layers. These guardrails filter out hallucinations and enforce alignment."

**Our Implementation**:
- `apply_guardrails` method filters search results based on:
  - Minimum relevance threshold (default: 0.7)
  - Required metadata fields validation
  - Semantic similarity checks
- Prevents low-quality or irrelevant results from being returned
- Provides transparency about what was filtered and why

### 5. Hybrid Caching System
**From Instacart**: "Live traffic is routed based on a cache-hit. High-frequency 'head' queries are served instantly with cache, while 'tail' queries are handled by a real-time model."

**Our Implementation**:
- Query classifications and rewrites are cached
- Cache hit rate is tracked and reported
- Frequently used queries are served instantly
- Reduces API calls and improves latency

## New MCP Tools

### `classify_query`
Classify a query to understand its intent and categories.

**Example**:
```json
{
  "query": "find conversations about MCP server setup",
  "context": {"domain": "software development"}
}
```

**Returns**:
- Query type (search, question, command, etc.)
- Confidence score
- Categories/topics
- Intent description

### `rewrite_query`
Generate query rewrites to improve search recall.

**Example**:
```json
{
  "query": "MCP configuration",
  "rewrite_types": ["synonym", "broader", "expansion"]
}
```

**Returns**:
- List of rewritten queries
- Type of each rewrite
- Confidence scores
- Reasoning for each rewrite

### `enhanced_search`
Comprehensive search that combines all QU techniques.

**Example**:
```json
{
  "query": "Pinecone vector storage",
  "top_k": 5,
  "use_rewrites": true,
  "min_relevance": 0.7
}
```

**Process**:
1. Classifies the query
2. Generates rewrites (if enabled)
3. Searches with original + rewritten queries
4. Applies guardrails to filter results
5. Returns top-k filtered results

**Returns**:
- Query classification
- Generated rewrites
- Guardrail filtering stats
- Filtered search results
- Cache statistics

## Architecture

```
User Query
    ↓
Query Understanding Engine
    ├─ Check Cache (Hybrid Approach)
    ├─ Context-Engineering (RAG)
    │   └─ Retrieve Domain Context
    ├─ Query Classification
    ├─ Query Rewrites
    └─ Guardrails
        └─ Filter Results
            └─ Return Top-K
```

## Benefits

1. **Better Recall**: Query rewrites help find relevant context even with imperfect queries
2. **Higher Precision**: Guardrails filter out low-quality results
3. **Faster Responses**: Caching reduces API calls for frequent queries
4. **Domain Awareness**: RAG injects domain-specific knowledge into queries
5. **Transparency**: Clear reporting of classification, rewrites, and filtering

## Usage Examples

### Basic Query Classification
```python
# In your MCP client
result = await mcp.classify_query({
    "query": "how to save context to Pinecone"
})
```

### Generate Query Rewrites
```python
result = await mcp.rewrite_query({
    "query": "MCP server",
    "rewrite_types": ["synonym", "broader"]
})
```

### Enhanced Search
```python
result = await mcp.enhanced_search({
    "query": "vector embeddings",
    "top_k": 5,
    "use_rewrites": True,
    "min_relevance": 0.75
})
```

## Configuration

You can customize the query understanding engine by modifying `query_understanding.py`:

- **Domain Context**: Add your own topics and patterns to `domain_context`
- **Cache Settings**: Adjust cache behavior in `QueryUnderstandingEngine`
- **Guardrail Thresholds**: Modify `min_relevance` defaults
- **Rewrite Types**: Customize which rewrite types are generated

## Performance Considerations

- **Caching**: Frequently used queries are cached, reducing API calls
- **Batch Processing**: Multiple queries can be processed efficiently
- **Guardrails**: Filtering happens after retrieval, so initial search may return more results than needed
- **LLM Calls**: Classification and rewrites use GPT-4o-mini for cost efficiency

## Future Enhancements

Potential improvements inspired by Instacart's approach:

1. **Fine-Tuning**: Fine-tune smaller models on domain-specific data (like Instacart's "student" model)
2. **Offline Pipeline**: Create an offline pipeline for high-frequency queries (like Instacart's "teacher")
3. **Structured Retrieval Labels**: Add structured tagging for better categorization
4. **Multi-Query Expansion**: Expand single queries into multiple parallel searches
5. **Semantic Similarity Validation**: Enhanced semantic similarity checks in guardrails

## References

- [Instacart's Intent Engine Article](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac)
- [Query Understanding Best Practices](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac)

