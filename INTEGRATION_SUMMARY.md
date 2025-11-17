# Instacart Query Understanding Integration Summary

## Overview

This document summarizes how concepts from Instacart's Query Understanding (QU) system have been integrated into the Infinite Context MCP project.

**Reference Article**: [Building The Intent Engine: How Instacart is Revamping Query Understanding with LLMs](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac)

## What Was Integrated

### 1. Context-Engineering (RAG) ✅
**Instacart's Approach**: "We build data pipelines that retrieve and inject Instacart-specific context, such as conversion history and catalog data, directly into the prompt."

**Our Implementation**:
- Created `QueryUnderstandingEngine` class that retrieves domain-specific context
- Context includes common topics, query patterns, and relevant domain knowledge
- Context is injected into LLM prompts before query processing
- Located in: `query_understanding.py` → `_get_relevant_domain_context()`

### 2. Query Classification ✅
**Instacart's Approach**: "Accurately classifying queries into our product taxonomy is essential. It directly powers recall and ranking."

**Our Implementation**:
- `classify_query` tool classifies queries into types (search, question, command, etc.)
- Determines intent and relevant categories
- Uses RAG-enhanced prompts with domain context
- Results cached for performance
- Located in: `main.py` → `classify_query()` method

### 3. Query Rewrites ✅
**Instacart's Approach**: "Query rewrites are critical for improving recall... we design specialized prompts for three distinct rewrite types: Substitutes, Broader queries, and Synonyms."

**Our Implementation**:
- `rewrite_query` tool generates multiple types of rewrites:
  - Synonyms: Alternative phrasings
  - Broader: More general queries
  - Expansion: Queries with related terms
  - Substitutes: Alternative queries
- Uses specialized prompts with chain-of-thought reasoning
- Results cached for frequently used queries
- Located in: `query_understanding.py` → `generate_rewrites()`

### 4. Post-Processing Guardrails ✅
**Instacart's Approach**: "We refine LLM outputs through validation layers. These guardrails filter out hallucinations and enforce alignment."

**Our Implementation**:
- `apply_guardrails` method filters search results based on:
  - Minimum relevance threshold (default: 0.7)
  - Required metadata fields validation
  - Semantic similarity checks
- Prevents low-quality or irrelevant results
- Provides transparency about filtering
- Located in: `query_understanding.py` → `apply_guardrails()`

### 5. Hybrid Caching System ✅
**Instacart's Approach**: "Live traffic is routed based on a cache-hit. High-frequency 'head' queries are served instantly with cache, while 'tail' queries are handled by a real-time model."

**Our Implementation**:
- Query classifications and rewrites are cached in memory
- Cache hit rate tracked and reported
- Frequently used queries served instantly
- Reduces API calls and improves latency
- Located in: `query_understanding.py` → `QueryUnderstandingEngine` class

### 6. Enhanced Search ✅
**Instacart's Approach**: Combines classification, rewrites, and guardrails into a unified search experience.

**Our Implementation**:
- `enhanced_search` tool combines all QU techniques:
  1. Classifies the query
  2. Generates rewrites (if enabled)
  3. Searches with original + rewritten queries
  4. Applies guardrails to filter results
  5. Returns top-k filtered results
- Located in: `main.py` → `enhanced_search()` method

## Architecture Comparison

### Instacart's Architecture
```
Query → Classification → Rewrites → Search → Guardrails → Results
         ↓
      Cache (for head queries)
```

### Our Architecture
```
Query → QueryUnderstandingEngine
         ├─ Check Cache
         ├─ Context-Engineering (RAG)
         ├─ Classification
         ├─ Rewrites
         └─ Guardrails
             └─ Enhanced Search Results
```

## Key Differences

1. **Scale**: Instacart handles millions of queries; our system is designed for MCP context management
2. **Domain**: Instacart focuses on e-commerce/product search; we focus on conversation context
3. **Fine-Tuning**: Instacart uses fine-tuned models; we use prompt engineering (fine-tuning can be added later)
4. **Offline Pipeline**: Instacart has offline "teacher" pipeline; we use real-time processing (can be added)

## Files Added/Modified

### New Files
- `query_understanding.py` - Core Query Understanding Engine implementation
- `QUERY_UNDERSTANDING.md` - Detailed documentation
- `test_query_understanding.py` - Test suite
- `INTEGRATION_SUMMARY.md` - This file

### Modified Files
- `main.py` - Integrated QueryUnderstandingEngine and added new tools
- `README.md` - Updated with new features

## Usage Example

```python
# Classify a query
result = await mcp.classify_query({
    "query": "find conversations about MCP server setup"
})

# Generate rewrites
result = await mcp.rewrite_query({
    "query": "MCP configuration",
    "rewrite_types": ["synonym", "broader", "expansion"]
})

# Enhanced search with all features
result = await mcp.enhanced_search({
    "query": "vector embeddings",
    "top_k": 5,
    "use_rewrites": True,
    "min_relevance": 0.7
})
```

## Benefits Achieved

1. ✅ **Better Recall**: Query rewrites help find relevant context even with imperfect queries
2. ✅ **Higher Precision**: Guardrails filter out low-quality results
3. ✅ **Faster Responses**: Caching reduces API calls for frequent queries
4. ✅ **Domain Awareness**: RAG injects domain-specific knowledge
5. ✅ **Transparency**: Clear reporting of classification, rewrites, and filtering

## Future Enhancements (Inspired by Instacart)

1. **Fine-Tuning**: Fine-tune smaller models (like Llama-3-8B) on domain-specific data
2. **Offline Pipeline**: Create offline "teacher" pipeline for high-frequency queries
3. **Structured Retrieval Labels**: Add structured tagging for better categorization
4. **Multi-Query Expansion**: Expand single queries into multiple parallel searches
5. **Semantic Similarity Validation**: Enhanced semantic similarity checks in guardrails

## Testing

Run the test suite to see Query Understanding in action:

```bash
python test_query_understanding.py
```

## References

- [Instacart's Intent Engine Article](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac)
- [Query Understanding Documentation](QUERY_UNDERSTANDING.md)
- [Usage Guide](USAGE.md)

