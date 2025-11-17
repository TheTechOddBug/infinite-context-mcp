# Quick MCP Test Examples

## Using Enhanced Search

### Example 1: Basic Search
```
Use enhanced_search to find past discussions about query understanding
```

### Example 2: Search with Auto-Refine (default)
```
Search for "MCP server configuration" using enhanced_search
```

### Example 3: Search with Specific Parameters
```
Use enhanced_search with query "smart action", top_k 5, min_relevance 0.6
```

## Using Smart Action

### Example 1: Simple Search
```
Use smart_action to find conversations about Pinecone
```

### Example 2: Save Conversation
```
Use smart_action to save this conversation about testing the MCP
```

### Example 3: Multi-Action
```
Use smart_action to search for "query understanding" and show me the stats
```

## Using Query Classification

### Example 1: Classify a Query
```
Use classify_query to understand what type of query "how do I save context?" is
```

## Using Query Rewrites

### Example 1: Generate Rewrites
```
Use rewrite_query to generate alternatives for "MCP configuration"
```

## Quick Test Commands

Copy and paste these directly into Cursor/Claude:

1. **Test Enhanced Search:**
   ```
   Use enhanced_search to find discussions about enhanced search features
   ```

2. **Test Auto-Refine (poor query):**
   ```
   Search for "xyz123testquery" using enhanced_search
   ```

3. **Test Recommendations:**
   ```
   Use enhanced_search with query "context management" and top_k 3
   ```

4. **Test Smart Action Save:**
   ```
   Use smart_action to save this conversation about testing MCP queries
   ```

5. **Test Classification:**
   ```
   Classify the query "find past conversations about MCP"
   ```

6. **Test Rewrites:**
   ```
   Generate query rewrites for "vector storage"
   ```

7. **Test Memory Stats:**
   ```
   Show me memory statistics
   ```

## Expected Outputs

### Enhanced Search Output Should Show:
- ✅ Query classification (type, intent, categories)
- ✅ Generated rewrites
- ✅ Search results with relevance scores
- ✅ Auto-refined results (if initial results were poor)
- ✅ Follow-up query recommendations

### Smart Action Output Should Show:
- ✅ Detected action
- ✅ Extracted parameters
- ✅ Results from the action
- ✅ For saves: Conversation analysis (if context provided)

