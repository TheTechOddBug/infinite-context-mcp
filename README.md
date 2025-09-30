# Infinite Context Claude

A sophisticated chat system that uses Claude with infinite context through intelligent conversation compression and vector storage using Pinecone.

## Features

- **Infinite Context**: Automatically compresses long conversations while preserving important information
- **Vector Search**: Uses Pinecone for semantic search across conversation history
- **Smart Compression**: Uses Claude to intelligently summarize conversations
- **High-Quality Embeddings**: Uses OpenAI's text-embedding-3-large model

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy the example environment file and fill in your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` and add your actual API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_custom_index_name_here  # Optional, defaults to "infinite-context-index"
```

### 3. Get API Keys

- **Anthropic**: Get your API key from [Anthropic Console](https://console.anthropic.com/)
- **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Pinecone**: Get your API key from [Pinecone Console](https://app.pinecone.io/)

### 4. Run the Application

```bash
python main.py
```

## How It Works

1. **Context Management**: Monitors token usage and compresses conversations when they approach Claude's limit
2. **Intelligent Compression**: Uses Claude to extract key information including:
   - Founder data and company associations
   - Duplicate detection across systems
   - Decisions and actions taken
   - Statistics and metrics
3. **Vector Storage**: Stores compressed conversations in Pinecone for semantic search
4. **Context Retrieval**: Searches past conversations to provide relevant context for new questions

## Usage

Once running, simply type your messages and press Enter. The system will:

- Track context usage
- Automatically compress long conversations
- Search relevant past context
- Provide comprehensive responses

Type `quit`, `exit`, or `bye` to end the conversation.

## Configuration

Key parameters you can adjust in the code:

- `MAX_TOKENS`: Claude's context limit (default: 150,000)
- `THRESHOLD`: Compression trigger percentage (default: 75%)
- `PINECONE_INDEX_NAME`: Pinecone index name (default: "infinite-context-index")
- `embedding_model`: OpenAI embedding model (default: "text-embedding-3-large")

## Security

**Important Security Notes:**

- Never commit your `.env` file to version control
- The `.gitignore` file is configured to exclude sensitive files
- All API keys are loaded from environment variables
- The Pinecone index name is configurable via environment variable
- Use unique index names for different projects to avoid data conflicts

## Troubleshooting

- **Missing API Keys**: Make sure all three API keys are set in your `.env` file
- **Pinecone Index**: The system will automatically create the Pinecone index on first run
- **Rate Limits**: The system includes caching to minimize API calls
