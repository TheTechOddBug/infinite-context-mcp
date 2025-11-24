#!/bin/bash
# Quick start script for ChatGPT API integration

echo "🚀 Starting Infinite Context API Server for ChatGPT..."
echo ""

cd "$(dirname "$0")"
source venv/bin/activate

echo "📡 API will be available at:"
echo "   - http://localhost:8000"
echo "   - http://localhost:8000/docs (Swagger UI)"
echo "   - http://localhost:8000/tools (OpenAI function format)"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""

python api/api_server.py

