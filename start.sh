#!/bin/bash

# ORA Backend Startup Script

echo "🚀 Starting ORA Spiritual Conversations API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before running the server."
    echo "   - Add your OpenAI API key"
    echo "   - Add your Anthropic API key (optional)"
    exit 1
fi

# Start the FastAPI server
echo "✨ Starting FastAPI server on http://localhost:8000..."
echo "📖 API Documentation available at http://localhost:8000/docs"
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000