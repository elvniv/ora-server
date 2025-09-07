import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime

from models import (
    ChatMessage, ChatResponse, VerseRequest, VerseContext, SearchVersesRequest
)
from services.bible_service import KJVBibleService
from services.ai_service import SpiritualAIService
from config import get_settings

# Load environment variables
load_dotenv(override=False)
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="ORA Spiritual Conversations API",
    description="AI-powered spiritual guidance with KJV Bible integration",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize services
bible_service = KJVBibleService()
ai_service = SpiritualAIService()

@app.get("/")
async def root():
    return {
        "message": "ORA Spiritual Conversations API",
        "version": "1.0.0",
        "endpoints": [
            "/chat",
            "/verse/{reference}",
            "/verse/context",
            "/search/verses",
            "/versions",
            "/chapter/{book}/{chapter}",
            "/versions/download/{code}"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessage):
    """Main chat endpoint for spiritual conversations"""
    try:
        response = await ai_service.generate_response(message)
        
        # If verse was recommended, fetch the full text
        if response.verse_recommendation and response.verse_recommendation.get("verse_reference"):
            verse_data = await bible_service.get_verse(response.verse_recommendation["verse_reference"])
            response.verse_recommendation["verse_text"] = verse_data.get("text", "")
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verse/{reference}")
async def get_verse(reference: str, translation: Optional[str] = None):
    """Get a specific Bible verse by reference"""
    try:
        verse = await bible_service.get_verse(reference, translation)
        return verse
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verse/context", response_model=VerseContext)
async def get_verse_context(request: VerseRequest):
    """Get detailed context and explanation for a verse"""
    try:
        from services.context_service import get_verse_context_data
        context_data = await get_verse_context_data(request, bible_service)
        return context_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/verses")
async def search_verses(request: SearchVersesRequest):
    """Search for verses by topic or keyword"""
    try:
        verses = await bible_service.search_verses_by_topic(request.query)
        return {
            "query": request.query,
            "results": verses[:request.limit],
            "count": len(verses[:request.limit])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": settings.has_openai_key,
            "anthropic": settings.has_anthropic_key,
            "bible_api": "active"
        }
    }

@app.get("/versions")
async def list_versions():
    """List supported translations from current provider."""
    return { "versions": bible_service.get_supported_versions() }

@app.get("/chapter/{book}/{chapter}")
async def get_chapter(book: str, chapter: int, translation: Optional[str] = None):
    data = await bible_service.get_chapter(book, chapter, translation)
    return data

@app.get("/versions/download/{code}")
async def download_version(code: str):
    """Stream a whole-Bible JSON for a translation code (local preferred)."""
    # Use service internals to resolve local path or remote URL
    try:
        # Reach into service: if cached, serialize; else trigger load to get path or data
        data = await bible_service._get_version_json(code.lower())
        if not data:
            raise HTTPException(status_code=404, detail="Version not found")

        def iter_json():
            import json
            yield json.dumps(data).encode("utf-8")

        return StreamingResponse(iter_json(), media_type="application/json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)