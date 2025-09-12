import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables FIRST
load_dotenv(override=False)

from models import (
    ChatMessage, ChatResponse, VerseRequest, VerseContext, SearchVersesRequest
)
from services.bible_service import BibleService
from services.ai_service import SpiritualAIService
from config import get_settings

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
bible_service = BibleService()
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

@app.post("/verse/explain")
async def explain_verse_context(request: dict):
    """AI-powered verse context explanation"""
    try:
        verse_reference = request.get('verse_reference')
        verse_text = request.get('verse_text')
        user_question = request.get('user_question', '')
        
        if not verse_reference or not verse_text:
            raise HTTPException(status_code=400, detail="Verse reference and text are required")
        
        # Create a context-focused prompt for the AI
        explanation_prompt = f"""
        As a biblical scholar and pastor, please explain the context and meaning of this verse:
        
        Verse: {verse_reference}
        Text: "{verse_text}"
        
        {f"User's specific question: {user_question}" if user_question else ""}
        
        Please provide:
        1. Historical and cultural context
        2. What this verse means in everyday language
        3. How it applies to modern life
        4. Any important theological insights
        
        Keep the explanation clear, practical, and encouraging. Avoid overly academic language.
        """
        
        # Use the AI service to generate the explanation
        response = await ai_service.generate_contextual_explanation(explanation_prompt)
        
        return {
            "verse_reference": verse_reference,
            "verse_text": verse_text,
            "explanation": response,
            "success": True
        }
        
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

@app.post("/action/{action_type}")
async def handle_alternative_action(action_type: str, request: dict):
    """Handle alternative action requests (prayer, breathing, etc.)"""
    import random
    user_message = request.get('user_message', '')
    
    try:
        if action_type == "prayer":
            prayers = {
                "peace": "Lord, grant me peace that surpasses understanding. Calm my anxious heart and help me trust in your perfect plan.",
                "hope": "God, when darkness feels overwhelming, remind me that you are my light. Fill me with hope that comes from knowing you.",
                "patience": "Father, help me respond with patience and grace. Teach me to pause, breathe, and choose love over anger.",
                "general": "God, be near to me in this moment. Guide my thoughts and guard my heart. Help me feel your presence."
            }
            prayer_type = "peace" if "anxious" in user_message.lower() else "hope" if "sad" in user_message.lower() else "patience" if "angry" in user_message.lower() else "general"
            return {"content": prayers[prayer_type], "type": "prayer"}
            
        elif action_type == "breathing":
            return {
                "content": "Let's breathe together. Inhale slowly for 4 counts... hold for 4... exhale for 6. God is with you in this breath. You are safe. Repeat as needed.",
                "type": "breathing_exercise"
            }
            
        elif action_type == "affirmation":
            affirmations = [
                "You are deeply loved by God, exactly as you are.",
                "This feeling will pass. You have weathered storms before.",
                "God's strength is made perfect in your weakness.",
                "You are not alone in this struggle."
            ]
            return {"content": random.choice(affirmations), "type": "affirmation"}
            
        elif action_type == "reframe":
            return {
                "content": "What if this situation is an opportunity for growth? What might God be teaching you through this challenge? Sometimes our struggles become our greatest teachers.",
                "type": "reframe"
            }
        
        elif action_type == "another_verse":
            # Get another verse recommendation
            user_id = request.get('user_id', 'anonymous')
            from models import ChatMessage
            chat_msg = ChatMessage(
                message=user_message or "I need encouragement",
                user_id=user_id,
                user_name="User",
                spiritual_goal="encouragement"
            )
            
            # Use the v2 AI service for verse recommendations
            from services.ai_service_v2 import FriendlyAIService
            ai_service = FriendlyAIService()
            verse_rec = await ai_service._get_verse_recommendation(
                chat_msg.message, user_id, "niv"
            )
            
            if verse_rec:
                return {
                    "content": f"{verse_rec['verse_reference']}: {verse_rec['verse_text']}",
                    "explanation": verse_rec.get('explanation', 'Here\'s another verse that came to mind'),
                    "type": "verse"
                }
            else:
                return {
                    "content": "Psalm 46:1: God is our refuge and strength, an ever-present help in trouble.",
                    "explanation": "here's a verse that always brings me comfort",
                    "type": "verse"
                }
        
        elif action_type == "different_angle":
            # Generate a different perspective on the same issue
            from services.ai_service_v2 import FriendlyAIService
            ai_service = FriendlyAIService()
            
            # Create a reframing prompt
            reframe_message = f"Help me see this differently: {user_message}"
            user_id = request.get('user_id', 'anonymous')
            
            from models import ChatMessage
            chat_msg = ChatMessage(
                message=reframe_message,
                user_id=user_id,
                user_name="User",
                spiritual_goal="perspective"
            )
            
            # Get a response with a different angle
            try:
                response = await ai_service.generate_response(chat_msg)
                return {
                    "content": response.response,
                    "type": "different_angle"
                }
            except Exception:
                # Fallback response
                angles = [
                    "you know what? sometimes the struggles we face today are preparing us for something bigger tomorrow",
                    "what if this challenge is actually revealing how strong you really are?",
                    "maybe this situation is less about what's happening and more about who you're becoming through it",
                    "i wonder if there's something in this that God wants to use to help others later"
                ]
                return {
                    "content": random.choice(angles),
                    "type": "different_angle"
                }
            
        else:
            return {"content": "I'm here to support you. How can I help you process what you're feeling right now?", "type": "support"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tone/{tone_type}")
async def switch_tone(tone_type: str, request: dict):
    """Handle tone switching requests"""
    user_message = request.get('user_message', '')
    user_id = request.get('user_id', 'anonymous')
    
    try:
        if tone_type not in ['friend', 'pastor', 'coach']:
            raise HTTPException(status_code=400, detail="Invalid tone type")
        
        from models import ChatMessage
        from services.ai_service_v2 import FriendlyAIService
        
        # Create modified message with tone instruction
        tone_prompt = f"Respond to this in {tone_type} tone: {user_message}"
        
        chat_msg = ChatMessage(
            message=tone_prompt,
            user_id=user_id,
            user_name="User",
            spiritual_goal=f"{tone_type}_tone"
        )
        
        ai_service = FriendlyAIService()
        
        # Override the tone for this request
        original_tone = ai_service.tone
        ai_service.tone = tone_type
        
        try:
            response = await ai_service.generate_response(chat_msg)
            return {
                "content": response.response,
                "tone": tone_type,
                "type": "tone_switch"
            }
        finally:
            # Restore original tone
            ai_service.tone = original_tone
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)