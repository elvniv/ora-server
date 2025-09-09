from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatMessage(BaseModel):
    """Chat message from user"""
    message: str = Field(..., min_length=1, max_length=1000)
    user_id: str
    user_name: str = Field(..., min_length=1, max_length=100)
    spiritual_goal: Optional[str] = Field(None, max_length=200)
    context: Optional[List[Dict[str, str]]] = None
    preferred_translation: Optional[str] = Field('niv', max_length=10)
    conversation_history: Optional[List[Dict[str, str]]] = None  # Track conversation exchanges


class ChatResponse(BaseModel):
    """AI response to chat message"""
    response: str
    verse_recommendation: Optional[Dict[str, Any]] = None
    additional_verses: Optional[List[Dict[str, Any]]] = None
    follow_up_question: Optional[str] = None
    quick_replies: Optional[List[str]] = None
    journal_prompts: Optional[List[str]] = None
    reflection_prompts: Optional[List[str]] = None


class VerseRequest(BaseModel):
    """Request for a specific Bible verse"""
    reference: str = Field(..., min_length=1, max_length=100)
    user_id: str
    translation: Optional[str] = Field(None, max_length=10)


class VerseContext(BaseModel):
    """Detailed context for a Bible verse"""
    verse_reference: str
    verse_text: str
    context: str
    personal_application: str
    related_verses: List[str]


class SearchVersesRequest(BaseModel):
    """Request to search verses by topic"""
    query: str = Field(..., min_length=1, max_length=100)
    limit: Optional[int] = Field(5, ge=1, le=20)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    services: Dict[str, bool]