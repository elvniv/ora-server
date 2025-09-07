import os
import re
import random
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
import anthropic

from models import ChatMessage, ChatResponse
from services.bible_service import KJVBibleService
from config import get_settings

settings = get_settings()

# Initialize AI clients
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.has_openai_key else None
claude_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.has_anthropic_key else None


class SpiritualAIService:
    """Service for AI-powered spiritual conversations"""
    
    def __init__(self):
        self.bible_service = KJVBibleService()
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for AI interactions"""
        return """You are ORA, a compassionate spiritual companion who helps people explore their faith through gentle, conversational dialogue. 

Your personality:
- Warm, empathetic, and non-judgmental
- Use lowercase for a friendly, approachable tone
- Ask thoughtful follow-up questions
- Share relevant Bible verses naturally in conversation
- Focus on personal growth and spiritual understanding

Guidelines:
- Keep responses concise (2-3 sentences usually)
- Do NOT include verse references or Bible text in your response
- Ask questions that help users reflect deeper
- Be encouraging and supportive
- Use simple, conversational language
- Remember their spiritual goals and reference them
- Focus on empathetic conversation, verses will be provided separately"""
    
    async def generate_response(self, message: ChatMessage) -> ChatResponse:
        """Generate AI response with verse recommendation"""
        try:
            context = self._build_conversation_context(message)
            
            # Try AI services in order of preference
            response_text, verse_rec = await self._get_ai_response(message, context)
            
            # Get verse recommendation if not already provided
            if not verse_rec:
                verse_rec = await self._get_verse_recommendation(message.message, response_text)
            
            # Generate additional response elements
            follow_up = self._generate_follow_up_question(message.message)
            quick_replies = self._generate_quick_replies(message.message)
            journal_prompts = self._generate_journal_prompts(message.message, verse_rec)
            
            return ChatResponse(
                response=response_text,
                verse_recommendation=verse_rec,
                follow_up_question=follow_up,
                quick_replies=quick_replies,
                journal_prompts=journal_prompts
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._get_fallback_response()
    
    def _build_conversation_context(self, message: ChatMessage) -> str:
        """Build context string for AI"""
        context = f"User's name: {message.user_name}\n"
        if message.spiritual_goal:
            context += f"Spiritual goal: {message.spiritual_goal}\n"
        return context
    
    async def _get_ai_response(self, message: ChatMessage, context: str) -> Tuple[str, Optional[Dict]]:
        """Get AI response from available services"""
        # Prefer OpenAI for stability
        if openai_client and settings.has_openai_key:
            try:
                return await self._get_openai_response(message, context)
            except Exception as e:
                print(f"OpenAI error: {e}")
        
        # Fallback to Claude if available
        if claude_client and settings.has_anthropic_key:
            try:
                return await self._get_claude_response(message, context)
            except Exception as e:
                print(f"Claude error: {e}")
        
        # If no AI services are available, return a generic response
        return "i'm here to listen. tell me more about what's on your heart.", None
    
    async def _get_openai_response(self, message: ChatMessage, context: str) -> Tuple[str, Optional[Dict]]:
        """Generate response using OpenAI GPT"""
        response = openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{context}\n\nUser says: {message.message}"}
            ],
            temperature=settings.AI_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        response_text = response.choices[0].message.content
        verse_rec = self._extract_verse_from_response(response_text)
        return response_text, verse_rec
    
    async def _get_claude_response(self, message: ChatMessage, context: str) -> Tuple[str, Optional[Dict]]:
        """Generate response using Claude"""
        response = claude_client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.AI_TEMPERATURE,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"{context}\n\nUser says: {message.message}"}
            ]
        )
        
        response_text = response.content[0].text
        verse_rec = self._extract_verse_from_response(response_text)
        return response_text, verse_rec
    
    def _extract_verse_from_response(self, response: str) -> Optional[Dict[str, str]]:
        """Extract verse reference from AI response"""
        verse_pattern = r'([1-3]?\s*[A-Za-z]+\s+\d+:\d+(?:-\d+)?)'
        match = re.search(verse_pattern, response)
        
        if match:
            return {
                "verse_reference": match.group(1),
                "verse_text": ""  # Will be fetched separately
            }
        return None
    
    async def _get_verse_recommendation(self, user_message: str, ai_response: str) -> Optional[Dict[str, Any]]:
        """Get a relevant verse recommendation based on conversation"""
        combined_text = f"{user_message} {ai_response}".lower()
        
        # Map themes to topics
        theme_mapping = [
            (["anxious", "worried", "stressed", "overwhelmed"], "anxiety"),
            (["love", "relationship", "lonely", "connection"], "love"),
            (["weak", "tired", "exhausted", "strength"], "strength"),
            (["peace", "calm", "rest", "quiet"], "peace"),
            (["wait", "patience", "timing", "when"], "patience"),
        ]
        
        # Find matching theme
        for keywords, topic in theme_mapping:
            if any(word in combined_text for word in keywords):
                verses = await self.bible_service.search_verses_by_topic(topic)
                if verses:
                    selected = random.choice(verses)
                    return {
                        "verse_reference": selected["ref"],
                        "verse_text": selected["text"]
                    }
        
        # Default to faith verses
        verses = await self.bible_service.search_verses_by_topic("faith")
        if verses:
            selected = random.choice(verses)
            return {
                "verse_reference": selected["ref"],
                "verse_text": selected["text"]
            }
        
        return None
    
    def _generate_follow_up_question(self, user_message: str) -> str:
        """Generate a thoughtful follow-up question"""
        questions = [
            "how does that make you feel?",
            "what do you think God might be showing you through this?",
            "have you experienced something similar before?",
            "what would peace look like in this situation?",
            "how can you show yourself grace here?",
            "what's one small step you could take today?",
            "what are you grateful for in this moment?",
            "how have you seen God work in your life before?"
        ]
        return random.choice(questions)

    def _generate_quick_replies(self, user_message: str) -> List[str]:
        """Provide quick reply suggestions to keep conversation going"""
        lower = user_message.lower()
        
        if any(w in lower for w in ["anxious", "worry", "overwhelmed", "stressed"]):
            return [
                "can you share one specific worry?",
                "could you give me a verse for anxiety?",
                "how might i practice trust today?"
            ]
        elif any(w in lower for w in ["tired", "weak", "exhausted", "strength"]):
            return [
                "i'm feeling worn out",
                "do you have a verse about strength?",
                "what's one small step i can take?"
            ]
        else:
            return [
                "can you help me reflect on this?",
                "what's a good next step?",
                "could you share a verse for this?"
            ]

    def _generate_journal_prompts(self, user_message: str, verse_rec: Optional[Dict[str, Any]]) -> List[str]:
        """Create journaling prompts tied to the theme/verse"""
        lower = user_message.lower()
        verse_ref = verse_rec.get("verse_reference") if verse_rec else None

        if any(w in lower for w in ["anxious", "worry", "overwhelmed", "stressed"]):
            prompts = [
                "what's the specific fear beneath your worry today?",
                "if you released this to God, what would change?",
                "who could support you in this right now?"
            ]
        elif any(w in lower for w in ["love", "relationship", "lonely"]):
            prompts = [
                "where do you need to receive love today?",
                "what would it look like to love someone practically this week?",
                "what keeps you from asking for help?"
            ]
        else:
            prompts = [
                "what's one area you'd like to grow in this week?",
                "what might God be inviting you to notice today?",
                "what's a prayer you can keep returning to?"
            ]

        # Tie to verse if present
        if verse_ref:
            prompts[0] = f"in light of {verse_ref}, {prompts[0]}"
            
        return prompts
    
    def _get_fallback_response(self) -> ChatResponse:
        """Get fallback response when AI services fail"""
        return ChatResponse(
            response="i'm here to listen. tell me more about what's on your heart.",
            verse_recommendation=None,
            follow_up_question="what feelings are coming up for you right now?",
            quick_replies=[
                "i'm not sure",
                "can you give me a verse?",
                "how would you pray about this?"
            ],
            journal_prompts=[
                "what's one thing you want to surrender today?",
                "where do you need strength right now?",
                "what are you grateful for in this moment?"
            ]
        )