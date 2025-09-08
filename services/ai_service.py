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
        return """You are ORA, a compassionate spiritual companion who listens deeply before offering any spiritual support.

Listening First Approach:
- ALWAYS acknowledge what they've shared with empathy
- Show you heard them by reflecting back their emotions
- Ask follow-up questions to understand more before offering spiritual insight
- Build context through conversation, don't rush to verses or solutions

Response Pattern:
1. Acknowledge their sharing ("i hear that you're feeling...")
2. Express empathy ("that sounds really difficult" or "i can sense your heart in this")
3. Ask a caring follow-up question to understand more

Examples of Good Listening:
User: "I'm stressed about work"
You: "i'm sorry you're feeling stressed. work pressure can be really overwhelming. what specifically about work is weighing on you most?"

User: "I feel alone"
You: "that sounds like such a heavy feeling to carry. loneliness can feel so isolating. what's been making you feel most disconnected lately?"

User: "I'm worried about my family"
You: "i can hear the concern in your heart for your family. that kind of worry shows how much you care about them. what specifically has you most concerned?"

Conversation Flow:
- First response: Listen + empathize + ask for more context
- Second response: Understand deeper + gentle spiritual question
- Third+ response: Can include spiritual reflection if they've shared enough

Never:
- Jump straight to spiritual advice
- Offer verses without understanding context
- Dismiss their feelings with spiritual platitudes
- Rush to solutions

Always:
- Make them feel heard first
- Ask caring questions about their experience
- Show genuine interest in their story
- Build trust through patient listening"""
    
    async def generate_response(self, message: ChatMessage) -> ChatResponse:
        """Generate AI response with verse recommendation"""
        try:
            context = self._build_conversation_context(message)
            
            # Try AI services in order of preference
            response_text, verse_rec = await self._get_ai_response(message, context)
            
            # Only provide verses after meaningful conversation, not immediately
            # Check if this is a first-time mention or needs more context
            should_include_verse = self._should_include_verse(message.message, context)
            if should_include_verse and not verse_rec:
                verse_rec = await self._get_verse_recommendation(
                    message.message, 
                    response_text, 
                    message.preferred_translation or 'niv'
                )
            
            # Get additional related verses
            additional_verses = await self._get_additional_verses(message.message, verse_rec)
            
            # Generate additional response elements
            follow_up = self._generate_follow_up_question(message.message)
            quick_replies = self._generate_quick_replies(message.message)
            journal_prompts = self._generate_journal_prompts(message.message, verse_rec)
            reflection_prompts = self._generate_reflection_prompts(message.message, verse_rec)
            
            return ChatResponse(
                response=response_text,
                verse_recommendation=verse_rec,
                additional_verses=additional_verses,
                follow_up_question=follow_up,
                quick_replies=quick_replies,
                journal_prompts=journal_prompts,
                reflection_prompts=reflection_prompts
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Re-raise to surface the error
            raise e
    
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
                # Try Claude as backup
                if claude_client and settings.has_anthropic_key:
                    try:
                        return await self._get_claude_response(message, context)
                    except Exception as e2:
                        print(f"Claude error: {e2}")
                        raise Exception("Both AI services failed")
                raise e
        
        # Try Claude if OpenAI not available
        if claude_client and settings.has_anthropic_key:
            try:
                return await self._get_claude_response(message, context)
            except Exception as e:
                print(f"Claude error: {e}")
                raise e
        
        # If no AI services are available, raise error
        raise Exception("No AI services configured")
    
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
    
    async def _get_verse_recommendation(self, user_message: str, ai_response: str, translation: str = 'niv') -> Optional[Dict[str, Any]]:
        """Get a relevant verse recommendation based on conversation in user's preferred translation"""
        combined_text = f"{user_message} {ai_response}".lower()
        
        # Map themes to verse references
        theme_verse_refs = {
            "anxiety": ["Philippians 4:6-7", "1 Peter 5:7", "Matthew 6:34"],
            "love": ["1 Corinthians 13:4-7", "1 John 4:19", "John 13:34"],
            "strength": ["Isaiah 40:31", "Philippians 4:13", "2 Timothy 1:7"],
            "peace": ["John 14:27", "Isaiah 26:3", "Philippians 4:7"],
            "patience": ["James 1:3-4", "Romans 12:12", "Galatians 6:9"],
            "faith": ["Hebrews 11:1", "2 Corinthians 5:7", "Mark 11:24"]
        }
        
        theme_keywords = [
            (["anxious", "worried", "stressed", "overwhelmed"], "anxiety"),
            (["love", "relationship", "lonely", "connection"], "love"),
            (["weak", "tired", "exhausted", "strength"], "strength"),
            (["peace", "calm", "rest", "quiet"], "peace"),
            (["wait", "patience", "timing", "when"], "patience"),
        ]
        
        # Find matching theme
        selected_theme = "faith"  # default
        for keywords, theme in theme_keywords:
            if any(word in combined_text for word in keywords):
                selected_theme = theme
                break
        
        # Get a random verse reference from the theme
        verse_refs = theme_verse_refs.get(selected_theme, theme_verse_refs["faith"])
        selected_ref = random.choice(verse_refs)
        
        # Fetch the verse in user's preferred translation
        try:
            verse_data = await self.bible_service.get_verse(selected_ref, translation)
            if verse_data and verse_data.get("text"):
                return {
                    "verse_reference": verse_data.get("reference", selected_ref),
                    "verse_text": verse_data.get("text")
                }
        except Exception as e:
            print(f"Error fetching verse {selected_ref} in {translation}: {e}")
        
        return None
    
    def _should_include_verse(self, user_message: str, context: str) -> bool:
        """Determine if we should include a verse or just listen first"""
        # Don't include verses on very first interactions or surface-level shares
        if not context or len(context.strip()) < 50:
            return False
            
        # Look for indicators that they've shared enough for meaningful verse support
        indicators_for_verse = [
            "feel", "feeling", "emotion", "heart", "soul", "pray", "prayer", 
            "God", "faith", "believe", "trust", "hope", "peace", "strength",
            "been struggling with", "going through", "dealing with", "working through"
        ]
        
        message_lower = user_message.lower()
        context_lower = context.lower() if context else ""
        
        # Check if they've shared deeper context in current message or previous context
        deep_sharing_indicators = sum(1 for indicator in indicators_for_verse if indicator in message_lower or indicator in context_lower)
        
        # Include verse if they've shared meaningful context (3+ indicators)
        return deep_sharing_indicators >= 3
    
    def _generate_follow_up_question(self, user_message: str) -> str:
        """Generate a thoughtful follow-up question for deeper reflection"""
        lower = user_message.lower()
        
        # Empathetic, context-seeking questions based on what they shared
        if any(word in lower for word in ["stressed", "stress", "overwhelmed", "pressure"]):
            questions = [
                "what specifically is creating the most stress for you right now?",
                "when did you first start feeling this overwhelm?",
                "what does the stress feel like in your body and heart?",
                "what would help you feel most supported in this?"
            ]
        elif any(word in lower for word in ["sad", "down", "depressed", "heavy"]):
            questions = [
                "what's been weighing heaviest on your heart?",
                "when did you first start feeling this way?",
                "what does this sadness feel like for you?",
                "who or what usually brings you comfort when you feel like this?"
            ]
        elif any(word in lower for word in ["worried", "worry", "anxious", "anxiety", "afraid"]):
            questions = [
                "what thoughts keep running through your mind?",
                "what are you most afraid might happen?",
                "when do you notice the worry is strongest?",
                "what helps quiet your mind when anxiety rises?"
            ]
        elif any(word in lower for word in ["angry", "frustrated", "mad", "upset"]):
            questions = [
                "what happened that stirred up these feelings?",
                "what's underneath the anger - hurt, disappointment, fear?",
                "how long have you been carrying this frustration?",
                "what would it feel like to release some of this anger?"
            ]
        elif any(word in lower for word in ["lonely", "alone", "isolated", "disconnected"]):
            questions = [
                "what's been making you feel most alone lately?",
                "when do you feel the loneliness most strongly?",
                "what kind of connection are you longing for?",
                "who in your life usually helps you feel less alone?"
            ]
        else:
            # General caring follow-up questions
            questions = [
                "tell me more about what's going on in your heart.",
                "what's been on your mind lately?",
                "how are you really doing with all of this?",
                "what would help you feel most heard right now?",
                "what part of this feels most important to talk through?"
            ]
        
        return random.choice(questions)

    def _generate_quick_replies(self, user_message: str) -> List[str]:
        """Provide quick reply suggestions that encourage deeper sharing"""
        return [
            "tell me more about how you're feeling",
            "what's really on your heart?",
            "help me understand what you're experiencing",
            "i want to process this with you",
            "let's explore this together"
        ]

    async def _get_additional_verses(self, user_message: str, primary_verse: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Get 2-3 additional related verses"""
        if not primary_verse:
            return None
            
        combined_text = user_message.lower()
        additional = []
        
        # Get theme-based verses
        theme_verses = {
            "anxiety": [
                {"verse_reference": "Philippians 4:6-7", "verse_text": "Do not be anxious about anything, but in every situation, by prayer and petition, with thanksgiving, present your requests to God."},
                {"verse_reference": "1 Peter 5:7", "verse_text": "Cast all your anxiety on him because he cares for you."},
            ],
            "strength": [
                {"verse_reference": "Isaiah 40:31", "verse_text": "But those who hope in the Lord will renew their strength."},
                {"verse_reference": "2 Corinthians 12:9", "verse_text": "My grace is sufficient for you, for my power is made perfect in weakness."},
            ],
            "love": [
                {"verse_reference": "1 Corinthians 13:4-7", "verse_text": "Love is patient, love is kind..."},
                {"verse_reference": "Romans 8:38-39", "verse_text": "Nothing can separate us from the love of God."},
            ],
            "faith": [
                {"verse_reference": "Hebrews 11:1", "verse_text": "Now faith is confidence in what we hope for and assurance about what we do not see."},
                {"verse_reference": "Matthew 17:20", "verse_text": "If you have faith as small as a mustard seed..."},
            ]
        }
        
        # Find matching theme and add 1-2 verses
        for theme, verses in theme_verses.items():
            if theme in combined_text:
                additional.extend(verses[:2])
                break
        
        # If no theme match, add general encouragement
        if not additional:
            additional = [
                {"verse_reference": "Jeremiah 29:11", "verse_text": "For I know the plans I have for you, declares the Lord, plans to prosper you and not to harm you."},
                {"verse_reference": "Psalm 23:1", "verse_text": "The Lord is my shepherd, I lack nothing."},
            ]
        
        return additional[:2]  # Return max 2 additional verses
    
    def _generate_reflection_prompts(self, user_message: str, verse_rec: Optional[Dict[str, Any]]) -> List[str]:
        """Generate reflection prompts that connect the verse to personal devotion"""
        verse_ref = verse_rec.get("verse_reference") if verse_rec else "this passage"
        
        return [
            f"How does {verse_ref} speak to your current season of life?",
            "What is one way you can live out this truth today?",
            "Write a prayer inspired by this scripture.",
            "What does this reveal about God's character?",
            "How might this verse change your perspective on your situation?"
        ]
    
    def _generate_journal_prompts(self, user_message: str, verse_rec: Optional[Dict[str, Any]]) -> List[str]:
        """Create journaling prompts that encourage personal reflection"""
        base_prompts = [
            "what is God showing me through this experience?",
            "how am I being invited to grow in this season?",
            "what would I want to say to God about this?",
            "where do I see God's presence in my story right now?",
            "what is my heart truly longing for?"
        ]
        
        return random.sample(base_prompts, min(3, len(base_prompts)))
    
    # Removed fallback response - we want real AI responses only