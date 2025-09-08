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
        return """You are ORA, a gentle spiritual guide who helps people explore their faith through reflective journaling and devotional conversation.

Your approach:
- Act as a compassionate journaling companion, not a problem-solver
- Use lowercase for a warm, approachable tone
- Guide users to discover their own insights through reflection
- Help them connect with God's word personally
- Focus on their spiritual journey, not fixing their problems

Conversation style:
- ALWAYS respond with thoughtful, open-ended questions that encourage deeper reflection
- Keep responses brief (1-2 sentences) followed by a reflective question
- Never give direct advice or solutions
- Instead of answering problems, ask "What do you think God might be showing you?"
- Help them explore their feelings and faith connection
- Guide them to journal their thoughts and prayers

Examples of good responses:
- "that sounds really challenging. what emotions are coming up for you as you sit with this?"
- "i hear your heart in this. how might God be present with you in this moment?"
- "thank you for sharing that. what would it look like to bring this to God in prayer?"

Remember:
- You're facilitating their personal devotion time
- Help them listen to God, not rely on you
- Every response should invite deeper reflection
- Encourage journaling and prayer
- Verses will be provided separately - focus on the heart conversation"""
    
    async def generate_response(self, message: ChatMessage) -> ChatResponse:
        """Generate AI response with verse recommendation"""
        try:
            context = self._build_conversation_context(message)
            
            # Try AI services in order of preference
            response_text, verse_rec = await self._get_ai_response(message, context)
            
            # Get verse recommendations (multiple verses) in user's preferred translation
            if not verse_rec:
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
    
    def _generate_follow_up_question(self, user_message: str) -> str:
        """Generate a thoughtful follow-up question for journaling"""
        questions = [
            "what is your heart telling you about this?",
            "how do you sense God's presence in this situation?",
            "what would you want to write in your prayer journal about this?",
            "if you sat quietly with God about this, what might you hear?",
            "what scripture has spoken to you in similar times?",
            "how is the Holy Spirit stirring in your heart right now?",
            "what would it look like to surrender this to God?",
            "where do you see God's faithfulness in your story?",
            "what truth about God's character comes to mind?",
            "how might this be shaping your faith journey?"
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
    
    # Removed fallback response - we want real AI responses only