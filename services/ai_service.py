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
        return """You are ORA, a compassionate spiritual companion who meets people where they are.

CRITICAL: Match their energy and input length
- If they write 1-2 words: Respond with 1-2 sentences, gentle and brief
- If they write a few sentences: Match with similar length, warm and supportive  
- If they write paragraphs: Respond with fuller thoughts, deeper engagement
- Mirror their emotional energy - if quiet/low, be gentle; if expressive, be more engaged

Response Style:
- ALWAYS acknowledge their feelings with empathy first
- Match their communication style and energy level
- Provide spiritual insight appropriate to their sharing depth
- Keep responses warm, conversational, and supportive
- Never use emojis in your responses

Response Pattern:
1. Acknowledge their sharing (match their energy level)
2. Express empathy (brief for short messages, fuller for longer ones)
3. Offer spiritual perspective appropriate to their sharing depth
4. Ask MAX 1-2 follow-ups total before providing verses

Examples by Length:
Short input - User: "stressed"
You: "i hear that stress. you're not carrying it alone."

Medium input - User: "I'm really stressed about work lately"  
You: "i'm sorry work has been weighing on you. that kind of pressure can feel overwhelming. remember there's peace available even in busy seasons."

Long input - User: "I've been so stressed about work lately. My boss keeps piling on more projects and I feel like I can't keep up. I'm worried I'm going to disappoint everyone and maybe even lose my job. It's affecting my sleep and I just feel anxious all the time."
You: "i can really hear how overwhelmed you're feeling right now. work pressure like that - with the fear of disappointing people and job security concerns - that's so much to carry. it makes complete sense that it's affecting your sleep and creating anxiety. you don't have to bear this weight alone. even when everything feels uncertain, there's a peace that can anchor you through these storms."

Verse Recommendations:
- After 1-2 exchanges, recommend a relevant verse
- ALWAYS explain WHY you chose that specific verse for their situation
- Connect the verse directly to what they've shared

Conversation Flow:
- First response: Match their energy + acknowledge + empathize
- Second response: Gentle wisdom + possible follow-up  
- Third response: Verse + explanation of why it fits their situation

Never:
- Use emojis or special characters
- Ask excessive questions 
- Overwhelm them with length if they're brief
- Under-respond if they've shared deeply

Always:
- Match their energy and length
- Meet them where they are emotionally
- Explain verse choices clearly"""
    
    async def generate_response(self, message: ChatMessage) -> ChatResponse:
        """Generate AI response with verse recommendation"""
        try:
            context = self._build_conversation_context(message)
            
            # Analyze user input to match energy and length
            input_analysis = self._analyze_user_input(message.message)
            context += f"\nUser input analysis: {input_analysis}\n"
            
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
    
    def _analyze_user_input(self, user_message: str) -> str:
        """Analyze user input length and energy to guide response matching"""
        message_length = len(user_message.strip())
        word_count = len(user_message.strip().split())
        
        # Analyze length category
        if word_count <= 3:
            length_category = "very short (1-3 words)"
            response_guidance = "Respond with 1-2 brief, gentle sentences"
        elif word_count <= 10:
            length_category = "short (4-10 words)"
            response_guidance = "Respond with 2-3 supportive sentences"
        elif word_count <= 30:
            length_category = "medium (11-30 words)"
            response_guidance = "Respond with 3-4 sentences, matching their depth"
        else:
            length_category = "long (30+ words)"
            response_guidance = "Respond with fuller engagement, deeper thoughts"
        
        # Analyze emotional energy
        energy_indicators = {
            "high": ["!", "really", "so", "very", "extremely", "absolutely", "totally"],
            "urgent": ["help", "need", "can't", "won't", "never", "always", "everything", "nothing"],
            "low": [".", "i guess", "maybe", "i don't know", "whatever", "tired", "exhausted"]
        }
        
        message_lower = user_message.lower()
        energy_level = "neutral"
        
        if sum(1 for word in energy_indicators["high"] if word in message_lower) >= 2:
            energy_level = "high energy - match with engaged, warm response"
        elif any(word in message_lower for word in energy_indicators["urgent"]):
            energy_level = "urgent/stressed - respond with calm, reassuring tone"
        elif any(word in message_lower for word in energy_indicators["low"]):
            energy_level = "low energy - respond gently, don't overwhelm"
        
        return f"{length_category} - {response_guidance}. Energy: {energy_level}"
    
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
        """Get a relevant verse recommendation with explanation based on conversation"""
        combined_text = f"{user_message} {ai_response}".lower()
        
        # Map themes to verse references with explanations
        theme_verse_data = {
            "anxiety": {
                "verses": ["Philippians 4:6-7", "1 Peter 5:7", "Matthew 6:34"],
                "explanations": {
                    "Philippians 4:6-7": "i chose this verse because it directly addresses anxiety and worry. when you're feeling stressed, paul reminds us we can bring everything to god in prayer instead of carrying it alone.",
                    "1 Peter 5:7": "this verse is perfect for your situation because it reminds us that god actually cares about what's weighing on you. you can literally cast your worries on him.",
                    "Matthew 6:34": "jesus spoke these words knowing how overwhelming worry can be. this verse helps us focus on today instead of getting consumed by tomorrow's unknowns."
                }
            },
            "love": {
                "verses": ["1 Corinthians 13:4-7", "1 John 4:19", "John 13:34"],
                "explanations": {
                    "1 Corinthians 13:4-7": "i chose this because it beautifully describes what love looks like in action. when relationships feel hard, this reminds us what love truly is.",
                    "1 John 4:19": "this verse is perfect because it reminds us that all our ability to love comes from being loved by god first. it takes the pressure off.",
                    "John 13:34": "jesus gave us this command knowing how much we need love and connection. it's a reminder that loving others is central to following him."
                }
            },
            "strength": {
                "verses": ["Isaiah 40:31", "Philippians 4:13", "2 Timothy 1:7"],
                "explanations": {
                    "Isaiah 40:31": "when you're feeling weak or exhausted, this verse promises that god renews your strength. even eagles get tired, but god doesn't.",
                    "Philippians 4:13": "i chose this because paul wrote it during his own difficult circumstances. it's a reminder that christ's strength works through us when we feel insufficient.",
                    "2 Timothy 1:7": "this verse speaks directly to times when we feel powerless. god has given you a spirit of power, love, and sound mind - not fear or weakness."
                }
            },
            "peace": {
                "verses": ["John 14:27", "Isaiah 26:3", "Philippians 4:7"],
                "explanations": {
                    "John 14:27": "jesus spoke these words knowing his disciples would face troubled times. his peace is different from temporary calm - it's lasting.",
                    "Isaiah 26:3": "this verse is perfect because it connects peace with trusting god. when your mind is stayed on him, perfect peace follows.",
                    "Philippians 4:7": "this peace surpasses understanding - it doesn't depend on circumstances making sense. it guards your heart and mind."
                }
            },
            "patience": {
                "verses": ["James 1:3-4", "Romans 12:12", "Galatians 6:9"],
                "explanations": {
                    "James 1:3-4": "when waiting feels hard, james reminds us that testing produces patience, and patience produces completeness. there's purpose in the process.",
                    "Romans 12:12": "this verse is perfect for difficult seasons - it connects patience with hope and prayer. all three work together.",
                    "Galatians 6:9": "when you're tempted to give up, paul reminds us not to grow weary. the right season will come."
                }
            },
            "faith": {
                "verses": ["Hebrews 11:1", "2 Corinthians 5:7", "Mark 11:24"],
                "explanations": {
                    "Hebrews 11:1": "this verse beautifully defines what faith actually is - confidence in what we hope for even when we can't see it yet.",
                    "2 Corinthians 5:7": "when everything feels uncertain, this reminds us that walking by faith rather than sight is actually the normal christian life.",
                    "Mark 11:24": "jesus connects faith with prayer here. when we pray believing, we can trust god to work even before we see the results."
                }
            }
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
        
        # Get verse data for the theme
        theme_data = theme_verse_data.get(selected_theme, theme_verse_data["faith"])
        selected_ref = random.choice(theme_data["verses"])
        
        # Fetch the verse in user's preferred translation
        try:
            verse_data = await self.bible_service.get_verse(selected_ref, translation)
            if verse_data and verse_data.get("text"):
                return {
                    "verse_reference": verse_data.get("reference", selected_ref),
                    "verse_text": verse_data.get("text"),
                    "explanation": theme_data["explanations"].get(selected_ref, "i thought this verse would encourage you in this situation.")
                }
        except Exception as e:
            print(f"Error fetching verse {selected_ref} in {translation}: {e}")
        
        return None
    
    def _should_include_verse(self, user_message: str, context: str) -> bool:
        """Determine if we should include a verse - be generous with verse sharing"""
        # Always include verses after the first exchange or if they share any meaningful content
        
        # Look for any emotional, spiritual, or personal sharing indicators
        indicators_for_verse = [
            "feel", "feeling", "emotion", "heart", "soul", "pray", "prayer", 
            "God", "faith", "believe", "trust", "hope", "peace", "strength",
            "struggling", "going through", "dealing with", "working through",
            "stressed", "worry", "worried", "anxious", "sad", "lonely", "alone",
            "afraid", "scared", "angry", "frustrated", "tired", "overwhelmed",
            "help", "need", "want", "wish", "difficult", "hard", "tough",
            "family", "work", "relationship", "life", "future", "past"
        ]
        
        message_lower = user_message.lower()
        
        # Include verse if they share ANY meaningful content (just 1+ indicators)
        sharing_indicators = sum(1 for indicator in indicators_for_verse if indicator in message_lower)
        
        # Be much more generous - include verse with just 1 indicator or if message is longer than 20 chars
        return sharing_indicators >= 1 or len(user_message.strip()) > 20
    
    def _generate_follow_up_question(self, user_message: str) -> str:
        """Generate a brief follow-up question - less frequently and lighter"""
        lower = user_message.lower()
        
        # Only generate follow-up questions 40% of the time to reduce frequency
        if random.random() > 0.4:
            return None
        
        # Simpler, less probing follow-up questions
        if any(word in lower for word in ["stressed", "stress", "overwhelmed", "pressure"]):
            questions = [
                "what would bring you peace in this situation?",
                "how can i pray for you about this stress?"
            ]
        elif any(word in lower for word in ["sad", "down", "depressed", "heavy"]):
            questions = [
                "what brings you comfort when you feel this way?",
                "how can i support you through this?"
            ]
        elif any(word in lower for word in ["worried", "worry", "anxious", "anxiety", "afraid"]):
            questions = [
                "what helps you find peace when worry comes?",
                "would you like me to pray about this with you?"
            ]
        elif any(word in lower for word in ["angry", "frustrated", "mad", "upset"]):
            questions = [
                "what would help you find peace in this situation?",
                "how can i support you through this frustration?"
            ]
        elif any(word in lower for word in ["lonely", "alone", "isolated", "disconnected"]):
            questions = [
                "what kind of connection would mean most to you?",
                "how can i encourage you in this loneliness?"
            ]
        else:
            # Lighter, more supportive follow-up questions
            questions = [
                "how can i best support you in this?",
                "what would be most helpful for you right now?",
                "is there anything specific you'd like to talk through?"
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