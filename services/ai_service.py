import os
import re
import random
import json
import asyncio
import hashlib
from typing import Optional, List, Dict, Any, Tuple, Set
from openai import AsyncOpenAI
import anthropic
from collections import deque, Counter
from threading import Lock
from datetime import datetime, timedelta

from models import ChatMessage, ChatResponse
from services.bible_service import BibleService
from config import get_settings

settings = get_settings()

# Initialize async AI clients
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.has_openai_key else None
claude_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.has_anthropic_key else None


class SpiritualAIService:
    """Service for AI-powered spiritual conversations"""
    
    def __init__(self):
        self.bible_service = BibleService()
        # Track recent usage per user to avoid repetition
        self.recent_questions = {}  # user_id -> deque of recent questions
        self.recent_verses = {}  # user_id -> deque of recent verse references
        self.recent_openers = {}  # user_id -> deque of recent opening phrases
        self.recent_structures = {}  # user_id -> deque of recent response structures
        self.previous_responses = {}  # user_id -> deque of previous responses for bigram check
        self.recent_responses = {}  # user_id -> list of recent fallback responses
        self.conversation_mode = {}  # user_id -> 'direct_answer' or 'dialogue'
        self.user_seeds = {}  # user_id -> random seed for deterministic selection
        self.user_request_times = {}  # user_id -> deque of request timestamps for rate limiting
        self.turn_counts = {}  # user_id -> turn number for cadence control
        self.last_cleanup = datetime.now()
        self._lock = Lock()  # Thread safety for shared state
        
        # Verse cooldown tracking
        self.global_last_seen = {}  # verse_ref -> timestamp
        self.user_last_seen = {}  # user_id -> {verse_ref -> timestamp}
        
        # Style banks for variety
        self._init_style_banks()
        
        # Load verse topics for better matching
        self._load_verse_topics()
        
        # Testing configuration
        self.testing_mode = os.getenv('TESTING_MODE', 'false').lower() == 'true'
        self.deterministic_seed = int(os.getenv('DETERMINISTIC_SEED', '42')) if self.testing_mode else None
        
        # Rate limiting config (30 requests per minute per user)
        self.rate_limit = int(os.getenv('RATE_LIMIT_PER_MINUTE', '30'))
        
        # Content safety patterns
        self.unsafe_patterns = [
            r'ignore.*previous.*instructions',
            r'forget.*everything',
            r'system.*prompt',
            r'you.*are.*now',
            r'act.*as.*if',
            r'pretend.*you.*are'
        ]
    
    def _build_system_prompt(self, mode: str, word_count: int, turn_number: int, user_context: Dict) -> str:
        """Build request-scoped system prompt with flexible length guidance"""
        # Flexible length band instead of exact matching
        min_words = max(5, int(word_count * 0.8))
        max_words = int(word_count * 1.2) + 5
        
        base_prompt = f"""You are ORA, a compassionate spiritual companion who meets people where they are - like a caring pastor or trusted friend.

LENGTH GUIDANCE: Target {min_words}-{max_words} words (flexible band, not rigid)
- Short messages deserve concise responses
- Longer shares can have fuller engagement
- Match their emotional investment level

STYLE VARIETY:
- Avoid these recent openers: {list(user_context.get('recent_openers', []))}
- Avoid these recent structures: {list(user_context.get('recent_structures', []))}
- Use varied sentence lengths and rhythms
- Never use emojis; keep lowercase for gentle tone

CADENCE RULES:
- Turn {turn_number}: {'Ask a question if needed' if turn_number % 2 == 1 or turn_number <= 2 else 'Offer a reflective statement instead of questions'}
- After turn 2: Alternate between questions and reflective statements
- Never ask excessive questions in a row

EMOTIONAL MATCHING:
- Match their energy level precisely
- Acknowledge feelings genuinely
- Be like a caring pastor or trusted friend"

Response Pattern Based on Message Length:
FOR VERY SHORT MESSAGES (1-5 words):
- Respond with: acknowledgment + 1 brief question
- Example: "i hear you. what's heaviest right now?"

FOR SHORT MESSAGES (1 sentence):
- Respond with: empathy + 1-2 short follow-up questions
- Keep total response under 20 words

FOR MEDIUM MESSAGES (2-3 sentences):
- Respond with: acknowledgment + empathy + 2 questions
- Match their sentence count

FOR LONG MESSAGES (paragraph+):
- Full engagement with deeper questions and reflection
- Can ask 2-3 thoughtful questions

IMPORTANT: Only offer verses after 3-4 exchanges, and keep verse explanations brief if they're being brief

Examples by Length:
Very short (1-3 words) - User: "stressed"
You: "i hear you. what's heaviest?"

Short (1 sentence) - User: "I'm really stressed about work lately"  
You: "work stress is so heavy. what part feels hardest?"

Medium (2-3 sentences) - User: "I'm stressed about work. My boss keeps adding projects. I can't keep up."
You: "that sounds overwhelming. how long has this been building? is your boss aware of your workload?"

Long (paragraph) - User: "I've been so stressed about work lately. My boss keeps piling on more projects and I feel like I can't keep up. I'm worried I'm going to disappoint everyone and maybe even lose my job. It's affecting my sleep and I just feel anxious all the time."
You: "i can really hear how overwhelmed you're feeling right now. work pressure like that - with the fear of disappointing people and job security concerns - that's so much to carry. it makes complete sense that it's affecting your sleep and creating anxiety. how long have you been feeling this weight? and is your boss aware of how much you're juggling?"

Verse Recommendations:
- Only after 3-4 meaningful exchanges where you've truly understood their situation
- ALWAYS explain WHY you chose that specific verse for their situation
- Connect the verse directly to what they've shared
- Make sure you've asked enough questions to understand the depth and context first

Conversation Flow:
- First response: Match their energy + acknowledge + ask 1 question (or 0 if they asked a direct question)
- Second response: Deeper empathy + optionally ask about context (50% chance)
- Third response: Offer verse with explanation or gentle wisdom
- Fourth+ response: Mix of support, verses, and occasional questions

IMPORTANT RULES:
- If user asks a DIRECT QUESTION (ends with ? or starts with what/how/why/when/where/who/can/should/does):
  * Answer the question FIRST and DIRECTLY
  * Then optionally add 0-1 short follow-up (not always)
  * Do NOT probe or ask questions when they asked for information
- Ask AT MOST one question per response
- NEVER repeat a question you've already asked in this conversation
- If you notice repetition, switch to statements or different support styles

Never:
- Use emojis or special characters
- Ask excessive questions 
- Repeat questions from earlier in conversation
- Overwhelm them with length if they're brief
- Under-respond if they've shared deeply

Always:
- Match their energy and length
- Meet them where they are emotionally
- Vary your response patterns
- Answer direct questions directly
- Explain verse choices clearly"""
        
        if mode == 'direct_answer':
            base_prompt += "\n\nMODE: DIRECT ANSWER - Answer their question directly first, then optionally add brief context. No probing questions."
        
        return base_prompt
    
    def _init_style_banks(self):
        """Initialize style banks for varied responses"""
        self.style_banks = {
            'acknowledgments': [
                "i hear you", "that makes sense", "i can feel how heavy this is", 
                "thanks for sharing that", "i understand", "that sounds really tough",
                "you're not alone in this", "i can sense your heart in this", 
                "that resonates deeply", "i'm honored you shared that",
                "i feel the weight of what you're carrying", "your words matter",
                "i see you in this struggle", "that takes courage to share"
            ],
            'validations': [
                "your feelings are completely valid", "it makes total sense you'd feel this way",
                "anyone in your situation would struggle", "you're being so honest about this",
                "this is a lot to process", "you're handling more than most people realize",
                "it's okay to feel overwhelmed", "your heart is in the right place",
                "you're asking the right questions", "this shows how much you care"
            ],
            'transitions': [
                "tell me more about", "help me understand", "what does that look like for you",
                "i'm curious about", "walk me through", "paint me a picture of",
                "when you think about", "as you sit with this", "in your experience",
                "from where you're standing"
            ],
            'closers': [
                "you're not walking this alone", "there's hope in this story",
                "god sees every detail of your struggle", "your story matters",
                "healing takes time, and that's okay", "you're exactly where you need to be",
                "grace is bigger than this moment", "tomorrow can hold new possibilities",
                "you're stronger than you realize", "love is working even when it's hidden"
            ],
            'question_starters': [
                "what's stirring", "what comes up when", "how does it feel when",
                "what would it mean if", "where do you find", "what helps you",
                "what's different about", "what would change if", "when do you feel",
                "what's hardest about"
            ]
        }
    
    def _choose_variant(self, user_id: str, bank_key: str, options: List[str]) -> str:
        """Choose a variant from style bank with anti-repeat logic"""
        with self._lock:
            if user_id not in self.recent_openers:
                self.recent_openers[user_id] = deque(maxlen=8)
            
            used = self.recent_openers[user_id] if bank_key == 'acknowledgments' else deque()
            available = [opt for opt in options if opt not in used] or options
            
            # Use user seed for consistency
            if user_id in self.user_seeds and not self.testing_mode:
                random.seed(self.user_seeds[user_id] + len(str(used)))
            elif self.testing_mode:
                random.seed(self.deterministic_seed)
            
            choice = random.choice(available)
            
            if bank_key == 'acknowledgments':
                self.recent_openers[user_id].append(choice)
            
            return choice
    
    def _on_cooldown(self, user_id: str, verse_ref: str) -> bool:
        """Check if verse is on cooldown (global 30 days, user 7 days)"""
        now = datetime.now()
        
        # Global cooldown (30 days)
        if verse_ref in self.global_last_seen:
            global_last = self.global_last_seen[verse_ref]
            if now - global_last < timedelta(days=30):
                return True
        
        # Per-user cooldown (7 days)  
        if user_id in self.user_last_seen and verse_ref in self.user_last_seen[user_id]:
            user_last = self.user_last_seen[user_id][verse_ref]
            if now - user_last < timedelta(days=7):
                return True
        
        return False
    
    def _mark_verse_used(self, user_id: str, verse_ref: str):
        """Mark verse as used for cooldown tracking"""
        now = datetime.now()
        self.global_last_seen[verse_ref] = now
        
        if user_id not in self.user_last_seen:
            self.user_last_seen[user_id] = {}
        self.user_last_seen[user_id][verse_ref] = now
    
    def _diversify_by_book(self, verses: List[Dict], limit: int = 5) -> List[Dict]:
        """Diversify verse selection by book to avoid repetition"""
        seen_books = set()
        diversified = []
        
        for verse in verses:
            book = verse['ref'].split()[0] if isinstance(verse, dict) and 'ref' in verse else verse.split()[0]
            if book not in seen_books:
                diversified.append(verse)
                seen_books.add(book)
                if len(diversified) >= limit:
                    break
        
        return diversified or verses[:limit]
    
    def _too_similar_bigrams(self, text_a: str, text_b: str) -> bool:
        """Check if two texts share >70% bigrams (avoid repetitive responses)"""
        if not text_a or not text_b:
            return False
            
        words_a = text_a.lower().split()
        words_b = text_b.lower().split()
        
        if len(words_a) < 2 or len(words_b) < 2:
            return False
        
        bigrams_a = {(words_a[i], words_a[i+1]) for i in range(len(words_a)-1)}
        bigrams_b = {(words_b[i], words_b[i+1]) for i in range(len(words_b)-1)}
        
        if not bigrams_a or not bigrams_b:
            return False
            
        overlap = len(bigrams_a & bigrams_b)
        return overlap / max(len(bigrams_a), len(bigrams_b)) > 0.7
    
    async def generate_response(self, message: ChatMessage) -> ChatResponse:
        """Generate AI response with verse recommendation"""
        try:
            # Cleanup old user states periodically (every hour)
            if datetime.now() - self.last_cleanup > timedelta(hours=1):
                self._cleanup_old_states()
            
            # Get or create user tracking with thread safety
            user_id = getattr(message, 'user_id', 'default')
            
            # Check rate limit
            if not self._check_rate_limit(user_id):
                return ChatResponse(
                    response="please slow down a bit. let's take a moment to breathe together.",
                    verse_recommendation=None,
                    additional_verses=None,
                    follow_up_question=None,
                    quick_replies=["i understand", "tell me more"],
                    journal_prompts=None,
                    reflection_prompts=None
                )
            
            # Check content safety
            if self._check_unsafe_content(message.message):
                return ChatResponse(
                    response="i'm here to offer spiritual support and biblical wisdom. how can i help you in your faith journey today?",
                    verse_recommendation=None,
                    additional_verses=None,
                    follow_up_question=None,
                    quick_replies=["tell me about faith", "i need encouragement", "share a verse"],
                    journal_prompts=None,
                    reflection_prompts=None
                )
            
            with self._lock:
                if user_id not in self.recent_questions:
                    self.recent_questions[user_id] = deque(maxlen=10)
                    self.recent_verses[user_id] = deque(maxlen=10)
                    self.recent_openers[user_id] = deque(maxlen=8)
                    self.recent_structures[user_id] = deque(maxlen=6)
                    self.previous_responses[user_id] = deque(maxlen=3)
                    self.conversation_mode[user_id] = 'dialogue'
                    self.user_request_times[user_id] = deque(maxlen=self.rate_limit)
                    self.turn_counts[user_id] = 0
                    # Create deterministic seed from user_id for consistent randomness
                    self.user_seeds[user_id] = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
                
                # Increment turn count
                self.turn_counts[user_id] = self.turn_counts.get(user_id, 0) + 1
            
            # Check if this is a direct question or complaint about repetition
            is_direct_question = self._is_direct_question(message.message)
            is_repetition_complaint = self._is_repetition_complaint(message.message)
            
            # Set mode based on message type
            if is_direct_question or is_repetition_complaint:
                self.conversation_mode[user_id] = 'direct_answer'
            
            # Create request-scoped system prompt with user context
            word_count = len(message.message.strip().split())
            turn_number = self.turn_counts[user_id]
            user_context = {
                'recent_openers': list(self.recent_openers[user_id]),
                'recent_structures': list(self.recent_structures[user_id])
            }
            system_prompt = self._build_system_prompt(
                self.conversation_mode[user_id], 
                word_count, 
                turn_number, 
                user_context
            )
            
            context = self._build_conversation_context(message)
            
            # Add recent questions to context to prevent repetition
            if self.recent_questions[user_id]:
                context += f"\nQuestions already asked (DO NOT REPEAT): {list(self.recent_questions[user_id])}\n"
            
            # Analyze user input to match energy and length
            input_analysis = self._analyze_user_input(message.message)
            word_count = len(message.message.strip().split())
            context += f"\nUser input analysis: {input_analysis}\n"
            context += f"\nCRITICAL: User wrote {word_count} words. You MUST match this length!\n"
            
            if is_repetition_complaint:
                context += "\nUSER IS COMPLAINING ABOUT REPETITION. Apologize briefly, then provide a thoughtful response with NO questions.\n"
            
            # Track conversation depth
            exchange_count = self._get_exchange_count(message)
            context += f"\nConversation exchange count: {exchange_count}\n"
            context += f"\nIMPORTANT: This is exchange #{exchange_count}. Only provide verses after 3-4 exchanges of meaningful dialogue.\n"
            
            # Add strict length enforcement
            if word_count <= 5:
                context += f"\nSTRICT RULE: Your response must be UNDER {word_count * 3} words total. Be extremely brief.\n"
            
            # Try AI services in order of preference with request-scoped prompt
            response_text, verse_rec = await self._get_ai_response(message, context, system_prompt)
            
            # Check for bigram similarity and regenerate if too similar to recent responses
            if user_id in self.previous_responses:
                for prev_response in self.previous_responses[user_id]:
                    if self._too_similar_bigrams(response_text, prev_response):
                        # Try to regenerate once with modified prompt
                        retry_prompt = system_prompt + "\\n\\nIMPORTANT: Vary your phrasing and structure from your previous responses. Use different sentence patterns."
                        response_text, verse_rec = await self._get_ai_response(message, context, retry_prompt)
                        break
            
            # Store this response for future bigram checking
            if user_id not in self.previous_responses:
                self.previous_responses[user_id] = deque(maxlen=3)
            self.previous_responses[user_id].append(response_text)
            
            # Only provide verses after 3-4 exchanges of meaningful conversation
            # This ensures we understand their situation better first
            should_include_verse = exchange_count >= 3 and self._should_include_verse(message.message, context)
            if should_include_verse and not verse_rec:
                verse_rec = await self._get_verse_recommendation(
                    message.message, 
                    response_text, 
                    message.preferred_translation or 'niv',
                    user_id
                )
            
            # Only get additional verses if we're including a primary verse
            additional_verses = await self._get_additional_verses(message.message, verse_rec) if verse_rec else None
            
            # Generate follow-up questions more frequently in early exchanges
            follow_up = None
            if self.conversation_mode[user_id] != 'direct_answer':
                follow_up = self._generate_contextual_follow_up(message.message, exchange_count, user_id)
            
            # Reset mode after direct answer
            if self.conversation_mode[user_id] == 'direct_answer':
                self.conversation_mode[user_id] = 'dialogue'
            
            quick_replies = self._generate_varied_quick_replies(message.message, user_id)
            alternative_actions = self._generate_alternative_actions(message.message, verse_rec, exchange_count)
            journal_prompts = self._generate_journal_prompts(message.message, verse_rec) if verse_rec else None
            reflection_prompts = self._generate_reflection_prompts(message.message, verse_rec) if verse_rec else None
            
            return ChatResponse(
                response=response_text,
                verse_recommendation=verse_rec,
                additional_verses=additional_verses,
                follow_up_question=follow_up,
                quick_replies=quick_replies,
                alternative_actions=alternative_actions,
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
    
    def _get_exchange_count(self, message: ChatMessage) -> int:
        """Track how many exchanges have happened in conversation"""
        # Check if conversation_history is provided
        if hasattr(message, 'conversation_history') and message.conversation_history:
            # Count user messages in history
            return len([msg for msg in message.conversation_history if msg.get('role') == 'user']) + 1
        # Default to 1 for first exchange
        return 1
    
    def _analyze_user_input(self, user_message: str) -> str:
        """Analyze user input length and energy to guide response matching"""
        message_length = len(user_message.strip())
        word_count = len(user_message.strip().split())
        
        # Better sentence detection using regex
        import re
        sentence_pattern = r'[.!?]+[\s]|[.!?]+$'
        sentences = re.split(sentence_pattern, user_message)
        sentence_count = len([s for s in sentences if s.strip() and len(s.strip()) > 2])
        
        # Analyze length category - BE VERY STRICT
        if word_count <= 3:
            length_category = "very short (1-3 words)"
            response_guidance = "MAXIMUM 10 words total. One short sentence only. Example: 'i hear you. what's heaviest?'"
        elif word_count <= 10:
            length_category = "short (4-10 words)"
            response_guidance = "MAXIMUM 15-20 words. 1-2 very short sentences."
        elif word_count <= 30:
            length_category = "medium (11-30 words)"
            response_guidance = f"Match their {sentence_count} sentences. Keep response under {word_count + 10} words."
        else:
            length_category = "long (30+ words)"
            response_guidance = "Can engage fully with paragraph-length response"
        
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
    
    async def _get_ai_response(self, message: ChatMessage, context: str, system_prompt: str) -> Tuple[str, Optional[Dict]]:
        """Get AI response from available services with request-scoped prompt"""
        # Prefer OpenAI for stability
        if openai_client and settings.has_openai_key:
            try:
                return await self._get_openai_response(message, context, system_prompt)
            except Exception as e:
                print(f"OpenAI error: {e}")
                # Try Claude as backup
                if claude_client and settings.has_anthropic_key:
                    try:
                        return await self._get_claude_response(message, context, system_prompt)
                    except Exception as e2:
                        print(f"Claude error: {e2}")
                        # Return graceful fallback instead of raising
                        return self._get_fallback_response(message), None
                # Single provider failure - return fallback
                return self._get_fallback_response(message), None
        
        # Try Claude if OpenAI not available
        if claude_client and settings.has_anthropic_key:
            try:
                return await self._get_claude_response(message, context, system_prompt)
            except Exception as e:
                print(f"Claude error: {e}")
                return self._get_fallback_response(message), None
        
        # If no AI services are available, return fallback
        return self._get_fallback_response(message), None
    
    async def _get_openai_response(self, message: ChatMessage, context: str, system_prompt: str) -> Tuple[str, Optional[Dict]]:
        """Generate response using async OpenAI GPT with request-scoped prompt"""
        response = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\nUser says: {message.message}"}
            ],
            temperature=0.8 if not self.testing_mode else 0.1,  # Low temp for testing
            max_tokens=settings.MAX_TOKENS,
            frequency_penalty=0.3,  # Reduce repetition
            presence_penalty=0.3,  # Encourage topic diversity
            seed=self.deterministic_seed  # Deterministic for testing
        )
        
        response_text = response.choices[0].message.content
        verse_rec = self._extract_verse_from_response(response_text)
        return response_text, verse_rec
    
    async def _get_claude_response(self, message: ChatMessage, context: str, system_prompt: str) -> Tuple[str, Optional[Dict]]:
        """Generate response using async Claude with request-scoped prompt"""
        response = await claude_client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=settings.MAX_TOKENS,
            temperature=0.8 if not self.testing_mode else 0.1,  # Low temp for testing
            system=system_prompt,
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
    
    def _is_direct_question(self, message: str) -> bool:
        """Check if user is asking a direct question with comprehensive detection"""
        msg = message.strip().lower()
        
        # Check for question mark anywhere
        if '?' in msg:
            return True
            
        # Question words at start
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'should', 'could', 'would', 'does', 'do', 'is', 'are', 'will', 'did', 'have', 'has', 'might', 'may']
        
        # Question phrases that indicate seeking information
        question_phrases = [
            'wondering', 'curious', 'want to know', 'need to know', 'tell me', 'explain', 'help me understand',
            'would love to know', 'wondering about', 'curious about', 'interested in knowing',
            'can you tell me', 'do you know', 'any idea', 'thoughts on', 'what do you think',
            'help me figure out', 'trying to understand', 'confused about', 'unclear about',
            'seeking', 'looking for', 'need guidance', 'need help', 'looking to understand'
        ]
        
        # Implicit questions (statements that seek information)
        implicit_questions = [
            "i don't understand", 'not sure', "i'm confused", "doesn't make sense",
            'hard to figure out', 'not clear', 'i wonder', 'makes me wonder'
        ]
        
        # Check start of message
        first_word = msg.split()[0] if msg.split() else ''
        if first_word in question_words:
            return True
            
        # Check for question phrases
        for phrase in question_phrases + implicit_questions:
            if phrase in msg:
                return True
        
        # Check for em-dash questions like "curious about this — what do you think"
        if '—' in msg or '--' in msg:
            parts = msg.replace('—', '--').split('--')
            for part in parts:
                part = part.strip()
                if part and (part.split()[0] in question_words if part.split() else False):
                    return True
                
        return False
    
    def _is_repetition_complaint(self, message: str) -> bool:
        """Check if user is complaining about repetition"""
        msg = message.lower()
        complaints = ['you already asked', 'you asked that', 'asked before', 'keep asking', 'same question', 'repetitive', 'repeating']
        return any(complaint in msg for complaint in complaints)
    
    def _load_verse_topics(self):
        """Load verse topics from JSON file with better path handling"""
        try:
            # More robust path resolution
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            topics_path = os.path.join(base_dir, 'data', 'verse_topics.json')
            
            if not os.path.exists(topics_path):
                print(f"Warning: verse_topics.json not found at {topics_path}")
                self.verse_topics = {}
                return
                
            with open(topics_path, 'r') as f:
                data = json.load(f)
                self.verse_topics = data['topics']
                print(f"Loaded {len(self.verse_topics)} verse topics")
        except Exception as e:
            print(f"Could not load verse topics: {e}")
            self.verse_topics = {}
    
    def _cleanup_old_states(self):
        """Clean up old user states to prevent memory bloat"""
        with self._lock:
            # In production, would check last activity time
            # For now, just reset the cleanup timer
            self.last_cleanup = datetime.now()
    
    def _get_fallback_response(self, message: ChatMessage) -> str:
        """Return graceful fallback when AI services are unavailable"""
        msg_lower = message.message.lower()
        
        # Context-aware fallback responses based on message content
        if any(word in msg_lower for word in ['anxious', 'worried', 'stress', 'overwhelmed', 'nervous']):
            responses = [
                "i can sense the weight you're carrying. anxiety is real, and so is God's peace that's bigger than our worries. what's stirring up the most worry right now?",
                "feeling anxious is completely human. even Jesus felt troubled at times. what would help you feel more grounded in this moment?",
                "that anxiety sounds heavy. you don't have to carry it alone. what's one small thing that usually brings you some calm?",
                "i hear that worry in your words. God sees every anxious thought and wants to meet you there. what's behind the biggest concern?"
            ]
        elif any(word in msg_lower for word in ['sad', 'depressed', 'down', 'heavy', 'lost', 'empty']):
            responses = [
                "sadness can feel so isolating, but you're not walking through this alone. what's weighing heaviest on your heart today?",
                "i can feel the heaviness in what you've shared. even in dark valleys, there's hope to be found. what's one thing that still brings you even a tiny spark?",
                "that sounds really hard. it's okay to feel sad - even Jesus wept. what would comfort look like for you right now?",
                "the weight you're describing sounds exhausting. you matter deeply, even when it doesn't feel that way. what's making this season feel so heavy?"
            ]
        elif any(word in msg_lower for word in ['grateful', 'thankful', 'blessed', 'joy', 'happy', 'celebrating']):
            responses = [
                "i love hearing that gratitude in your voice! gratitude has this amazing way of expanding our hearts. what's bringing you the most joy today?",
                "that's beautiful to hear! when we notice God's goodness, it changes everything. what specific blessing is standing out to you?",
                "your thankfulness is contagious! there's something powerful about celebrating the good. what made this moment feel special?",
                "i can feel the lightness in your words! gratitude is such a gift. what's one thing you want to remember about this feeling?"
            ]
        elif any(word in msg_lower for word in ['faith', 'doubt', 'believe', 'god', 'pray', 'bible']):
            responses = [
                "faith journeys are rarely straight lines, and that's completely normal. what's stirring in your heart about your relationship with God?",
                "spiritual questions often lead to the deepest growth. what aspect of faith feels most important to explore right now?",
                "i appreciate you bringing your spiritual thoughts here. what's one thing about God or faith that you're curious about?",
                "faith can feel mysterious and personal. what's been on your mind spiritually lately?"
            ]
        elif any(word in msg_lower for word in ['relationship', 'family', 'friend', 'marriage', 'conflict']):
            responses = [
                "relationships can be both our greatest joy and our deepest challenge. what's happening in this relationship that's on your heart?",
                "people are complicated, aren't they? even good relationships have their struggles. what feels most important to work through?",
                "it sounds like there's some relational complexity here. what would healthy connection look like in this situation?",
                "relationships matter so much to our well-being. what's one thing you're hoping for in this relationship?"
            ]
        elif '?' in message.message:
            responses = [
                "that's a thoughtful question. sometimes the best answers come when we explore together. what draws you to ask about this?",
                "i love that you're asking questions - that shows a heart that wants to grow. what sparked this particular question for you?",
                "questions often reveal what matters most to us. what would a helpful answer look like for you?",
                "that's worth exploring. what's behind this question - what are you hoping to understand better?"
            ]
        else:
            # General empathetic responses
            responses = [
                "thank you for sharing that with me. i can tell this matters to you. what feels most important to explore about this?",
                "i'm here to listen and walk through this with you. what would be most helpful to talk through right now?",
                "your words matter, and so do your experiences. what's one aspect of this that you'd like to understand better?",
                "i hear you, and i want to understand more. what would it look like for this situation to feel more manageable?",
                "it sounds like you have a lot going on internally. what's one piece of this that feels most pressing to you?"
            ]
        
        # Select response with some randomness but avoid recent ones
        user_id = getattr(message, 'user_id', 'default')
        if user_id not in self.recent_responses:
            self.recent_responses[user_id] = []
        
        # Filter out recently used responses
        available = [r for r in responses if r not in self.recent_responses[user_id][-2:]]
        if not available:  # If all recent, reset
            available = responses
            
        response = random.choice(available)
        
        # Track recent responses
        self.recent_responses[user_id].append(response)
        if len(self.recent_responses[user_id]) > 5:
            self.recent_responses[user_id] = self.recent_responses[user_id][-5:]
            
        return response
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limit"""
        if self.testing_mode:
            return True  # Skip rate limiting in testing
            
        now = datetime.now()
        with self._lock:
            if user_id not in self.user_request_times:
                self.user_request_times[user_id] = deque(maxlen=self.rate_limit)
            
            # Remove requests older than 1 minute
            while (self.user_request_times[user_id] and 
                   now - self.user_request_times[user_id][0] > timedelta(minutes=1)):
                self.user_request_times[user_id].popleft()
            
            # Check if under rate limit
            if len(self.user_request_times[user_id]) >= self.rate_limit:
                return False
            
            # Add current request
            self.user_request_times[user_id].append(now)
            return True
    
    def _check_unsafe_content(self, message: str) -> bool:
        """Check for potential prompt injection or unsafe content"""
        message_lower = message.lower()
        
        # Check for prompt injection patterns
        for pattern in self.unsafe_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        
        # Check for excessive special characters (potential injection)
        special_chars = sum(1 for c in message if c in '{}[]()<>|\\')
        if len(message) > 0 and special_chars / len(message) > 0.1:
            return True
            
        return False
    
    async def _get_verse_recommendation(self, user_message: str, ai_response: str, translation: str = 'niv', user_id: str = 'default') -> Optional[Dict[str, Any]]:
        """Get a relevant verse recommendation with explanation based on conversation"""
        combined_text = f"{user_message} {ai_response}".lower()
        
        # Use topic-based verse matching if available
        if self.verse_topics:
            # Score each topic based on keyword matches
            topic_scores = Counter()
            for topic_name, topic_data in self.verse_topics.items():
                keywords = topic_data.get('keywords', [])
                for keyword in keywords:
                    if keyword in combined_text:
                        topic_scores[topic_name] += 1
            
            # Get the best matching topic(s)
            if topic_scores:
                # Get top 2 topics for variety
                top_topics = topic_scores.most_common(2)
                selected_topic = random.choice([t[0] for t in top_topics])
                topic_data = self.verse_topics[selected_topic]
                
                # Get weighted verse selection
                verses_with_weights = topic_data.get('verses', [])
                
                # Filter out verses on cooldown
                available = [v for v in verses_with_weights if not self._on_cooldown(user_id, v['ref'])]
                
                if not available:
                    # If all are on cooldown, use any verse from the topic  
                    available = verses_with_weights
                
                # Diversify by book to avoid repetition from same books
                available = self._diversify_by_book(available, limit=10)
                
                # Weighted selection with seeding for consistency
                if available:
                    # Use user seed for deterministic selection
                    if user_id in self.user_seeds and not self.testing_mode:
                        random.seed(self.user_seeds[user_id] + len(combined_text))  # Add text length for variation
                    elif self.testing_mode:
                        random.seed(self.deterministic_seed)
                    
                    weights = [v.get('weight', 1) for v in available]
                    selected = random.choices(available, weights=weights, k=1)[0]
                    selected_ref = selected['ref']
                    
                    # Mark verse as used for cooldown tracking
                    self._mark_verse_used(user_id, selected_ref)
                    
                    # Also track in recent verses for immediate deduplication
                    if user_id not in self.recent_verses:
                        self.recent_verses[user_id] = deque(maxlen=15)
                    self.recent_verses[user_id].append(selected_ref)
                    
                    # Fetch the verse
                    try:
                        verse_data = await self.bible_service.get_verse(selected_ref, translation)
                        if verse_data and verse_data.get("text"):
                            # Generate contextual explanation
                            explanation = f"i chose {selected_ref} because it speaks directly to what you're experiencing with {selected_topic.replace('_', ' ')}"
                            return {
                                "verse_reference": verse_data.get("reference", selected_ref),
                                "verse_text": verse_data.get("text"),
                                "explanation": explanation
                            }
                    except Exception as e:
                        print(f"Error fetching verse {selected_ref}: {e}")
        
        # Fallback to simple keyword matching
        simple_verses = {
            "anxiety": ["Philippians 4:6-7", "1 Peter 5:7", "Matthew 6:34"],
            "love": ["1 Corinthians 13:4-7", "1 John 4:19", "John 13:34"],
            "strength": ["Isaiah 40:31", "Philippians 4:13", "2 Timothy 1:7"],
            "peace": ["John 14:27", "Isaiah 26:3", "Philippians 4:7"],
            "hope": ["Jeremiah 29:11", "Romans 15:13", "Hebrews 11:1"],
            "faith": ["Hebrews 11:1", "2 Corinthians 5:7", "Mark 11:24"]
        }
        
        # Find best matching category
        for category, keywords in [
            ("anxiety", ["anxious", "worried", "stress", "overwhelm"]),
            ("love", ["love", "relationship", "lonely", "heart"]),
            ("strength", ["weak", "tired", "strength", "power"]),
            ("peace", ["peace", "calm", "rest", "quiet"]),
            ("hope", ["hope", "future", "dream", "possibility"]),
            ("faith", ["faith", "believe", "trust", "doubt"])
        ]:
            if any(kw in combined_text for kw in keywords):
                verses = simple_verses[category]
                available = [v for v in verses if v not in self.recent_verses.get(user_id, [])]
                if not available:
                    available = verses
                    
                selected_ref = random.choice(available)
                
                if user_id not in self.recent_verses:
                    self.recent_verses[user_id] = deque(maxlen=15)
                self.recent_verses[user_id].append(selected_ref)
                
                try:
                    verse_data = await self.bible_service.get_verse(selected_ref, translation)
                    if verse_data and verse_data.get("text"):
                        return {
                            "verse_reference": verse_data.get("reference", selected_ref),
                            "verse_text": verse_data.get("text"),
                            "explanation": f"this verse came to mind as it relates to {category}"
                        }
                except Exception as e:
                    print(f"Error fetching verse {selected_ref}: {e}")
                break
        
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
    
    def _generate_contextual_follow_up(self, user_message: str, exchange_count: int, user_id: str) -> Optional[str]:
        """Generate thoughtful follow-up questions based on exchange count and message length"""
        lower = user_message.lower()
        word_count = len(user_message.strip().split())
        
        # Always ask follow-up questions in early exchanges (1-3)
        # Reduce frequency after that
        if exchange_count <= 2:
            # Always ask follow-ups early to understand better
            pass
        elif exchange_count == 3:
            # 70% chance on third exchange
            if random.random() > 0.7:
                return None
        else:
            # 30% chance after third exchange
            if random.random() > 0.3:
                return None
        
        # Adjust question length based on user's message length
        if word_count <= 5:
            # VERY SHORT questions for brief messages
            if exchange_count == 1:
                if any(word in lower for word in ["stressed", "stress", "overwhelmed"]):
                    questions = ["what's heaviest?", "how long?", "work or home?"]
                elif any(word in lower for word in ["sad", "down", "depressed"]):
                    questions = ["what happened?", "how long?", "want to share?"]
                elif any(word in lower for word in ["worried", "worry", "anxious"]):
                    questions = ["about what?", "biggest fear?", "how long?"]
                elif any(word in lower for word in ["angry", "mad", "frustrated"]):
                    questions = ["at who?", "what happened?", "how long?"]
                else:
                    questions = ["tell me more?", "what's hardest?", "how long?"]
            else:
                questions = ["how can i help?", "what do you need?", "feeling better?"]
            # Check if we've already asked this question
            selected = None
            attempts = 0
            while attempts < 10:
                candidate = random.choice(questions)
                if candidate not in self.recent_questions[user_id]:
                    selected = candidate
                    self.recent_questions[user_id].append(selected)
                    break
                attempts += 1
            
            return selected
        
        # Regular length questions for longer messages
        if exchange_count == 1:
            # First exchange - understand the situation better
            if any(word in lower for word in ["stressed", "stress", "overwhelmed", "pressure"]):
                questions = [
                    "what's been weighing on you most heavily?",
                    "how long have you been carrying this stress?",
                    "is this something new or has it been building for a while?"
                ]
            elif any(word in lower for word in ["sad", "down", "depressed", "heavy"]):
                questions = [
                    "what's making your heart feel heavy right now?",
                    "has something specific happened, or is it more of a season you're in?",
                    "how long have you been feeling this way?"
                ]
            elif any(word in lower for word in ["worried", "worry", "anxious", "anxiety", "afraid"]):
                questions = [
                    "what specific worries are keeping you up at night?",
                    "is there one main thing you're anxious about, or several things?",
                    "how is this anxiety affecting your daily life?"
                ]
            elif any(word in lower for word in ["angry", "frustrated", "mad", "upset"]):
                questions = [
                    "what triggered these feelings for you?",
                    "is this frustration with a situation or a person?",
                    "how long have you been holding onto this anger?"
                ]
            elif any(word in lower for word in ["lonely", "alone", "isolated", "disconnected"]):
                questions = [
                    "what's making you feel so alone right now?",
                    "has something changed in your relationships recently?",
                    "how long have you been feeling disconnected?"
                ]
            else:
                questions = [
                    "tell me more about what's on your heart?",
                    "what's really weighing on you right now?",
                    "help me understand what you're going through?"
                ]
        elif exchange_count == 2:
            # Second exchange - dig deeper into the weight and impact
            questions = [
                "how is this affecting the other areas of your life?",
                "what does this mean for you personally?",
                "who else knows what you're going through?",
                "what's the hardest part about this for you?",
                "what would need to change for you to feel some relief?"
            ]
        else:
            # Third+ exchange - more supportive, less probing
            questions = [
                "how can i best support you through this?",
                "what would bring you the most comfort right now?",
                "what do you need most in this moment?"
            ]
        
        return random.choice(questions)

    def _generate_varied_quick_replies(self, user_message: str, user_id: str) -> List[str]:
        """Provide varied quick reply suggestions"""
        msg_lower = user_message.lower()
        
        # Topic-aware quick replies
        if any(word in msg_lower for word in ['anxious', 'worried', 'stress']):
            options = [
                "tell me about your biggest worry",
                "what helps you feel calm?",
                "let's find peace together",
                "show me verses about anxiety",
                "i need a prayer for peace"
            ]
        elif any(word in msg_lower for word in ['sad', 'depressed', 'down']):
            options = [
                "what's making your heart heavy?",
                "i need encouragement",
                "show me verses about hope",
                "help me see God's presence",
                "i want to feel less alone"
            ]
        elif any(word in msg_lower for word in ['angry', 'frustrated', 'mad']):
            options = [
                "help me process this anger",
                "i need wisdom for this situation",
                "show me verses about forgiveness",
                "what would God say about this?",
                "i want to find peace"
            ]
        else:
            # General options with variety
            all_options = [
                "tell me more about how you're feeling",
                "what's really on your heart?",
                "help me understand better",
                "i want to process this with you",
                "let's explore this together",
                "show me a verse for today",
                "i need prayer",
                "what does the Bible say?",
                "help me find clarity",
                "i'm feeling lost",
                "guide me through this",
                "i need encouragement"
            ]
            # Randomly select from general options
            options = random.sample(all_options, min(8, len(all_options)))
        
        # Return 3-5 varied options
        return random.sample(options, min(random.randint(3, 5), len(options)))
    
    def _generate_alternative_actions(self, user_message: str, verse_rec: Optional[Dict], exchange_count: int) -> List[Dict[str, str]]:
        """Generate alternative action buttons to break the verse-only loop"""
        actions = []
        msg_lower = user_message.lower()
        
        # Always include "show another verse" if we have a verse
        if verse_rec:
            actions.append({"type": "another_verse", "label": "show another verse"})
        
        # Context-aware actions based on user's message
        if any(word in msg_lower for word in ['anxious', 'worried', 'stress', 'overwhelmed']):
            actions.extend([
                {"type": "prayer", "label": "prayer for peace"},
                {"type": "breathing", "label": "breathing exercise"},
                {"type": "affirmation", "label": "calming truth"}
            ])
        elif any(word in msg_lower for word in ['sad', 'depressed', 'down', 'heavy']):
            actions.extend([
                {"type": "prayer", "label": "prayer for hope"},
                {"type": "comfort", "label": "words of comfort"},
                {"type": "gratitude", "label": "gratitude practice"}
            ])
        elif any(word in msg_lower for word in ['angry', 'frustrated', 'mad']):
            actions.extend([
                {"type": "prayer", "label": "prayer for patience"},
                {"type": "reframe", "label": "reframe this situation"},
                {"type": "forgiveness", "label": "forgiveness reflection"}
            ])
        else:
            # General actions for any situation
            actions.extend([
                {"type": "prayer", "label": "short prayer"},
                {"type": "journal", "label": "journaling prompt"},
                {"type": "reflection", "label": "gentle reflection"}
            ])
        
        # Add "different angle" option after a few exchanges
        if exchange_count >= 2:
            actions.append({"type": "different_angle", "label": "try a different angle"})
        
        # Limit to 4 actions to avoid overwhelming
        return actions[:4]

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
    
    async def generate_contextual_explanation(self, explanation_prompt: str) -> str:
        """Generate AI-powered contextual explanation for verses"""
        try:
            # Try AI services in order of preference
            if openai_client and settings.has_openai_key:
                try:
                    response = openai_client.chat.completions.create(
                        model=settings.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a wise, compassionate biblical scholar and pastor who explains Scripture in practical, encouraging ways that connect to everyday life."},
                            {"role": "user", "content": explanation_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI error: {e}")
                    # Fall back to Claude if available
                    if claude_client and settings.has_anthropic_key:
                        try:
                            response = claude_client.messages.create(
                                model=settings.CLAUDE_MODEL,
                                max_tokens=800,
                                temperature=0.7,
                                system="You are a wise, compassionate biblical scholar and pastor who explains Scripture in practical, encouraging ways that connect to everyday life.",
                                messages=[{"role": "user", "content": explanation_prompt}]
                            )
                            return response.content[0].text
                        except Exception as e2:
                            print(f"Claude error: {e2}")
                            raise Exception("Both AI services failed")
                    raise e
            
            # Try Claude if OpenAI not available
            if claude_client and settings.has_anthropic_key:
                try:
                    response = claude_client.messages.create(
                        model=settings.CLAUDE_MODEL,
                        max_tokens=800,
                        temperature=0.7,
                        system="You are a wise, compassionate biblical scholar and pastor who explains Scripture in practical, encouraging ways that connect to everyday life.",
                        messages=[{"role": "user", "content": explanation_prompt}]
                    )
                    return response.content[0].text
                except Exception as e:
                    print(f"Claude error: {e}")
                    raise e
            
            # If no AI services are available, raise error
            raise Exception("No AI services configured")
            
        except Exception as e:
            print(f"Error generating contextual explanation: {e}")
            return "I'm having trouble explaining this verse right now. Please try again in a moment."
    
    # Removed fallback response - we want real AI responses only