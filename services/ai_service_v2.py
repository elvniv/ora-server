import os
import re
import random
import json
import asyncio
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from openai import AsyncOpenAI
import anthropic
from collections import deque, Counter
from threading import Lock
from datetime import datetime, timedelta

from models import ChatMessage, ChatResponse
from services.bible_service import BibleService
from services.redis_service import RedisStateService
from services.embedding_service import EmbeddingService
from config import get_settings

settings = get_settings()

# Initialize async AI clients
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.has_openai_key else None
claude_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.has_anthropic_key else None


class FriendlyAIService:
    """Rebuilt AI service with friend tone and anti-monotony guardrails"""
    
    def __init__(self):
        self.bible_service = BibleService()
        self.tone = "friend"  # Fixed to friend tone
        
        # Guardrails
        self.max_questions_per_turn = 1
        self.verse_frequency_limit = 3  # Max 1 verse every 3 turns
        self.length_band_tolerance = 0.2  # ±20%
        
        # Services
        self.redis_service = RedisStateService()
        self.embedding_service = EmbeddingService()
        self._lock = Lock()  # Keep for verse corpus access
        
        # Load verse corpus
        self._load_verse_corpus()
        
        # Testing config
        self.testing_mode = os.getenv('TESTING_MODE', 'false').lower() == 'true'
        
        # Initialize embeddings (async, happens in background)
        asyncio.create_task(self._initialize_embeddings())
        
    def _load_verse_corpus(self):
        """Load expanded verse corpus with metadata"""
        try:
            corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'verse_corpus.json')
            if os.path.exists(corpus_path):
                with open(corpus_path, 'r') as f:
                    self.verse_corpus = json.load(f)
            else:
                # Fallback to smaller corpus for now
                self._build_basic_corpus()
        except Exception as e:
            print(f"Error loading verse corpus: {e}")
            self._build_basic_corpus()
    
    def _build_basic_corpus(self):
        """Build basic verse corpus until we have the full one"""
        self.verse_corpus = {
            "anxiety": [
                {"ref": "Philippians 4:6-7", "book": "Philippians", "genre": "epistle", "weight": 10},
                {"ref": "1 Peter 5:7", "book": "1 Peter", "genre": "epistle", "weight": 9},
                {"ref": "Matthew 6:34", "book": "Matthew", "genre": "gospel", "weight": 8},
                {"ref": "Psalm 55:22", "book": "Psalms", "genre": "poetry", "weight": 7},
                {"ref": "Isaiah 41:10", "book": "Isaiah", "genre": "prophecy", "weight": 8},
                {"ref": "John 14:27", "book": "John", "genre": "gospel", "weight": 7}
            ],
            "peace": [
                {"ref": "John 14:27", "book": "John", "genre": "gospel", "weight": 10},
                {"ref": "Isaiah 26:3", "book": "Isaiah", "genre": "prophecy", "weight": 9},
                {"ref": "Philippians 4:7", "book": "Philippians", "genre": "epistle", "weight": 8},
                {"ref": "Psalm 29:11", "book": "Psalms", "genre": "poetry", "weight": 7}
            ],
            "strength": [
                {"ref": "Isaiah 40:31", "book": "Isaiah", "genre": "prophecy", "weight": 10},
                {"ref": "Philippians 4:13", "book": "Philippians", "genre": "epistle", "weight": 10},
                {"ref": "2 Corinthians 12:9", "book": "2 Corinthians", "genre": "epistle", "weight": 9}
            ]
        }
    
    async def _initialize_embeddings(self):
        """Initialize embeddings for verse corpus"""
        try:
            await self.embedding_service.build_verse_embeddings(self.verse_corpus)
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
    
    def _get_user_state(self, user_id: str) -> Dict:
        """Get user conversation state from Redis"""
        state = self.redis_service.get_user_state(user_id)
        
        # Add convenience methods for questions and verses
        state['recent_questions'] = self.redis_service.get_recent_questions(user_id)
        state['recent_verses'] = self.redis_service.get_recent_verses(user_id)
        state['conversation_seed'] = self.redis_service.get_conversation_seed(user_id, "seed")
        
        return state
    
    def _plan_response(self, message: str, user_state: Dict) -> Dict[str, Any]:
        """Mini planner: decide tone, verse inclusion, target length"""
        word_count = len(message.strip().split())
        turn_count = user_state['turn_count']
        last_verse_turn = user_state['last_verse_turn']
        
        # Length band (±20%)
        min_words = max(5, int(word_count * (1 - self.length_band_tolerance)))
        max_words = int(word_count * (1 + self.length_band_tolerance)) + 5
        
        # Verse inclusion logic
        turns_since_verse = turn_count - last_verse_turn
        can_include_verse = turns_since_verse >= self.verse_frequency_limit
        should_include_verse = can_include_verse and turn_count >= 2 and word_count >= 8
        
        # Direct question detection
        is_direct_question = self._is_direct_question(message)
        
        return {
            'tone': self.tone,
            'include_verse': should_include_verse and not is_direct_question,
            'target_len_min': min_words,
            'target_len_max': max_words,
            'ask_question': not is_direct_question and turn_count % 2 == 1,  # Alternate cadence
            'is_direct_question': is_direct_question
        }
    
    def _is_direct_question(self, message: str) -> bool:
        """Enhanced question detection"""
        msg = message.strip().lower()
        
        if '?' in msg:
            return True
            
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'should', 'could', 'would']
        question_phrases = ['wondering', 'curious', 'want to know', 'tell me', 'explain', 'help me understand']
        
        first_word = msg.split()[0] if msg.split() else ''
        if first_word in question_words:
            return True
            
        return any(phrase in msg for phrase in question_phrases)
    
    def _build_system_prompt(self, plan: Dict, user_context: Dict) -> str:
        """Build request-scoped system prompt with friend tone"""
        recent_questions = list(user_context.get('recent_questions', []))
        
        base_prompt = f"""You are ORA, a caring friend who listens deeply and offers gentle spiritual wisdom. You're like that friend who really gets it - warm, genuine, and never preachy.

FRIEND TONE GUIDELINES:
- Talk like you're having coffee with a close friend
- Use "I" statements: "I hear you", "I can feel that weight", "I've been there too"
- Be conversational, not counselor-y: "man, that's rough" vs "I understand your struggle"
- Lowercase for warmth, contractions for naturalness
- Share the emotional load: "we've all felt that crushing feeling"

LENGTH TARGET: {plan['target_len_min']}-{plan['target_len_max']} words (flexible)
- Match their investment level
- Short message = short, caring response
- Longer share = more engaged reply

CONVERSATION RULES:
- {"Answer their question directly first, then maybe add one caring thought" if plan['is_direct_question'] else "You can ask ONE question if it feels natural, but don't probe"}
- Recently asked questions (avoid): {recent_questions}
- {"Include a verse recommendation if you find one that really fits" if plan['include_verse'] else "Focus on connection and understanding - no verses this turn"}
- Vary your sentence structure - some short, some flowing
- Never be repetitive or templated

FRIEND AUTHENTICITY:
- "ugh, that's so hard" instead of "that must be difficult"  
- "i get why you'd feel that way" instead of "your feelings are valid"
- "honestly, i think..." instead of formal wisdom
- Use pauses: "you know what? that reminds me of..."
"""
        
        return base_prompt
    
    async def _generate_candidates(self, message: str, system_prompt: str) -> List[Dict]:
        """Generate 3 response candidates in structured JSON"""
        candidate_prompt = f"""
{system_prompt}

User message: "{message}"

Generate 3 different response options as JSON array. Each must be valid JSON with:
{{"open": "opening phrase", "body": "main response", "closer": "ending phrase", "ask_question": true/false, "question": "optional question"}}

Vary the rhythm - mix short/long sentences, different punctuation patterns.
"""
        
        try:
            # In testing mode, use deterministic mock responses
            if self.testing_mode:
                return await self._generate_mock_candidates(message)
            
            # Production candidate generation via API
            if openai_client:
                response = await openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[{"role": "user", "content": candidate_prompt}],
                    temperature=0.8,
                    max_tokens=800
                )
                content = response.choices[0].message.content
            else:
                # Fallback to structured mock if no API
                return await self._generate_structured_fallback(message)
            
            # Parse JSON candidates
            candidates = json.loads(content)
            if isinstance(candidates, list) and len(candidates) > 0:
                return candidates[:3]  # Max 3
            else:
                return await self._generate_structured_fallback(message)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, extract from text 
            return await self._parse_unstructured_response(content)
        except Exception as e:
            print(f"Error generating candidates: {e}")
            return await self._generate_structured_fallback(message)
    
    async def _generate_mock_candidates(self, message: str) -> List[Dict]:
        """Generate deterministic mock candidates for testing"""
        # Enhanced mock response sets with more variety
        mock_response_sets = [
            # Anxiety/worry responses
            [
                {"open": "honestly", "body": "that sounds really tough, and i can feel the weight of what you're carrying", "closer": "you're not walking this alone", "ask_question": True, "question": "what's been the hardest part for you?"},
                {"open": "hey", "body": "i get it - that anxious feeling can be so overwhelming", "closer": "want to talk through what's stirring up?", "ask_question": True, "question": "what's triggering this most right now?"},
                {"open": "ugh", "body": "that feeling is the worst, isn't it? like your mind won't stop racing", "closer": "i'm here and listening", "ask_question": False, "question": ""}
            ],
            # Hope/scripture responses  
            [
                {"open": "you know", "body": "that verse hits different when life feels chaotic - it's like god saying he's weaving even the messy parts", "closer": "there's so much hope in that promise", "ask_question": True, "question": "where do you most need to see god working?"},
                {"open": "oh", "body": "that's such a beautiful verse - it's basically god saying he sees the bigger picture", "closer": "even when we can only see this moment", "ask_question": False, "question": ""},
                {"open": "honestly", "body": "romans 8:28 reminds us that nothing - not even our struggles - is wasted in god's hands", "closer": "that's pretty incredible when you think about it", "ask_question": True, "question": "what part feels hardest to trust right now?"}
            ],
            # General support responses
            [
                {"open": "i hear you", "body": "that sounds like you're dealing with a lot right now", "closer": "how are you holding up?", "ask_question": True, "question": "what's been helping you get through?"},
                {"open": "man", "body": "that's heavy stuff you're processing", "closer": "it's okay to not have it all figured out", "ask_question": False, "question": ""},
                {"open": "thanks for sharing", "body": "it takes courage to open up about what you're going through", "closer": "your heart matters in all this", "ask_question": True, "question": "what do you need most right now?"}
            ]
        ]
        
        # Use message content to pick appropriate response set
        import hashlib
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['anxious', 'worried', 'stress', 'overwhelm']):
            response_set = 0
        elif any(word in message_lower for word in ['romans', 'verse', 'bible', 'scripture', 'god']):
            response_set = 1  
        else:
            response_set = 2
            
        # Add deterministic variation within the set
        hash_val = int(hashlib.md5(message.encode()).hexdigest(), 16)
        
        # Select base responses and add variety
        base_responses = mock_response_sets[response_set]
        
        # Rotate responses based on hash for variety
        rotated_responses = base_responses[hash_val % len(base_responses):] + base_responses[:hash_val % len(base_responses)]
        
        return rotated_responses[:3]
    
    async def _generate_structured_fallback(self, message: str) -> List[Dict]:
        """Generate structured fallback responses when API unavailable"""
        # Analyze message sentiment and create appropriate responses
        msg_lower = message.lower()
        
        openers = ["honestly", "hey", "i hear you", "you know", "ugh", "man"]
        
        if any(word in msg_lower for word in ['anxious', 'worried', 'stress']):
            return [
                {"open": "honestly", "body": "anxiety can feel so overwhelming - i get why you'd be struggling with this", "closer": "you're not alone", "ask_question": True, "question": "what's weighing on you most?"},
                {"open": "hey", "body": "that anxious feeling is rough, and it's okay to not be okay", "closer": "", "ask_question": False, "question": ""},
                {"open": "i hear you", "body": "stress like that can feel all-consuming sometimes", "closer": "want to talk through what's going on?", "ask_question": True, "question": "what's been the biggest trigger?"}
            ]
        elif any(word in msg_lower for word in ['sad', 'down', 'depressed', 'hopeless']):
            return [
                {"open": "ugh", "body": "that heavy feeling is so hard to carry", "closer": "i'm here with you in this", "ask_question": False, "question": ""},
                {"open": "honestly", "body": "when everything feels dark, it's hard to see any light", "closer": "but you're not walking this alone", "ask_question": True, "question": "what's been the hardest part?"},
                {"open": "i hear you", "body": "those deep struggles take so much out of you", "closer": "your heart matters", "ask_question": False, "question": ""}
            ]
        else:
            return [
                {"open": "i hear you", "body": "sounds like you're processing some important stuff", "closer": "thanks for sharing", "ask_question": True, "question": "what's on your heart?"},
                {"open": "honestly", "body": "that takes courage to open up about", "closer": "i'm listening", "ask_question": False, "question": ""},
                {"open": "you know", "body": "life throws us these moments that just hit different", "closer": "how are you doing with all this?", "ask_question": True, "question": "what do you need right now?"}
            ]
    
    async def _parse_unstructured_response(self, content: str) -> List[Dict]:
        """Parse unstructured API response into candidate format"""
        # Simple parsing fallback - split by numbers or line breaks
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        candidates = []
        for i, line in enumerate(lines[:3]):
            # Simple heuristic parsing
            if '?' in line:
                question = line.split('?')[0] + '?'
                body = line.replace(question, '').strip()
                candidates.append({"open": "honestly", "body": body or line, "closer": "", "ask_question": True, "question": question})
            else:
                candidates.append({"open": "i hear you", "body": line, "closer": "", "ask_question": False, "question": ""})
                
        return candidates or await self._generate_structured_fallback("")
    
    def _rerank_candidates(self, candidates: List[Dict], plan: Dict, user_context: Dict, last_response: str = "") -> Dict:
        """Enhanced reranking with multiple quality criteria"""
        scores = []
        
        for candidate in candidates:
            score = 0
            response_text = f"{candidate['open']} {candidate['body']} {candidate['closer']}".strip()
            word_count = len(response_text.split())
            
            # 1. Length band compliance (0-15 points)
            target_mid = (plan['target_len_min'] + plan['target_len_max']) / 2
            length_deviation = abs(word_count - target_mid) / target_mid
            if length_deviation <= 0.1:  # Within 10% of target
                score += 15
            elif length_deviation <= 0.2:  # Within 20% (the tolerance)
                score += 10
            elif length_deviation <= 0.4:  # Within 40%
                score += 5
            
            # 2. Bigram overlap avoidance (0-12 points)
            if last_response:
                overlap = self._bigram_overlap(response_text, last_response)
                if overlap < 0.3:
                    score += 12
                elif overlap < 0.5:
                    score += 8
                elif overlap < 0.7:
                    score += 4
            else:
                score += 12  # First response gets full points
            
            # 3. Question handling adherence (0-10 points)
            if plan['ask_question'] and candidate['ask_question']:
                question = candidate.get('question', '')
                if question and question not in user_context.get('recent_questions', []):
                    score += 10
                elif question:
                    score += 5  # Repeated question but still following plan
            elif not plan['ask_question'] and not candidate['ask_question']:
                score += 10
            
            # 4. Friend tone authenticity (0-8 points)
            friend_indicators = ['i ', 'you know', 'honestly', 'ugh', 'hey', 'that\'s ', 'what\'s ', 'it\'s like']
            friend_score = sum(1 for indicator in friend_indicators if indicator in response_text.lower())
            score += min(friend_score * 2, 8)
            
            # 5. Sentence variety (0-6 points)
            sentences = [s.strip() for s in response_text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            if len(sentences) > 1:
                # Check for variety in sentence lengths
                lengths = [len(s.split()) for s in sentences]
                if len(set(lengths)) > 1:  # Different lengths
                    score += 6
                else:
                    score += 3
            
            # 6. Emotional resonance (0-5 points)
            emotion_words = ['feel', 'heart', 'soul', 'tough', 'hard', 'heavy', 'burden', 'struggle', 'hope', 'peace']
            emotion_score = min(len([w for w in emotion_words if w in response_text.lower()]), 3)
            score += emotion_score + (2 if emotion_score > 0 else 0)
            
            # 7. Avoid over-repetition of common phrases (penalty)
            common_phrases = ['i hear you', 'that sounds', 'you know what']
            repeated_phrases = sum(1 for phrase in common_phrases if phrase in response_text.lower())
            if repeated_phrases > 1:
                score -= 5
            
            scores.append((score, candidate, response_text))
        
        # Return best candidate with debug info
        best = max(scores, key=lambda x: x[0])
        
        if self.testing_mode:
            print(f"Reranking scores: {[(s[0], s[2][:50] + '...') for s in sorted(scores, reverse=True)]}")
        
        return {"candidate": best[1], "response_text": best[2], "score": best[0]}
    
    def _bigram_overlap(self, text_a: str, text_b: str) -> float:
        """Calculate bigram overlap ratio"""
        if not text_a or not text_b:
            return 0.0
            
        words_a = text_a.lower().split()
        words_b = text_b.lower().split()
        
        if len(words_a) < 2 or len(words_b) < 2:
            return 0.0
            
        bigrams_a = {(words_a[i], words_a[i+1]) for i in range(len(words_a)-1)}
        bigrams_b = {(words_b[i], words_b[i+1]) for i in range(len(words_b)-1)}
        
        if not bigrams_a:
            return 0.0
            
        return len(bigrams_a & bigrams_b) / len(bigrams_a)
    
    def _diversify_by_book(self, candidates: List[Dict], limit: int = 5) -> List[Dict]:
        """Diversify verse candidates by book"""
        seen_books = set()
        diversified = []
        
        for verse in candidates:
            book = verse.get('book', '')
            if book not in seen_books:
                diversified.append(verse)
                seen_books.add(book)
                if len(diversified) >= limit:
                    break
        
        return diversified or candidates[:limit]
    
    def _on_cooldown(self, verse_ref: str, user_id: str) -> bool:
        """Check verse cooldown (30d global, user-specific logic)"""
        # Global cooldown (30 days)
        if self.redis_service.is_verse_on_cooldown(verse_ref, cooldown_days=30):
            return True
        
        # User-specific cooldown (last 10 verses)
        recent_verses = self.redis_service.get_recent_verses(user_id, limit=10)
        if verse_ref in recent_verses:
            return True
            
        return False
    
    async def _get_verse_recommendation(self, message: str, user_id: str, translation: str = 'niv') -> Optional[Dict]:
        """Get verse recommendation using semantic embeddings"""
        try:
            # Get recent verses to exclude
            recent_verses = self.redis_service.get_recent_verses(user_id, limit=10)
            
            # Use embeddings for semantic search
            similar_verses = await self.embedding_service.find_similar_verses(
                query=message,
                top_k=15,
                exclude_refs=recent_verses,
                book_diversity=True
            )
            
            # Filter by cooldown
            available = []
            for verse_ref, similarity, verse_data in similar_verses:
                if not self._on_cooldown(verse_ref, user_id):
                    available.append((verse_ref, similarity, verse_data))
            
            if not available and similar_verses:
                # Fallback: use top verse even if on cooldown
                available = [similar_verses[0]]
            
            if not available:
                # Final fallback to basic topic matching
                return await self._get_basic_verse_recommendation(message, user_id, translation)
            
            # Deterministic selection based on user seed
            user_state = self._get_user_state(user_id)
            random.seed(user_state['conversation_seed'] + user_state['turn_count'])
            
            # Weight by similarity score and verse weight
            weighted_candidates = []
            for verse_ref, similarity, verse_data in available[:5]:  # Top 5
                combined_weight = max(0.1, similarity * verse_data.get('weight', 5))  # Ensure positive weight
                weighted_candidates.append((verse_ref, combined_weight, verse_data))
            
            if not weighted_candidates:
                return await self._get_basic_verse_recommendation(message, user_id, translation)
            
            # Select based on combined weights
            weights = [weight for _, weight, _ in weighted_candidates]
            if sum(weights) == 0:
                # Fallback to equal weights
                weights = [1.0] * len(weighted_candidates)
                
            selected_idx = random.choices(range(len(weighted_candidates)), weights=weights, k=1)[0]
            verse_ref, _, verse_data = weighted_candidates[selected_idx]
            
            selected = {
                'ref': verse_ref,
                'book': verse_data.get('book', ''),
                'genre': verse_data.get('genre', ''),
                'weight': verse_data.get('weight', 5)
            }
        
        except Exception as e:
            print(f"Error in embedding-based verse selection: {e}")
            return await self._get_basic_verse_recommendation(message, user_id, translation)
        
        try:
            verse_data = await self.bible_service.get_verse(selected['ref'], translation)
            if verse_data and verse_data.get('text'):
                return {
                    'verse_reference': selected['ref'],
                    'verse_text': verse_data['text'],
                    'explanation': f"this verse spoke to me when I read your message - {selected['ref']} feels like it fits what you're going through"
                }
        except Exception as e:
            print(f"Error fetching verse: {e}")
            
        return None
    
    async def _get_basic_verse_recommendation(self, message: str, user_id: str, translation: str = 'niv') -> Optional[Dict]:
        """Fallback verse recommendation using topic matching"""
        topics = ['anxiety', 'peace', 'strength', 'hope', 'depression']
        matched_topic = None
        
        msg_lower = message.lower()
        if any(word in msg_lower for word in ['anxious', 'worried', 'stress']):
            matched_topic = 'anxiety'
        elif any(word in msg_lower for word in ['peace', 'calm', 'rest']):
            matched_topic = 'peace'  
        elif any(word in msg_lower for word in ['weak', 'tired', 'strength']):
            matched_topic = 'strength'
        elif any(word in msg_lower for word in ['hope', 'future', 'better']):
            matched_topic = 'hope'
        elif any(word in msg_lower for word in ['sad', 'depressed', 'down']):
            matched_topic = 'depression'
        else:
            matched_topic = random.choice(topics)
        
        if matched_topic not in self.verse_corpus:
            return None
            
        # Get candidates and apply cooldown filter
        candidates = self.verse_corpus[matched_topic]
        available = [v for v in candidates if not self._on_cooldown(v['ref'], user_id)]
        
        if not available:
            available = candidates[:3]  # Fallback to top 3 if all on cooldown
            
        if not available:
            return None
            
        # Deterministic seed for consistency
        user_state = self._get_user_state(user_id)
        random.seed(user_state['conversation_seed'] + user_state['turn_count'])
        
        selected = random.choice(available)
        
        try:
            verse_data = await self.bible_service.get_verse(selected['ref'], translation)
            if verse_data and verse_data.get('text'):
                return {
                    'verse_reference': selected['ref'],
                    'verse_text': verse_data['text'],
                    'explanation': f"this verse came to mind when I read your message - {selected['ref']} feels like it fits"
                }
        except Exception as e:
            print(f"Error fetching basic verse: {e}")
            
        return None
    
    async def generate_response(self, message: ChatMessage) -> ChatResponse:
        """Main response generation with new architecture"""
        try:
            user_id = message.user_id
            user_state = self._get_user_state(user_id)
            
            # Update turn count in Redis
            self.redis_service.update_user_state(user_id, {
                'turn_count': user_state['turn_count'] + 1
            })
            user_state['turn_count'] += 1
            
            # Phase 1: Plan the response
            plan = self._plan_response(message.message, user_state)
            
            # Get last response for bigram check
            last_response = getattr(user_state, 'last_response', '')
            
            # Phase 1: Generate candidates  
            system_prompt = self._build_system_prompt(plan, user_state)
            candidates = await self._generate_candidates(message.message, system_prompt)
            
            # Phase 1: Rerank candidates
            best = self._rerank_candidates(candidates, plan, user_state, last_response)
            response_text = best['response_text']
            selected_candidate = best['candidate']
            
            # Track question if asked
            if selected_candidate.get('ask_question') and selected_candidate.get('question'):
                self.redis_service.add_recent_question(user_id, selected_candidate['question'])
            
            # Phase 1: Get verse if planned
            verse_rec = None
            if plan['include_verse']:
                verse_rec = await self._get_verse_recommendation(
                    message.message, user_id, message.preferred_translation or 'niv'
                )
                if verse_rec:
                    # Update state and add to recent verses
                    self.redis_service.update_user_state(user_id, {
                        'last_verse_turn': user_state['turn_count']
                    })
                    self.redis_service.add_recent_verse(user_id, verse_rec['verse_reference'])
                    # Mark verse as used for global cooldown
                    self.redis_service.mark_verse_used(verse_rec['verse_reference'])
            
            # Store for next bigram check
            self.redis_service.update_user_state(user_id, {
                'last_response': response_text
            })
            
            # Quick replies (friend tone)
            quick_replies = self._generate_friend_quick_replies(message.message)
            
            # Alternative actions
            alternative_actions = [
                {"type": "another_verse", "label": "show me another verse"},
                {"type": "different_angle", "label": "different angle"},
                {"type": "prayer", "label": "short prayer"}
            ]
            
            return ChatResponse(
                response=response_text,
                verse_recommendation=verse_rec,
                additional_verses=None,
                follow_up_question=selected_candidate.get('question') if selected_candidate.get('ask_question') else None,
                quick_replies=quick_replies,
                alternative_actions=alternative_actions,
                journal_prompts=None,
                reflection_prompts=None
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Graceful fallback
            return ChatResponse(
                response="hey, i'm here with you. tell me more about what's on your heart?",
                verse_recommendation=None,
                additional_verses=None,
                follow_up_question=None,
                quick_replies=["i'm struggling", "need encouragement", "just want to talk"],
                alternative_actions=[{"type": "prayer", "label": "short prayer"}],
                journal_prompts=None,
                reflection_prompts=None
            )
    
    def _generate_friend_quick_replies(self, message: str) -> List[str]:
        """Generate friend-toned quick replies"""
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ['anxious', 'worried', 'stress']):
            return ["what's weighing on you most?", "want to talk through it?", "need a moment to breathe?"]
        elif any(word in msg_lower for word in ['sad', 'down', 'heavy']):
            return ["i'm here for you", "want to share more?", "need some encouragement?"] 
        else:
            return ["tell me more", "i'm listening", "what's on your heart?", "need support?"]