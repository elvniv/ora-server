import redis
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


class RedisStateService:
    """Redis-based state management for user conversations and cooldowns"""
    
    def __init__(self):
        """Initialize Redis connection"""
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.testing_mode = os.getenv('TESTING_MODE', 'false').lower() == 'true'
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError:
            if not self.testing_mode:
                print("Warning: Redis connection failed, falling back to in-memory storage")
            self.redis_client = None
            self._fallback_storage = {}
    
    def _key_recent_questions(self, user_id: str) -> str:
        """Generate key for user's recent questions"""
        return f"u:{user_id}:recent_q"
    
    def _key_recent_verses(self, user_id: str) -> str:
        """Generate key for user's recent verses"""
        return f"u:{user_id}:recent_v"
    
    def _key_user_state(self, user_id: str) -> str:
        """Generate key for user's general state"""
        return f"u:{user_id}:state"
    
    def _key_verse_cooldown(self, verse_ref: str) -> str:
        """Generate key for global verse cooldown"""
        return f"g:last_seen_verse:{verse_ref}"
    
    def get_recent_questions(self, user_id: str, limit: int = 10) -> List[str]:
        """Get user's recent questions"""
        key = self._key_recent_questions(user_id)
        if self.redis_client:
            try:
                questions = self.redis_client.lrange(key, 0, limit - 1)
                return questions
            except redis.RedisError:
                return []
        else:
            # Fallback storage
            return list(self._fallback_storage.get(key, []))[-limit:]
    
    def add_recent_question(self, user_id: str, question: str, max_questions: int = 10):
        """Add a question to user's recent questions list"""
        key = self._key_recent_questions(user_id)
        if self.redis_client:
            try:
                # Add to left, trim to max size
                self.redis_client.lpush(key, question)
                self.redis_client.ltrim(key, 0, max_questions - 1)
                # Set expiry (30 days)
                self.redis_client.expire(key, 30 * 24 * 3600)
            except redis.RedisError:
                pass
        else:
            # Fallback storage
            if key not in self._fallback_storage:
                self._fallback_storage[key] = []
            self._fallback_storage[key].insert(0, question)
            self._fallback_storage[key] = self._fallback_storage[key][:max_questions]
    
    def get_recent_verses(self, user_id: str, limit: int = 20) -> List[str]:
        """Get user's recent verses"""
        key = self._key_recent_verses(user_id)
        if self.redis_client:
            try:
                verses = self.redis_client.lrange(key, 0, limit - 1)
                return verses
            except redis.RedisError:
                return []
        else:
            return list(self._fallback_storage.get(key, []))[-limit:]
    
    def add_recent_verse(self, user_id: str, verse_ref: str, max_verses: int = 20):
        """Add a verse to user's recent verses list"""
        key = self._key_recent_verses(user_id)
        if self.redis_client:
            try:
                self.redis_client.lpush(key, verse_ref)
                self.redis_client.ltrim(key, 0, max_verses - 1)
                self.redis_client.expire(key, 30 * 24 * 3600)
            except redis.RedisError:
                pass
        else:
            if key not in self._fallback_storage:
                self._fallback_storage[key] = []
            self._fallback_storage[key].insert(0, verse_ref)
            self._fallback_storage[key] = self._fallback_storage[key][:max_verses]
    
    def get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get user's conversation state"""
        key = self._key_user_state(user_id)
        default_state = {
            'turn_count': 0,
            'last_verse_turn': -10,  # Allow verses early in conversation
            'last_response': '',
            'conversation_seed': None
        }
        
        if self.redis_client:
            try:
                state_json = self.redis_client.get(key)
                if state_json:
                    return {**default_state, **json.loads(state_json)}
                return default_state
            except (redis.RedisError, json.JSONDecodeError):
                return default_state
        else:
            return {**default_state, **self._fallback_storage.get(key, {})}
    
    def update_user_state(self, user_id: str, state_updates: Dict[str, Any]):
        """Update user's conversation state"""
        key = self._key_user_state(user_id)
        current_state = self.get_user_state(user_id)
        current_state.update(state_updates)
        
        if self.redis_client:
            try:
                self.redis_client.setex(key, 7 * 24 * 3600, json.dumps(current_state))  # 7 days
            except redis.RedisError:
                pass
        else:
            self._fallback_storage[key] = current_state
    
    def is_verse_on_cooldown(self, verse_ref: str, cooldown_days: int = 30) -> bool:
        """Check if verse is on global cooldown"""
        key = self._key_verse_cooldown(verse_ref)
        if self.redis_client:
            try:
                last_used = self.redis_client.get(key)
                if last_used:
                    last_used_dt = datetime.fromisoformat(last_used)
                    return datetime.now() - last_used_dt < timedelta(days=cooldown_days)
                return False
            except (redis.RedisError, ValueError):
                return False
        else:
            last_used = self._fallback_storage.get(key)
            if last_used:
                try:
                    last_used_dt = datetime.fromisoformat(last_used)
                    return datetime.now() - last_used_dt < timedelta(days=cooldown_days)
                except ValueError:
                    return False
            return False
    
    def mark_verse_used(self, verse_ref: str):
        """Mark verse as recently used for cooldown"""
        key = self._key_verse_cooldown(verse_ref)
        now = datetime.now().isoformat()
        
        if self.redis_client:
            try:
                # Set with 30 day expiry
                self.redis_client.setex(key, 30 * 24 * 3600, now)
            except redis.RedisError:
                pass
        else:
            self._fallback_storage[key] = now
    
    def get_conversation_seed(self, user_id: str, message: str) -> int:
        """Get deterministic seed for conversation variety"""
        import hashlib
        state = self.get_user_state(user_id)
        
        if not state.get('conversation_seed'):
            # Create seed from user_id and current time (or deterministic for testing)
            if self.testing_mode:
                seed_input = f"{user_id}:testing"
            else:
                seed_input = f"{user_id}:{datetime.now().strftime('%Y-%m-%d')}"
            
            conversation_seed = int(hashlib.md5(seed_input.encode()).hexdigest(), 16) % 10000
            self.update_user_state(user_id, {'conversation_seed': conversation_seed})
            return conversation_seed
        
        return state['conversation_seed']