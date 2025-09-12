import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
import asyncio
from config import get_settings

settings = get_settings()

# Initialize async OpenAI client for embeddings
embeddings_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.has_openai_key else None


class EmbeddingService:
    """Service for generating and working with verse embeddings"""
    
    def __init__(self):
        self.embeddings_cache = {}
        self.verse_embeddings = {}
        self.testing_mode = os.getenv('TESTING_MODE', 'false').lower() == 'true'
        
        # Try to load pre-computed embeddings
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load pre-computed embeddings from cache file"""
        try:
            embeddings_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'data', 
                'verse_embeddings.json'
            )
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'r') as f:
                    data = json.load(f)
                    # Convert list embeddings back to numpy arrays
                    for verse_ref, embedding_data in data.items():
                        self.verse_embeddings[verse_ref] = {
                            'embedding': np.array(embedding_data['embedding']),
                            'text': embedding_data['text'],
                            'tags': embedding_data.get('tags', [])
                        }
                print(f"Loaded {len(self.verse_embeddings)} verse embeddings from cache")
        except Exception as e:
            print(f"Could not load cached embeddings: {e}")
    
    def _save_embeddings(self):
        """Save embeddings to cache file"""
        try:
            embeddings_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'data', 
                'verse_embeddings.json'
            )
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for verse_ref, data in self.verse_embeddings.items():
                serializable_data[verse_ref] = {
                    'embedding': data['embedding'].tolist(),
                    'text': data['text'],
                    'tags': data.get('tags', [])
                }
            
            with open(embeddings_path, 'w') as f:
                json.dump(serializable_data, f)
            print(f"Saved {len(self.verse_embeddings)} verse embeddings to cache")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text string"""
        if self.testing_mode:
            # Return deterministic mock embedding for testing
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            # Create a pseudo-random but deterministic 1536-dimensional vector
            np.random.seed(hash_val % 2**32)
            return np.random.normal(0, 1, 1536)
        
        if not embeddings_client:
            return None
            
        try:
            response = await embeddings_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            return np.array(embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    async def build_verse_embeddings(self, verse_corpus: Dict[str, List[Dict]]):
        """Build embeddings for all verses in the corpus"""
        print("Building verse embeddings...")
        
        for topic, verses in verse_corpus.items():
            print(f"Processing {len(verses)} verses for topic: {topic}")
            
            for verse_data in verses:
                verse_ref = verse_data['ref']
                
                # Skip if we already have this embedding
                if verse_ref in self.verse_embeddings:
                    continue
                
                # Create embedding text from reference, topic, and tags
                embedding_text = f"{verse_ref} {topic} " + " ".join(verse_data.get('tags', []))
                
                embedding = await self.get_embedding(embedding_text)
                if embedding is not None:
                    self.verse_embeddings[verse_ref] = {
                        'embedding': embedding,
                        'text': embedding_text,
                        'tags': verse_data.get('tags', []),
                        'topic': topic,
                        'book': verse_data.get('book', ''),
                        'genre': verse_data.get('genre', ''),
                        'weight': verse_data.get('weight', 5)
                    }
                
                # Small delay to respect API rate limits
                if not self.testing_mode:
                    await asyncio.sleep(0.1)
        
        # Save to cache
        self._save_embeddings()
        print(f"Completed building embeddings for {len(self.verse_embeddings)} verses")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norms == 0:
                return 0.0
            return float(dot_product / norms)
        except Exception:
            return 0.0
    
    async def find_similar_verses(self, 
                                 query: str, 
                                 top_k: int = 10,
                                 exclude_refs: List[str] = None,
                                 book_diversity: bool = True) -> List[Tuple[str, float, Dict]]:
        """Find verses most similar to the query"""
        if not self.verse_embeddings:
            return []
        
        exclude_refs = exclude_refs or []
        query_embedding = await self.get_embedding(query)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        similarities = []
        for verse_ref, verse_data in self.verse_embeddings.items():
            if verse_ref in exclude_refs:
                continue
                
            similarity = self.cosine_similarity(query_embedding, verse_data['embedding'])
            similarities.append((verse_ref, similarity, verse_data))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not book_diversity:
            return similarities[:top_k]
        
        # Apply book diversification
        diversified = []
        seen_books = set()
        
        for verse_ref, similarity, verse_data in similarities:
            book = verse_data.get('book', '')
            if book not in seen_books or len(diversified) < top_k // 2:
                diversified.append((verse_ref, similarity, verse_data))
                seen_books.add(book)
                if len(diversified) >= top_k:
                    break
        
        # Fill remaining slots with highest similarities regardless of book
        if len(diversified) < top_k:
            remaining_needed = top_k - len(diversified)
            diversified_refs = {ref for ref, _, _ in diversified}
            
            for verse_ref, similarity, verse_data in similarities:
                if verse_ref not in diversified_refs:
                    diversified.append((verse_ref, similarity, verse_data))
                    remaining_needed -= 1
                    if remaining_needed <= 0:
                        break
        
        return diversified[:top_k]
    
    def get_topic_keywords(self, message: str) -> List[str]:
        """Extract potential topic keywords from a message"""
        message_lower = message.lower()
        
        keyword_mappings = {
            'anxiety': ['anxious', 'worried', 'worry', 'nervous', 'stress', 'stressed', 'panic', 'fear', 'afraid', 'overwhelmed'],
            'depression': ['depressed', 'sad', 'hopeless', 'down', 'discouraged', 'despair', 'lonely', 'empty', 'worthless'],
            'hope': ['hope', 'future', 'tomorrow', 'better', 'optimistic', 'faith', 'trust', 'believe', 'confident'],
            'peace': ['peace', 'calm', 'rest', 'quiet', 'tranquil', 'serene', 'still', 'comfort'],
            'strength': ['strength', 'strong', 'power', 'energy', 'courage', 'brave', 'endure', 'persevere', 'overcome']
        }
        
        detected_topics = []
        for topic, keywords in keyword_mappings.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics or ['hope']  # Default to hope if no specific topic detected