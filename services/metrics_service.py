import json
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio


class ConversationMetrics:
    """Service for tracking conversation quality and variety metrics"""
    
    def __init__(self):
        self.session_metrics = {}  # session_id -> metrics
        self.global_metrics = defaultdict(list)
        self.response_cache = deque(maxlen=1000)  # Recent responses for analysis
        self.testing_mode = False
    
    def start_session(self, session_id: str, user_id: str):
        """Start tracking a new conversation session"""
        self.session_metrics[session_id] = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'turns': [],
            'response_times': [],
            'variety_scores': [],
            'repetition_warnings': 0,
            'verse_count': 0,
            'question_count': 0,
            'length_compliance': [],
            'tone_consistency': [],
            'user_satisfaction_indicators': []
        }
    
    def log_turn(self, session_id: str, turn_data: Dict[str, Any]):
        """Log a single conversation turn with quality metrics"""
        if session_id not in self.session_metrics:
            self.start_session(session_id, turn_data.get('user_id', 'unknown'))
        
        session = self.session_metrics[session_id]
        
        # Basic turn info
        turn_metrics = {
            'timestamp': datetime.now(),
            'user_message': turn_data.get('user_message', ''),
            'ai_response': turn_data.get('ai_response', ''),
            'response_time': turn_data.get('response_time', 0),
            'verse_included': bool(turn_data.get('verse_recommendation')),
            'question_asked': '?' in turn_data.get('ai_response', ''),
            'word_count': len(turn_data.get('ai_response', '').split()),
            'target_length_min': turn_data.get('target_length_min', 0),
            'target_length_max': turn_data.get('target_length_max', 100),
        }
        
        # Calculate quality scores
        turn_metrics.update(self._calculate_turn_quality(turn_data, session))
        
        session['turns'].append(turn_metrics)
        self.response_cache.append(turn_data.get('ai_response', ''))
        
        # Update session aggregates
        self._update_session_aggregates(session_id, turn_metrics)
    
    def _calculate_turn_quality(self, turn_data: Dict, session: Dict) -> Dict[str, float]:
        """Calculate various quality metrics for a turn"""
        ai_response = turn_data.get('ai_response', '')
        metrics = {}
        
        # 1. Length compliance score
        word_count = len(ai_response.split())
        target_min = turn_data.get('target_length_min', 10)
        target_max = turn_data.get('target_length_max', 50)
        target_mid = (target_min + target_max) / 2
        
        if target_min <= word_count <= target_max:
            metrics['length_compliance'] = 1.0
        else:
            deviation = abs(word_count - target_mid) / target_mid
            metrics['length_compliance'] = max(0, 1 - deviation)
        
        # 2. Repetition avoidance score
        recent_responses = [turn['ai_response'] for turn in session['turns'][-5:]]
        if recent_responses:
            similarities = [self._calculate_similarity(ai_response, resp) for resp in recent_responses]
            avg_similarity = sum(similarities) / len(similarities)
            metrics['variety_score'] = max(0, 1 - avg_similarity)
        else:
            metrics['variety_score'] = 1.0
        
        # 3. Friend tone authenticity score
        friend_indicators = ['i ', 'you know', 'honestly', 'ugh', 'hey', 'that\'s ', 'what\'s ']
        friend_score = sum(1 for indicator in friend_indicators if indicator in ai_response.lower())
        metrics['friend_tone_score'] = min(1.0, friend_score / 3)  # Max score at 3+ indicators
        
        # 4. Emotional resonance score
        emotion_words = ['feel', 'heart', 'soul', 'tough', 'hard', 'heavy', 'hope', 'peace', 'love']
        emotion_score = sum(1 for word in emotion_words if word in ai_response.lower())
        metrics['emotional_resonance'] = min(1.0, emotion_score / 2)  # Max score at 2+ words
        
        # 5. Question appropriateness score
        has_question = '?' in ai_response
        should_ask_question = turn_data.get('plan_ask_question', False)
        if has_question == should_ask_question:
            metrics['question_appropriateness'] = 1.0
        else:
            metrics['question_appropriateness'] = 0.5
        
        # 6. Verse timing score (if verse was included)
        if turn_data.get('verse_recommendation'):
            turns_since_last_verse = len(session['turns']) - max([
                i for i, turn in enumerate(session['turns']) 
                if turn.get('verse_included', False)
            ] + [0])
            
            # Ideal spacing is 3-5 turns
            if 3 <= turns_since_last_verse <= 5:
                metrics['verse_timing_score'] = 1.0
            else:
                metrics['verse_timing_score'] = 0.7
        else:
            metrics['verse_timing_score'] = 1.0  # No penalty for not including verse
        
        return metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _update_session_aggregates(self, session_id: str, turn_metrics: Dict):
        """Update session-level aggregate metrics"""
        session = self.session_metrics[session_id]
        
        # Response times
        if turn_metrics.get('response_time', 0) > 0:
            session['response_times'].append(turn_metrics['response_time'])
        
        # Variety tracking
        session['variety_scores'].append(turn_metrics.get('variety_score', 0))
        
        # Length compliance
        session['length_compliance'].append(turn_metrics.get('length_compliance', 0))
        
        # Counts
        if turn_metrics.get('verse_included'):
            session['verse_count'] += 1
        if turn_metrics.get('question_asked'):
            session['question_count'] += 1
        
        # Repetition warnings
        if turn_metrics.get('variety_score', 1) < 0.3:
            session['repetition_warnings'] += 1
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics summary for a session"""
        if session_id not in self.session_metrics:
            return {}
        
        session = self.session_metrics[session_id]
        turns = session['turns']
        
        if not turns:
            return {'error': 'No turns recorded'}
        
        summary = {
            'session_info': {
                'session_id': session_id,
                'user_id': session['user_id'],
                'start_time': session['start_time'].isoformat(),
                'duration_minutes': (datetime.now() - session['start_time']).seconds / 60,
                'total_turns': len(turns)
            },
            'quality_metrics': {
                'avg_variety_score': sum(session['variety_scores']) / len(session['variety_scores']) if session['variety_scores'] else 0,
                'avg_length_compliance': sum(session['length_compliance']) / len(session['length_compliance']) if session['length_compliance'] else 0,
                'avg_friend_tone_score': sum(t.get('friend_tone_score', 0) for t in turns) / len(turns),
                'avg_emotional_resonance': sum(t.get('emotional_resonance', 0) for t in turns) / len(turns),
                'avg_question_appropriateness': sum(t.get('question_appropriateness', 1) for t in turns) / len(turns),
                'repetition_warnings': session['repetition_warnings']
            },
            'conversation_flow': {
                'verse_frequency': session['verse_count'] / len(turns) if len(turns) > 0 else 0,
                'question_frequency': session['question_count'] / len(turns) if len(turns) > 0 else 0,
                'avg_response_time': sum(session['response_times']) / len(session['response_times']) if session['response_times'] else 0
            },
            'recent_turns': [
                {
                    'user_message': turn['user_message'][:100] + '...' if len(turn['user_message']) > 100 else turn['user_message'],
                    'ai_response': turn['ai_response'][:150] + '...' if len(turn['ai_response']) > 150 else turn['ai_response'],
                    'quality_score': (
                        turn.get('variety_score', 0) + 
                        turn.get('length_compliance', 0) + 
                        turn.get('friend_tone_score', 0) + 
                        turn.get('emotional_resonance', 0)
                    ) / 4
                }
                for turn in turns[-5:]  # Last 5 turns
            ]
        }
        
        # Overall quality score (0-100)
        quality_factors = [
            summary['quality_metrics']['avg_variety_score'] * 25,  # 25% weight
            summary['quality_metrics']['avg_length_compliance'] * 20,  # 20% weight
            summary['quality_metrics']['avg_friend_tone_score'] * 25,  # 25% weight
            summary['quality_metrics']['avg_emotional_resonance'] * 15,  # 15% weight
            summary['quality_metrics']['avg_question_appropriateness'] * 15  # 15% weight
        ]
        
        summary['overall_quality_score'] = sum(quality_factors)
        
        # Quality grade
        score = summary['overall_quality_score']
        if score >= 90:
            summary['quality_grade'] = 'A'
        elif score >= 80:
            summary['quality_grade'] = 'B'
        elif score >= 70:
            summary['quality_grade'] = 'C'
        elif score >= 60:
            summary['quality_grade'] = 'D'
        else:
            summary['quality_grade'] = 'F'
        
        return summary
    
    def export_metrics(self, session_id: str = None) -> Dict[str, Any]:
        """Export metrics for analysis"""
        if session_id:
            return {
                'single_session': self.get_session_summary(session_id)
            }
        else:
            return {
                'all_sessions': {
                    sid: self.get_session_summary(sid) 
                    for sid in self.session_metrics.keys()
                },
                'global_stats': self._calculate_global_stats()
            }
    
    def _calculate_global_stats(self) -> Dict[str, Any]:
        """Calculate global statistics across all sessions"""
        if not self.session_metrics:
            return {}
        
        all_sessions = [self.get_session_summary(sid) for sid in self.session_metrics.keys()]
        valid_sessions = [s for s in all_sessions if 'quality_metrics' in s]
        
        if not valid_sessions:
            return {}
        
        return {
            'total_sessions': len(valid_sessions),
            'avg_quality_score': sum(s['overall_quality_score'] for s in valid_sessions) / len(valid_sessions),
            'avg_variety_score': sum(s['quality_metrics']['avg_variety_score'] for s in valid_sessions) / len(valid_sessions),
            'avg_turns_per_session': sum(s['session_info']['total_turns'] for s in valid_sessions) / len(valid_sessions),
            'quality_distribution': {
                'A': sum(1 for s in valid_sessions if s['quality_grade'] == 'A'),
                'B': sum(1 for s in valid_sessions if s['quality_grade'] == 'B'),
                'C': sum(1 for s in valid_sessions if s['quality_grade'] == 'C'),
                'D': sum(1 for s in valid_sessions if s['quality_grade'] == 'D'),
                'F': sum(1 for s in valid_sessions if s['quality_grade'] == 'F'),
            }
        }