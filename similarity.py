import re
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

class TextSimilarity:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer()
        
    def _preprocess(self, text: str) -> str:
        """Clean and normalize text using proper libraries"""
        # Expand contractions
        text = contractions.fix(text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize and remove stopwords
        tokens = [word for word in word_tokenize(text) if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def _extract_features(self, text1: str, text2: str) -> dict:
        """Extract all similarity features"""
        tokens1, tokens2 = text1.split(), text2.split()
        set1, set2 = set(tokens1), set(tokens2)
        
        # Common and total words
        common_words = len(set1 & set2)
        total_words = len(set1 | set2)
        word_share = common_words / total_words if total_words > 0 else 0
        
        # Stopwords features
        stops1 = set(w for w in tokens1 if w in self.stop_words)
        stops2 = set(w for w in tokens2 if w in self.stop_words)
        common_stops = len(stops1 & stops2)
        
        # Safe division constant
        SAFE_DIV = 0.0001
        
        # Token features
        token_features = {
            'cwc_min': common_words / (min(len(set1), len(set2)) + SAFE_DIV),
            'cwc_max': common_words / (max(len(set1), len(set2)) + SAFE_DIV),
            'csc_min': common_stops / (min(len(stops1), len(stops2)) + SAFE_DIV),
            'csc_max': common_stops / (max(len(stops1), len(stops2)) + SAFE_DIV),
            'ctc_min': common_words / (min(len(tokens1), len(tokens2)) + SAFE_DIV),
            'ctc_max': common_words / (max(len(tokens1), len(tokens2)) + SAFE_DIV),
            'last_word_eq': int(tokens1[-1] == tokens2[-1]),
            'first_word_eq': int(tokens1[0] == tokens2[0]),
            'abs_len_diff': abs(len(text1) - len(text2)),
            'mean_len': (len(text1) + len(text2)) / 2,
            'word_share': word_share
        }
        
        # Fuzzy features
        fuzzy_features = {
            'fuzz_ratio': fuzz.ratio(text1, text2) / 100,
            'fuzz_partial_ratio': fuzz.partial_ratio(text1, text2) / 100,
            'token_sort_ratio': fuzz.token_sort_ratio(text1, text2) / 100,
            'token_set_ratio': fuzz.token_set_ratio(text1, text2) / 100
        }
        
        return {**token_features, **fuzzy_features}
    
    def _get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Get semantic similarity using TF-IDF"""
        vectors = self.tfidf.fit_transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate combined similarity score (0-1)"""
        # Preprocess texts
        clean1 = self._preprocess(text1)
        clean2 = self._preprocess(text2)
        
        # Extract features
        features = self._extract_features(clean1, clean2)
        semantic_sim = self._get_semantic_similarity(clean1, clean2)
        
        # Combine features with weights
        weights = {
            'semantic_similarity': 0.4,
            'word_share': 0.2,
            'token_set_ratio': 0.15,
            'cwc_max': 0.1,
            'mean_len': 0.05,
            'first_word_eq': 0.05,
            'last_word_eq': 0.05
        }
        
        # Calculate weighted score
        score = (weights['semantic_similarity'] * semantic_sim +
                 weights['word_share'] * features['word_share'] +
                 weights['token_set_ratio'] * features['token_set_ratio'] +
                 weights['cwc_max'] * features['cwc_max'] +
                 weights['mean_len'] * (1 - features['abs_len_diff'] / max(len(clean1), len(clean2), 1)) +
                 weights['first_word_eq'] * features['first_word_eq'] +
                 weights['last_word_eq'] * features['last_word_eq'])
        
        return max(0, min(1, score))

def get_similarity_score(text1: str, text2: str) -> float:
    """Public interface for similarity calculation"""
    analyzer = TextSimilarity()
    return analyzer.calculate_similarity(text1, text2)