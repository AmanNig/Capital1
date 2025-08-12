"""
Language Detection Module
Handles Hindi, English, and code-mixed text detection
"""

import re
from typing import Dict, List, Tuple, Optional
from langdetect import detect, detect_langs, LangDetectException
from transformers import pipeline
import numpy as np


class LanguageDetector:
    """
    Advanced language detector for Hindi/English code-mixed text
    """
    
    def __init__(self, use_transformer: bool = True):
        self.use_transformer = use_transformer
        self.hindi_patterns = [
            r'[\u0900-\u097F]',  # Devanagari script
            r'[क-ह]',  # Hindi consonants
            r'[अ-औ]',  # Hindi vowels
            r'[ा-ौ]',  # Hindi matras
        ]
        
        # Common Hindi words and phrases
        self.hindi_indicators = {
            'agricultural': [
                'फसल', 'खेती', 'किसान', 'बीज', 'पानी', 'खाद', 'मिट्टी',
                'सिंचाई', 'कटाई', 'बुवाई', 'रोपाई', 'पौधा', 'पत्ती',
                'फूल', 'फल', 'जड़', 'तना', 'शाखा', 'पत्ता'
            ],
            'common': [
                'है', 'हैं', 'था', 'थी', 'थे', 'क्या', 'कैसे', 'कहाँ',
                'कब', 'कौन', 'क्यों', 'में', 'पर', 'से', 'को', 'का',
                'की', 'के', 'और', 'या', 'लेकिन', 'फिर', 'अब', 'आज'
            ]
        }
        
        if use_transformer:
            try:
                self.transformer_detector = pipeline(
                    "text-classification",
                    model="papluca/xlm-roberta-base-language-detection",
                    top_k=None  # return all scores without using deprecated return_all_scores
                )
            except Exception as e:
                print(f"Transformer model not available: {e}")
                self.use_transformer = False
    
    def detect_language(self, text: str) -> Dict[str, float]:
        """
        Detect language composition of the text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with language probabilities
        """
        if not text.strip():
            return {'unknown': 1.0}
        
        # Method 1: Rule-based detection
        rule_based = self._rule_based_detection(text)
        
        # Method 2: Langdetect
        try:
            langdetect_result = detect_langs(text)
            langdetect_scores = {lang.lang: lang.prob for lang in langdetect_result}
        except LangDetectException:
            langdetect_scores = {'unknown': 1.0}
        
        # Method 3: Transformer-based (if available)
        transformer_scores = {}
        if self.use_transformer:
            try:
                transformer_result = self.transformer_detector(text)
                # Handle both shapes: list of dicts, or list containing a list of dicts
                if isinstance(transformer_result, list) and transformer_result and isinstance(transformer_result[0], list):
                    candidates = transformer_result[0]
                else:
                    candidates = transformer_result if isinstance(transformer_result, list) else []

                transformer_scores = {
                    item['label'].lower(): item['score']
                    for item in candidates
                }
            except Exception:
                pass
        
        # Combine results with weighted averaging
        final_scores = self._combine_detection_methods(
            rule_based, langdetect_scores, transformer_scores
        )
        
        return final_scores
    
    def _rule_based_detection(self, text: str) -> Dict[str, float]:
        """Rule-based language detection using patterns and word lists"""
        text_lower = text.lower()
        
        # Count Hindi characters
        hindi_char_count = 0
        for pattern in self.hindi_patterns:
            hindi_char_count += len(re.findall(pattern, text))
        
        # Count Hindi words
        hindi_word_count = 0
        words = text_lower.split()
        for word in words:
            for hindi_words in self.hindi_indicators.values():
                if word in hindi_words:
                    hindi_word_count += 1
                    break
        
        # Calculate scores
        total_chars = len(text.replace(' ', ''))
        total_words = len(words)
        
        if total_chars == 0:
            return {'unknown': 1.0}
        
        hindi_score = (hindi_char_count / total_chars) * 0.7 + (hindi_word_count / max(total_words, 1)) * 0.3
        english_score = 1 - hindi_score
        
        return {
            'hi': max(0, hindi_score),
            'en': max(0, english_score)
        }
    
    def _combine_detection_methods(
        self, 
        rule_based: Dict[str, float],
        langdetect: Dict[str, float],
        transformer: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine multiple detection methods with weighted averaging"""
        
        # Normalize scores
        def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
            total = sum(scores.values())
            return {k: v/total if total > 0 else 0 for k, v in scores.items()}
        
        rule_based = normalize_scores(rule_based)
        langdetect = normalize_scores(langdetect)
        
        # Weights for different methods
        weights = {
            'rule_based': 0.4,
            'langdetect': 0.3,
            'transformer': 0.3
        }
        
        # Combine scores
        combined = {}
        all_languages = set(rule_based.keys()) | set(langdetect.keys()) | set(transformer.keys())
        
        for lang in all_languages:
            score = 0
            score += rule_based.get(lang, 0) * weights['rule_based']
            score += langdetect.get(lang, 0) * weights['langdetect']
            score += transformer.get(lang, 0) * weights['transformer']
            combined[lang] = score
        
        return normalize_scores(combined)
    
    def is_code_mixed(self, text: str, threshold: float = 0.3) -> bool:
        """
        Detect if text is code-mixed (contains both Hindi and English)
        
        Args:
            text: Input text
            threshold: Minimum probability for both languages to be considered code-mixed
            
        Returns:
            True if code-mixed, False otherwise
        """
        scores = self.detect_language(text)
        
        hindi_prob = scores.get('hi', 0)
        english_prob = scores.get('en', 0)
        
        return hindi_prob >= threshold and english_prob >= threshold
    
    def get_primary_language(self, text: str) -> str:
        """
        Get the primary language of the text
        
        Returns:
            Language code ('hi', 'en', 'unknown')
        """
        scores = self.detect_language(text)
        return max(scores.items(), key=lambda x: x[1])[0]
