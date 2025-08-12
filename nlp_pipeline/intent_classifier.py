"""
Intent Classification Module
Classifies agricultural queries into different intents
"""

import re
from typing import Dict, List, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os


class IntentClassifier:
    """
    Advanced intent classifier for agricultural queries
    Supports both rule-based and ML-based classification
    """
    
    def __init__(self, use_transformer: bool = True, model_path: Optional[str] = None):
        self.use_transformer = use_transformer
        self.model_path = model_path
        
        # Define intent categories
        self.intents = {
            'crop_advice': {
                'description': 'Queries about crop cultivation, diseases, and farming practices',
                'keywords': [
                    'crop', 'disease', 'pest', 'fertilizer', 'irrigation', 'sowing',
                    'harvesting', 'yield', 'quality', 'variety', 'season', 'grown',
                    'growing', 'plant', 'planting', 'cultivation', 'remedy', 'remedies',
                    'problem', 'issue', 'not', 'properly', 'well', 'healthy', 'unhealthy',
                    'treatment', 'solution', 'advice', 'help', 'guide', 'practice',
                    'फसल', 'रोग', 'कीट', 'खाद', 'सिंचाई', 'बुवाई', 'कटाई', 'उगाना',
                    'लगाना', 'समस्या', 'उपाय', 'इलाज', 'सलाह', 'मदद'
                ]
            },
            'policy_query': {
                'description': 'Queries about government schemes, subsidies, and policies',
                'keywords': [
                    'scheme', 'subsidy', 'loan', 'insurance', 'policy', 'government',
                    'support', 'assistance', 'benefit', 'eligibility', 'application',
                    'योजना', 'सब्सिडी', 'ऋण', 'बीमा', 'सरकार', 'सहायता'
                ]
            },
            'price_query': {
                'description': 'Queries about crop prices, market rates, and selling',
                'keywords': [
                    'price', 'rate', 'market', 'selling', 'buying', 'mandi', 'auction',
                    'cost', 'profit', 'loss', 'demand', 'supply', 'export', 'import',
                    'rupees', 'rs', 'quintal', 'ton', 'kg', 'per', 'worth', 'value',
                    'भाव', 'कीमत', 'मंडी', 'बिक्री', 'खरीद', 'लाभ', 'हानि', 'रुपये'
                ]
            },
            'weather_query': {
                'description': 'Queries about weather conditions and forecasts',
                'keywords': [
                    'weather', 'rain', 'temperature', 'humidity', 'forecast', 'climate',
                    'drought', 'flood', 'storm', 'season', 'monsoon', 'winter', 'summer',
                    'मौसम', 'बारिश', 'तापमान', 'सूखा', 'बाढ़', 'मानसून'
                ]
            },
            'technical_support': {
                'description': 'Queries about technical farming issues and equipment',
                'keywords': [
                    'equipment', 'machine', 'tractor', 'pump', 'technology', 'digital',
                    'app', 'software', 'sensor', 'automation', 'precision', 'smart',
                    'मशीन', 'ट्रैक्टर', 'पंप', 'तकनीक', 'सेंसर', 'स्मार्ट'
                ]
            },
            'general_inquiry': {
                'description': 'General agricultural inquiries and information',
                'keywords': [
                    'information', 'help', 'guide', 'how', 'what', 'when', 'where',
                    'why', 'general', 'basic', 'overview', 'details', 'explain',
                    'जानकारी', 'मदद', 'गाइड', 'कैसे', 'क्या', 'कब', 'कहाँ'
                ]
            }
        }
        
        # Initialize rule-based classifier
        self._init_rule_based_classifier()
        
        # Initialize ML-based classifier
        if use_transformer:
            self._init_transformer_classifier()
        
        # Initialize TF-IDF classifier
        self._init_tfidf_classifier()
    
    def _init_rule_based_classifier(self):
        """Initialize rule-based classification patterns"""
        self.rule_patterns = {}
        
        for intent, config in self.intents.items():
            patterns = []
            for keyword in config['keywords']:
                # Create regex patterns for exact word matching
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                patterns.append(pattern)
            
            # Add common question patterns
            question_patterns = [
                r'\b(how|what|when|where|why|which)\b.*\b' + re.escape(keyword.lower()) + r'\b'
                for keyword in config['keywords']
            ]
            patterns.extend(question_patterns)
            
            self.rule_patterns[intent] = patterns
    
    def _init_transformer_classifier(self):
        """Initialize transformer-based classifier"""
        try:
            # Use a multilingual model for Hindi/English
            model_name = "microsoft/Multilingual-MiniLM-L12-H384"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(self.intents)
            )
            
            # Create zero-shot classification pipeline
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            print(f"Transformer model initialization failed: {e}")
            self.use_transformer = False
    
    def _init_tfidf_classifier(self):
        """Initialize TF-IDF based classifier"""
        try:
            # Create training data from intent keywords
            training_texts = []
            training_labels = []
            
            for intent, config in self.intents.items():
                for keyword in config['keywords']:
                    training_texts.append(keyword)
                    training_labels.append(intent)
            
            # Create and train TF-IDF classifier
            self.tfidf_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
                ('clf', MultinomialNB())
            ])
            
            self.tfidf_classifier.fit(training_texts, training_labels)
            
        except Exception as e:
            print(f"TF-IDF classifier initialization failed: {e}")
            self.tfidf_classifier = None
    
    def classify_intent(self, text: str) -> Dict[str, float]:
        """
        Classify the intent of the given text
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with intent probabilities
        """
        if not text.strip():
            return {'general_inquiry': 1.0}
        
        # Method 1: Rule-based classification
        rule_scores = self._rule_based_classification(text)
        
        # Method 2: TF-IDF classification
        tfidf_scores = self._tfidf_classification(text)
        
        # Method 3: Transformer-based classification
        transformer_scores = {}
        if self.use_transformer:
            transformer_scores = self._transformer_classification(text)
        
        # Combine results
        final_scores = self._combine_classification_results(
            rule_scores, tfidf_scores, transformer_scores
        )
        
        return final_scores
    
    def _rule_based_classification(self, text: str) -> Dict[str, float]:
        """Rule-based intent classification with improved context handling"""
        text_lower = text.lower()
        scores = {intent: 0.0 for intent in self.intents}
        
        # Special context patterns for better accuracy
        context_patterns = {
            'crop_advice': [
                r'\b(not|not properly|not well|problem|issue|disease|pest|remedy|remedies|solution|treatment)\b',
                r'\b(how to|what to|when to|where to|why is|which is)\b.*\b(grow|plant|cultivate|treat|solve|fix)\b',
                r'\b(grown|growing|planted|planting)\b.*\b(not|problem|issue|disease|pest)\b',
                r'\b(give me|tell me|show me|help me)\b.*\b(remedy|solution|advice|treatment)\b'
            ],
            'price_query': [
                r'\b(price|rate|cost|worth|value)\b.*\b(rupees|rs|quintal|ton|kg|per)\b',
                r'\b(what is|how much|tell me)\b.*\b(price|rate|cost|worth)\b',
                r'\b(mandi|market|auction)\b.*\b(price|rate|selling|buying)\b'
            ],
            'policy_query': [
                r'\b(scheme|subsidy|loan|insurance|policy|government)\b.*\b(how|what|when|where)\b',
                r'\b(apply|application|eligibility|benefit)\b.*\b(scheme|subsidy|loan|insurance)\b'
            ],
            'weather_query': [
                r'\b(weather|rain|temperature|humidity|forecast)\b.*\b(today|tomorrow|week|month)\b',
                r'\b(what is|how is|tell me)\b.*\b(weather|rain|temperature)\b'
            ]
        }
        
        # Apply context patterns first (higher weight)
        for intent, patterns in context_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 2  # Higher weight for context patterns
            scores[intent] += score
        
        # Apply regular keyword patterns
        for intent, patterns in self.rule_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches)
            scores[intent] += score
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _tfidf_classification(self, text: str) -> Dict[str, float]:
        """TF-IDF based intent classification"""
        if self.tfidf_classifier is None:
            return {intent: 0.0 for intent in self.intents}
        
        try:
            # Get prediction probabilities
            proba = self.tfidf_classifier.predict_proba([text])[0]
            intent_names = self.tfidf_classifier.classes_
            
            scores = {intent: 0.0 for intent in self.intents}
            for intent, prob in zip(intent_names, proba):
                scores[intent] = prob
            
            return scores
            
        except Exception as e:
            print(f"TF-IDF classification failed: {e}")
            return {intent: 0.0 for intent in self.intents}
    
    def _transformer_classification(self, text: str) -> Dict[str, float]:
        """Transformer-based intent classification"""
        try:
            # Prepare candidate labels
            candidate_labels = list(self.intents.keys())
            candidate_labels = [self.intents[label]['description'] for label in candidate_labels]
            
            # Perform zero-shot classification
            result = self.zero_shot_classifier(
                text,
                candidate_labels,
                hypothesis_template="This text is about {}."
            )
            
            # Map results back to intent names
            scores = {intent: 0.0 for intent in self.intents}
            for label, score in zip(result['labels'], result['scores']):
                # Find corresponding intent
                for intent, config in self.intents.items():
                    if config['description'] == label:
                        scores[intent] = score
                        break
            
            return scores
            
        except Exception as e:
            print(f"Transformer classification failed: {e}")
            return {intent: 0.0 for intent in self.intents}
    
    def _combine_classification_results(
        self,
        rule_scores: Dict[str, float],
        tfidf_scores: Dict[str, float],
        transformer_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine results from different classification methods with adaptive weighting"""
        
        # Check if rule-based classification is confident (high score for one intent)
        max_rule_score = max(rule_scores.values()) if rule_scores.values() else 0
        rule_confidence = max_rule_score > 0.6  # High confidence threshold
        
        # Adaptive weights based on rule confidence
        if rule_confidence:
            weights = {
                'rule': 0.6,      # Higher weight when rule-based is confident
                'tfidf': 0.2,
                'transformer': 0.2
            }
        else:
            weights = {
                'rule': 0.3,
                'tfidf': 0.3,
                'transformer': 0.4
            }
        
        # Combine scores
        combined = {intent: 0.0 for intent in self.intents}
        
        for intent in self.intents:
            score = 0
            score += rule_scores.get(intent, 0) * weights['rule']
            score += tfidf_scores.get(intent, 0) * weights['tfidf']
            score += transformer_scores.get(intent, 0) * weights['transformer']
            combined[intent] = score
        
        # Normalize scores
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}
        
        return combined
    
    def get_primary_intent(self, text: str) -> str:
        """
        Get the primary intent of the text
        
        Returns:
            Intent name with highest probability
        """
        scores = self.classify_intent(text)
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_intent_confidence(self, text: str) -> float:
        """
        Get confidence score for the primary intent
        
        Returns:
            Confidence score (0-1)
        """
        scores = self.classify_intent(text)
        return max(scores.values())
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.tfidf_classifier:
            with open(path, 'wb') as f:
                pickle.dump(self.tfidf_classifier, f)
    
    def load_model(self, path: str):
        """Load a trained model"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.tfidf_classifier = pickle.load(f)
