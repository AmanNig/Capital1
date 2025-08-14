"""
Advanced Intent Classification Module
Uses multiple ML approaches for better intent classification
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# For semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# For zero-shot classification
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AdvancedIntentClassifier:
    """
    Advanced intent classifier using multiple ML approaches
    """
    
    def __init__(self, use_semantic: bool = True, use_zero_shot: bool = True):
        self.use_semantic = use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE
        self.use_zero_shot = use_zero_shot and TRANSFORMERS_AVAILABLE
        
        # Intent definitions with examples
        self.intents = {
            'greeting': {
                'description': 'Greetings, hellos, and casual introductions',
                'examples': [
                    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                    "namaste", "namaskar", "salaam", "howdy", "what's up", "sup",
                    "hello there", "hi there", "hey there", "good day", "good night",
                    "hello sir", "hello ma'am", "hi everyone", "hey everyone",
                    "नमस्ते", "नमस्कार", "सलाम", "हैलो", "हाय", "कैसे हो",
                    "कैसा चल रहा है", "क्या हाल है", "सुप्रभात", "सुसंध्या",
                    "hello how are you", "hi how are you", "hey how are you",
                    "hello how are you doing", "hi how are you doing",
                    "how are you", "how do you do", "pleasure to meet you",
                    "nice to meet you", "good to see you", "long time no see"
                ]
            },
            'farewell': {
                'description': 'Goodbyes, farewells, and ending conversations',
                'examples': [
                    "goodbye", "bye", "see you", "see you later", "see you soon",
                    "take care", "take it easy", "have a good day", "have a nice day",
                    "good night", "sleep well", "sweet dreams", "until next time",
                    "farewell", "adios", "ciao", "au revoir", "auf wiedersehen",
                    "alvida", "फिर मिलेंगे", "अलविदा", "खुदा हाफिज", "गुड बाय",
                    "बाय", "फिर मिलते हैं", "अच्छा रहो", "ख्याल रखना",
                    "goodbye for now", "see you around", "catch you later",
                    "talk to you later", "until we meet again", "so long",
                    "have a great day", "enjoy your day", "stay safe"
                ]
            },
            'introduction': {
                'description': 'Self-introductions and asking about identity',
                'examples': [
                    "who are you", "what are you", "tell me about yourself",
                    "what can you do", "what is your name", "what do you do",
                    "introduce yourself", "what is this", "what is this bot",
                    "are you a bot", "are you ai", "are you artificial intelligence",
                    "what kind of bot are you", "what is your purpose",
                    "what is your function", "what services do you provide",
                    "आप कौन हैं", "आप क्या हैं", "अपने बारे में बताइए",
                    "आपका नाम क्या है", "आप क्या करते हैं", "आप कैसे काम करते हैं",
                    "आप किस तरह का बॉट हैं", "आपका उद्देश्य क्या है",
                    "आप क्या सेवाएं प्रदान करते हैं", "आप कैसे मदद कर सकते हैं",
                    "my name is", "i am", "i'm", "this is", "calling from",
                    "i work as", "i do", "i study", "i live in", "i'm from"
                ]
            },
            'gratitude': {
                'description': 'Thank you messages and expressions of gratitude',
                'examples': [
                    "thank you", "thanks", "thank you so much", "thanks a lot",
                    "thank you very much", "thanks very much", "much obliged",
                    "appreciate it", "appreciate that", "thanks for your help",
                    "thank you for helping", "thanks for the information",
                    "thank you for your time", "thanks for everything",
                    "धन्यवाद", "शुक्रिया", "बहुत बहुत धन्यवाद", "आभार",
                    "धन्यवाद आपकी मदद के लिए", "शुक्रिया जानकारी के लिए",
                    "आपका बहुत बहुत शुक्रिया", "आपकी कृपा के लिए धन्यवाद",
                    "grateful", "gratitude", "blessed", "fortunate", "lucky",
                    "thank you kindly", "many thanks", "thanks a million",
                    "thank you from the bottom of my heart", "owe you one"
                ]
            },
            'wellbeing': {
                'description': 'Asking about health, mood, and general wellbeing',
                'examples': [
                    "how are you", "how are you doing", "how do you feel",
                    "are you okay", "are you alright", "how is everything",
                    "how is it going", "how have you been", "what's new",
                    "how is your day", "how is your day going", "feeling good",
                    "are you feeling well", "how is your health", "how is life",
                    "कैसे हो", "कैसा चल रहा है", "सब ठीक है", "कैसा लग रहा है",
                    "कैसा महसूस कर रहे हो", "दिन कैसा जा रहा है", "सब बढ़िया",
                    "तबीयत कैसी है", "जिंदगी कैसी चल रही है", "सब कुछ ठीक है",
                    "i'm fine", "i'm good", "i'm okay", "i'm well", "i'm great",
                    "doing well", "feeling great", "all good", "everything is fine",
                    "life is good", "can't complain", "pretty good", "not bad"
                ]
            },
            'crop_advice': {
                'description': 'Queries about crop cultivation, diseases, and farming practices',
                'examples': [
                    "my crops are not growing properly",
                    "how to treat plant diseases",
                    "what fertilizer should I use for wheat",
                    "my rice plants have yellow leaves",
                    "when to irrigate cotton crop",
                    "how to increase crop yield",
                    "pest control methods for vegetables",
                    "soil preparation for farming",
                    "crop rotation techniques",
                    "organic farming methods",
                    "crop disease treatment",
                    "plant health issues",
                    "fertilizer application timing",
                    "irrigation schedule for crops",
                    "harvesting techniques",
                    "seed selection advice",
                    "crop protection methods",
                    "soil health improvement",
                    "weed control strategies",
                    "crop nutrition management",
                    "मेरी फसल में रोग लग गया है",
                    "गेहूं की फसल में पीले पत्ते आ रहे हैं",
                    "फसल में कीट लग गए हैं",
                    "पौधों को कैसे बचाएं",
                    "खाद कब डालनी चाहिए",
                    "फसल की देखभाल कैसे करें",
                    "पौधों में रोग का इलाज",
                    "फसल उत्पादन बढ़ाने के तरीके",
                    "सिंचाई का सही समय",
                    "फसल संरक्षण के उपाय"
                ]
            },
            'price_query': {
                'description': 'Queries about crop prices, market rates, and selling',
                'examples': [
                    "what is the current price of rice",
                    "wheat price in mandi today",
                    "how much does cotton sell for",
                    "market rates for vegetables",
                    "price of sugarcane per quintal",
                    "auction prices for crops",
                    "export prices for agricultural products",
                    "wholesale rates in market",
                    "crop selling prices",
                    "commodity prices today",
                    "current market prices",
                    "crop price trends",
                    "mandi rates today",
                    "selling price for crops",
                    "market value of agricultural products",
                    "price per quintal",
                    "commodity exchange rates",
                    "agricultural commodity prices",
                    "crop auction rates",
                    "wholesale market prices",
                    "गेहूं का भाव क्या है आज",
                    "धान का दाम कितना है",
                    "मंडी में सब्जियों का भाव",
                    "कपास का रेट क्या है",
                    "आज का मंडी भाव",
                    "फसल का बाजार भाव",
                    "कृषि उत्पादों की कीमत",
                    "मंडी दरें आज",
                    "फसल बिक्री मूल्य",
                    "कृषि वस्तुओं का भाव"
                ]
            },
            'policy_query': {
                'description': 'Queries about government schemes, subsidies, and policies',
                'examples': [
                    "how to apply for agricultural loan",
                    "government subsidy for farmers",
                    "crop insurance scheme details",
                    "PM Kisan scheme application",
                    "subsidy on farm equipment",
                    "loan eligibility for farmers",
                    "government support programs",
                    "agricultural policy information",
                    "farmer benefit schemes",
                    "how to get agricultural subsidy",
                    "सरकारी योजना कैसे मिलेगी",
                    "किसान ऋण कैसे लें",
                    "पीएम किसान योजना में आवेदन",
                    "फसल बीमा योजना",
                    "सरकारी सब्सिडी कैसे मिलेगी"
                ]
            },
            'weather_query': {
                'description': 'Queries about weather conditions and forecasts',
                'examples': [
                    "weather forecast for farming",
                    "when will it rain",
                    "temperature for crop growth",
                    "monsoon prediction for agriculture",
                    "drought conditions in region",
                    "humidity levels for crops",
                    "weather impact on harvest",
                    "climate conditions for sowing",
                    "rainfall forecast for next week",
                    "weather suitable for farming",
                    "आज का मौसम कैसा है",
                    "बारिश कब होगी",
                    "मानसून का पूर्वानुमान",
                    "तापमान कैसा रहेगा",
                    "खेती के लिए मौसम"
                ]
            },
            'technical_support': {
                'description': 'Queries about technical farming issues and equipment',
                'examples': [
                    "tractor maintenance tips",
                    "irrigation system problems",
                    "farm equipment troubleshooting",
                    "digital farming app issues",
                    "sensor calibration for farming",
                    "automated irrigation setup",
                    "precision farming technology",
                    "smart farming equipment",
                    "agricultural software help",
                    "farm machinery repair",
                    "ट्रैक्टर में समस्या आ गई है",
                    "सिंचाई सिस्टम में तकनीकी समस्या",
                    "फार्म मशीनरी की मरम्मत",
                    "डिजिटल खेती ऐप की समस्या",
                    "कृषि उपकरणों की देखभाल"
                ]
            },
            'general_inquiry': {
                'description': 'General agricultural inquiries and information',
                'examples': [
                    "what is organic farming",
                    "basic farming information",
                    "agricultural practices overview",
                    "farming guide for beginners",
                    "general crop information",
                    "agricultural education resources",
                    "farming knowledge base",
                    "agricultural facts and figures",
                    "farming basics explained",
                    "agricultural information sources",
                    "खेती के बारे में जानकारी चाहिए",
                    "जैविक खेती क्या है",
                    "कृषि के बारे में सामान्य जानकारी",
                    "खेती की बुनियादी जानकारी",
                    "कृषि शिक्षा संसाधन"
                ]
            }
        }
        
        # Initialize different classifiers
        self._init_classifiers()
        
        # Initialize semantic model if available
        if self.use_semantic:
            self._init_semantic_model()
        
        # Initialize zero-shot classifier if available
        if self.use_zero_shot:
            self._init_zero_shot_classifier()
    
    def _init_classifiers(self):
        """Initialize multiple ML classifiers with improved parameters"""
        # TF-IDF + Naive Bayes (improved parameters)
        self.tfidf_nb = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3), 
                max_features=3000,
                min_df=1,
                max_df=0.95,
                stop_words='english'
            )),
            ('nb', MultinomialNB(alpha=0.1))
        ])
        
        # TF-IDF + Logistic Regression (improved parameters)
        self.tfidf_lr = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3), 
                max_features=3000,
                min_df=1,
                max_df=0.95,
                stop_words='english'
            )),
            ('lr', LogisticRegression(
                max_iter=2000, 
                random_state=42,
                C=1.0,
                class_weight='balanced'
            ))
        ])
        
        # Count Vectorizer + Random Forest (improved parameters)
        self.count_rf = Pipeline([
            ('count', CountVectorizer(
                ngram_range=(1, 3), 
                max_features=2500,
                min_df=1,
                max_df=0.95,
                stop_words='english'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, 
                random_state=42,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced'
            ))
        ])
        
        # Train classifiers with examples
        self._train_classifiers()
    
    def _init_semantic_model(self):
        """Initialize semantic similarity model"""
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Create embeddings for all intent examples
            self.intent_embeddings = {}
            for intent, config in self.intents.items():
                embeddings = self.semantic_model.encode(config['examples'])
                self.intent_embeddings[intent] = embeddings
        except Exception as e:
            print(f"Semantic model initialization failed: {e}")
            self.use_semantic = False
    
    def _init_zero_shot_classifier(self):
        """Initialize zero-shot classifier"""
        try:
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception as e:
            print(f"Zero-shot classifier initialization failed: {e}")
            self.use_zero_shot = False
    
    def _train_classifiers(self):
        """Train all classifiers with intent examples"""
        # Prepare training data
        texts = []
        labels = []
        
        for intent, config in self.intents.items():
            for example in config['examples']:
                texts.append(example)
                labels.append(intent)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train classifiers
        self.tfidf_nb.fit(X_train, y_train)
        self.tfidf_lr.fit(X_train, y_train)
        self.count_rf.fit(X_train, y_train)
        
        # Evaluate and print results
        print("Classifier Training Results:")
        for name, clf in [("TF-IDF + NB", self.tfidf_nb), 
                         ("TF-IDF + LR", self.tfidf_lr), 
                         ("Count + RF", self.count_rf)]:
            y_pred = clf.predict(X_test)
            print(f"{name}: {np.mean(y_pred == y_test):.3f} accuracy")
    
    def classify_intent(self, text: str) -> Dict[str, float]:
        """Classify intent using multiple approaches with rule-based fallback"""
        if not text.strip():
            return {'general_inquiry': 1.0}
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        scores = {intent: 0.0 for intent in self.intents}
        
        # Method 1: Traditional ML classifiers
        ml_scores = self._get_ml_scores(processed_text)
        
        # Method 2: Semantic similarity
        semantic_scores = {}
        if self.use_semantic:
            semantic_scores = self._get_semantic_scores(processed_text)
        
        # Method 3: Zero-shot classification
        zero_shot_scores = {}
        if self.use_zero_shot:
            zero_shot_scores = self._get_zero_shot_scores(processed_text)
        
        # Method 4: Rule-based classification (fallback)
        rule_scores = self._get_rule_based_scores(processed_text)
        
        # Combine all scores
        scores = self._combine_scores(ml_scores, semantic_scores, zero_shot_scores, rule_scores)
        
        # Post-process scores for edge cases
        scores = self._post_process_scores(scores, text)
        
        return scores
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Handle common abbreviations and variations
        text = text.replace("what's", "what is")
        text = text.replace("how's", "how is")
        text = text.replace("where's", "where is")
        text = text.replace("when's", "when is")
        
        return text
    
    def _post_process_scores(self, scores: Dict[str, float], original_text: str) -> Dict[str, float]:
        """Post-process scores to handle edge cases"""
        # Check for very short queries
        if len(original_text.split()) < 3:
            # For very short queries, boost general_inquiry
            if scores.get('general_inquiry', 0) > 0.1:
                scores['general_inquiry'] = min(1.0, scores['general_inquiry'] * 1.2)
        
        # Check for very long queries
        if len(original_text.split()) > 20:
            # For very long queries, boost the highest scoring intent
            max_intent = max(scores.items(), key=lambda x: x[1])[0]
            if scores[max_intent] > 0.2:
                scores[max_intent] = min(1.0, scores[max_intent] * 1.1)
        
        # Ensure minimum confidence for the highest scoring intent
        max_score = max(scores.values()) if scores.values() else 0
        if max_score < 0.1:
            # If all scores are very low, boost the highest one
            max_intent = max(scores.items(), key=lambda x: x[1])[0]
            scores[max_intent] = 0.3
            # Reduce others proportionally
            for intent in scores:
                if intent != max_intent:
                    scores[intent] = scores[intent] * 0.5
        
        return scores
    
    def _get_rule_based_scores(self, text: str) -> Dict[str, float]:
        """Get scores using rule-based keyword matching"""
        text_lower = text.lower()
        scores = {intent: 0.0 for intent in self.intents}
        
        # Define strong keyword patterns for each intent
        keyword_patterns = {
            'crop_advice': [
                r'\b(not|not properly|not well|problem|issue|disease|pest|remedy|remedies|solution|treatment)\b',
                r'\b(how to|what to|when to|where to|why is|which is)\b.*\b(grow|plant|cultivate|treat|solve|fix)\b',
                r'\b(grown|growing|planted|planting)\b.*\b(not|problem|issue|disease|pest)\b',
                r'\b(give me|tell me|show me|help me)\b.*\b(remedy|solution|advice|treatment)\b',
                r'\b(crop|plant|seed|fertilizer|irrigation|harvest|yield)\b.*\b(advice|help|guide|method|technique)\b',
                r'\b(suitable|optimal|ideal|best|proper)\b.*\b(conditions|environment|climate|soil|temperature)\b',
                r'\b(what are|tell me|show me)\b.*\b(conditions|requirements|needs)\b.*\b(grow|plant|cultivate)\b'
            ],
            'price_query': [
                r'\b(price|rate|cost|worth|value)\b.*\b(rupees|rs|quintal|ton|kg|per)\b',
                r'\b(what is|how much|tell me)\b.*\b(price|rate|cost|worth)\b',
                r'\b(mandi|market|auction)\b.*\b(price|rate|selling|buying)\b',
                r'\b(bhav|dam|mulya|keemat)\b',  # Hindi price terms
                r'\b(market|mandi|bazaar)\b.*\b(rate|price|cost)\b'
            ],
            'policy_query': [
                r'\b(scheme|subsidy|loan|insurance|policy|government)\b.*\b(how|what|when|where)\b',
                r'\b(apply|application|eligibility|benefit)\b.*\b(scheme|subsidy|loan|insurance)\b',
                r'\b(yojana|sarkar|sarkari|subsidy|loan)\b',  # Hindi policy terms
                r'\b(government|sarkar)\b.*\b(scheme|yojana|policy)\b'
            ],
            'weather_query': [
                r'\b(weather|rain|temperature|humidity|forecast)\b.*\b(today|tomorrow|week|month)\b',
                r'\b(what is|how is|tell me)\b.*\b(weather|rain|temperature)\b',
                r'\b(mausam|baarish|temperature|humidity)\b',  # Hindi weather terms
                r'\b(forecast|prediction)\b.*\b(weather|rain|temperature)\b'
            ],
            'technical_support': [
                r'\b(tractor|equipment|machine|system|technology)\b.*\b(problem|issue|repair|maintenance)\b',
                r'\b(technical|technology|digital|automated)\b.*\b(help|support|issue)\b',
                r'\b(repair|maintenance|troubleshoot|fix)\b.*\b(equipment|machine|system)\b',
                r'\b(tractor|machine|equipment)\b.*\b(problem|issue|repair)\b'
            ],
            'general_inquiry': [
                r'\b(what is|tell me about|information about|guide for)\b.*\b(farming|agriculture|crop)\b',
                r'\b(basic|general|overview|introduction)\b.*\b(farming|agriculture)\b',
                r'\b(how to start|beginner|basic)\b.*\b(farming|agriculture)\b'
            ]
        }
        
        # Calculate scores based on pattern matches
        for intent, patterns in keyword_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 0.3  # Weight for each match
            
            # Normalize score
            scores[intent] = min(1.0, score)
        
        return scores
    
    def _get_ml_scores(self, text: str) -> Dict[str, float]:
        """Get scores from traditional ML classifiers"""
        scores = {intent: 0.0 for intent in self.intents}
        
        # Get probabilities from each classifier
        try:
            nb_proba = self.tfidf_nb.predict_proba([text])[0]
            lr_proba = self.tfidf_lr.predict_proba([text])[0]
            rf_proba = self.count_rf.predict_proba([text])[0]
            
            # Get class names
            classes = self.tfidf_nb.classes_
            
            # Average probabilities
            for i, intent in enumerate(classes):
                avg_prob = (nb_proba[i] + lr_proba[i] + rf_proba[i]) / 3
                scores[intent] = avg_prob
                
        except Exception as e:
            print(f"ML classification failed: {e}")
        
        return scores
    
    def _get_semantic_scores(self, text: str) -> Dict[str, float]:
        """Get scores using semantic similarity with improved scoring"""
        scores = {intent: 0.0 for intent in self.intents}
        
        try:
            # Encode input text
            text_embedding = self.semantic_model.encode([text])
            
            # Calculate similarities with all intent examples
            for intent, embeddings in self.intent_embeddings.items():
                similarities = cosine_similarity(text_embedding, embeddings)[0]
                
                # Use multiple similarity metrics for better scoring
                max_similarity = np.max(similarities)
                avg_similarity = np.mean(similarities)
                top_k_similarity = np.mean(np.sort(similarities)[-3:])  # Top 3 similarities
                
                # Combine metrics with weights
                combined_score = (0.5 * max_similarity + 0.3 * avg_similarity + 0.2 * top_k_similarity)
                
                # Apply sigmoid-like transformation to boost scores
                boosted_score = 1 / (1 + np.exp(-5 * (combined_score - 0.5)))
                
                scores[intent] = boosted_score
                
        except Exception as e:
            print(f"Semantic classification failed: {e}")
        
        return scores
    
    def _get_zero_shot_scores(self, text: str) -> Dict[str, float]:
        """Get scores using zero-shot classification"""
        scores = {intent: 0.0 for intent in self.intents}
        
        try:
            # Prepare candidate labels
            candidate_labels = [config['description'] for config in self.intents.values()]
            
            # Get zero-shot results
            result = self.zero_shot(
                text,
                candidate_labels,
                hypothesis_template="This text is about {}."
            )
            
            # Map back to intent names
            for label, score in zip(result['labels'], result['scores']):
                for intent, config in self.intents.items():
                    if config['description'] == label:
                        scores[intent] = score
                        break
                        
        except Exception as e:
            print(f"Zero-shot classification failed: {e}")
        
        return scores
    
    def _combine_scores(self, ml_scores: Dict[str, float], 
                       semantic_scores: Dict[str, float], 
                       zero_shot_scores: Dict[str, float],
                       rule_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine scores from different methods with improved weighting"""
        
        combined = {intent: 0.0 for intent in self.intents}
        
        # Check if we have valid scores from each method
        has_ml = any(score > 0.1 for score in ml_scores.values())
        has_semantic = semantic_scores and any(score > 0.1 for score in semantic_scores.values())
        has_zero_shot = zero_shot_scores and any(score > 0.1 for score in zero_shot_scores.values())
        has_rule = any(score > 0.1 for score in rule_scores.values())
        
        # Dynamic weights based on available methods and their quality
        if has_ml and has_semantic and has_zero_shot and has_rule:
            weights = {'ml': 0.25, 'semantic': 0.25, 'zero_shot': 0.20, 'rule': 0.30}
        elif has_ml and has_semantic and has_rule:
            weights = {'ml': 0.30, 'semantic': 0.30, 'zero_shot': 0.0, 'rule': 0.40}
        elif has_ml and has_zero_shot and has_rule:
            weights = {'ml': 0.30, 'zero_shot': 0.30, 'semantic': 0.0, 'rule': 0.40}
        elif has_semantic and has_zero_shot and has_rule:
            weights = {'semantic': 0.35, 'zero_shot': 0.25, 'ml': 0.0, 'rule': 0.40}
        elif has_ml and has_semantic and has_zero_shot:
            weights = {'ml': 0.35, 'semantic': 0.35, 'zero_shot': 0.30, 'rule': 0.0}
        elif has_ml and has_semantic:
            weights = {'ml': 0.45, 'semantic': 0.55, 'zero_shot': 0.0, 'rule': 0.0}
        elif has_ml and has_zero_shot:
            weights = {'ml': 0.45, 'zero_shot': 0.55, 'semantic': 0.0, 'rule': 0.0}
        elif has_semantic and has_zero_shot:
            weights = {'semantic': 0.50, 'zero_shot': 0.50, 'ml': 0.0, 'rule': 0.0}
        elif has_rule:
            weights = {'rule': 1.0, 'ml': 0.0, 'semantic': 0.0, 'zero_shot': 0.0}
        elif has_ml:
            weights = {'ml': 1.0, 'semantic': 0.0, 'zero_shot': 0.0, 'rule': 0.0}
        elif has_semantic:
            weights = {'semantic': 1.0, 'ml': 0.0, 'zero_shot': 0.0, 'rule': 0.0}
        elif has_zero_shot:
            weights = {'zero_shot': 1.0, 'ml': 0.0, 'semantic': 0.0, 'rule': 0.0}
        else:
            # Fallback to equal distribution
            weights = {'ml': 0.25, 'semantic': 0.25, 'zero_shot': 0.25, 'rule': 0.25}
        
        # Combine scores with confidence boosting
        for intent in self.intents:
            score = 0
            
            # ML scores (already probabilities)
            if has_ml:
                ml_score = ml_scores.get(intent, 0)
                score += ml_score * weights['ml']
            
            # Semantic scores (need normalization and boosting)
            if has_semantic:
                semantic_score = semantic_scores.get(intent, 0)
                # Boost semantic scores to make them more competitive
                semantic_score = min(1.0, semantic_score * 1.5)
                score += semantic_score * weights['semantic']
            
            # Zero-shot scores (already probabilities)
            if has_zero_shot:
                zero_shot_score = zero_shot_scores.get(intent, 0)
                score += zero_shot_score * weights['zero_shot']
            
            # Rule-based scores (boosted for confidence)
            if has_rule:
                rule_score = rule_scores.get(intent, 0)
                # Boost rule-based scores when they're high
                if rule_score > 0.3:
                    rule_score = min(1.0, rule_score * 1.5)
                score += rule_score * weights['rule']
            
            combined[intent] = score
        
        # Apply confidence boosting for clear winners
        max_score = max(combined.values()) if combined.values() else 0
        if max_score > 0.3:  # If we have a reasonably confident prediction
            # Boost the highest score and reduce others
            for intent in combined:
                if combined[intent] == max_score:
                    combined[intent] = min(1.0, combined[intent] * 1.3)  # Boost winner
                else:
                    combined[intent] = combined[intent] * 0.7  # Reduce others
        
        # Apply confidence calibration
        combined = self._calibrate_confidence(combined)
        
        # Normalize to ensure sum = 1
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}
        
        return combined
    
    def _calibrate_confidence(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Calibrate confidence scores to be more realistic"""
        calibrated = {}
        
        # Find the maximum score
        max_score = max(scores.values()) if scores.values() else 0
        
        if max_score > 0:
            # Apply sigmoid-like calibration
            for intent, score in scores.items():
                # Normalize to 0-1 range
                normalized = score / max_score
                
                # Apply calibration curve
                if normalized > 0.7:
                    # High confidence: boost slightly
                    calibrated[intent] = min(1.0, normalized * 1.1)
                elif normalized > 0.4:
                    # Medium confidence: keep as is
                    calibrated[intent] = normalized
                else:
                    # Low confidence: reduce slightly
                    calibrated[intent] = normalized * 0.9
        else:
            calibrated = scores
        
        return calibrated
    
    def get_primary_intent(self, text: str) -> str:
        """Get primary intent"""
        scores = self.classify_intent(text)
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def get_intent_confidence(self, text: str) -> float:
        """Get confidence score"""
        scores = self.classify_intent(text)
        return max(scores.values())
    
    def add_training_example(self, text: str, intent: str):
        """Add new training example to improve classification"""
        if intent in self.intents:
            self.intents[intent]['examples'].append(text)
            # Retrain classifiers
            self._train_classifiers()
            # Update semantic embeddings
            if self.use_semantic:
                self._init_semantic_model()
    
    def save_model(self, path: str):
        """Save the trained model"""
        model_data = {
            'intents': self.intents,
            'tfidf_nb': self.tfidf_nb,
            'tfidf_lr': self.tfidf_lr,
            'count_rf': self.count_rf
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """Load a trained model"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.intents = model_data['intents']
            self.tfidf_nb = model_data['tfidf_nb']
            self.tfidf_lr = model_data['tfidf_lr']
            self.count_rf = model_data['count_rf']
            
            # Reinitialize semantic and zero-shot
            if self.use_semantic:
                self._init_semantic_model()
            if self.use_zero_shot:
                self._init_zero_shot_classifier()
