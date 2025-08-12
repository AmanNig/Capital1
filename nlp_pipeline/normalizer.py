"""
Text Normalization Module
Handles slang, local words, and agricultural terminology normalization
"""

import re
from typing import Dict, List, Optional
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextNormalizer:
    """
    Advanced text normalizer for Hindi/English agricultural text
    """
    
    def __init__(self):
        self.indic_normalizer = IndicNormalizerFactory().get_normalizer("hi")
        self.lemmatizer = WordNetLemmatizer()
        
        # Hindi to English agricultural mappings
        self.hindi_to_english = {
            # Water and irrigation
            'पानी देना': 'irrigate',
            'सिंचाई': 'irrigation',
            'पानी': 'water',
            'नहर': 'canal',
            'कुंआ': 'well',
            'ट्यूबवेल': 'tubewell',
            
            # Crops
            'गेहूं': 'wheat',
            'धान': 'rice',
            'मक्का': 'maize',
            'बाजरा': 'bajra',
            'ज्वार': 'jowar',
            'चना': 'chickpea',
            'मूंग': 'mungbean',
            'अरहर': 'pigeonpea',
            'सोयाबीन': 'soybean',
            'कपास': 'cotton',
            'गन्ना': 'sugarcane',
            'आलू': 'potato',
            'प्याज': 'onion',
            'टमाटर': 'tomato',
            
            # Farming activities
            'खेती': 'farming',
            'बुवाई': 'sowing',
            'रोपाई': 'transplanting',
            'कटाई': 'harvesting',
            'जुताई': 'plowing',
            'खाद': 'fertilizer',
            'कीटनाशक': 'pesticide',
            'बीज': 'seed',
            'पौधा': 'plant',
            'फसल': 'crop',
            
            # Common slang/local words
            'किसान': 'farmer',
            'खेत': 'field',
            'मिट्टी': 'soil',
            'मौसम': 'weather',
            'बारिश': 'rain',
            'सूखा': 'drought',
            'बाढ़': 'flood',
            'तापमान': 'temperature',
            'नमी': 'moisture',
            'उर्वरता': 'fertility',
            
            # Policy and scheme related
            'योजना': 'scheme',
            'सब्सिडी': 'subsidy',
            'ऋण': 'loan',
            'बीमा': 'insurance',
            'मंडी': 'market',
            'भाव': 'price',
            'कीमत': 'price',
            'बिक्री': 'sale',
            'खरीद': 'purchase',
            
            # Time and dates
            'आज': 'today',
            'कल': 'tomorrow',
            'परसों': 'day after tomorrow',
            'बीते': 'past',
            'आने वाले': 'upcoming',
            'महीना': 'month',
            'साल': 'year',
            'मौसम': 'season'
        }
        
        # English slang to standard mappings
        self.english_slang = {
            'crop': 'crop',
            'farm': 'farm',
            'field': 'field',
            'soil': 'soil',
            'water': 'water',
            'seed': 'seed',
            'plant': 'plant',
            'harvest': 'harvest',
            'irrigate': 'irrigate',
            'fertilize': 'fertilize',
            'spray': 'spray',
            'price': 'price',
            'market': 'market',
            'scheme': 'scheme',
            'subsidy': 'subsidy',
            'loan': 'loan',
            'insurance': 'insurance'
        }
        
        # Common abbreviations
        self.abbreviations = {
            'kg': 'kilogram',
            'ha': 'hectare',
            'acre': 'acre',
            'rs': 'rupees',
            'inr': 'rupees',
            'govt': 'government',
            'dept': 'department',
            'min': 'minimum',
            'max': 'maximum',
            'temp': 'temperature',
            'humidity': 'humidity',
            'rainfall': 'rainfall'
        }
    
    def normalize_text(self, text: str, target_language: str = 'en') -> str:
        """
        Normalize text by handling slang, local words, and standardizing terminology
        
        Args:
            text: Input text to normalize
            target_language: Target language for normalization ('en' or 'hi')
            
        Returns:
            Normalized text
        """
        if not text.strip():
            return text
        
        # Step 1: Basic cleaning
        text = self._basic_cleaning(text)
        
        # Step 2: Handle Hindi text
        if target_language == 'en':
            text = self._normalize_hindi_to_english(text)
        
        # Step 3: Handle English slang
        text = self._normalize_english_slang(text)
        
        # Step 4: Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Step 5: Standardize agricultural terms
        text = self._standardize_agricultural_terms(text)
        
        # Step 6: Final cleaning
        text = self._final_cleaning(text)
        
        return text
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\!\?\:\;\(\)]', '', text)
        
        return text.strip()
    
    def _normalize_hindi_to_english(self, text: str) -> str:
        """Convert Hindi agricultural terms to English"""
        normalized_text = text
        
        # Sort by length (longest first) to avoid partial matches
        sorted_mappings = sorted(
            self.hindi_to_english.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for hindi_term, english_term in sorted_mappings:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(hindi_term) + r'\b'
            normalized_text = re.sub(pattern, english_term, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    def _normalize_english_slang(self, text: str) -> str:
        """Normalize English slang terms"""
        normalized_text = text
        
        for slang, standard in self.english_slang.items():
            pattern = r'\b' + re.escape(slang) + r'\b'
            normalized_text = re.sub(pattern, standard, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        normalized_text = text
        
        for abbr, full_form in self.abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            normalized_text = re.sub(pattern, full_form, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    def _standardize_agricultural_terms(self, text: str) -> str:
        """Standardize agricultural terminology"""
        # Common variations to standard forms
        standardizations = {
            'irrigation': 'irrigation',
            'watering': 'irrigation',
            'sowing': 'sowing',
            'planting': 'sowing',
            'harvesting': 'harvesting',
            'cutting': 'harvesting',
            'fertilizing': 'fertilizing',
            'manuring': 'fertilizing',
            'pesticide': 'pesticide',
            'insecticide': 'pesticide',
            'herbicide': 'pesticide',
            'crop': 'crop',
            'cultivation': 'crop',
            'farming': 'farming',
            'agriculture': 'farming'
        }
        
        normalized_text = text
        for variant, standard in standardizations.items():
            pattern = r'\b' + re.escape(variant) + r'\b'
            normalized_text = re.sub(pattern, standard, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    def _final_cleaning(self, text: str) -> str:
        """Final text cleaning and standardization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize first letter of sentences
        text = '. '.join(s.capitalize() for s in text.split('. '))
        
        return text.strip()
    
    def get_normalized_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract and normalize entities from text
        
        Returns:
            Dictionary with entity types and their normalized values
        """
        entities = {
            'crops': [],
            'activities': [],
            'locations': [],
            'dates': [],
            'quantities': []
        }
        
        # Extract crop names
        crop_patterns = [
            r'\b(wheat|rice|maize|bajra|jowar|chickpea|mungbean|pigeonpea|soybean|cotton|sugarcane|potato|onion|tomato)\b',
            r'\b(गेहूं|धान|मक्का|बाजरा|ज्वार|चना|मूंग|अरहर|सोयाबीन|कपास|गन्ना|आलू|प्याज|टमाटर)\b'
        ]
        
        for pattern in crop_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['crops'].extend(matches)
        
        # Extract activities
        activity_patterns = [
            r'\b(irrigate|sowing|harvesting|fertilizing|spraying|plowing|transplanting)\b',
            r'\b(सिंचाई|बुवाई|कटाई|खाद|कीटनाशक|जुताई|रोपाई)\b'
        ]
        
        for pattern in activity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['activities'].extend(matches)
        
        # Remove duplicates and normalize
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
            entities[entity_type] = [self.normalize_text(item) for item in entities[entity_type]]
        
        return entities
