"""
Entity Extraction Module
Extracts key entities from agricultural queries
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import spacy
from transformers import pipeline
import json


class EntityExtractor:
    """
    Advanced entity extractor for agricultural text
    Extracts crops, locations, dates, quantities, and other entities
    """
    
    def __init__(self, use_spacy: bool = True, use_transformer: bool = True):
        self.use_spacy = use_spacy
        self.use_transformer = use_transformer
        
        # Load spaCy model if available
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Initialize transformer NER if available
        if use_transformer:
            try:
                self.transformer_ner = pipeline(
                    "token-classification",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
            except Exception as e:
                print(f"Transformer NER not available: {e}")
                self.use_transformer = False
        
        # Entity patterns and dictionaries
        self._init_entity_patterns()
    
    def _init_entity_patterns(self):
        """Initialize entity extraction patterns"""
        
        # Crop entities
        self.crop_entities = {
            'cereals': [
                'wheat', 'rice', 'maize', 'bajra', 'jowar', 'ragi', 'barley', 'oats',
                'गेहूं', 'धान', 'मक्का', 'बाजरा', 'ज्वार', 'रागी', 'जौ', 'जई'
            ],
            'pulses': [
                'chickpea', 'pigeonpea', 'mungbean', 'urdbean', 'lentil', 'pea',
                'चना', 'अरहर', 'मूंग', 'उड़द', 'मसूर', 'मटर'
            ],
            'oilseeds': [
                'soybean', 'groundnut', 'mustard', 'sesame', 'sunflower', 'castor',
                'सोयाबीन', 'मूंगफली', 'सरसों', 'तिल', 'सूरजमुखी', 'अरंडी'
            ],
            'vegetables': [
                'potato', 'onion', 'tomato', 'brinjal', 'cabbage', 'cauliflower',
                'आलू', 'प्याज', 'टमाटर', 'बैंगन', 'पत्तागोभी', 'फूलगोभी'
            ],
            'fruits': [
                'mango', 'banana', 'apple', 'orange', 'grapes', 'papaya',
                'आम', 'केला', 'सेब', 'संतरा', 'अंगूर', 'पपीता'
            ],
            'cash_crops': [
                'cotton', 'sugarcane', 'tobacco', 'jute', 'tea', 'coffee',
                'कपास', 'गन्ना', 'तंबाकू', 'जूट', 'चाय', 'कॉफी'
            ]
        }
        
        # Location entities
        self.location_entities = {
            'states': [
                'punjab', 'haryana', 'uttar pradesh', 'madhya pradesh', 'rajasthan',
                'gujarat', 'maharashtra', 'karnataka', 'tamil nadu', 'andhra pradesh',
                'पंजाब', 'हरियाणा', 'उत्तर प्रदेश', 'मध्य प्रदेश', 'राजस्थान',
                'गुजरात', 'महाराष्ट्र', 'कर्नाटक', 'तमिलनाडु', 'आंध्र प्रदेश'
            ],
            'districts': [
                'ludhiana', 'amritsar', 'jalandhar', 'patiala', 'bathinda',
                'hisar', 'rohtak', 'karnal', 'gurgaon', 'faridabad',
                'लुधियाना', 'अमृतसर', 'जालंधर', 'पटियाला', 'बठिंडा'
            ]
        }
        
        # Activity entities
        self.activity_entities = {
            'farming_activities': [
                'sowing', 'planting', 'transplanting', 'irrigation', 'fertilizing',
                'pesticide application', 'harvesting', 'threshing', 'storage',
                'बुवाई', 'रोपाई', 'सिंचाई', 'खाद', 'कीटनाशक', 'कटाई', 'मड़ाई'
            ],
            'equipment': [
                'tractor', 'harvester', 'thresher', 'pump', 'sprayer', 'seeder',
                'ट्रैक्टर', 'हार्वेस्टर', 'थ्रेशर', 'पंप', 'स्प्रेयर', 'सीडर'
            ]
        }
        
        # Date and time patterns
        self.date_patterns = [
            r'\b(today|tomorrow|yesterday|next week|last week|next month|last month)\b',
            r'\b(आज|कल|परसों|अगले हफ्ते|पिछले हफ्ते|अगले महीने|पिछले महीने)\b',
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',  # DD/MM/YYYY
            r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+(\d{1,2})\b'
        ]
        
        # Quantity patterns
        self.quantity_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilogram|kilograms|quintal|quintals|ton|tons|acre|acres|hectare|hectares)\b',
            r'\b(\d+(?:\.\d+)?)\s*(किलो|क्विंटल|टन|एकड़|हेक्टेयर)\b',
            r'\b(\d+(?:\.\d+)?)\s*(rs|rupees|inr|lakh|lakhs|crore|crores)\b',
            r'\b(\d+(?:\.\d+)?)\s*(रुपये|लाख|करोड़)\b'
        ]
        
        # Weather and environmental patterns
        self.weather_patterns = [
            r'\b(rain|rainfall|drought|flood|storm|cyclone|temperature|humidity|wind)\b',
            r'\b(बारिश|सूखा|बाढ़|तूफान|तापमान|नमी|हवा)\b'
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all entities from the given text
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary with entity types and their extracted values
        """
        if not text.strip():
            return {}
        
        entities = {
            'crops': [],
            'locations': [],
            'activities': [],
            'dates': [],
            'quantities': [],
            'weather': [],
            'equipment': [],
            'organizations': [],
            'persons': []
        }
        
        # Extract using different methods
        if self.use_spacy:
            spacy_entities = self._extract_spacy_entities(text)
            self._merge_entities(entities, spacy_entities)
        
        if self.use_transformer:
            transformer_entities = self._extract_transformer_entities(text)
            self._merge_entities(entities, transformer_entities)
        
        # Extract using pattern matching
        pattern_entities = self._extract_pattern_entities(text)
        self._merge_entities(entities, pattern_entities)
        
        # Remove duplicates and normalize
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_spacy_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using spaCy"""
        entities = {entity_type: [] for entity_type in [
            'crops', 'locations', 'activities', 'dates', 'quantities', 
            'weather', 'equipment', 'organizations', 'persons'
        ]}
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.8,
                'source': 'spacy'
            }
            
            # Map spaCy entity types to our categories
            if ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(entity_info)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(entity_info)
            elif ent.label_ in ['QUANTITY', 'MONEY', 'PERCENT']:
                entities['quantities'].append(entity_info)
            elif ent.label_ in ['ORG']:
                entities['organizations'].append(entity_info)
            elif ent.label_ in ['PERSON']:
                entities['persons'].append(entity_info)
        
        return entities
    
    def _extract_transformer_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using transformer NER"""
        entities = {entity_type: [] for entity_type in [
            'crops', 'locations', 'activities', 'dates', 'quantities', 
            'weather', 'equipment', 'organizations', 'persons'
        ]}
        
        try:
            results = self.transformer_ner(text)
            
            for result in results:
                entity_info = {
                    'text': result['word'],
                    'start': result['start'],
                    'end': result['end'],
                    'confidence': result['score'],
                    'source': 'transformer'
                }
                
                # Map transformer labels to our categories
                label = result['entity_group']
                if label in ['LOC', 'GPE']:
                    entities['locations'].append(entity_info)
                elif label in ['DATE', 'TIME']:
                    entities['dates'].append(entity_info)
                elif label in ['MONEY', 'QUANTITY']:
                    entities['quantities'].append(entity_info)
                elif label in ['ORG']:
                    entities['organizations'].append(entity_info)
                elif label in ['PERSON']:
                    entities['persons'].append(entity_info)
        
        except Exception as e:
            print(f"Transformer NER extraction failed: {e}")
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using pattern matching"""
        entities = {entity_type: [] for entity_type in [
            'crops', 'locations', 'activities', 'dates', 'quantities', 
            'weather', 'equipment', 'organizations', 'persons'
        ]}
        
        text_lower = text.lower()
        
        # Extract crops
        for category, crops in self.crop_entities.items():
            for crop in crops:
                pattern = r'\b' + re.escape(crop.lower()) + r'\b'
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    entities['crops'].append({
                        'text': text[match.start():match.end()],
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9,
                        'source': 'pattern',
                        'category': category
                    })
        
        # Extract locations
        for category, locations in self.location_entities.items():
            for location in locations:
                pattern = r'\b' + re.escape(location.lower()) + r'\b'
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    entities['locations'].append({
                        'text': text[match.start():match.end()],
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9,
                        'source': 'pattern',
                        'category': category
                    })
        
        # Extract activities
        for category, activities in self.activity_entities.items():
            for activity in activities:
                pattern = r'\b' + re.escape(activity.lower()) + r'\b'
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    entities['activities'].append({
                        'text': text[match.start():match.end()],
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9,
                        'source': 'pattern',
                        'category': category
                    })
        
        # Extract dates
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities['dates'].append({
                    'text': text[match.start():match.end()],
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'source': 'pattern'
                })
        
        # Extract quantities
        for pattern in self.quantity_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities['quantities'].append({
                    'text': text[match.start():match.end()],
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'source': 'pattern'
                })
        
        # Extract weather terms
        for pattern in self.weather_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities['weather'].append({
                    'text': text[match.start():match.end()],
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'source': 'pattern'
                })
        
        return entities
    
    def _merge_entities(self, target: Dict[str, List[Dict[str, Any]]], source: Dict[str, List[Dict[str, Any]]]):
        """Merge entities from different sources"""
        for entity_type in target:
            if entity_type in source:
                target[entity_type].extend(source[entity_type])
    
    def _deduplicate_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Remove duplicate entities and normalize"""
        deduplicated = {}
        
        for entity_type, entity_list in entities.items():
            unique_entities = []
            seen_texts = set()
            
            for entity in entity_list:
                text_lower = entity['text'].lower()
                if text_lower not in seen_texts:
                    seen_texts.add(text_lower)
                    unique_entities.append(entity)
            
            deduplicated[entity_type] = unique_entities
        
        return deduplicated
    
    def get_entity_summary(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """
        Get a summary of extracted entities
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Summary with entity types and their text values
        """
        summary = {}
        
        for entity_type, entity_list in entities.items():
            summary[entity_type] = [entity['text'] for entity in entity_list]
        
        return summary
    
    def extract_specific_entities(self, text: str, entity_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract specific entity types from text
        
        Args:
            text: Input text
            entity_types: List of entity types to extract
            
        Returns:
            Dictionary with requested entity types
        """
        all_entities = self.extract_entities(text)
        return {entity_type: all_entities.get(entity_type, []) for entity_type in entity_types}
