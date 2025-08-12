"""
Main NLP Pipeline
Orchestrates all components for end-to-end query processing
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from .language_detector import LanguageDetector
from .normalizer import TextNormalizer
from .advanced_intent_classifier import AdvancedIntentClassifier
from .entity_extractor import EntityExtractor
import json
import time


class QueryResult(BaseModel):
    """Result model for query processing"""
    original_text: str
    normalized_text: str
    language_detection: Dict[str, float]
    primary_language: str
    is_code_mixed: bool
    intent_classification: Dict[str, float]
    primary_intent: str
    intent_confidence: float
    entities: Dict[str, List[Dict[str, Any]]]
    entity_summary: Dict[str, List[str]]
    processing_time: float
    metadata: Dict[str, Any]


class QueryProcessingPipeline:
    """
    Main pipeline for processing agricultural queries
    Combines language detection, normalization, intent classification, and entity extraction
    """
    
    def __init__(
        self,
        use_transformer: bool = True,
        use_spacy: bool = True,
        use_semantic: bool = True,
        use_zero_shot: bool = True,
        model_path: Optional[str] = None
    ):
        """
        Initialize the pipeline with all components
        
        Args:
            use_transformer: Whether to use transformer models
            use_spacy: Whether to use spaCy for entity extraction
            use_semantic: Whether to use semantic similarity in intent classification
            use_zero_shot: Whether to use zero-shot classification in intent classification
            model_path: Path to saved models
        """
        self.language_detector = LanguageDetector(use_transformer=use_transformer)
        self.normalizer = TextNormalizer()
        self.intent_classifier = AdvancedIntentClassifier(
            use_semantic=use_semantic,
            use_zero_shot=use_zero_shot
        )
        self.entity_extractor = EntityExtractor(
            use_spacy=use_spacy,
            use_transformer=use_transformer
        )
        
        # Pipeline configuration
        self.config = {
            'use_transformer': use_transformer,
            'use_spacy': use_spacy,
            'use_semantic': use_semantic,
            'use_zero_shot': use_zero_shot,
            'normalize_text': True,
            'extract_entities': True,
            'classify_intent': True
        }
    
    def process_query(self, text: str, config: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Process a query through the complete pipeline
        
        Args:
            text: Input query text
            config: Optional configuration overrides
            
        Returns:
            QueryResult with all processing results
        """
        start_time = time.time()
        
        # Use provided config or default
        pipeline_config = {**self.config, **(config or {})}
        
        # Step 1: Language Detection
        language_detection = self.language_detector.detect_language(text)
        primary_language = self.language_detector.get_primary_language(text)
        is_code_mixed = self.language_detector.is_code_mixed(text)
        
        # Step 2: Text Normalization
        normalized_text = text
        if pipeline_config['normalize_text']:
            normalized_text = self.normalizer.normalize_text(
                text, 
                target_language='en'
            )
        
        # Step 3: Intent Classification
        intent_classification = {'general_inquiry': 1.0}
        primary_intent = 'general_inquiry'
        intent_confidence = 1.0
        
        if pipeline_config['classify_intent']:
            intent_classification = self.intent_classifier.classify_intent(normalized_text)
            primary_intent = self.intent_classifier.get_primary_intent(normalized_text)
            intent_confidence = self.intent_classifier.get_intent_confidence(normalized_text)
        
        # Step 4: Entity Extraction
        entities = {}
        entity_summary = {}
        
        if pipeline_config['extract_entities']:
            entities = self.entity_extractor.extract_entities(normalized_text)
            entity_summary = self.entity_extractor.get_entity_summary(entities)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            'pipeline_config': pipeline_config,
            'components_used': {
                'language_detector': True,
                'normalizer': pipeline_config['normalize_text'],
                'intent_classifier': pipeline_config['classify_intent'],
                'entity_extractor': pipeline_config['extract_entities']
            }
        }
        
        return QueryResult(
            original_text=text,
            normalized_text=normalized_text,
            language_detection=language_detection,
            primary_language=primary_language,
            is_code_mixed=is_code_mixed,
            intent_classification=intent_classification,
            primary_intent=primary_intent,
            intent_confidence=intent_confidence,
            entities=entities,
            entity_summary=entity_summary,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def process_batch(self, texts: List[str], config: Optional[Dict[str, Any]] = None) -> List[QueryResult]:
        """
        Process multiple queries in batch
        
        Args:
            texts: List of input query texts
            config: Optional configuration overrides
            
        Returns:
            List of QueryResult objects
        """
        results = []
        for text in texts:
            result = self.process_query(text, config)
            results.append(result)
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline components
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'components': {
                'language_detector': {
                    'type': 'LanguageDetector',
                    'transformer_enabled': self.language_detector.use_transformer
                },
                'normalizer': {
                    'type': 'TextNormalizer',
                    'hindi_mappings': len(self.normalizer.hindi_to_english)
                },
                'intent_classifier': {
                    'type': 'AdvancedIntentClassifier',
                    'semantic_enabled': self.intent_classifier.use_semantic,
                    'zero_shot_enabled': self.intent_classifier.use_zero_shot,
                    'intents': list(self.intent_classifier.intents.keys())
                },
                'entity_extractor': {
                    'type': 'EntityExtractor',
                    'spacy_enabled': self.entity_extractor.use_spacy,
                    'transformer_enabled': self.entity_extractor.use_transformer
                }
            },
            'config': self.config
        }
    
    def save_pipeline(self, path: str):
        """Save pipeline components"""
        import pickle
        
        pipeline_data = {
            'config': self.config,
            'language_detector': self.language_detector,
            'normalizer': self.normalizer,
            'intent_classifier': self.intent_classifier,
            'entity_extractor': self.entity_extractor
        }
        
        with open(path, 'wb') as f:
            pickle.dump(pipeline_data, f)
    
    def load_pipeline(self, path: str):
        """Load pipeline components"""
        import pickle
        
        with open(path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.config = pipeline_data['config']
        self.language_detector = pipeline_data['language_detector']
        self.normalizer = pipeline_data['normalizer']
        self.intent_classifier = pipeline_data['intent_classifier']
        self.entity_extractor = pipeline_data['entity_extractor']
    
    def get_statistics(self, results: List[QueryResult]) -> Dict[str, Any]:
        """
        Get statistics from batch processing results
        
        Args:
            results: List of QueryResult objects
            
        Returns:
            Dictionary with processing statistics
        """
        if not results:
            return {}
        
        # Language statistics
        languages = [r.primary_language for r in results]
        language_counts = {}
        for lang in languages:
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Intent statistics
        intents = [r.primary_intent for r in results]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Entity statistics
        total_entities = 0
        entity_type_counts = {}
        for result in results:
            for entity_type, entity_list in result.entities.items():
                total_entities += len(entity_list)
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + len(entity_list)
        
        # Performance statistics
        processing_times = [r.processing_time for r in results]
        
        return {
            'total_queries': len(results),
            'languages': {
                'distribution': language_counts,
                'code_mixed_count': sum(1 for r in results if r.is_code_mixed)
            },
            'intents': {
                'distribution': intent_counts,
                'average_confidence': sum(r.intent_confidence for r in results) / len(results)
            },
            'entities': {
                'total_extracted': total_entities,
                'by_type': entity_type_counts,
                'average_per_query': total_entities / len(results)
            },
            'performance': {
                'average_processing_time': sum(processing_times) / len(processing_times),
                'min_processing_time': min(processing_times),
                'max_processing_time': max(processing_times)
            }
        }
    
    def export_results(self, results: List[QueryResult], format: str = 'json', path: Optional[str] = None) -> str:
        """
        Export processing results to file
        
        Args:
            results: List of QueryResult objects
            format: Export format ('json' or 'csv')
            path: Output file path (optional)
            
        Returns:
            Exported data as string or file path
        """
        if format.lower() == 'json':
            data = [result.dict() for result in results]
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                return path
            return json_str
        
        elif format.lower() == 'csv':
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for result in results:
                row = {
                    'original_text': result.original_text,
                    'normalized_text': result.normalized_text,
                    'primary_language': result.primary_language,
                    'is_code_mixed': result.is_code_mixed,
                    'primary_intent': result.primary_intent,
                    'intent_confidence': result.intent_confidence,
                    'processing_time': result.processing_time
                }
                
                # Add entity counts
                for entity_type, entity_list in result.entities.items():
                    row[f'{entity_type}_count'] = len(entity_list)
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            
            if path:
                df.to_csv(path, index=False, encoding='utf-8')
                return path
            return df.to_csv(index=False, encoding='utf-8')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
