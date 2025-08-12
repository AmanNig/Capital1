#!/usr/bin/env python3
"""
Agricultural NLP Chatbot
Interactive CLI chatbot for agricultural query processing
Provides intent classification, entity extraction, and language detection
"""

import sys
import json
import time
from typing import Dict, List, Any
from nlp_pipeline import QueryProcessingPipeline


class AgriculturalChatbot:
    """Interactive chatbot for agricultural query processing"""
    
    def __init__(self, use_transformer: bool = True, use_semantic: bool = True, use_zero_shot: bool = True):
        """Initialize the chatbot with pipeline components"""
        print("Initializing Agricultural NLP Chatbot...")
        print("Loading models and components...")
        
        self.pipeline = QueryProcessingPipeline(
            use_transformer=use_transformer,
            use_semantic=use_semantic,
            use_zero_shot=use_zero_shot
        )
        
        print("✅ Chatbot initialized successfully!")
        print()
    
    def print_welcome(self):
        """Print welcome message and instructions"""
        print("=" * 80)
        print("🌾 AGRICULTURAL NLP CHATBOT 🌾")
        print("=" * 80)
        print()
        print("This chatbot can help you with agricultural queries by providing:")
        print("• Language Detection (Hindi/English/Code-mixed)")
        print("• Intent Classification (6 categories)")
        print("• Entity Extraction (crops, locations, activities, etc.)")
        print("• Text Normalization")
        print()
        print("Supported Intent Categories:")
        print("• crop_advice     - Farming practices, diseases, cultivation")
        print("• policy_query    - Government schemes, subsidies, policies")
        print("• price_query     - Market rates, crop prices, selling")
        print("• weather_query   - Weather conditions, forecasts")
        print("• technical_support - Equipment, technology, digital farming")
        print("• general_inquiry - General agricultural information")
        print()
        print("Commands:")
        print("• Type your query and press Enter")
        print("• Type 'help' for this message")
        print("• Type 'stats' to see processing statistics")
        print("• Type 'quit' or 'exit' to close the chatbot")
        print("• Type 'clear' to clear the screen")
        print()
        print("Examples:")
        print("• 'मेरी फसल में रोग लग गया है'")
        print("• 'What is the price of rice in Punjab?'")
        print("• 'How to irrigate wheat crop properly?'")
        print("• 'सरकारी योजना कैसे मिलेगी?'")
        print()
        print("-" * 80)
        print()
    
    def format_confidence(self, confidence: float) -> str:
        """Format confidence score with color indicators"""
        if confidence >= 0.8:
            return f"🟢 {confidence:.3f} (High)"
        elif confidence >= 0.6:
            return f"🟡 {confidence:.3f} (Medium)"
        else:
            return f"🔴 {confidence:.3f} (Low)"
    
    def print_language_info(self, result):
        """Print language detection results"""
        print("🌐 LANGUAGE DETECTION:")
        print(f"   Primary Language: {result.primary_language.upper()}")
        print(f"   Code-mixed: {'Yes' if result.is_code_mixed else 'No'}")
        
        # Show language scores
        lang_scores = result.language_detection
        if lang_scores:
            print("   Language Scores:")
            for lang, score in sorted(lang_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"     • {lang.upper()}: {score:.3f}")
        print()
    
    def print_intent_info(self, result):
        """Print intent classification results"""
        print("🎯 INTENT CLASSIFICATION:")
        print(f"   Primary Intent: {result.primary_intent.replace('_', ' ').title()}")
        print(f"   Confidence: {self.format_confidence(result.intent_confidence)}")
        
        # Show all intent scores
        intent_scores = result.intent_classification
        if intent_scores:
            print("   All Intent Scores:")
            for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True):
                intent_name = intent.replace('_', ' ').title()
                confidence_indicator = self.format_confidence(score)
                print(f"     • {intent_name}: {confidence_indicator}")
        print()
    
    def print_entity_info(self, result):
        """Print entity extraction results"""
        print("🏷️ ENTITY EXTRACTION:")
        
        if not result.entity_summary:
            print("   No entities found")
        else:
            for entity_type, entities in result.entity_summary.items():
                if entities:
                    entity_type_name = entity_type.replace('_', ' ').title()
                    print(f"   {entity_type_name}: {', '.join(entities)}")
        
        # Show detailed entities if available
        if result.entities:
            total_entities = sum(len(entities) for entities in result.entities.values())
            if total_entities > 0:
                print(f"   Total Entities Found: {total_entities}")
        print()
    
    def print_processing_info(self, result):
        """Print processing information"""
        print("⚡ PROCESSING INFO:")
        print(f"   Processing Time: {result.processing_time:.3f} seconds")
        print(f"   Normalized Text: {result.normalized_text}")
        print()
    
    def print_result(self, query: str, result):
        """Print complete processing result"""
        print("=" * 80)
        print(f"📝 QUERY: {query}")
        print("=" * 80)
        print()
        
        self.print_language_info(result)
        self.print_intent_info(result)
        self.print_entity_info(result)
        self.print_processing_info(result)
        
        print("=" * 80)
        print()
    
    def get_intent_description(self, intent: str) -> str:
        """Get description for intent category"""
        descriptions = {
            'crop_advice': 'Farming practices, crop diseases, cultivation techniques',
            'policy_query': 'Government schemes, subsidies, policies',
            'price_query': 'Market rates, crop prices, selling information',
            'weather_query': 'Weather conditions, forecasts, climate information',
            'technical_support': 'Equipment, technology, digital farming',
            'general_inquiry': 'General agricultural information'
        }
        return descriptions.get(intent, 'Unknown intent')
    
    def print_help(self):
        """Print help information"""
        print("\n" + "=" * 80)
        print("📖 HELP & INFORMATION")
        print("=" * 80)
        print()
        print("This chatbot processes agricultural queries and provides:")
        print()
        print("🌐 Language Detection:")
        print("   • Detects Hindi, English, and code-mixed text")
        print("   • Shows confidence scores for each language")
        print("   • Identifies when text mixes multiple languages")
        print()
        print("🎯 Intent Classification:")
        print("   • Categorizes queries into 6 intent types")
        print("   • Uses advanced ML methods for better accuracy")
        print("   • Shows confidence scores for all intents")
        print()
        print("🏷️ Entity Extraction:")
        print("   • Extracts crops, locations, activities, quantities")
        print("   • Identifies dates, weather terms, equipment")
        print("   • Provides both summary and detailed entity lists")
        print()
        print("📝 Text Normalization:")
        print("   • Converts slang and local terms to standard forms")
        print("   • Handles Hindi-to-English agricultural terminology")
        print()
        print("Commands:")
        print("   help    - Show this help message")
        print("   stats   - Show processing statistics")
        print("   clear   - Clear the screen")
        print("   quit    - Exit the chatbot")
        print("   exit    - Exit the chatbot")
        print()
        print("=" * 80)
        print()
    
    def print_stats(self, stats: Dict[str, Any]):
        """Print processing statistics"""
        print("\n" + "=" * 80)
        print("📊 PROCESSING STATISTICS")
        print("=" * 80)
        print()
        
        if not stats:
            print("No statistics available yet. Process some queries first!")
            print()
            return
        
        print(f"📈 Total Queries Processed: {stats.get('total_queries', 0)}")
        print()
        
        # Language statistics
        languages = stats.get('languages', {})
        if languages:
            print("🌐 Language Distribution:")
            lang_dist = languages.get('distribution', {})
            for lang, count in lang_dist.items():
                print(f"   • {lang.upper()}: {count}")
            print(f"   • Code-mixed: {languages.get('code_mixed_count', 0)}")
            print()
        
        # Intent statistics
        intents = stats.get('intents', {})
        if intents:
            print("🎯 Intent Distribution:")
            intent_dist = intents.get('distribution', {})
            for intent, count in intent_dist.items():
                intent_name = intent.replace('_', ' ').title()
                print(f"   • {intent_name}: {count}")
            avg_conf = intents.get('average_confidence', 0)
            print(f"   • Average Confidence: {avg_conf:.3f}")
            print()
        
        # Entity statistics
        entities = stats.get('entities', {})
        if entities:
            print("🏷️ Entity Statistics:")
            total_entities = entities.get('total_extracted', 0)
            avg_per_query = entities.get('average_per_query', 0)
            print(f"   • Total Entities: {total_entities}")
            print(f"   • Average per Query: {avg_per_query:.2f}")
            
            entity_types = entities.get('by_type', {})
            if entity_types:
                print("   • By Type:")
                for entity_type, count in entity_types.items():
                    type_name = entity_type.replace('_', ' ').title()
                    print(f"     - {type_name}: {count}")
            print()
        
        # Performance statistics
        performance = stats.get('performance', {})
        if performance:
            print("⚡ Performance Statistics:")
            avg_time = performance.get('average_processing_time', 0)
            min_time = performance.get('min_processing_time', 0)
            max_time = performance.get('max_processing_time', 0)
            print(f"   • Average Time: {avg_time:.3f} seconds")
            print(f"   • Min Time: {min_time:.3f} seconds")
            print(f"   • Max Time: {max_time:.3f} seconds")
            print()
        
        print("=" * 80)
        print()
    
    def clear_screen(self):
        """Clear the screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_welcome()
    
    def run(self):
        """Run the interactive chatbot"""
        self.print_welcome()
        
        # Statistics tracking
        all_results = []
        
        while True:
            try:
                # Get user input
                query = input("🌾 You: ").strip()
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Agricultural NLP Chatbot!")
                    print("Goodbye! 🌾")
                    break
                
                elif query.lower() == 'help':
                    self.print_help()
                    continue
                
                elif query.lower() == 'stats':
                    stats = self.pipeline.get_statistics(all_results) if all_results else {}
                    self.print_stats(stats)
                    continue
                
                elif query.lower() == 'clear':
                    self.clear_screen()
                    continue
                
                elif not query:
                    print("Please enter a query or type 'help' for assistance.")
                    continue
                
                # Process the query
                print("🔄 Processing...")
                start_time = time.time()
                
                result = self.pipeline.process_query(query)
                all_results.append(result)
                
                # Print results
                self.print_result(query, result)
                
                # Show intent description
                intent_desc = self.get_intent_description(result.primary_intent)
                print(f"💡 Intent Description: {intent_desc}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using Agricultural NLP Chatbot!")
                print("Goodbye! 🌾")
                break
            
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("Please try again or type 'help' for assistance.\n")


def main():
    """Main function to run the chatbot"""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Agricultural NLP Chatbot")
        parser.add_argument(
            "--no-transformer",
            action="store_true",
            help="Disable transformer models for faster processing"
        )
        parser.add_argument(
            "--no-semantic",
            action="store_true",
            help="Disable semantic similarity in intent classification"
        )
        parser.add_argument(
            "--no-zero-shot",
            action="store_true",
            help="Disable zero-shot classification in intent classification"
        )
        
        args = parser.parse_args()
        
        # Initialize chatbot
        chatbot = AgriculturalChatbot(
            use_transformer=not args.no_transformer,
            use_semantic=not args.no_semantic,
            use_zero_shot=not args.no_zero_shot
        )
        
        # Run the chatbot
        chatbot.run()
        
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure proper encoding for Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    
    main()
