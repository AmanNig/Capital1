#!/usr/bin/env python3
"""
Comparison script to demonstrate different intent classification approaches
"""

import json
from nlp_pipeline.intent_classifier import IntentClassifier
from nlp_pipeline.advanced_intent_classifier import AdvancedIntentClassifier


def format_scores(scores):
    """Format scores for display"""
    return {k: f"{v:.3f}" for k, v in scores.items()}


def compare_classifiers():
    """Compare keyword-based vs ML-based classification"""
    
    print("=" * 80)
    print("INTENT CLASSIFICATION COMPARISON")
    print("=" * 80)
    
    # Initialize classifiers
    print("Initializing classifiers...")
    keyword_classifier = IntentClassifier(use_transformer=False)  # No transformers for fair comparison
    ml_classifier = AdvancedIntentClassifier(use_semantic=True, use_zero_shot=True)
    
    # Test cases
    test_cases = [
        "my rices are not grown properly give me the remedies for this",
        "what is the price of rice in Punjab mandi",
        "how to apply for government subsidy scheme",
        "weather forecast for next week farming",
        "tractor maintenance and repair issues",
        "basic information about organic farming",
        "crop disease treatment methods",
        "market rates for wheat today",
        "loan application process for farmers",
        "irrigation system problems and solutions"
    ]
    
    print(f"\nTesting {len(test_cases)} queries...")
    print("-" * 80)
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        # Keyword-based classification
        keyword_scores = keyword_classifier.classify_intent(query)
        keyword_intent = keyword_classifier.get_primary_intent(query)
        keyword_conf = keyword_classifier.get_intent_confidence(query)
        
        # ML-based classification
        ml_scores = ml_classifier.classify_intent(query)
        ml_intent = ml_classifier.get_primary_intent(query)
        ml_conf = ml_classifier.get_intent_confidence(query)
        
        # Display results
        print(f"Keyword-based: {keyword_intent} (confidence: {keyword_conf:.3f})")
        print(f"ML-based:      {ml_intent} (confidence: {ml_conf:.3f})")
        
        # Show if they agree
        if keyword_intent == ml_intent:
            print("✅ AGREEMENT")
        else:
            print("❌ DISAGREEMENT")
        
        # Show detailed scores for disagreement cases
        if keyword_intent != ml_intent:
            print(f"\nDetailed scores:")
            print(f"Keyword: {format_scores(keyword_scores)}")
            print(f"ML:      {format_scores(ml_scores)}")


def demonstrate_ml_advantages():
    """Demonstrate advantages of ML-based classification"""
    
    print("\n" + "=" * 80)
    print("ML-BASED CLASSIFICATION ADVANTAGES")
    print("=" * 80)
    
    ml_classifier = AdvancedIntentClassifier(use_semantic=True, use_zero_shot=True)
    
    # Test cases that keyword matching would struggle with
    challenging_cases = [
        # Synonyms and paraphrases
        ("my rices are not grown properly give me the remedies for this", "crop_advice"),
        ("agricultural commodity rates", "price_query"),
        ("farmer assistance programs", "policy_query"),
        ("climate conditions for agriculture", "weather_query"),
        ("farm machinery issues", "technical_support"),
        ("farming basics for beginners", "general_inquiry"),
        
        # Complex queries
        ("I need help with my failing wheat crop due to lack of proper irrigation", "crop_advice"),
        ("What are the current market prices for agricultural commodities in the local mandi", "price_query"),
        ("How can I access government financial support for purchasing farming equipment", "policy_query"),
        ("Will the weather conditions be suitable for sowing crops next week", "weather_query"),
        ("My automated irrigation system is malfunctioning and needs technical support", "technical_support"),
        ("Can you provide general information about sustainable farming practices", "general_inquiry"),
        
        # Hindi queries
        ("मेरी फसल में रोग लग गया है", "crop_advice"),
        ("गेहूं का भाव क्या है आज", "price_query"),
        ("सरकारी योजना कैसे मिलेगी", "policy_query"),
        ("आज का मौसम कैसा है", "weather_query"),
        ("ट्रैक्टर में समस्या आ गई है", "technical_support"),
        ("खेती के बारे में जानकारी चाहिए", "general_inquiry"),
        
        # Code-mixed queries (Hindi + English)
        ("मेरे crops में disease लग गया है", "crop_advice"),
        ("wheat का price क्या है mandi में", "price_query"),
        ("government scheme कैसे apply करें", "policy_query"),
        ("weather forecast कैसा है आज", "weather_query"),
        ("tractor में problem आ गई है", "technical_support"),
        ("farming के बारे में information चाहिए", "general_inquiry"),
        
        # Complex Hindi queries
        ("मेरी गेहूं की फसल में पीले पत्ते आ रहे हैं और पौधे सूख रहे हैं", "crop_advice"),
        ("आज के मंडी में धान का क्या भाव है और कल का क्या रहेगा", "price_query"),
        ("पीएम किसान योजना में कैसे आवेदन करना है और कौन सी डॉक्यूमेंट्स चाहिए", "policy_query"),
        ("अगले हफ्ते बारिश का मौसम कैसा रहेगा और खेती के लिए क्या सही होगा", "weather_query"),
        ("मेरे सिंचाई सिस्टम में कुछ तकनीकी समस्या आ गई है", "technical_support"),
        ("जैविक खेती के बारे में पूरी जानकारी और इसके फायदे क्या हैं", "general_inquiry")
    ]
    
    print(f"\nTesting {len(challenging_cases)} challenging queries...")
    print("-" * 80)
    
    correct = 0
    for query, expected_intent in challenging_cases:
        predicted_intent = ml_classifier.get_primary_intent(query)
        confidence = ml_classifier.get_intent_confidence(query)
        
        is_correct = predicted_intent == expected_intent
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Query: {query[:60]}...")
        print(f"   Expected: {expected_intent}")
        print(f"   Predicted: {predicted_intent} (confidence: {confidence:.3f})")
        print()
    
    accuracy = correct / len(challenging_cases)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(challenging_cases)})")


def show_learning_capability():
    """Show how the ML classifier can learn from new examples"""
    
    print("\n" + "=" * 80)
    print("LEARNING CAPABILITY DEMONSTRATION")
    print("=" * 80)
    
    ml_classifier = AdvancedIntentClassifier(use_semantic=True, use_zero_shot=True)
    
    # Test a query that might be misclassified
    test_query = "my farm equipment is broken and I need help fixing it"
    
    print(f"Initial classification for: {test_query}")
    initial_intent = ml_classifier.get_primary_intent(test_query)
    initial_conf = ml_classifier.get_intent_confidence(test_query)
    print(f"Result: {initial_intent} (confidence: {initial_conf:.3f})")
    
    # Add training example
    print(f"\nAdding training example: '{test_query}' -> 'technical_support'")
    ml_classifier.add_training_example(test_query, "technical_support")
    
    # Test again
    print(f"\nRe-classifying the same query...")
    new_intent = ml_classifier.get_primary_intent(test_query)
    new_conf = ml_classifier.get_intent_confidence(test_query)
    print(f"Result: {new_intent} (confidence: {new_conf:.3f})")
    
    if new_intent == "technical_support":
        print("✅ Successfully learned!")
    else:
        print("❌ Still needs more training examples")


def test_hindi_queries():
    """Test Hindi and code-mixed queries specifically"""
    
    print("\n" + "=" * 80)
    print("HINDI AND CODE-MIXED QUERIES TEST")
    print("=" * 80)
    
    # Initialize classifiers
    keyword_classifier = IntentClassifier(use_transformer=False)
    ml_classifier = AdvancedIntentClassifier(use_semantic=True, use_zero_shot=True)
    
    # Hindi test cases
    hindi_cases = [
        # Pure Hindi queries
        ("मेरी फसल में रोग लग गया है", "crop_advice"),
        ("गेहूं का भाव क्या है आज", "price_query"),
        ("सरकारी योजना कैसे मिलेगी", "policy_query"),
        ("आज का मौसम कैसा है", "weather_query"),
        ("ट्रैक्टर में समस्या आ गई है", "technical_support"),
        ("खेती के बारे में जानकारी चाहिए", "general_inquiry"),
        
        # Code-mixed queries
        ("मेरे crops में disease लग गया है", "crop_advice"),
        ("wheat का price क्या है mandi में", "price_query"),
        ("government scheme कैसे apply करें", "policy_query"),
        ("weather forecast कैसा है आज", "weather_query"),
        ("tractor में problem आ गई है", "technical_support"),
        ("farming के बारे में information चाहिए", "general_inquiry"),
        
        # Complex Hindi queries
        ("मेरी गेहूं की फसल में पीले पत्ते आ रहे हैं और पौधे सूख रहे हैं", "crop_advice"),
        ("आज के मंडी में धान का क्या भाव है और कल का क्या रहेगा", "price_query"),
        ("पीएम किसान योजना में कैसे आवेदन करना है", "policy_query"),
        ("अगले हफ्ते बारिश का मौसम कैसा रहेगा", "weather_query"),
        ("मेरे सिंचाई सिस्टम में कुछ तकनीकी समस्या आ गई है", "technical_support"),
        ("जैविक खेती के बारे में पूरी जानकारी चाहिए", "general_inquiry")
    ]
    
    print(f"\nTesting {len(hindi_cases)} Hindi and code-mixed queries...")
    print("-" * 80)
    
    keyword_correct = 0
    ml_correct = 0
    
    for i, (query, expected) in enumerate(hindi_cases, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        # Keyword-based classification
        keyword_intent = keyword_classifier.get_primary_intent(query)
        keyword_conf = keyword_classifier.get_intent_confidence(query)
        
        # ML-based classification
        ml_intent = ml_classifier.get_primary_intent(query)
        ml_conf = ml_classifier.get_intent_confidence(query)
        
        # Check accuracy
        keyword_right = keyword_intent == expected
        ml_right = ml_intent == expected
        
        if keyword_right:
            keyword_correct += 1
        if ml_right:
            ml_correct += 1
        
        # Display results
        print(f"Expected: {expected}")
        print(f"Keyword:  {keyword_intent} (confidence: {keyword_conf:.3f}) {'✅' if keyword_right else '❌'}")
        print(f"ML:       {ml_intent} (confidence: {ml_conf:.3f}) {'✅' if ml_right else '❌'}")
        
        # Show disagreement
        if keyword_intent != ml_intent:
            print("⚠️  DISAGREEMENT between methods")
    
    # Final accuracy comparison
    print(f"\n" + "=" * 50)
    print("ACCURACY COMPARISON FOR HINDI QUERIES")
    print("=" * 50)
    print(f"Keyword-based: {keyword_correct}/{len(hindi_cases)} ({keyword_correct/len(hindi_cases):.1%})")
    print(f"ML-based:      {ml_correct}/{len(hindi_cases)} ({ml_correct/len(hindi_cases):.1%})")
    
    if ml_correct > keyword_correct:
        print("✅ ML-based classifier performs better on Hindi queries!")
    elif keyword_correct > ml_correct:
        print("❌ Keyword-based classifier performs better on Hindi queries!")
    else:
        print("🤝 Both methods perform equally on Hindi queries!")


def main():
    """Main function"""
    try:
        compare_classifiers()
        demonstrate_ml_advantages()
        test_hindi_queries()
        show_learning_capability()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("ML-based classification advantages:")
        print("1. Better handling of synonyms and paraphrases")
        print("2. Semantic understanding beyond exact keyword matching")
        print("3. Ability to learn from new examples")
        print("4. More robust to variations in query formulation")
        print("5. Higher accuracy on complex, multi-part queries")
        print("6. Better performance on Hindi and code-mixed queries")
        print("\nThe advanced classifier combines:")
        print("- Traditional ML (TF-IDF + Naive Bayes, Logistic Regression, Random Forest)")
        print("- Semantic similarity using sentence transformers")
        print("- Zero-shot classification using transformers")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install sentence-transformers scikit-learn")


if __name__ == "__main__":
    main()
