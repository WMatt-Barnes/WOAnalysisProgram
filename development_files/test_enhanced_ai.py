#!/usr/bin/env python3
"""
Test script for Enhanced AI Failure Mode Classifier
Demonstrates the new low-risk classification methods
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the current directory to the path to import the AI classifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_failure_classifier import AIClassifier, ExpertSystemClassifier, ContextualPatternClassifier, TimeSeriesPatternClassifier
    print("✓ Enhanced AI Classifier imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Enhanced AI Classifier: {e}")
    sys.exit(1)

def test_expert_system():
    """Test the Expert System classifier"""
    print("\n=== Testing Expert System Classifier ===")
    
    expert = ExpertSystemClassifier()
    
    test_cases = [
        "Bearing making loud noise and excessive vibration",
        "Seal leaking fluid around the shaft",
        "Motor overheating and thermal trip activated",
        "Pump experiencing cavitation in suction line",
        "Electrical fault causing short circuit",
        "Valve stuck in closed position",
        "Belt broken and needs replacement",
        "Equipment showing signs of corrosion"
    ]
    
    for description in test_cases:
        code, desc, confidence = expert.classify(description)
        print(f"Description: {description}")
        print(f"  → Code: {code}, Description: {desc}, Confidence: {confidence:.3f}")
        print()

def test_contextual_patterns():
    """Test the Contextual Pattern classifier"""
    print("\n=== Testing Contextual Pattern Classifier ===")
    
    contextual = ContextualPatternClassifier()
    
    test_cases = [
        "Pump suction pressure low causing cavitation noise",
        "Motor current high and temperature alarm active",
        "Valve actuator not responding to position command",
        "Compressor discharge temperature too high",
        "Fan blade damaged causing imbalance vibration"
    ]
    
    for description in test_cases:
        code, desc, confidence = contextual.classify_with_context(description)
        print(f"Description: {description}")
        print(f"  → Code: {code}, Description: {desc}, Confidence: {confidence:.3f}")
        print()

def test_temporal_analysis():
    """Test the Temporal Analysis classifier"""
    print("\n=== Testing Temporal Analysis Classifier ===")
    
    temporal = TimeSeriesPatternClassifier()
    
    # Create sample work order data
    sample_data = {
        'Equipment #': ['PUMP-001', 'PUMP-001', 'PUMP-001', 'MOTOR-002', 'MOTOR-002'],
        'Failure Code': ['Bearing Failure', 'Bearing Failure', 'Seal Leak', 'Motor Overheating', 'Motor Overheating'],
        'Description': [
            'Bearing noise and vibration',
            'Bearing failure detected',
            'Seal leak around shaft',
            'Motor overheating',
            'Motor temperature high'
        ],
        'Reported Date': [
            '2024-01-15', '2024-02-15', '2024-03-15', '2024-01-20', '2024-02-20'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    temporal.analyze_temporal_patterns(df)
    
    test_cases = [
        ("PUMP-001", "Bearing making unusual noise"),
        ("MOTOR-002", "Motor running hot"),
        ("PUMP-001", "Seal leaking again"),
        ("UNKNOWN", "Random failure description")
    ]
    
    for equipment, description in test_cases:
        code, desc, confidence = temporal.classify_with_temporal_context(
            description, equipment, datetime.now()
        )
        print(f"Equipment: {equipment}, Description: {description}")
        print(f"  → Code: {code}, Description: {desc}, Confidence: {confidence:.3f}")
        print()

def test_enhanced_ai_classifier():
    """Test the complete Enhanced AI Classifier"""
    print("\n=== Testing Complete Enhanced AI Classifier ===")
    
    # Create a simple failure dictionary
    dict_data = {
        'Keyword': ['bearing', 'seal', 'motor', 'pump', 'valve'],
        'Code': ['Bearing Failure', 'Seal Leak', 'Motor Overheating', 'Pump Cavitation', 'Electrical Fault'],
        'Description': ['Bearing Failure', 'Seal Leak', 'Motor Overheating', 'Pump Cavitation', 'Valve Stuck']
    }
    
    dict_df = pd.DataFrame(dict_data)
    dict_path = "test_dictionary.xlsx"
    dict_df.to_excel(dict_path, index=False)
    
    try:
        # Initialize the enhanced AI classifier
        ai_classifier = AIClassifier(
            confidence_threshold=0.5,
            cache_file="test_ai_cache.json"
        )
        
        # Load the dictionary
        if ai_classifier.load_failure_dictionary(dict_path):
            print("✓ Dictionary loaded successfully")
        else:
            print("✗ Failed to load dictionary")
            return
        
        # Test classification
        test_descriptions = [
            "Bearing making loud noise and excessive vibration",
            "Seal leaking fluid around the shaft",
            "Motor overheating and thermal trip activated",
            "Pump experiencing cavitation in suction line",
            "Valve stuck in closed position"
        ]
        
        print("\nTesting Enhanced AI Classification:")
        for description in test_descriptions:
            result = ai_classifier.classify_hybrid(description, lambda desc: ('No Failure Mode Identified', 'No Match', 0.0))
            print(f"Description: {description}")
            print(f"  → Method: {result.method}")
            print(f"  → Code: {result.code}, Description: {result.description}")
            print(f"  → Confidence: {result.confidence:.3f}")
            print(f"  → Reasoning: {result.reasoning}")
            print()
        
        # Test batch classification
        print("Testing Batch Classification:")
        results = ai_classifier.batch_classify(test_descriptions, lambda desc: ('No Failure Mode Identified', 'No Match', 0.0))
        
        for i, result in enumerate(results):
            print(f"Batch {i+1}: {result.method} → {result.code} (confidence: {result.confidence:.3f})")
        
        # Get statistics
        stats = ai_classifier.get_classification_stats()
        print(f"\nClassification Statistics:")
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
        print(f"  Methods used: {stats['methods_used']}")
        
    except Exception as e:
        print(f"✗ Error testing enhanced AI classifier: {e}")
    finally:
        # Clean up test files
        if os.path.exists(dict_path):
            os.remove(dict_path)
        if os.path.exists("test_ai_cache.json"):
            os.remove("test_ai_cache.json")

def main():
    """Run all tests"""
    print("Enhanced AI Failure Mode Classifier - Test Suite")
    print("=" * 60)
    
    try:
        test_expert_system()
        test_contextual_patterns()
        test_temporal_analysis()
        test_enhanced_ai_classifier()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("\nEnhanced AI Classifier Features:")
        print("  • Expert System: Rule-based classification with weighted conditions")
        print("  • Contextual Patterns: Equipment-specific failure pattern recognition")
        print("  • Temporal Analysis: Historical pattern analysis for recurring failures")
        print("  • Sentence Embeddings: Semantic similarity matching")
        print("  • SpaCy NLP: Advanced linguistic analysis")
        print("  • Dictionary Fallback: Traditional keyword matching")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 