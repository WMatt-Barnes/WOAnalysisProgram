"""
Test script for SpaCy-enhanced AI failure classification
Demonstrates the enhanced NLP capabilities with SpaCy integration
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_failure_classifier import AIClassifier, AIClassificationResult
    print("✓ AI classifier module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AI classifier: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

def create_sample_dictionary():
    """Create a sample failure mode dictionary for testing"""
    sample_data = {
        'Keyword': [
            'pump failure, pump broken, pump not working, pump malfunction',
            'motor failure, motor burned out, motor seized, motor overheating',
            'bearing failure, bearing worn, bearing noise, bearing vibration',
            'leak, leaking, fluid leak, oil leak, seal leak',
            'valve failure, valve stuck, valve malfunction, valve actuator',
            'sensor failure, sensor reading, sensor error, transmitter fault',
            'electrical failure, electrical problem, wiring issue, circuit fault',
            'mechanical failure, mechanical problem, equipment failure, component failure'
        ],
        'Code': ['Bearing Failure', 'Bearing Failure', 'Bearing Failure', 'Seal Leak', 'Seal Leak', 'Motor Overheating', 'Motor Overheating', 'Pump Cavitation'],
        'Description': [
            'Pump Failure',
            'Motor Failure', 
            'Bearing Failure',
            'Fluid Leak',
            'Valve Failure',
            'Sensor Failure',
            'Electrical Failure',
            'Mechanical Failure'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_excel('sample_failure_dictionary.xlsx', index=False)
    print("✓ Created sample failure dictionary: sample_failure_dictionary.xlsx")
    return 'sample_failure_dictionary.xlsx'

def test_spacy_analysis():
    """Test SpaCy NLP analysis capabilities"""
    print("\n" + "="*60)
    print("SPACY NLP ANALYSIS TEST")
    print("="*60)
    
    # Create sample data
    dict_file = create_sample_dictionary()
    
    # Initialize AI classifier
    print("\nInitializing AI classifier with SpaCy...")
    try:
        classifier = AIClassifier(
            confidence_threshold=0.3,
            cache_file="test_spacy_cache.json"
        )
        print("✓ AI classifier initialized")
    except Exception as e:
        print(f"✗ Failed to initialize AI classifier: {e}")
        return
    
    # Check if SpaCy is available
    if not classifier.nlp:
        print("✗ SpaCy model not available. Install with: python -m spacy download en_core_web_sm")
        return
    
    print("✓ SpaCy model loaded successfully")
    
    # Load failure dictionary
    print("\nLoading failure dictionary...")
    if not classifier.load_failure_dictionary(dict_file):
        print("✗ Failed to load failure dictionary")
        return
    print("✓ Failure dictionary loaded")
    
    # Test SpaCy analysis on various descriptions
    test_descriptions = [
        "Pump P-101 failed due to bearing wear and excessive vibration",
        "Motor M-205 overheated and tripped on thermal protection",
        "Oil leak detected from pump seal on compressor C-301",
        "Pressure sensor PT-401 giving erratic readings",
        "Valve actuator stuck in closed position",
        "Electrical panel showing fault codes and circuit breaker tripped",
        "Mechanical seal failure on centrifugal pump",
        "Temperature transmitter TT-102 reading high and causing alarm"
    ]
    
    print("\nTesting SpaCy NLP analysis:")
    for i, description in enumerate(test_descriptions, 1):
        print(f"\nTest {i}: {description}")
        
        try:
            # Analyze with SpaCy
            analysis = classifier.analyze_with_spacy(description)
            
            if analysis:
                print(f"  Equipment Types: {analysis.get('equipment_types', [])}")
                print(f"  Failure Indicators: {analysis.get('failure_indicators', [])}")
                print(f"  Technical Terms: {analysis.get('technical_terms', [])[:5]}...")  # Show first 5
                print(f"  Entities: {analysis.get('entities', [])}")
                print(f"  Noun Chunks: {analysis.get('noun_chunks', [])[:3]}...")  # Show first 3
                print(f"  Failure Verbs: {analysis.get('failure_verbs', [])}")
            else:
                print("  No analysis results")
                
        except Exception as e:
            print(f"  Error: {e}")

def test_spacy_classification():
    """Test SpaCy-based classification"""
    print("\n" + "="*60)
    print("SPACY CLASSIFICATION TEST")
    print("="*60)
    
    # Create sample data
    dict_file = create_sample_dictionary()
    
    # Initialize AI classifier
    print("\nInitializing AI classifier...")
    try:
        classifier = AIClassifier(
            confidence_threshold=0.3,
            cache_file="test_spacy_classification_cache.json"
        )
        print("✓ AI classifier initialized")
    except Exception as e:
        print(f"✗ Failed to initialize AI classifier: {e}")
        return
    
    # Load failure dictionary
    print("\nLoading failure dictionary...")
    if not classifier.load_failure_dictionary(dict_file):
        print("✗ Failed to load failure dictionary")
        return
    print("✓ Failure dictionary loaded")
    
    # Test SpaCy classification
    test_cases = [
        ("Pump P-101 failed due to bearing wear and excessive vibration", "Should match: 1.1 (Pump Failure) or 1.3 (Bearing Failure)"),
        ("Motor M-205 overheated and tripped on thermal protection", "Should match: 1.2 (Motor Failure)"),
        ("Oil leak detected from pump seal on compressor C-301", "Should match: 2.1 (Fluid Leak)"),
        ("Pressure sensor PT-401 giving erratic readings", "Should match: 3.1 (Sensor Failure)"),
        ("Valve actuator stuck in closed position", "Should match: 2.2 (Valve Failure)"),
        ("Electrical panel showing fault codes and circuit breaker tripped", "Should match: 3.2 (Electrical Failure)")
    ]
    
    print("\nTesting SpaCy classification:")
    for description, expected in test_cases:
        print(f"\nDescription: {description}")
        print(f"Expected: {expected}")
        
        try:
            # Test SpaCy classification
            result = classifier.classify_with_spacy(description)
            if result:
                print(f"Result: {result.code} ({result.description})")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Method: {result.method}")
                print(f"Reasoning: {result.reasoning}")
                if result.equipment_type:
                    print(f"Equipment Type: {result.equipment_type}")
                if result.failure_indicators:
                    print(f"Failure Indicators: {result.failure_indicators}")
            else:
                print("Result: No classification")
                
        except Exception as e:
            print(f"Error: {e}")

def test_hybrid_classification_with_spacy():
    """Test hybrid classification including SpaCy"""
    print("\n" + "="*60)
    print("HYBRID CLASSIFICATION WITH SPACY TEST")
    print("="*60)
    
    # Create sample data
    dict_file = create_sample_dictionary()
    
    # Initialize AI classifier
    print("\nInitializing AI classifier...")
    try:
        classifier = AIClassifier(
            confidence_threshold=0.3,
            cache_file="test_hybrid_spacy_cache.json"
        )
        print("✓ AI classifier initialized")
    except Exception as e:
        print(f"✗ Failed to initialize AI classifier: {e}")
        return
    
    # Load failure dictionary
    print("\nLoading failure dictionary...")
    if not classifier.load_failure_dictionary(dict_file):
        print("✗ Failed to load failure dictionary")
        return
    print("✓ Failure dictionary loaded")
    
    # Test hybrid classification
    test_descriptions = [
        "Pump failure due to bearing wear and excessive vibration",
        "Motor overheated and stopped working",
        "Oil leak from pump seal",
        "Pressure sensor giving wrong readings",
        "Valve stuck in closed position",
        "Electrical fault in control panel"
    ]
    
    print("\nTesting hybrid classification (SpaCy + Embeddings + Dictionary):")
    for i, description in enumerate(test_descriptions, 1):
        print(f"\nTest {i}: {description}")
        
        try:
            # Test hybrid classification
            result = classifier.classify_hybrid(description, lambda desc: ("0.0", "No Failure Mode Identified", ""))
            
            print(f"  Code: {result.code}")
            print(f"  Description: {result.description}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Method: {result.method}")
            print(f"  Reasoning: {result.reasoning}")
            
            if result.equipment_type:
                print(f"  Equipment Type: {result.equipment_type}")
            if result.failure_indicators:
                print(f"  Failure Indicators: {result.failure_indicators}")
                
        except Exception as e:
            print(f"  Error: {e}")

def test_batch_processing_with_spacy():
    """Test batch processing with SpaCy integration"""
    print("\n" + "="*60)
    print("BATCH PROCESSING WITH SPACY TEST")
    print("="*60)
    
    # Create sample data
    dict_file = create_sample_dictionary()
    
    # Initialize AI classifier
    print("\nInitializing AI classifier...")
    try:
        classifier = AIClassifier(
            confidence_threshold=0.3,
            cache_file="test_batch_spacy_cache.json"
        )
        print("✓ AI classifier initialized")
    except Exception as e:
        print(f"✗ Failed to initialize AI classifier: {e}")
        return
    
    # Load failure dictionary
    print("\nLoading failure dictionary...")
    if not classifier.load_failure_dictionary(dict_file):
        print("✗ Failed to load failure dictionary")
        return
    print("✓ Failure dictionary loaded")
    
    # Create sample work orders
    sample_work_orders = [
        "Pump P-101 failed due to bearing wear",
        "Motor M-205 overheated and tripped",
        "Oil leak from pump seal",
        "Pressure sensor giving erratic readings",
        "Valve actuator stuck in closed position",
        "Electrical panel showing fault codes",
        "Mechanical seal failure on pump",
        "Temperature transmitter reading high",
        "Compressor vibration exceeding limits",
        "Control valve not responding to signals"
    ]
    
    print(f"\nProcessing {len(sample_work_orders)} work orders with hybrid classification...")
    
    try:
        # Batch process
        results = classifier.batch_classify(sample_work_orders, lambda desc: ("0.0", "No Failure Mode Identified", ""))
        
        print(f"✓ Processed {len(results)} work orders")
        
        # Show results summary
        method_counts = {}
        confidence_ranges = {'high': 0, 'medium': 0, 'low': 0}
        equipment_types = set()
        failure_indicators = set()
        
        for i, result in enumerate(results):
            method_counts[result.method] = method_counts.get(result.method, 0) + 1
            
            if result.confidence >= 0.8:
                confidence_ranges['high'] += 1
            elif result.confidence >= 0.5:
                confidence_ranges['medium'] += 1
            else:
                confidence_ranges['low'] += 1
            
            if result.equipment_type:
                equipment_types.add(result.equipment_type)
            if result.failure_indicators:
                failure_indicators.update(result.failure_indicators)
            
            print(f"  {i+1}. {sample_work_orders[i][:50]}...")
            print(f"     → {result.code} ({result.description}) - {result.method} - {result.confidence:.3f}")
        
        print(f"\nClassification Methods Used:")
        for method, count in method_counts.items():
            print(f"  {method}: {count}")
        
        print(f"\nConfidence Distribution:")
        print(f"  High (≥0.8): {confidence_ranges['high']}")
        print(f"  Medium (≥0.5): {confidence_ranges['medium']}")
        print(f"  Low (<0.5): {confidence_ranges['low']}")
        
        print(f"\nEquipment Types Detected: {list(equipment_types)}")
        print(f"Failure Indicators Found: {list(failure_indicators)}")
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")

def main():
    """Run all SpaCy integration tests"""
    print("SPACY-ENHANCED AI FAILURE CLASSIFICATION TEST")
    print("="*60)
    
    # Test SpaCy analysis
    test_spacy_analysis()
    
    # Test SpaCy classification
    test_spacy_classification()
    
    # Test hybrid classification
    test_hybrid_classification_with_spacy()
    
    # Test batch processing
    test_batch_processing_with_spacy()
    
    print("\n" + "="*60)
    print("SPACY INTEGRATION TEST COMPLETE")
    print("="*60)
    print("\nKey Benefits of SpaCy Integration:")
    print("1. Named Entity Recognition (NER) for equipment identification")
    print("2. Part-of-Speech tagging for better keyword matching")
    print("3. Noun chunk extraction for technical term identification")
    print("4. Equipment type detection from work order descriptions")
    print("5. Failure indicator extraction for improved classification")
    print("6. Enhanced confidence scoring based on linguistic analysis")

if __name__ == "__main__":
    main() 