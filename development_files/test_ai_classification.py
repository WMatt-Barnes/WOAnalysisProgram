"""
Test script for AI-based failure classification
Demonstrates the AI classifier functionality with sample data
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
            'pump failure, pump broken, pump not working',
            'motor failure, motor burned out, motor seized',
            'bearing failure, bearing worn, bearing noise',
            'leak, leaking, fluid leak, oil leak',
            'valve failure, valve stuck, valve malfunction',
            'sensor failure, sensor reading, sensor error',
            'electrical failure, electrical problem, wiring issue',
            'mechanical failure, mechanical problem, equipment failure'
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

def create_sample_work_orders():
    """Create sample work order descriptions for testing"""
    sample_descriptions = [
        "Pump #3 failed to start during morning shift",
        "Motor bearings making unusual noise and vibration",
        "Oil leak detected around pump housing",
        "Pressure sensor showing erratic readings",
        "Electrical panel tripped during operation",
        "Valve stuck in closed position",
        "Bearing temperature alarm activated",
        "Pump impeller damaged due to cavitation",
        "Motor overheated and shut down",
        "Fluid leak from pipe connection",
        "Sensor calibration required",
        "Mechanical seal failure on pump",
        "Electrical wiring damaged by heat",
        "Valve actuator not responding to commands",
        "Bearing failure causing shaft misalignment"
    ]
    
    sample_data = {
        'Work Order': [f'WO{i+1:03d}' for i in range(len(sample_descriptions))],
        'Description': sample_descriptions,
        'Asset': ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E'] * 3,
        'Equipment #': [f'EQ{i+1:03d}' for i in range(len(sample_descriptions))],
        'Work Type': ['Corrective', 'Preventive', 'Emergency'] * 5,
        'Reported Date': [datetime.now().strftime('%m/%d/%Y') for _ in range(len(sample_descriptions))]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_excel('sample_work_orders.xlsx', index=False)
    print("✓ Created sample work orders: sample_work_orders.xlsx")
    return 'sample_work_orders.xlsx'

def test_ai_classifier():
    """Test the AI classifier with sample data"""
    print("\n" + "="*60)
    print("AI CLASSIFICATION TEST")
    print("="*60)
    
    # Create sample data
    dict_file = create_sample_dictionary()
    wo_file = create_sample_work_orders()
    
    # Initialize AI classifier
    print("\nInitializing AI classifier...")
    try:
        classifier = AIClassifier(
            confidence_threshold=0.7,
            cache_file="test_ai_cache.json",
            model_name="gpt-3.5-turbo"
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
    
    # Load work orders
    print("\nLoading work orders...")
    try:
        wo_df = pd.read_excel(wo_file)
        print(f"✓ Loaded {len(wo_df)} work orders")
    except Exception as e:
        print(f"✗ Failed to load work orders: {e}")
        return
    
    # Test individual classification
    print("\nTesting individual classifications...")
    test_descriptions = [
        "Pump failure due to bearing wear",
        "Motor overheated and stopped working",
        "Oil leak from pump seal",
        "Pressure sensor giving wrong readings"
    ]
    
    for i, description in enumerate(test_descriptions, 1):
        print(f"\nTest {i}: {description}")
        try:
            # Test with embeddings (local)
            result = classifier.classify_with_embeddings(description)
            if result:
                print(f"  Embeddings: {result.code} ({result.description}) - Confidence: {result.confidence:.2f}")
            else:
                print("  Embeddings: No result")
            
            # Test with OpenAI (if available)
            result = classifier.classify_with_openai(description)
            if result:
                print(f"  OpenAI: {result.code} ({result.description}) - Confidence: {result.confidence:.2f}")
                print(f"  Reasoning: {result.reasoning}")
            else:
                print("  OpenAI: No result (API key may be required)")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test batch classification
    print("\nTesting batch classification...")
    try:
        descriptions = wo_df['Description'].tolist()
        results = classifier.batch_classify(descriptions, lambda desc: ("0.0", "No Failure Mode Identified", ""))
        
        print(f"✓ Processed {len(results)} descriptions")
        
        # Show results summary
        method_counts = {}
        confidence_ranges = {'high': 0, 'medium': 0, 'low': 0}
        
        for result in results:
            method_counts[result.method] = method_counts.get(result.method, 0) + 1
            
            if result.confidence >= 0.8:
                confidence_ranges['high'] += 1
            elif result.confidence >= 0.5:
                confidence_ranges['medium'] += 1
            else:
                confidence_ranges['low'] += 1
        
        print(f"\nClassification Methods Used:")
        for method, count in method_counts.items():
            print(f"  {method}: {count}")
        
        print(f"\nConfidence Distribution:")
        print(f"  High (≥0.8): {confidence_ranges['high']}")
        print(f"  Medium (≥0.5): {confidence_ranges['medium']}")
        print(f"  Low (<0.5): {confidence_ranges['low']}")
        
    except Exception as e:
        print(f"✗ Batch classification failed: {e}")
    
    # Show statistics
    print("\nAI Classification Statistics:")
    try:
        stats = classifier.get_classification_stats()
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    except Exception as e:
        print(f"  Failed to get stats: {e}")
    
    # Export training data
    print("\nExporting training data...")
    try:
        if classifier.export_training_data("training_data.json", wo_df):
            print("✓ Training data exported to training_data.json")
        else:
            print("✗ Failed to export training data")
    except Exception as e:
        print(f"✗ Export failed: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

def test_without_openai():
    """Test with only local embeddings (no OpenAI API required)"""
    print("\n" + "="*60)
    print("LOCAL EMBEDDINGS TEST (No OpenAI Required)")
    print("="*60)
    
    # Create sample data
    dict_file = create_sample_dictionary()
    
    # Initialize AI classifier without OpenAI
    print("\nInitializing AI classifier (embeddings only)...")
    try:
        classifier = AIClassifier(
            confidence_threshold=0.7,
            cache_file="test_embeddings_cache.json"
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
    
    # Test embeddings classification
    test_cases = [
        ("Pump failure due to bearing wear", "Should match: 1.1 (Pump Failure) or 1.3 (Bearing Failure)"),
        ("Motor overheated and stopped working", "Should match: 1.2 (Motor Failure)"),
        ("Oil leak from pump seal", "Should match: 2.1 (Fluid Leak)"),
        ("Pressure sensor giving wrong readings", "Should match: 3.1 (Sensor Failure)")
    ]
    
    print("\nTesting embeddings classification:")
    for description, expected in test_cases:
        print(f"\nDescription: {description}")
        print(f"Expected: {expected}")
        
        try:
            result = classifier.classify_with_embeddings(description)
            if result:
                print(f"Result: {result.code} ({result.description})")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Reasoning: {result.reasoning}")
            else:
                print("Result: No classification")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("AI Failure Classification Test Script")
    print("This script tests the AI-based failure classification functionality")
    
    # Test with local embeddings first (no API key required)
    test_without_openai()
    
    # Test full AI functionality (requires OpenAI API key)
    print("\n" + "="*60)
    print("FULL AI TEST (Requires OpenAI API Key)")
    print("="*60)
    print("To test OpenAI functionality, set your OPENAI_API_KEY environment variable")
    print("or enter it when prompted in the main application.")
    
    # Uncomment the line below to test with OpenAI (requires API key)
    # test_ai_classifier()
    
    print("\nTest completed. Check the generated files:")
    print("- sample_failure_dictionary.xlsx")
    print("- sample_work_orders.xlsx")
    print("- test_embeddings_cache.json")
    print("- training_data.json") 