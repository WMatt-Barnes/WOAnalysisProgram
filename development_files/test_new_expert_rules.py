"""
Test script for new expert rules: Packing Leak, Lubrication Failure, and Signal Fault
Demonstrates the enhanced expert system classification
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_failure_classifier import AIClassifier, ExpertSystemClassifier
    print("✓ AI classifier module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AI classifier: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

def create_test_dictionary():
    """Create a test failure mode dictionary"""
    test_data = {
        'Keyword': [
            'bearing failure, bearing noise, bearing vibration',
            'seal leak, mechanical seal, packing leak',
            'motor overheating, motor temperature high',
            'pump cavitation, cavitation noise',
            'electrical fault, short circuit',
            'valve stuck, valve seized',
            'belt failure, belt broken',
            'corrosion, rust formation',
            'packing leak, gland leak, stuffing box',
            'lubrication failure, oil level low, dry running',
            'signal fault, false alarm, wrong reading'
        ],
        'Code': ['Bearing Failure', 'Seal Leak', 'Motor Overheating', 'Pump Cavitation', 'Electrical Fault', 'Valve Stuck/Seized', 'Belt Failure', 'Corrosion', 'Packing Leak', 'Lubrication Failure', 'Signal Fault/Indication Error'],
        'Description': [
            'Bearing Failure',
            'Seal Leak',
            'Motor Overheating',
            'Pump Cavitation',
            'Electrical Fault',
            'Valve Stuck/Seized',
            'Belt Failure',
            'Corrosion',
            'Packing Leak',
            'Lubrication Failure',
            'Signal Fault/Indication Error'
        ]
    }
    
    df = pd.DataFrame(test_data)
    df.to_excel('test_failure_dictionary.xlsx', index=False)
    print("✓ Created test failure dictionary: test_failure_dictionary.xlsx")
    return 'test_failure_dictionary.xlsx'

def test_expert_system():
    """Test the expert system with new failure types"""
    print("\n" + "="*60)
    print("TESTING EXPERT SYSTEM WITH NEW FAILURE TYPES")
    print("="*60)
    
    # Initialize expert system
    expert_system = ExpertSystemClassifier()
    
    # Test cases for new failure types
    test_cases = [
        # Packing Leak tests
        {
            'description': 'Packing leak detected on pump gland',
            'expected': 'Packing Leak'
        },
        {
            'description': 'Stuffing box leaking fluid',
            'expected': 'Packing Leak'
        },
        {
            'description': 'Gland packing needs replacement due to leak',
            'expected': 'Packing Leak'
        },
        
        # Lubrication Failure tests
        {
            'description': 'Lubrication failure causing bearing damage',
            'expected': 'Lubrication Failure'
        },
        {
            'description': 'Oil level low in lubrication system',
            'expected': 'Lubrication Failure'
        },
        {
            'description': 'Dry running detected due to no lubrication',
            'expected': 'Lubrication Failure'
        },
        {
            'description': 'Grease not flowing to bearing points',
            'expected': 'Lubrication Failure'
        },
        
        # Signal Fault tests (differentiated from electrical fault)
        {
            'description': 'Signal fault in pressure indication',
            'expected': 'Signal Fault/Indication Error'
        },
        {
            'description': 'False alarm on temperature sensor',
            'expected': 'Signal Fault/Indication Error'
        },
        {
            'description': 'Wrong reading on flow meter display',
            'expected': 'Signal Fault/Indication Error'
        },
        {
            'description': 'Sensor reading incorrect values',
            'expected': 'Signal Fault/Indication Error'
        },
        {
            'description': 'Display error on control panel',
            'expected': 'Signal Fault/Indication Error'
        },
        
        # Electrical fault (should not be confused with signal fault)
        {
            'description': 'Electrical fault in motor wiring',
            'expected': 'Electrical Fault'
        },
        {
            'description': 'Short circuit in electrical panel',
            'expected': 'Electrical Fault'
        },
        
        # Mixed cases to test differentiation
        {
            'description': 'Electrical fault causing signal error',
            'expected': 'Electrical Fault'  # Should prioritize electrical fault
        },
        {
            'description': 'Packing leak with lubrication failure',
            'expected': 'Packing Leak'  # Should prioritize packing leak (higher weight)
        }
    ]
    
    print(f"{'Test Case':<40} {'Expected':<10} {'Actual':<10} {'Confidence':<10} {'Status'}")
    print("-" * 80)
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        description = test_case['description']
        expected = test_case['expected']
        
        # Classify using expert system
        code, desc, confidence = expert_system.classify(description)
        
        # Determine status
        status = "✓ PASS" if code == expected else "✗ FAIL"
        if code == expected:
            passed += 1
        
        print(f"{description[:39]:<40} {expected:<10} {code:<10} {confidence:<10.2f} {status}")
    
    print("-" * 80)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed, total

def test_ai_classifier_integration():
    """Test the full AI classifier with new rules"""
    print("\n" + "="*60)
    print("TESTING FULL AI CLASSIFIER INTEGRATION")
    print("="*60)
    
    # Create test dictionary
    dict_path = create_test_dictionary()
    
    # Initialize AI classifier
    classifier = AIClassifier(confidence_threshold=0.3)
    
    if not classifier.load_failure_dictionary(dict_path):
        print("✗ Failed to load failure dictionary")
        return
    
    # Test cases for AI classification
    test_descriptions = [
        "Packing leak detected on pump gland seal",
        "Lubrication failure causing bearing to run dry",
        "Signal fault in pressure indication system",
        "Electrical fault in motor control circuit",
        "False alarm on temperature sensor reading",
        "Oil level low in lubrication reservoir",
        "Wrong reading on flow meter display",
        "Gland packing needs replacement"
    ]
    
    print(f"{'Description':<50} {'Method':<20} {'Code':<8} {'Confidence':<10}")
    print("-" * 90)
    
    for description in test_descriptions:
        # Use hybrid classification
        result = classifier.classify_hybrid(description, lambda x: ('No Failure Mode Identified', 'No Match', 0.0))
        
        print(f"{description[:49]:<50} {result.method:<20} {result.code:<8} {result.confidence:<10.2f}")
    
    print("-" * 90)

def test_differentiation():
    """Test differentiation between similar failure types"""
    print("\n" + "="*60)
    print("TESTING FAILURE TYPE DIFFERENTIATION")
    print("="*60)
    
    expert_system = ExpertSystemClassifier()
    
    # Test cases to ensure proper differentiation
    differentiation_tests = [
        {
            'category': 'Packing vs Seal Leak',
            'tests': [
                ('Packing leak on gland', 'Packing Leak'),
                ('Mechanical seal leak', 'Seal Leak'),
                ('Seal packing failure', 'Packing Leak'),  # Should match packing
                ('Packing seal damaged', 'Packing Leak')   # Should match packing
            ]
        },
        {
            'category': 'Signal vs Electrical Fault',
            'tests': [
                ('Signal fault in sensor', 'Signal Fault/Indication Error'),
                ('Electrical fault in wiring', 'Electrical Fault'),
                ('Signal error in display', 'Signal Fault/Indication Error'),
                ('Electrical short circuit', 'Electrical Fault'),
                ('False alarm signal', 'Signal Fault/Indication Error'),
                ('Electrical ground fault', 'Electrical Fault')
            ]
        },
        {
            'category': 'Lubrication vs Other Failures',
            'tests': [
                ('Lubrication failure', 'Lubrication Failure'),
                ('Oil level low', 'Lubrication Failure'),
                ('Dry running condition', 'Lubrication Failure'),
                ('Bearing failure due to lubrication', 'Bearing Failure'),  # Should match bearing
                ('Lubricant contamination', 'Lubrication Failure')
            ]
        }
    ]
    
    for category in differentiation_tests:
        print(f"\n{category['category']}:")
        print(f"{'Description':<40} {'Expected':<10} {'Actual':<10} {'Status'}")
        print("-" * 70)
        
        for description, expected in category['tests']:
            code, desc, confidence = expert_system.classify(description)
            status = "✓" if code == expected else "✗"
            print(f"{description[:39]:<40} {expected:<10} {code:<10} {status}")

def main():
    """Main test function"""
    print("Enhanced Expert System Test - New Failure Types")
    print("Testing: Packing Leak, Lubrication Failure, Signal Fault")
    
    # Test expert system rules
    passed, total = test_expert_system()
    
    # Test AI classifier integration
    test_ai_classifier_integration()
    
    # Test differentiation
    test_differentiation()
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Expert System Tests: {passed}/{total} passed")
    print("AI Classifier Integration: Tested")
    print("Failure Type Differentiation: Tested")
    print("\nNew Expert Rules Added:")
    print("- Packing Leak (Code: 9.1)")
    print("- Lubrication Failure (Code: 10.1)")
    print("- Signal Fault/Indication Error (Code: 11.1)")
    print("\nKey Features:")
    print("- Packing leak differentiated from seal leak")
    print("- Signal fault differentiated from electrical fault")
    print("- Lubrication failure covers oil, grease, and dry running")
    print("- Enhanced contextual patterns for equipment-specific failures")

if __name__ == "__main__":
    main() 