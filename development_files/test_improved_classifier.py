"""
Test script for improved AI classifier with enhanced rules and patterns
"""

import sys
import os
from ai_failure_classifier import AIClassifier, AIClassificationResult

def test_improved_classifier():
    """Test the improved classifier with new expert rules and patterns"""
    print("Testing Improved AI Classifier")
    print("="*50)
    
    # Initialize classifier
    classifier = AIClassifier(
        confidence_threshold=0.3,
        cache_file="test_cache.json"
    )
    
    # Test cases based on training data analysis
    test_cases = [
        # Faulty Signal/Indication/Alarm (most common - 169 occurrences)
        {
            'description': 'Aux Boiler TDL showing malfunction',
            'expected_code': 'Faulty Signal/Indication/Alarm',
            'expected_desc': 'Faulty Signal/Indication/Alarm'
        },
        {
            'description': 'Transmitter alarm fault',
            'expected_code': 'Faulty Signal/Indication/Alarm',
            'expected_desc': 'Faulty Signal/Indication/Alarm'
        },
        {
            'description': 'Signal error on pressure transmitter',
            'expected_code': 'Faulty Signal/Indication/Alarm',
            'expected_desc': 'Faulty Signal/Indication/Alarm'
        },
        
        # Leakage (168 occurrences)
        {
            'description': '010PSV009B is leaking by and tubing fittings have a small drip',
            'expected_code': 'Leakage',
            'expected_desc': 'Leakage'
        },
        {
            'description': 'Steam leak on tubing to 010FT3108',
            'expected_code': 'Leakage',
            'expected_desc': 'Leakage'
        },
        {
            'description': 'Valve leaking from body flange',
            'expected_code': 'Leakage',
            'expected_desc': 'Leakage'
        },
        
        # Looseness (83 occurrences)
        {
            'description': 'I/A Dryer condensate trap not functioning properly',
            'expected_code': 'Looseness',
            'expected_desc': 'Looseness'
        },
        {
            'description': 'Loose connection on motor',
            'expected_code': 'Looseness',
            'expected_desc': 'Looseness'
        },
        
        # Contamination (72 occurrences)
        {
            'description': 'Troubleshoot oil analyzer on NT-1700B',
            'expected_code': 'Contamination',
            'expected_desc': 'Contamination'
        },
        {
            'description': 'Troubleshoot oil pressure',
            'expected_code': 'Contamination',
            'expected_desc': 'Contamination'
        },
        
        # Control failure (63 occurrences)
        {
            'description': 'Valve in tracking and not meeting limit switch',
            'expected_code': 'Control failure',
            'expected_desc': 'Control failure'
        },
        {
            'description': 'Control valve not responding to signal',
            'expected_code': 'Control failure',
            'expected_desc': 'Control failure'
        },
        
        # Calibration (60 occurrences)
        {
            'description': 'HP-MP Letdown 015PV1501C will need to be calibrated/troubleshot during outage',
            'expected_code': 'Calibration',
            'expected_desc': 'Calibration'
        },
        {
            'description': 'Transmitter needs to be calibrated',
            'expected_code': 'Calibration',
            'expected_desc': 'Calibration'
        },
        
        # Power Supply (52 occurrences)
        {
            'description': 'Reconnect power to wall louvers in Fire water house B',
            'expected_code': 'Power Supply',
            'expected_desc': 'Power Supply'
        },
        {
            'description': 'Power supply tripped',
            'expected_code': 'Power Supply',
            'expected_desc': 'Power Supply'
        },
        
        # Open circuit (52 occurrences)
        {
            'description': 'Replace plastic chains at ECR transformers',
            'expected_code': 'Open circuit',
            'expected_desc': 'Open circuit'
        },
        {
            'description': 'Broken electrical connection',
            'expected_code': 'Open circuit',
            'expected_desc': 'Open circuit'
        },
        
        # Seal (48 occurrences)
        {
            'description': 'Seal leak on pump',
            'expected_code': 'Seal',
            'expected_desc': 'Seal'
        },
        {
            'description': 'Mechanical seal alarming',
            'expected_code': 'Seal',
            'expected_desc': 'Seal'
        },
        
        # Overheating (47 occurrences)
        {
            'description': 'Motor overheating',
            'expected_code': 'Overheating',
            'expected_desc': 'Overheating'
        },
        {
            'description': 'Equipment temperature high',
            'expected_code': 'Overheating',
            'expected_desc': 'Overheating'
        },
        
        # Equipment-specific tests
        {
            'description': 'Transmitter freezing up',
            'expected_code': 'transmitter_freezing',
            'expected_desc': 'Transmitter Freezing'
        },
        {
            'description': 'Valve stuck and will not open',
            'expected_code': 'valve_stuck',
            'expected_desc': 'Valve Stuck'
        },
        {
            'description': 'Pump cavitation noise',
            'expected_code': 'pump_cavitation',
            'expected_desc': 'Pump Cavitation'
        },
        {
            'description': 'Fan bearing noise',
            'expected_code': 'fan_bearing_failure',
            'expected_desc': 'Fan Bearing Failure'
        },
        {
            'description': 'Change motor on P-04003-A',
            'expected_code': 'motor_change_motor',
            'expected_desc': 'Motor Change Motor'
        }
    ]
    
    # Run tests
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print("-" * 40)
        
        try:
            # Classify using the improved classifier
            result = classifier.classify_hybrid(
                test_case['description'], 
                lambda desc: ('No Failure Mode Identified', 'No Failure Mode Identified', 0.0)
            )
            
            print(f"Result: {result.code} - {result.description}")
            print(f"Method: {result.method}")
            print(f"Confidence: {result.confidence:.2f}")
            
            # Check if classification matches expected
            if (result.code == test_case['expected_code'] or 
                result.description == test_case['expected_desc'] or
                test_case['expected_desc'] in result.description):
                print("✓ PASS")
                passed += 1
            else:
                print(f"✗ FAIL - Expected: {test_case['expected_code']} - {test_case['expected_desc']}")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    # Summary
    print(f"\n" + "="*50)
    print(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The improved classifier is working correctly.")
    else:
        print("⚠️  Some tests failed. Review the results above.")
    
    # Clean up
    if os.path.exists("test_cache.json"):
        os.remove("test_cache.json")
    
    return passed == total

def test_equipment_context_detection():
    """Test equipment context detection improvements"""
    print("\nTesting Equipment Context Detection")
    print("="*50)
    
    classifier = AIClassifier(confidence_threshold=0.3)
    
    # Test equipment detection
    equipment_tests = [
        ('valve leaking from body flange', 'valve'),
        ('pump cavitation noise', 'pump'),
        ('transmitter freezing up', 'transmitter'),
        ('fan bearing noise', 'fan'),
        ('motor overheating', 'motor'),
        ('compressor vibration', 'compressor'),
        ('turbine blade damage', 'turbine'),
        ('vessel corrosion', 'vessel'),
        ('agitator seal leak', 'agitator'),
        ('mixer impeller damage', 'mixer'),
        ('reactor temperature fault', 'reactor'),
        ('boiler water level fault', 'boiler'),
        ('generator electrical fault', 'generator'),
        ('exchanger tube leak', 'exchanger')
    ]
    
    passed = 0
    total = len(equipment_tests)
    
    for description, expected_equipment in equipment_tests:
        try:
            result = classifier.classify_hybrid(
                description,
                lambda desc: ('No Failure Mode Identified', 'No Failure Mode Identified', 0.0)
            )
            
            print(f"Description: {description}")
            print(f"Detected Equipment: {result.equipment_type}")
            print(f"Expected: {expected_equipment}")
            
            if result.equipment_type == expected_equipment:
                print("✓ PASS")
                passed += 1
            else:
                print("✗ FAIL")
            print()
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    print(f"Equipment Detection: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed == total

if __name__ == "__main__":
    print("Testing Improved AI Classifier with Enhanced Rules and Patterns")
    print("Based on Training Data Analysis")
    print("="*60)
    
    # Run tests
    classifier_tests = test_improved_classifier()
    equipment_tests = test_equipment_context_detection()
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"Classifier Tests: {'PASSED' if classifier_tests else 'FAILED'}")
    print(f"Equipment Detection: {'PASSED' if equipment_tests else 'FAILED'}")
    
    if classifier_tests and equipment_tests:
        print("\n🎉 All improvements working correctly!")
        print("The enhanced AI classifier now includes:")
        print("- 10 new expert rules based on training data analysis")
        print("- Enhanced equipment context detection")
        print("- Improved pattern matching for common failure modes")
        print("- Better handling of equipment-specific failures")
    else:
        print("\n⚠️  Some improvements need attention.") 