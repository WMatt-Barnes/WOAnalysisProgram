"""
Test script for new equipment types in contextual patterns
Tests the enhanced equipment recognition and classification
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_failure_classifier import ContextualPatternClassifier, ExpertSystemClassifier
    print("✓ AI classifier module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AI classifier: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)

def test_equipment_detection():
    """Test equipment type detection for new equipment types"""
    print("\n" + "="*60)
    print("TESTING EQUIPMENT TYPE DETECTION")
    print("="*60)
    
    classifier = ContextualPatternClassifier()
    
    # Test cases for new equipment types
    test_cases = [
        # Control Valve tests
        {
            'description': 'Control valve actuator failure',
            'expected': 'control_valve'
        },
        {
            'description': 'Positioner signal fault on control valve',
            'expected': 'control_valve'
        },
        {
            'description': 'Automatic valve calibration error',
            'expected': 'control_valve'
        },
        
        # Turbine tests
        {
            'description': 'Turbine blade damage detected',
            'expected': 'turbine'
        },
        {
            'description': 'Steam turbine governor failure',
            'expected': 'turbine'
        },
        {
            'description': 'Turbine rotor vibration high',
            'expected': 'turbine'
        },
        
        # Vessel tests
        {
            'description': 'Storage vessel corrosion found',
            'expected': 'vessel'
        },
        {
            'description': 'Tank pressure fault alarm',
            'expected': 'vessel'
        },
        {
            'description': 'Vessel level fault detected',
            'expected': 'vessel'
        },
        
        # Agitator/Mixer tests
        {
            'description': 'Agitator impeller damage',
            'expected': 'agitator'
        },
        {
            'description': 'Mixer bearing failure',
            'expected': 'mixer'
        },
        {
            'description': 'Agitation shaft seal leak',
            'expected': 'agitator'
        },
        
        # Transmitter tests
        {
            'description': 'Pressure transmitter signal fault',
            'expected': 'transmitter'
        },
        {
            'description': 'Temperature transmitter calibration error',
            'expected': 'transmitter'
        },
        {
            'description': 'Flow transmitter communication fault',
            'expected': 'transmitter'
        },
        
        # Reactor tests
        {
            'description': 'Reactor temperature fault',
            'expected': 'reactor'
        },
        {
            'description': 'Reaction pressure fault alarm',
            'expected': 'reactor'
        },
        {
            'description': 'Catalyst agitation failure',
            'expected': 'reactor'
        },
        
        # Boiler tests
        {
            'description': 'Boiler water level fault',
            'expected': 'boiler'
        },
        {
            'description': 'Steam boiler burner failure',
            'expected': 'boiler'
        },
        {
            'description': 'Boiler tube leak detected',
            'expected': 'boiler'
        },
        
        # Generator tests
        {
            'description': 'Electrical generator bearing failure',
            'expected': 'generator'
        },
        {
            'description': 'Power generator excitation fault',
            'expected': 'generator'
        },
        {
            'description': 'Generator cooling failure',
            'expected': 'generator'
        },
        
        # Exchanger tests
        {
            'description': 'Heat exchanger tube leak',
            'expected': 'exchanger'
        },
        {
            'description': 'Shell and tube fouling detected',
            'expected': 'exchanger'
        },
        {
            'description': 'Exchanger pressure drop high',
            'expected': 'exchanger'
        }
    ]
    
    print(f"{'Description':<50} {'Expected':<15} {'Detected':<15} {'Status'}")
    print("-" * 90)
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        description = test_case['description']
        expected = test_case['expected']
        
        # Detect equipment type
        detected = classifier.detect_equipment_context(description)
        
        # Determine status
        status = "✓ PASS" if detected == expected else "✗ FAIL"
        if detected == expected:
            passed += 1
        
        print(f"{description[:49]:<50} {expected:<15} {detected:<15} {status}")
    
    print("-" * 90)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed, total

def test_contextual_classification():
    """Test contextual classification for new equipment types"""
    print("\n" + "="*60)
    print("TESTING CONTEXTUAL CLASSIFICATION")
    print("="*60)
    
    classifier = ContextualPatternClassifier()
    
    # Test cases for contextual classification
    test_cases = [
        # Control Valve specific failures
        {
            'description': 'Control valve positioner failure',
            'expected_failure': 'actuator_failure'
        },
        {
            'description': 'Control valve signal fault',
            'expected_failure': 'signal_fault'
        },
        {
            'description': 'Control valve calibration error',
            'expected_failure': 'calibration_error'
        },
        
        # Turbine specific failures
        {
            'description': 'Turbine blade damage',
            'expected_failure': 'blade_damage'
        },
        {
            'description': 'Steam turbine governor failure',
            'expected_failure': 'governor_failure'
        },
        {
            'description': 'Turbine steam leak',
            'expected_failure': 'steam_leak'
        },
        
        # Vessel specific failures
        {
            'description': 'Vessel corrosion detected',
            'expected_failure': 'corrosion'
        },
        {
            'description': 'Tank pressure fault',
            'expected_failure': 'pressure_fault'
        },
        {
            'description': 'Vessel level fault',
            'expected_failure': 'level_fault'
        },
        
        # Transmitter specific failures
        {
            'description': 'Transmitter signal fault',
            'expected_failure': 'signal_fault'
        },
        {
            'description': 'Transmitter calibration error',
            'expected_failure': 'calibration_error'
        },
        {
            'description': 'Transmitter sensor failure',
            'expected_failure': 'sensor_failure'
        },
        
        # Reactor specific failures
        {
            'description': 'Reactor temperature fault',
            'expected_failure': 'temperature_fault'
        },
        {
            'description': 'Reactor pressure fault',
            'expected_failure': 'pressure_fault'
        },
        {
            'description': 'Reactor agitation failure',
            'expected_failure': 'agitation_failure'
        },
        
        # Boiler specific failures
        {
            'description': 'Boiler water level fault',
            'expected_failure': 'water_level_fault'
        },
        {
            'description': 'Boiler burner failure',
            'expected_failure': 'burner_failure'
        },
        {
            'description': 'Boiler tube leak',
            'expected_failure': 'tube_leak'
        },
        
        # Generator specific failures
        {
            'description': 'Generator electrical fault',
            'expected_failure': 'electrical_fault'
        },
        {
            'description': 'Generator excitation fault',
            'expected_failure': 'excitation_fault'
        },
        {
            'description': 'Generator cooling failure',
            'expected_failure': 'cooling_failure'
        },
        
        # Exchanger specific failures
        {
            'description': 'Exchanger tube leak',
            'expected_failure': 'tube_leak'
        },
        {
            'description': 'Heat exchanger fouling',
            'expected_failure': 'fouling'
        },
        {
            'description': 'Exchanger pressure drop',
            'expected_failure': 'pressure_drop'
        }
    ]
    
    print(f"{'Description':<50} {'Expected':<20} {'Actual':<20} {'Confidence':<10} {'Status'}")
    print("-" * 110)
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        description = test_case['description']
        expected_failure = test_case['expected_failure']
        
        # Classify with context
        code, desc, confidence = classifier.classify_with_context(description)
        
        # Extract actual failure type from code
        actual_failure = code.split('_', 1)[1] if '_' in code else 'unknown'
        
        # Determine status
        status = "✓ PASS" if actual_failure == expected_failure else "✗ FAIL"
        if actual_failure == expected_failure:
            passed += 1
        
        print(f"{description[:49]:<50} {expected_failure:<20} {actual_failure:<20} {confidence:<10.2f} {status}")
    
    print("-" * 110)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed, total

def test_differentiation():
    """Test differentiation between similar equipment types"""
    print("\n" + "="*60)
    print("TESTING EQUIPMENT DIFFERENTIATION")
    print("="*60)
    
    classifier = ContextualPatternClassifier()
    
    # Test cases for differentiation
    differentiation_tests = [
        {
            'category': 'Control Valve vs Manual Valve',
            'tests': [
                ('Control valve actuator failure', 'control_valve'),
                ('Manual valve stuck', 'valve'),
                ('Automatic valve positioner fault', 'control_valve'),
                ('Manual valve seat damage', 'valve'),
                ('Control valve signal error', 'control_valve'),
                ('Manual valve leak', 'valve')
            ]
        },
        {
            'category': 'Agitator vs Mixer',
            'tests': [
                ('Agitator impeller damage', 'agitator'),
                ('Mixer bearing failure', 'mixer'),
                ('Agitation shaft seal leak', 'agitator'),
                ('Mixing impeller bent', 'mixer'),
                ('Agitator vibration high', 'agitator'),
                ('Mixer gear failure', 'mixer')
            ]
        },
        {
            'category': 'Transmitter vs Sensor',
            'tests': [
                ('Pressure transmitter signal fault', 'transmitter'),
                ('Temperature sensor failure', 'unknown'),  # Should not match transmitter
                ('Flow transmitter calibration error', 'transmitter'),
                ('Level sensor reading wrong', 'unknown'),  # Should not match transmitter
                ('Transmitter communication fault', 'transmitter'),
                ('Pressure sensor broken', 'unknown')  # Should not match transmitter
            ]
        }
    ]
    
    for category in differentiation_tests:
        print(f"\n{category['category']}:")
        print(f"{'Description':<50} {'Expected':<15} {'Detected':<15} {'Status'}")
        print("-" * 90)
        
        for description, expected in category['tests']:
            detected = classifier.detect_equipment_context(description)
            status = "✓" if detected == expected else "✗"
            print(f"{description[:49]:<50} {expected:<15} {detected:<15} {status}")

def test_expert_system_integration():
    """Test integration with expert system rules"""
    print("\n" + "="*60)
    print("TESTING EXPERT SYSTEM INTEGRATION")
    print("="*60)
    
    expert_system = ExpertSystemClassifier()
    contextual_classifier = ContextualPatternClassifier()
    
    # Test cases that should work with both systems
    test_cases = [
        {
            'description': 'Control valve signal fault',
            'expert_expected': 'Signal Fault/Indication Error',  # Signal fault
            'contextual_expected': 'control_valve_signal_fault'
        },
        {
            'description': 'Turbine bearing failure',
            'expert_expected': 'Bearing Failure',  # Bearing failure
            'contextual_expected': 'turbine_bearing_failure'
        },
        {
            'description': 'Vessel corrosion detected',
            'expert_expected': 'Corrosion',  # Corrosion
            'contextual_expected': 'vessel_corrosion'
        },
        {
            'description': 'Transmitter calibration error',
            'expert_expected': 'No Failure Mode Identified',  # No expert rule for calibration
            'contextual_expected': 'transmitter_calibration_error'
        }
    ]
    
    print(f"{'Description':<40} {'Expert':<10} {'Contextual':<20} {'Integration'}")
    print("-" * 80)
    
    for test_case in test_cases:
        description = test_case['description']
        expert_expected = test_case['expert_expected']
        contextual_expected = test_case['contextual_expected']
        
        # Expert system classification
        expert_code, expert_desc, expert_conf = expert_system.classify(description)
        
        # Contextual classification
        contextual_code, contextual_desc, contextual_conf = contextual_classifier.classify_with_context(description)
        
        # Determine integration status
        if expert_code != 'No Failure Mode Identified' and contextual_code != 'No Failure Mode Identified':
            integration_status = "Both systems"
        elif expert_code != 'No Failure Mode Identified':
            integration_status = "Expert only"
        elif contextual_code != 'No Failure Mode Identified':
            integration_status = "Contextual only"
        else:
            integration_status = "No match"
        
        print(f"{description[:39]:<40} {expert_code:<10} {contextual_code:<20} {integration_status}")

def main():
    """Main test function"""
    print("Enhanced Equipment Types Test")
    print("Testing: Turbine, Vessel, Agitator/Mixer, Transmitter, Control Valve, Reactor, Boiler, Generator, Exchanger")
    
    # Test equipment detection
    detection_passed, detection_total = test_equipment_detection()
    
    # Test contextual classification
    classification_passed, classification_total = test_contextual_classification()
    
    # Test differentiation
    test_differentiation()
    
    # Test expert system integration
    test_expert_system_integration()
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Equipment Detection Tests: {detection_passed}/{detection_total} passed")
    print(f"Contextual Classification Tests: {classification_passed}/{classification_total} passed")
    print("Equipment Differentiation: Tested")
    print("Expert System Integration: Tested")
    
    print("\nNew Equipment Types Added:")
    print("- Turbine (steam, blade, rotor, governor)")
    print("- Vessel (tank, pressure, level, temperature)")
    print("- Agitator/Mixer (impeller, mixing, agitation)")
    print("- Transmitter (signal, calibration, sensor)")
    print("- Control Valve (positioner, signal, calibration)")
    print("- Reactor (reaction, temperature, pressure, catalyst)")
    print("- Boiler (steam, water, burner, tube)")
    print("- Generator (electrical, power, voltage, excitation)")
    print("- Exchanger (tube, shell, heat, fouling)")
    
    print("\nKey Features:")
    print("- Control valve differentiated from manual valve")
    print("- Equipment-specific failure patterns")
    print("- Contextual word recognition")
    print("- Integration with expert system rules")

if __name__ == "__main__":
    main() 