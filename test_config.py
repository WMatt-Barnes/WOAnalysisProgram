#!/usr/bin/env python3
"""
Test script for the configuration functionality
"""

import json
import os
from WorkOrderAnalysisCur2 import load_app_config, save_app_config, CONFIG_FILE

def test_config_functionality():
    """Test the configuration loading and saving functionality"""
    
    print("Testing configuration functionality...")
    
    # Test 1: Load default configuration
    print("\n1. Testing default configuration loading...")
    config = load_app_config()
    print(f"Default config: {config}")
    
    # Test 2: Save custom configuration
    print("\n2. Testing configuration saving...")
    test_config = {
        'last_work_order_path': '/test/path/work_orders.xlsx',
        'last_dictionary_path': '/test/path/dictionary.xlsx',
        'last_output_directory': '/test/path/output',
        'ai_enabled': True,
        'confidence_threshold': 0.75
    }
    save_app_config(test_config)
    print(f"Saved test config: {test_config}")
    
    # Test 3: Load saved configuration
    print("\n3. Testing saved configuration loading...")
    loaded_config = load_app_config()
    print(f"Loaded config: {loaded_config}")
    
    # Test 4: Verify all keys are present
    print("\n4. Verifying all configuration keys...")
    required_keys = ['last_work_order_path', 'last_dictionary_path', 'last_output_directory', 'ai_enabled', 'confidence_threshold']
    for key in required_keys:
        if key in loaded_config:
            print(f"✓ {key}: {loaded_config[key]}")
        else:
            print(f"✗ Missing key: {key}")
    
    # Test 5: Clean up test file
    print("\n5. Cleaning up test file...")
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
        print(f"✓ Removed {CONFIG_FILE}")
    else:
        print(f"✗ {CONFIG_FILE} not found")
    
    print("\nConfiguration test completed!")

if __name__ == "__main__":
    test_config_functionality() 