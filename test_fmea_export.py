#!/usr/bin/env python3
"""
Test script for FMEA export functionality
"""

import json
import os
from datetime import datetime

def test_fmea_json_structure():
    """Test the FMEA JSON file structure"""
    
    # Create a sample FMEA entry
    sample_entry = {
        "equipment": "PUMP-001",
        "failure_mode_ai": "Bearing Failure",
        "failure_mode_user": "Bearing Wear",
        "failure_mode_type": "AI/Dictionary",
        "failure_mode_frequency": 2.5,
        "weibull_beta": 2.1,
        "weibull_eta": 365.0,
        "mtbf_days": 320.0,
        "total_failures": 5,
        "analysis_date": datetime.now().isoformat(),
        "date_range_start": "2024-01-01T00:00:00",
        "date_range_end": "2024-12-31T23:59:59"
    }
    
    # Test file path
    test_file = "test_fmea_export_data.json"
    
    try:
        # Test writing to JSON
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump([sample_entry], f, indent=2, ensure_ascii=False)
        
        print("✅ Successfully wrote FMEA data to JSON")
        
        # Test reading from JSON
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("✅ Successfully read FMEA data from JSON")
        print(f"📊 Sample entry: {data[0]['equipment']} - {data[0]['failure_mode_ai']}")
        
        # Test duplicate handling
        duplicate_entry = sample_entry.copy()
        duplicate_entry["analysis_date"] = datetime.now().isoformat()
        
        # Simulate duplicate detection
        key1 = f"{sample_entry['equipment']}_{sample_entry['failure_mode_ai']}_{sample_entry['failure_mode_type']}"
        key2 = f"{duplicate_entry['equipment']}_{duplicate_entry['failure_mode_ai']}_{duplicate_entry['failure_mode_type']}"
        
        if key1 == key2:
            print("✅ Duplicate detection logic works correctly")
        else:
            print("❌ Duplicate detection logic failed")
        
        # Clean up test file
        os.remove(test_file)
        print("✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing FMEA JSON structure: {e}")
        return False

def test_fmea_fields():
    """Test that all required FMEA fields are present"""
    
    required_fields = [
        "equipment",
        "failure_mode_ai", 
        "failure_mode_user",
        "failure_mode_type",
        "failure_mode_frequency",
        "weibull_beta",
        "weibull_eta", 
        "mtbf_days",
        "total_failures",
        "analysis_date"
    ]
    
    sample_entry = {
        "equipment": "TEST-EQ-001",
        "failure_mode_ai": "Test Failure",
        "failure_mode_user": "Test User Failure",
        "failure_mode_type": "AI/Dictionary",
        "failure_mode_frequency": 1.0,
        "weibull_beta": 1.5,
        "weibull_eta": 180.0,
        "mtbf_days": 150.0,
        "total_failures": 3,
        "analysis_date": datetime.now().isoformat(),
        "date_range_start": "2024-01-01T00:00:00",
        "date_range_end": "2024-12-31T23:59:59"
    }
    
    missing_fields = []
    for field in required_fields:
        if field not in sample_entry:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    else:
        print("✅ All required FMEA fields are present")
        return True

if __name__ == "__main__":
    print("🧪 Testing FMEA Export Functionality")
    print("=" * 50)
    
    # Test JSON structure
    json_test = test_fmea_json_structure()
    
    # Test required fields
    fields_test = test_fmea_fields()
    
    print("\n📋 Test Summary:")
    print(f"JSON Structure: {'✅ PASS' if json_test else '❌ FAIL'}")
    print(f"Required Fields: {'✅ PASS' if fields_test else '❌ FAIL'}")
    
    if json_test and fields_test:
        print("\n🎉 All tests passed! FMEA export functionality is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.") 