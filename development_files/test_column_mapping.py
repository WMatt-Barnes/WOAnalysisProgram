#!/usr/bin/env python3
"""
Test script for column mapping functionality
Demonstrates how the program handles different CMMS export formats
"""

import pandas as pd
import os
from datetime import datetime

def create_test_files():
    """Create test files with different column formats"""
    
    # Standard format (what the program expects)
    standard_data = {
        'Work Order': ['WO-001', 'WO-002', 'WO-003'],
        'Description': ['Pump failure', 'Motor replacement', 'Valve repair'],
        'Asset': ['Pump-01', 'Motor-02', 'Valve-03'],
        'Equipment #': ['EQ001', 'EQ002', 'EQ003'],
        'Work Type': ['Repair', 'Replace', 'Maintenance'],
        'Reported Date': ['01/15/2024', '01/20/2024', '01/25/2024']
    }
    
    # CMMS format 1 (different column names)
    cmms_format1_data = {
        'wo_number': ['WO-001', 'WO-002', 'WO-003'],
        'work_description': ['Pump failure', 'Motor replacement', 'Valve repair'],
        'asset_name': ['Pump-01', 'Motor-02', 'Valve-03'],
        'equipment_number': ['EQ001', 'EQ002', 'EQ003'],
        'work_type': ['Repair', 'Replace', 'Maintenance'],
        'date': ['01/15/2024', '01/20/2024', '01/25/2024']
    }
    
    # CMMS format 2 (abbreviated names)
    cmms_format2_data = {
        'WO': ['WO-001', 'WO-002', 'WO-003'],
        'Desc': ['Pump failure', 'Motor replacement', 'Valve repair'],
        'Asset': ['Pump-01', 'Motor-02', 'Valve-03'],
        'Eq#': ['EQ001', 'EQ002', 'EQ003'],
        'Type': ['Repair', 'Replace', 'Maintenance'],
        'Date': ['01/15/2024', '01/20/2024', '01/25/2024']
    }
    
    # Create test files
    os.makedirs('test_files', exist_ok=True)
    
    # Standard format
    standard_df = pd.DataFrame(standard_data)
    standard_df.to_excel('test_files/standard_format.xlsx', index=False)
    
    # CMMS format 1
    cmms1_df = pd.DataFrame(cmms_format1_data)
    cmms1_df.to_excel('test_files/cmms_format1.xlsx', index=False)
    
    # CMMS format 2
    cmms2_df = pd.DataFrame(cmms_format2_data)
    cmms2_df.to_excel('test_files/cmms_format2.xlsx', index=False)
    
    print("Test files created:")
    print("- test_files/standard_format.xlsx (standard format)")
    print("- test_files/cmms_format1.xlsx (CMMS format 1)")
    print("- test_files/cmms_format2.xlsx (CMMS format 2)")
    
    return ['test_files/standard_format.xlsx', 'test_files/cmms_format1.xlsx', 'test_files/cmms_format2.xlsx']

def demonstrate_column_mapping():
    """Demonstrate how column mapping works"""
    
    print("\n" + "="*60)
    print("COLUMN MAPPING DEMONSTRATION")
    print("="*60)
    
    # Create test files
    test_files = create_test_files()
    
    print("\n1. STANDARD FORMAT (no mapping needed):")
    print("   Columns: Work Order, Description, Asset, Equipment #, Work Type, Reported Date")
    print("   Status: ✅ Ready to process")
    
    print("\n2. CMMS FORMAT 1 (requires mapping):")
    print("   Columns: wo_number, work_description, asset_name, equipment_number, work_type, date")
    print("   Required mapping:")
    print("   - 'Work Order' → 'wo_number'")
    print("   - 'Description' → 'work_description'")
    print("   - 'Asset' → 'asset_name'")
    print("   - 'Equipment #' → 'equipment_number'")
    print("   - 'Work Type' → 'work_type'")
    print("   - 'Reported Date' → 'date'")
    
    print("\n3. CMMS FORMAT 2 (requires mapping):")
    print("   Columns: WO, Desc, Asset, Eq#, Type, Date")
    print("   Required mapping:")
    print("   - 'Work Order' → 'WO'")
    print("   - 'Description' → 'Desc'")
    print("   - 'Equipment #' → 'Eq#'")
    print("   - 'Work Type' → 'Type'")
    print("   - 'Reported Date' → 'Date'")
    print("   - 'Asset' → 'Asset' (no change needed)")
    
    print("\n" + "="*60)
    print("HOW TO USE COLUMN MAPPING IN THE PROGRAM:")
    print("="*60)
    print("1. Go to Tools → Column Mapping...")
    print("2. Select your CMMS export file")
    print("3. Map each required column to your file's column")
    print("4. Use 'Auto-Detect' to automatically find similar names")
    print("5. Save mappings for future use")
    print("6. Process files normally - the program will use your mappings")
    
    print("\n" + "="*60)
    print("BENEFITS:")
    print("="*60)
    print("✅ No need to manually rename columns in Excel")
    print("✅ Works with any CMMS export format")
    print("✅ Mappings are saved and reused")
    print("✅ Auto-detection for similar column names")
    print("✅ Batch processing works with mappings")
    print("✅ Status indicator shows when mappings are active")

if __name__ == "__main__":
    demonstrate_column_mapping() 