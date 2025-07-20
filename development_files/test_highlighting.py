#!/usr/bin/env python3
"""
Test script to verify work order highlighting on Crow-AMSAA plot.
"""

import tkinter as tk
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from WorkOrderAnalysisCur1 import FailureModeApp

def create_test_data():
    """Create test work order data for highlighting test."""
    # Create sample work order data
    dates = []
    for i in range(10):
        date = datetime(2024, 1, 1) + timedelta(days=i*30)  # Every 30 days
        dates.append(date.strftime('%m/%d/%Y'))
    
    test_data = {
        'Work Order': [f'WO{i+1:03d}' for i in range(10)],
        'Description': [f'Test failure {i+1}' for i in range(10)],
        'Asset': [f'Asset{i+1}' for i in range(10)],
        'Equipment #': ['EQ001'] * 10,
        'Work Type': ['Corrective'] * 10,
        'Reported Date': dates,
        'Failure Code': ['No Failure Mode Identified'] * 10,
        'Failure Description': ['Test Failure'] * 10,
        'Matched Keyword': ['test'] * 10
    }
    
    return pd.DataFrame(test_data)

def test_highlighting():
    """Test the highlighting functionality."""
    
    # Create a test window
    root = tk.Tk()
    root.title("Highlighting Test")
    root.geometry("800x600")
    
    try:
        app = FailureModeApp(root)
        
        # Create test data
        test_df = create_test_data()
        app.wo_df = test_df
        app.included_indices = set(test_df.index)
        
        print("Test data created:")
        print(f"Work orders: {len(test_df)}")
        print(f"Date range: {test_df['Reported Date'].min()} to {test_df['Reported Date'].max()}")
        
        # Update the table to create the plot
        print("\nUpdating table to create Crow-AMSAA plot...")
        app.update_table()
        
        # Wait a moment for the plot to render
        root.update()
        root.after(1000)  # Wait 1 second
        
        # Test highlighting each work order
        print("\nTesting highlighting for each work order...")
        for i in range(len(test_df)):
            work_order_idx = test_df.index[i]
            work_order_num = test_df.iloc[i]['Work Order']
            date = test_df.iloc[i]['Reported Date']
            
            print(f"Testing highlight for {work_order_num} (date: {date})")
            
            # Select the work order in the tree
            if app.tree:
                # Find the item in the tree
                for item in app.tree.get_children():
                    values = app.tree.item(item, 'values')
                    if len(values) > 2 and values[2] == work_order_num:
                        app.tree.selection_set(item)
                        app.tree.see(item)
                        root.update()
                        break
                
                # Wait a moment to see the highlight
                root.after(500)
                root.update()
                
                print(f"  - Selected {work_order_num} in tree")
            else:
                print("  - Tree not available")
        
        print("\nHighlighting test completed!")
        print("Check the Crow-AMSAA plot to see if points are highlighted in red when selected.")
        
        # Keep the window open for manual inspection
        print("\nWindow will stay open for 10 seconds for manual inspection...")
        root.after(10000, root.destroy)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        root.destroy()

if __name__ == "__main__":
    test_highlighting() 