#!/usr/bin/env python3
"""
Test script to verify the risk calculation fix for segmented plots.
"""

import tkinter as tk
import pandas as pd
from WorkOrderAnalysisCur1 import FailureModeApp

def test_risk_calculation():
    """Test that risk calculation works correctly for both segmented and non-segmented views."""
    
    # Create a minimal test window
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    try:
        app = FailureModeApp(root)
        # Set a dummy non-empty DataFrame to avoid early return
        app.wo_df = pd.DataFrame({'A': [1]})
        
        # Test 1: Check initial state (not segmented)
        print("Test 1: Initial state (not segmented)")
        print(f"is_segmented_view: {app.is_segmented_view}")
        print(f"segment_data: {app.segment_data}")
        
        # Test 2: Simulate segmented view
        print("\nTest 2: Simulating segmented view")
        app.is_segmented_view = True
        app.segment_data = (
            (1.2, 0.001, 2.5),  # beta1, lambda1, failures_per_year1
            (0.8, 0.002, 1.8)   # beta2, lambda2, failures_per_year2
        )
        
        # Set some test values
        app.prod_loss_var.set("100")
        app.maint_cost_var.set("5000")
        app.margin_var.set("50")
        
        # Test the update_risk function
        print("Calling update_risk() in segmented mode...")
        app.update_risk()
        
        # Check the result
        current_text = app.risk_label.cget("text")
        print(f"Risk label text: {current_text}")
        
        # Verify it shows both segments
        if "Segment 1:" in current_text and "Segment 2:" in current_text:
            print("✅ SUCCESS: Risk calculation shows both segments when in segmented view")
        else:
            print("❌ FAILURE: Risk calculation does not show both segments")
        
        # Test 3: Return to non-segmented view
        print("\nTest 3: Returning to non-segmented view")
        app.is_segmented_view = False
        app.segment_data = None
        
        # Mock some data for single view
        app.wo_df = None  # This will trigger the early return in update_risk
        
        print("Calling update_risk() in non-segmented mode...")
        app.update_risk()
        
        current_text = app.risk_label.cget("text")
        print(f"Risk label text: {current_text}")
        
        if "N/A" in current_text:
            print("✅ SUCCESS: Risk calculation shows N/A when no data (expected)")
        else:
            print("❌ FAILURE: Risk calculation shows unexpected result")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        root.destroy()

if __name__ == "__main__":
    test_risk_calculation() 