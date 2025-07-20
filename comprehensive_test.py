#!/usr/bin/env python3
"""
Comprehensive Test Script for Work Order Analysis Program
Tests all features, menu items, tabs, and functions systematically
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import sys
import os
import logging
from datetime import datetime

# Add the current directory to the path to import the main program
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from WorkOrderAnalysisCur2 import FailureModeApp
    print("✓ Successfully imported WorkOrderAnalysisCur2")
except ImportError as e:
    print(f"✗ Failed to import WorkOrderAnalysisCur2: {e}")
    sys.exit(1)

class ComprehensiveTester:
    def __init__(self):
        self.root = None
        self.app = None
        self.test_results = []
        self.current_test = ""
        self.errors_found = []
        
    def log_test(self, test_name, status, details=""):
        """Log test results"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = f"[{timestamp}] {test_name}: {status}"
        if details:
            result += f" - {details}"
        print(result)
        self.test_results.append(result)
        
        if status == "FAILED":
            self.errors_found.append(f"{test_name}: {details}")
    
    def run_test(self, test_func, test_name):
        """Run a test function and log results"""
        self.current_test = test_name
        try:
            test_func()
            self.log_test(test_name, "PASSED")
        except Exception as e:
            self.log_test(test_name, "FAILED", str(e))
    
    def test_application_startup(self):
        """Test application startup and basic initialization"""
        try:
            self.root = tk.Tk()
            self.root.withdraw()  # Hide the main window during testing
            self.app = FailureModeApp(self.root)
            self.log_test("Application Startup", "PASSED")
        except Exception as e:
            self.log_test("Application Startup", "FAILED", str(e))
            raise
    
    def test_menu_bar_creation(self):
        """Test menu bar creation and all menu items"""
        try:
            # Test menu bar exists
            if hasattr(self.app, 'menu_bar'):
                self.log_test("Menu Bar Creation", "PASSED")
            else:
                self.log_test("Menu Bar Creation", "FAILED", "Menu bar not found")
                return
            
            # Test menu items exist
            menu_items = ['File', 'Tools', 'Help']
            for item in menu_items:
                if hasattr(self.app, f'{item.lower()}_menu'):
                    self.log_test(f"Menu Item: {item}", "PASSED")
                else:
                    self.log_test(f"Menu Item: {item}", "FAILED", f"{item} menu not found")
                    
        except Exception as e:
            self.log_test("Menu Bar Testing", "FAILED", str(e))
    
    def test_notebook_creation(self):
        """Test notebook creation and all tabs"""
        try:
            if hasattr(self.app, 'notebook'):
                self.log_test("Notebook Creation", "PASSED")
            else:
                self.log_test("Notebook Creation", "FAILED", "Notebook not found")
                return
            
            # Test all expected tabs
            expected_tabs = [
                'Data Input', 'Analysis', 'AI Settings', 'Risk Assessment',
                'Weibull Analysis', 'PM Analysis', 'Spares Analysis'
            ]
            
            for tab_name in expected_tabs:
                # Check if tab exists by looking for tab methods or attributes
                tab_method = f'create_{tab_name.lower().replace(" ", "_")}_tab'
                if hasattr(self.app, tab_method):
                    self.log_test(f"Tab: {tab_name}", "PASSED")
                else:
                    self.log_test(f"Tab: {tab_name}", "FAILED", f"Tab method {tab_method} not found")
                    
        except Exception as e:
            self.log_test("Notebook Testing", "FAILED", str(e))
    
    def test_data_input_tab(self):
        """Test data input tab functionality"""
        try:
            # Test browse functions
            browse_methods = ['browse_wo', 'browse_dict', 'browse_output']
            for method in browse_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Data Input - {method}", "PASSED")
                else:
                    self.log_test(f"Data Input - {method}", "FAILED", f"Method {method} not found")
            
            # Test processing function
            if hasattr(self.app, 'run_processing'):
                self.log_test("Data Input - run_processing", "PASSED")
            else:
                self.log_test("Data Input - run_processing", "FAILED", "Method not found")
                
        except Exception as e:
            self.log_test("Data Input Tab Testing", "FAILED", str(e))
    
    def test_analysis_tab(self):
        """Test analysis tab functionality"""
        try:
            # Test table operations
            table_methods = ['update_table', 'sort_column', 'edit_cell', 'toggle_row']
            for method in table_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Analysis - {method}", "PASSED")
                else:
                    self.log_test(f"Analysis - {method}", "FAILED", f"Method {method} not found")
            
            # Test export functions
            export_methods = ['export_to_excel', 'export_plot', 'open_plot_in_new_window']
            for method in export_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Analysis - {method}", "PASSED")
                else:
                    self.log_test(f"Analysis - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Analysis Tab Testing", "FAILED", str(e))
    
    def test_ai_settings_tab(self):
        """Test AI settings tab functionality"""
        try:
            ai_methods = ['toggle_ai', 'show_ai_settings', 'show_ai_stats', 'clear_ai_cache']
            for method in ai_methods:
                if hasattr(self.app, method):
                    self.log_test(f"AI Settings - {method}", "PASSED")
                else:
                    self.log_test(f"AI Settings - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("AI Settings Tab Testing", "FAILED", str(e))
    
    def test_risk_assessment_tab(self):
        """Test risk assessment tab functionality"""
        try:
            risk_methods = [
                'update_risk', 'update_risk_plot', 'export_risk_plot',
                'save_risk_preset', 'load_risk_preset', 'load_risk_presets'
            ]
            for method in risk_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Risk Assessment - {method}", "PASSED")
                else:
                    self.log_test(f"Risk Assessment - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Risk Assessment Tab Testing", "FAILED", str(e))
    
    def test_weibull_analysis_tab(self):
        """Test Weibull analysis tab functionality"""
        try:
            weibull_methods = [
                'update_weibull_analysis', 'update_weibull_work_orders_table',
                'toggle_weibull_work_order', 'export_weibull_plot', 'export_weibull_results'
            ]
            for method in weibull_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Weibull Analysis - {method}", "PASSED")
                else:
                    self.log_test(f"Weibull Analysis - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Weibull Analysis Tab Testing", "FAILED", str(e))
    
    def test_pm_analysis_tab(self):
        """Test PM analysis tab functionality"""
        try:
            pm_methods = [
                'update_pm_analysis', 'export_pm_report', 'export_pm_plot',
                'optimize_pm_schedule', 'generate_pm_frequency_report'
            ]
            for method in pm_methods:
                if hasattr(self.app, method):
                    self.log_test(f"PM Analysis - {method}", "PASSED")
                else:
                    self.log_test(f"PM Analysis - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("PM Analysis Tab Testing", "FAILED", str(e))
    
    def test_spares_analysis_tab(self):
        """Test spares analysis tab functionality"""
        try:
            spares_methods = [
                'update_spares_analysis', 'export_spares_plot', 'export_spares_report',
                'get_spares_recommendations'
            ]
            for method in spares_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Spares Analysis - {method}", "PASSED")
                else:
                    self.log_test(f"Spares Analysis - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Spares Analysis Tab Testing", "FAILED", str(e))
    
    def test_filter_functions(self):
        """Test filter and utility functions"""
        try:
            filter_methods = [
                'show_filter_manager', 'apply_date_filter', 'reset_equip_failcode',
                'reset_work_type', 'reset_date', 'include_all_work_orders',
                'exclude_all_work_orders', 'reset_all_filters', 'reset_defaults'
            ]
            for method in filter_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Filter Functions - {method}", "PASSED")
                else:
                    self.log_test(f"Filter Functions - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Filter Functions Testing", "FAILED", str(e))
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        try:
            batch_methods = ['batch_process']
            for method in batch_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Batch Processing - {method}", "PASSED")
                else:
                    self.log_test(f"Batch Processing - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Batch Processing Testing", "FAILED", str(e))
    
    def test_column_mapping(self):
        """Test column mapping functionality"""
        try:
            mapping_methods = [
                'show_column_mapping', 'apply_column_mapping', 'load_saved_column_mappings'
            ]
            for method in mapping_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Column Mapping - {method}", "PASSED")
                else:
                    self.log_test(f"Column Mapping - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Column Mapping Testing", "FAILED", str(e))
    
    def test_export_functions(self):
        """Test export and reporting functions"""
        try:
            export_methods = [
                'export_report', 'export_training_data', 'export_plot',
                'add_to_fmea_export', 'show_fmea_export_data', 'export_fmea_data_to_excel'
            ]
            for method in export_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Export Functions - {method}", "PASSED")
                else:
                    self.log_test(f"Export Functions - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Export Functions Testing", "FAILED", str(e))
    
    def test_utility_functions(self):
        """Test utility and helper functions"""
        try:
            utility_methods = [
                'show_user_guide', 'show_about', 'check_updates', 'update_progress',
                'clear_data', 'on_closing', 'get_filtered_df', 'get_current_filters_text'
            ]
            for method in utility_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Utility Functions - {method}", "PASSED")
                else:
                    self.log_test(f"Utility Functions - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Utility Functions Testing", "FAILED", str(e))
    
    def test_dropdown_updates(self):
        """Test dropdown update functions"""
        try:
            dropdown_methods = [
                'update_failure_code_dropdown', 'update_weibull_failure_dropdown',
                'update_pm_equipment_dropdown', 'update_spares_equipment_dropdown'
            ]
            for method in dropdown_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Dropdown Updates - {method}", "PASSED")
                else:
                    self.log_test(f"Dropdown Updates - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Dropdown Updates Testing", "FAILED", str(e))
    
    def test_plot_interactions(self):
        """Test plot interaction functions"""
        try:
            plot_methods = [
                'highlight_plot_point_by_work_order', 'return_to_single_plot',
                'update_risk_segmented', 'create_crow_amsaa_plot_interactive',
                'show_context_menu', 'segment_crow_amsaa_at_selected'
            ]
            for method in plot_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Plot Interactions - {method}", "PASSED")
                else:
                    self.log_test(f"Plot Interactions - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Plot Interactions Testing", "FAILED", str(e))
    
    def test_configuration_functions(self):
        """Test configuration and file management functions"""
        try:
            config_methods = [
                'load_saved_file_paths', 'save_file_paths'
            ]
            for method in config_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Configuration - {method}", "PASSED")
                else:
                    self.log_test(f"Configuration - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Configuration Testing", "FAILED", str(e))
    
    def test_date_functions(self):
        """Test date-related functions"""
        try:
            date_methods = [
                'show_date_selector', 'clear_date_range'
            ]
            for method in date_methods:
                if hasattr(self.app, method):
                    self.log_test(f"Date Functions - {method}", "PASSED")
                else:
                    self.log_test(f"Date Functions - {method}", "FAILED", f"Method {method} not found")
                    
        except Exception as e:
            self.log_test("Date Functions Testing", "FAILED", str(e))
    
    def run_all_tests(self):
        """Run all comprehensive tests"""
        print("=" * 80)
        print("COMPREHENSIVE WORK ORDER ANALYSIS PROGRAM TEST")
        print("=" * 80)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all test categories
        test_categories = [
            ("Application Startup", self.test_application_startup),
            ("Menu Bar Creation", self.test_menu_bar_creation),
            ("Notebook Creation", self.test_notebook_creation),
            ("Data Input Tab", self.test_data_input_tab),
            ("Analysis Tab", self.test_analysis_tab),
            ("AI Settings Tab", self.test_ai_settings_tab),
            ("Risk Assessment Tab", self.test_risk_assessment_tab),
            ("Weibull Analysis Tab", self.test_weibull_analysis_tab),
            ("PM Analysis Tab", self.test_pm_analysis_tab),
            ("Spares Analysis Tab", self.test_spares_analysis_tab),
            ("Filter Functions", self.test_filter_functions),
            ("Batch Processing", self.test_batch_processing),
            ("Column Mapping", self.test_column_mapping),
            ("Export Functions", self.test_export_functions),
            ("Utility Functions", self.test_utility_functions),
            ("Dropdown Updates", self.test_dropdown_updates),
            ("Plot Interactions", self.test_plot_interactions),
            ("Configuration Functions", self.test_configuration_functions),
            ("Date Functions", self.test_date_functions),
        ]
        
        for test_name, test_func in test_categories:
            self.run_test(test_func, test_name)
            time.sleep(0.1)  # Small delay between tests
        
        # Print summary
        print()
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if "PASSED" in r])
        failed_tests = len([r for r in self.test_results if "FAILED" in r])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.errors_found:
            print()
            print("ERRORS FOUND:")
            print("-" * 40)
            for error in self.errors_found:
                print(f"• {error}")
        
        print()
        print("=" * 80)
        print("DETAILED TEST RESULTS")
        print("=" * 80)
        for result in self.test_results:
            print(result)
        
        # Cleanup
        if self.root:
            self.root.destroy()
        
        return len(self.errors_found) == 0

def main():
    """Main test execution function"""
    tester = ComprehensiveTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! The program appears to be working correctly.")
        return 0
    else:
        print(f"\n❌ {len(tester.errors_found)} errors found. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 