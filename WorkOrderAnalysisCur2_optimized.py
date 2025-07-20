import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import os
import re
import logging
import json
import warnings
from typing import Optional, Union, Any, List, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
DEFAULT_DICT_PATH = "failure_mode_dictionary.xlsx"
DEFAULT_CODE = "No Failure Mode Identified"
DEFAULT_DESC = "No Failure Mode Identified"
CUSTOM_CODE = "custom"
LOG_FILE = "matching_log.txt"
ABBREVIATIONS = {
    'comp': 'compressor',
    'leek': 'leak',
    'leeking': 'leaking',
    'brk': 'break',
    'mtr': 'motor',
}
THRESHOLD = 75  # Fuzzy matching threshold
DATE_FORMATS = ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]

# Required columns for work order files
REQUIRED_COLUMNS = {
    'Work Order': 'Work Order Number',
    'Description': 'Work Description', 
    'Asset': 'Asset Name',
    'Equipment #': 'Equipment Number',
    'Work Type': 'Work Type',
    'Reported Date': 'Date Reported',
    'Work Order Cost': 'Work Order Cost (Optional)',
    'User failure code': 'User Failure Code (Optional)'
}

# AI Configuration
AI_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for embeddings
AI_CACHE_FILE = "ai_classification_cache.json"

# Configuration files
CONFIG_FILE = "app_config.json"
COLUMN_MAPPING_FILE = "column_mapping.json"

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for deferred imports
_pandas_loaded = False
_matplotlib_loaded = False
_numpy_loaded = False
_scipy_loaded = False
_nltk_loaded = False
_rapidfuzz_loaded = False
_ai_modules_loaded = False
_analysis_modules_loaded = False

def load_pandas():
    """Deferred import of pandas"""
    global pd, _pandas_loaded
    if not _pandas_loaded:
        import pandas as pd
        _pandas_loaded = True
    return pd

def load_numpy():
    """Deferred import of numpy"""
    global np, _numpy_loaded
    if not _numpy_loaded:
        import numpy as np
        _numpy_loaded = True
    return np

def load_matplotlib():
    """Deferred import of matplotlib"""
    global plt, FigureCanvasTkAgg, FuncFormatter, Line2D, Rectangle, _matplotlib_loaded
    if not _matplotlib_loaded:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.ticker import FuncFormatter
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        _matplotlib_loaded = True
    return plt, FigureCanvasTkAgg, FuncFormatter, Line2D, Rectangle

def load_scipy():
    """Deferred import of scipy"""
    global scipy, _scipy_loaded
    if not _scipy_loaded:
        import scipy
        _scipy_loaded = True
    return scipy

def load_nltk():
    """Deferred import of nltk"""
    global nltk, word_tokenize, SnowballStemmer, stemmer, _nltk_loaded
    if not _nltk_loaded:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.stem import SnowballStemmer
        
        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            logging.error(f"Failed to download NLTK data: {e}")
        
        stemmer = SnowballStemmer("english")
        _nltk_loaded = True
    return nltk, word_tokenize, SnowballStemmer, stemmer

def load_rapidfuzz():
    """Deferred import of rapidfuzz"""
    global rapidfuzz, fuzz, _rapidfuzz_loaded
    if not _rapidfuzz_loaded:
        import rapidfuzz
        from rapidfuzz import fuzz
        _rapidfuzz_loaded = True
    return rapidfuzz, fuzz

def load_ai_modules():
    """Deferred import of AI modules"""
    global AIClassifier, AIClassificationResult, AI_AVAILABLE, _ai_modules_loaded
    if not _ai_modules_loaded:
        try:
            from ai_failure_classifier import AIClassifier, AIClassificationResult
            AI_AVAILABLE = True
        except ImportError:
            AI_AVAILABLE = False
            logging.warning("AI classifier not available. Install dependencies and ensure ai_failure_classifier.py is present.")
        _ai_modules_loaded = True
    return AIClassifier, AIClassificationResult, AI_AVAILABLE

def load_analysis_modules():
    """Deferred import of analysis modules"""
    global WeibullAnalysis, FMEAExport, PMAnalysis, SparesAnalysis, MODULES_AVAILABLE, _analysis_modules_loaded
    if not _analysis_modules_loaded:
        try:
            from weibull_analysis import WeibullAnalysis
            from fmea_export import FMEAExport
            from pm_analysis import PMAnalysis
            from spares_analysis import SparesAnalysis
            MODULES_AVAILABLE = True
        except ImportError as e:
            MODULES_AVAILABLE = False
            logging.warning(f"Analysis modules not available: {e}")
        _analysis_modules_loaded = True
    return WeibullAnalysis, FMEAExport, PMAnalysis, SparesAnalysis, MODULES_AVAILABLE

def load_app_config() -> dict:
    """Load application configuration from JSON file."""
    default_config = {
        'last_work_order_path': '',
        'last_dictionary_path': '',
        'last_output_directory': '',
        'ai_enabled': False,
        'ai_confidence_threshold': AI_CONFIDENCE_THRESHOLD,
        'fuzzy_threshold': THRESHOLD,
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
    
    return default_config

def save_app_config(config: dict):
    """Save application configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving config: {e}")

def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    if not text:
        return ""
    return re.sub(r'[^\w\s]', ' ', str(text).lower()).strip()

def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations in text."""
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
    return text

def compile_patterns(keywords: list) -> list:
    """Compile regex patterns for keywords."""
    patterns = []
    for keyword in keywords:
        if keyword and isinstance(keyword, str):
            # Escape special regex characters and create pattern
            escaped = re.escape(keyword.lower())
            pattern = re.compile(escaped, re.IGNORECASE)
            patterns.append(pattern)
    return patterns

def match_failure_mode(description: str, dictionary: list) -> tuple:
    """Match failure mode using fuzzy matching and pattern matching."""
    # Load required modules
    load_nltk()
    load_rapidfuzz()
    
    if not description or not dictionary:
        return DEFAULT_CODE, DEFAULT_DESC, 0
    
    # Normalize description
    desc = normalize_text(description)
    desc = expand_abbreviations(desc)
    
    best_match = None
    best_score = 0
    best_code = DEFAULT_CODE
    best_desc = DEFAULT_DESC
    
    for item in dictionary:
        if not isinstance(item, dict):
            continue
            
        failure_desc = item.get('Failure Description', '')
        failure_code = item.get('Failure Code', '')
        
        if not failure_desc:
            continue
        
        # Normalize failure description
        norm_failure_desc = normalize_text(failure_desc)
        norm_failure_desc = expand_abbreviations(norm_failure_desc)
        
        # Fuzzy matching
        score = fuzz.ratio(desc, norm_failure_desc)
        
        # Pattern matching bonus
        patterns = compile_patterns([failure_desc])
        for pattern in patterns:
            if pattern.search(desc):
                score += 10  # Bonus for pattern match
                break
        
        # Stemming comparison
        try:
            desc_stems = [stemmer.stem(word) for word in word_tokenize(desc)]
            failure_stems = [stemmer.stem(word) for word in word_tokenize(norm_failure_desc)]
            
            common_stems = set(desc_stems) & set(failure_stems)
            if common_stems:
                score += len(common_stems) * 2  # Bonus for common stems
        except Exception as e:
            logging.warning(f"Stemming error: {e}")
        
        if score > best_score:
            best_score = score
            best_code = failure_code
            best_desc = failure_desc
    
    return best_code, best_desc, best_score

def parse_date(date: str) -> Union[Any, Any]:
    """Parse date string to datetime object."""
    from datetime import datetime
    
    if not date or pd.isna(date):
        return None
    
    date_str = str(date).strip()
    
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try pandas parsing as fallback
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def calculate_mtbf(filtered_df: Any, included_indices: set) -> float:
    """Calculate Mean Time Between Failures."""
    load_pandas()
    load_numpy()
    
    if filtered_df.empty or not included_indices:
        return 0.0
    
    # Filter to included work orders
    included_df = filtered_df[filtered_df.index.isin(included_indices)]
    
    if included_df.empty:
        return 0.0
    
    # Calculate total time span
    dates = included_df['Reported Date'].dropna()
    if len(dates) < 2:
        return 0.0
    
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    
    if total_days <= 0:
        return 0.0
    
    # Calculate MTBF in days
    mtbf_days = total_days / len(included_df)
    
    return mtbf_days

def calculate_crow_amsaa_params(filtered_df: Any, included_indices: set) -> tuple:
    """Calculate Crow-AMSAA parameters."""
    load_pandas()
    load_numpy()
    load_scipy()
    
    if filtered_df.empty or not included_indices:
        return 0.0, 1.0, 0.0
    
    # Filter to included work orders
    included_df = filtered_df[filtered_df.index.isin(included_indices)]
    
    if len(included_df) < 3:
        return 0.0, 1.0, 0.0
    
    try:
        # Sort by date
        sorted_df = included_df.sort_values('Reported Date')
        
        # Calculate cumulative failures
        cumulative_failures = np.arange(1, len(sorted_df) + 1)
        
        # Calculate time from start
        start_time = sorted_df['Reported Date'].iloc[0]
        times = [(date - start_time).days for date in sorted_df['Reported Date']]
        times = np.array(times)
        
        # Avoid zero times
        if np.any(times == 0):
            times = times + 1
        
        # Log transformation for linear regression
        log_times = np.log(times)
        log_failures = np.log(cumulative_failures)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_times, log_failures)
        
        # Crow-AMSAA parameters
        beta = slope  # Shape parameter
        eta = np.exp(-intercept / beta)  # Scale parameter
        
        return eta, beta, r_value**2
        
    except Exception as e:
        logging.error(f"Error calculating Crow-AMSAA parameters: {e}")
        return 0.0, 1.0, 0.0

# Continue with the rest of your functions, but add load_*() calls at the beginning
# of functions that need the heavy modules

def create_crow_amsaa_plot(filtered_df: Any, included_indices: set, frame: Union[tk.Frame, ttk.LabelFrame]) -> tuple:
    """Create Crow-AMSAA plot."""
    load_matplotlib()
    load_pandas()
    load_numpy()
    
    # Clear previous plot
    for widget in frame.winfo_children():
        widget.destroy()
    
    if filtered_df.empty or not included_indices:
        label = ttk.Label(frame, text="No data available for Crow-AMSAA analysis")
        label.pack(pady=20)
        return None, None
    
    # Filter to included work orders
    included_df = filtered_df[filtered_df.index.isin(included_indices)]
    
    if len(included_df) < 3:
        label = ttk.Label(frame, text="Insufficient data for Crow-AMSAA analysis (minimum 3 failures required)")
        label.pack(pady=20)
        return None, None
    
    try:
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by date
        sorted_df = included_df.sort_values('Reported Date')
        
        # Calculate cumulative failures
        cumulative_failures = np.arange(1, len(sorted_df) + 1)
        
        # Calculate time from start
        start_time = sorted_df['Reported Date'].iloc[0]
        times = [(date - start_time).days for date in sorted_df['Reported Date']]
        times = np.array(times)
        
        # Avoid zero times
        if np.any(times == 0):
            times = times + 1
        
        # Plot actual data
        ax.plot(times, cumulative_failures, 'bo-', label='Actual Failures', markersize=6)
        
        # Calculate and plot Crow-AMSAA fit
        eta, beta, r_squared = calculate_crow_amsaa_params(filtered_df, included_indices)
        
        if eta > 0 and beta > 0:
            # Generate fitted curve
            time_range = np.linspace(times[0], times[-1], 100)
            fitted_failures = (time_range / eta) ** beta
            
            ax.plot(time_range, fitted_failures, 'r--', label=f'Crow-AMSAA Fit (β={beta:.2f}, η={eta:.1f})', linewidth=2)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cumulative Failures')
        ax.set_title('Crow-AMSAA Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R-squared value
        if r_squared > 0:
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        return fig, canvas
        
    except Exception as e:
        logging.error(f"Error creating Crow-AMSAA plot: {e}")
        label = ttk.Label(frame, text=f"Error creating plot: {str(e)}")
        label.pack(pady=20)
        return None, None

# Continue with the rest of your application...
# For brevity, I'll show the main application class structure

class FailureModeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WO Analysis Program")
        
        # Load configuration
        self.config = load_app_config()
        
        # Initialize variables
        self.work_order_df = None
        self.dictionary_df = None
        self.output_directory = self.config.get('last_output_directory', '')
        
        # Load modules when needed
        self.ai_classifier = None
        self.ai_available = False
        
        # Create GUI
        self.create_gui()
        
        # Load saved paths
        self.load_saved_file_paths()
    
    def create_gui(self):
        """Create the main GUI."""
        # Create menu bar
        self.create_menu_bar()
        
        # Create notebook for tabs
        self.create_notebook()
        
        # Create status bar
        self.create_status_bar()
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Work Order File", command=self.browse_wo)
        file_menu.add_command(label="Load Dictionary File", command=self.browse_dict)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_notebook(self):
        """Create the notebook with tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs (implement these methods)
        self.create_data_input_tab()
        self.create_analysis_tab()
    
    def create_data_input_tab(self):
        """Create the data input tab."""
        # Implementation here
        pass
    
    def create_analysis_tab(self):
        """Create the analysis tab."""
        # Implementation here
        pass
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_wo(self):
        """Browse for work order file."""
        filename = filedialog.askopenfilename(
            title="Select Work Order File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.config['last_work_order_path'] = filename
            save_app_config(self.config)
            self.status_var.set(f"Loaded: {os.path.basename(filename)}")
    
    def browse_dict(self):
        """Browse for dictionary file."""
        filename = filedialog.askopenfilename(
            title="Select Dictionary File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.config['last_dictionary_path'] = filename
            save_app_config(self.config)
            self.status_var.set(f"Loaded: {os.path.basename(filename)}")
    
    def show_user_guide(self):
        """Show user guide."""
        messagebox.showinfo("User Guide", "User guide content here...")
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About", "WO Analysis Program\nVersion 1.0\n\nA comprehensive work order analysis tool.")
    
    def load_saved_file_paths(self):
        """Load saved file paths."""
        # Implementation here
        pass

def main():
    """Main function."""
    root = tk.Tk()
    root.geometry("1200x800")
    
    # Set icon if available
    try:
        root.iconbitmap("icons/app_icon.ico")
    except:
        pass
    
    app = FailureModeApp(root)
    
    # Handle window closing
    def on_closing():
        app.config['last_output_directory'] = app.output_directory
        save_app_config(app.config)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main() 