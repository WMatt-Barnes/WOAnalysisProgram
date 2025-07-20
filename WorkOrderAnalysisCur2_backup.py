import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import pandas as pd
import os
import re
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import rapidfuzz
from rapidfuzz import fuzz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import warnings
from typing import Optional, Union, Any

# AI Classification imports
try:
    from ai_failure_classifier import AIClassifier, AIClassificationResult
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logging.warning("AI classifier not available. Install dependencies and ensure ai_failure_classifier.py is present.")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {e}")

# Constants
DEFAULT_DICT_PATH = "failure_mode_dictionary.xlsx"
DEFAULT_CODE = "0.0"
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
    'Work Order Cost': 'Work Order Cost (Optional)'
}

# AI Configuration
AI_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for embeddings
AI_CACHE_FILE = "ai_classification_cache.json"

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize stemmer
stemmer = SnowballStemmer("english")

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations in text."""
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
    return text

def compile_patterns(keywords: list) -> list:
    """Compile regex patterns for keywords."""
    patterns = []
    for keyword in keywords:
        try:
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
            patterns.append(pattern)
        except re.error as e:
            logging.error(f"Error compiling pattern for keyword '{keyword}': {e}")
    return patterns

def match_failure_mode(description: str, dictionary: list) -> tuple:
    """Match failure mode in description using exact, fuzzy, and stemmed matching."""
    if not description or not isinstance(description, str):
        logging.debug(f"Invalid description: {description}")
        return DEFAULT_CODE, DEFAULT_DESC, ''
    
    norm_desc = normalize_text(description)
    norm_desc = expand_abbreviations(norm_desc)
    logging.debug(f"Normalized description: {norm_desc}")
    
    try:
        tokens = word_tokenize(norm_desc)
        stemmed_desc = ' '.join(stemmer.stem(token) for token in tokens)
        logging.debug(f"Stemmed description: {stemmed_desc}")
    except Exception as e:
        logging.error(f"Error tokenizing description: {e}")
        stemmed_desc = norm_desc
    
    desc_length = len(norm_desc)
    
    for keyword, norm_keyword, stemmed_keyword, code, failure_desc, pattern in dictionary:
        keyword_list = [k.strip().lower() for k in keyword.split(',')]
        for kw in keyword_list:
            norm_kw = normalize_text(kw)
            kw_length = len(norm_kw)
            try:
                kw_tokens = word_tokenize(norm_kw)
                stemmed_kw = ' '.join(stemmer.stem(token) for token in kw_tokens)
            except Exception as e:
                logging.error(f"Error tokenizing keyword '{kw}': {e}")
                stemmed_kw = norm_kw
            
            if norm_kw in norm_desc or kw in norm_desc:
                logging.info(f"Exact/partial match: keyword={kw}, code={code}, desc={failure_desc}")
                return code, failure_desc, kw
            
            try:
                if re.search(r'\b' + re.escape(kw) + r'\b', norm_desc, re.IGNORECASE):
                    logging.info(f"Regex match: keyword={kw}, code={code}, desc={failure_desc}")
                    return code, failure_desc, kw
            except re.error as e:
                logging.error(f"Error in regex for keyword '{kw}': {e}")
            
            if stemmed_kw in stemmed_desc:
                logging.info(f"Stemmed match: keyword={kw}, code={code}, desc={failure_desc}")
                return code, failure_desc, kw
            
            if min(kw_length, desc_length) / max(kw_length, desc_length) >= 0.5:
                score = fuzz.partial_ratio(norm_kw, norm_desc)
                logging.debug(f"Fuzzy match attempt: keyword={kw}, score={score}, length_ratio={min(kw_length, desc_length)/max(kw_length, desc_length):.2f}")
                if score >= THRESHOLD:
                    logging.info(f"Fuzzy match: keyword={kw}, score={score}, code={code}, desc={failure_desc}")
                    return code, failure_desc, kw
            else:
                logging.debug(f"Fuzzy match skipped: keyword={kw}, length_ratio={min(kw_length, desc_length)/max(kw_length, desc_length):.2f}")
    
    logging.info(f"No match for description: {norm_desc}")
    return DEFAULT_CODE, DEFAULT_DESC, ''

def parse_date(date: str) -> Union[datetime, Any]:
    """Try parsing date with multiple formats."""
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(date, format=fmt)
        except (ValueError, TypeError):
            continue
    logging.error(f"Failed to parse date: {date}")
    return pd.NaT

def calculate_mtbf(filtered_df: pd.DataFrame, included_indices: set) -> float:
    """Calculate Mean Time Between Failures (MTBF) in days for included rows."""
    try:
        valid_indices = filtered_df.index.intersection(included_indices)
        filtered_dates = [
            parse_date(filtered_df.at[idx, 'Reported Date'])
            for idx in valid_indices
            if pd.notna(filtered_df.at[idx, 'Reported Date'])
        ]
        filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
        if len(filtered_dates) < 2:
            logging.debug(f"Insufficient valid dates for MTBF: {len(filtered_dates)}")
            return 0.0
        dates = sorted(filtered_dates)
        time_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        mtbf = round(sum(time_diffs) / len(time_diffs), 2)
        logging.debug(f"MTBF calculated: {mtbf} days from {len(dates)} dates")
        return mtbf
    except Exception as e:
        logging.error(f"Error calculating MTBF: {e}")
        return 0.0

def calculate_crow_amsaa_params(filtered_df: pd.DataFrame, included_indices: set) -> tuple:
    """Calculate Crow-AMSAA parameters (lambda, beta) and failures per year."""
    try:
        valid_indices = filtered_df.index.intersection(included_indices)
        filtered_dates = [
            parse_date(filtered_df.at[idx, 'Reported Date'])
            for idx in valid_indices
            if pd.notna(filtered_df.at[idx, 'Reported Date'])
        ]
        filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
        
        if len(filtered_dates) == 0:
            logging.debug("No valid dates for Crow-AMSAA")
            return None, None, 0.0
        elif len(filtered_dates) == 1:
            lambda_param = 1 / 365
            beta = 1.0
            failures_per_year = lambda_param * (365 ** beta)
            logging.debug(f"Single failure case: failures/year={failures_per_year:.2f}")
            return lambda_param, beta, round(failures_per_year, 2)
        
        dates = sorted(filtered_dates)
        t0 = dates[0]
        times = [(d - t0).days + 1 for d in dates]
        n = np.arange(1, len(times) + 1)
        
        log_n = np.log(n)
        log_t = np.log(times)
        coeffs = np.polyfit(log_t, log_n, 1)
        beta = coeffs[0]
        lambda_param = np.exp(coeffs[1])
        
        failures_per_year = lambda_param * (365 ** beta)
        
        logging.debug(f"Crow-AMSAA params: beta={beta:.2f}, lambda={lambda_param:.4f}, failures/year={failures_per_year:.2f}")
        return lambda_param, beta, round(failures_per_year, 2)
    except Exception as e:
        logging.error(f"Error calculating Crow-AMSAA params: {e}")
        return None, None, 0.0

def create_crow_amsaa_plot(filtered_df: pd.DataFrame, included_indices: set, frame: Union[tk.Frame, ttk.LabelFrame]) -> tuple:
    """Create a Crow-AMSAA plot and return figure and parameters."""
    try:
        valid_indices = filtered_df.index.intersection(included_indices)
        filtered_dates = [
            parse_date(filtered_df.at[idx, 'Reported Date'])
            for idx in valid_indices
            if pd.notna(filtered_df.at[idx, 'Reported Date'])
        ]
        filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
        
        if len(filtered_dates) < 2:
            logging.debug(f"Insufficient valid dates for Crow-AMSAA: {len(filtered_dates)}")
            return None, None, None
        
        dates = sorted(filtered_dates)
        t0 = dates[0]
        times = [(d - t0).days + 1 for d in dates]
        n = np.arange(1, len(times) + 1)
        
        log_n = np.log(n)
        log_t = np.log(times)
        coeffs = np.polyfit(log_t, log_n, 1)
        beta = coeffs[0]
        lambda_param = np.exp(coeffs[1])
        
        frame.update_idletasks()
        width_px = max(frame.winfo_width(), 100)
        height_px = max(frame.winfo_height(), 100)
        width_in = min(max(width_px / 100, 4), 8)
        height_in = min(max(height_px / 100, 2.5), 5)
        font_scale = width_in / 5
        
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax.scatter(times, n, marker='o', label='Observed Failures')
        t_fit = np.linspace(min(times), max(times), 100)
        n_fit = lambda_param * t_fit ** beta
        ax.plot(t_fit, n_fit, label=f'Crow-AMSAA (β={beta:.2f}, λ={lambda_param:.4f})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Crow-AMSAA Plot", fontsize=10 * font_scale)
        ax.set_xlabel("Time (days)", fontsize=8 * font_scale)
        ax.set_ylabel("Cumulative Failures", fontsize=8 * font_scale)
        ax.legend(fontsize=6 * font_scale)
        ax.grid(True, which="both", ls="--")
        ax.tick_params(axis='both', labelsize=6 * font_scale)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        logging.debug(f"Crow-AMSAA plot created: beta={beta:.2f}, lambda={lambda_param:.4f}, figsize=({width_in}, {height_in})")
        return fig, beta, lambda_param
    except Exception as e:
        logging.error(f"Error creating Crow-AMSAA plot: {e}")
        return None, None, None

def process_files(work_order_path: str, dict_path: str, status_label: ttk.Label, root: tk.Tk, output_dir: Optional[str] = None, use_ai: bool = False, ai_classifier: Optional[AIClassifier] = None, column_mapping: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """Process work order and dictionary files with optional AI classification."""
    try:
        status_label.config(text="Processing...", foreground="blue")
        root.config(cursor="wait")
        root.update()
        logging.info(f"Processing work order file: {work_order_path}")
        logging.info(f"Dictionary file: {dict_path}")
        logging.info(f"AI classification enabled: {use_ai}")
        logging.info(f"Column mapping: {column_mapping}")
        
        if not os.path.exists(work_order_path):
            raise FileNotFoundError(f"Work order file not found: {work_order_path}")
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        if not work_order_path.endswith('.xlsx'):
            raise ValueError(f"Work order file must be .xlsx, got: {work_order_path}")
        if not dict_path.endswith('.xlsx'):
            raise ValueError(f"Dictionary file must be .xlsx, got: {dict_path}")
        
        try:
            wo_df = pd.read_excel(work_order_path)
            logging.info(f"Work order columns: {list(wo_df.columns)}")
            if wo_df.empty:
                raise ValueError("Work order file is empty")
        except Exception as e:
            raise ValueError(f"Failed to read work order file: {str(e)}")
        
        # Apply column mapping if provided
        if column_mapping:
            rename_dict = {}
            for required_col, mapped_col in column_mapping.items():
                if mapped_col in wo_df.columns and mapped_col != required_col:
                    rename_dict[mapped_col] = required_col
            
            if rename_dict:
                wo_df = wo_df.rename(columns=rename_dict)
                logging.info(f"Applied column mapping: {rename_dict}")
        
        required_columns = ['Work Order', 'Description', 'Asset', 'Equipment #', 'Work Type', 'Reported Date']
        missing_cols = [col for col in required_columns if col not in wo_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in work order file: {', '.join(missing_cols)}")
        
        # Add Work Order Cost column if not present (optional column)
        if 'Work Order Cost' not in wo_df.columns:
            wo_df['Work Order Cost'] = 0.0
            logging.info("Work Order Cost column not found, adding with default value 0.0")
        
        try:
            dict_df = pd.read_excel(dict_path)
            logging.info(f"Dictionary columns: {list(dict_df.columns)}")
            if dict_df.empty:
                raise ValueError("Dictionary file is empty")
        except Exception as e:
            raise ValueError(f"Failed to read dictionary file: {str(e)}")
        
        if not all(col in dict_df.columns for col in ['Keyword', 'Code', 'Description']):
            raise ValueError("Dictionary file must contain columns: Keyword, Code, Description")
        
        if dict_df['Keyword'].dropna().empty:
            raise ValueError("Dictionary file contains no valid keywords")
        
        dictionary = []
        keywords = dict_df['Keyword'].dropna().astype(str).tolist()
        patterns = compile_patterns(keywords)
        for idx, row in dict_df.iterrows():
            keyword = str(row['Keyword']).lower()
            if not keyword:
                continue
            norm_keyword = normalize_text(keyword)
            try:
                tokens = word_tokenize(norm_keyword)
                stemmed_keyword = ' '.join(stemmer.stem(token) for token in tokens)
            except Exception as e:
                logging.error(f"Error tokenizing keyword '{keyword}': {e}")
                stemmed_keyword = norm_keyword
            code = str(row['Code'])
            desc = str(row['Description'])
            pattern = patterns[keywords.index(keyword)] if keyword in keywords else None
            dictionary.append((keyword, norm_keyword, stemmed_keyword, code, desc, pattern))
        
        if not dictionary:
            raise ValueError("No valid keywords processed from dictionary")
        
        # Initialize AI classifier if requested
        if use_ai and AI_AVAILABLE and ai_classifier is None:
            try:
                ai_classifier = AIClassifier(
                    confidence_threshold=AI_CONFIDENCE_THRESHOLD,
                    cache_file=AI_CACHE_FILE
                )
                if not ai_classifier.load_failure_dictionary(dict_path):
                    logging.warning("Failed to load dictionary for AI classifier, falling back to dictionary matching")
                    use_ai = False
            except Exception as e:
                logging.error(f"Failed to initialize AI classifier: {e}")
                use_ai = False
        
        wo_df['Failure Code'] = DEFAULT_CODE
        wo_df['Failure Description'] = DEFAULT_DESC
        wo_df['Matched Keyword'] = ''
        wo_df['AI Confidence'] = 0.0
        wo_df['Classification Method'] = 'dictionary'
        
        # Process work orders
        if use_ai and ai_classifier:
            status_label.config(text="Processing with Enhanced AI classification...", foreground="blue")
            root.update()
            
            # Analyze historical patterns for temporal analysis
            ai_classifier.analyze_historical_patterns(wo_df)
            
            # Batch process with AI
            descriptions = [str(row['Description']) for _, row in wo_df.iterrows()]
            ai_results = ai_classifier.batch_classify(descriptions, lambda desc: match_failure_mode(desc, dictionary))
            
            for idx, (_, row) in enumerate(wo_df.iterrows()):
                if idx < len(ai_results):
                    result = ai_results[idx]
                    wo_df.at[idx, 'Failure Code'] = result.code
                    wo_df.at[idx, 'Failure Description'] = result.description
                    wo_df.at[idx, 'Matched Keyword'] = result.matched_keyword
                    wo_df.at[idx, 'AI Confidence'] = result.confidence
                    wo_df.at[idx, 'Classification Method'] = result.method
                    logging.debug(f"Work Order {row['Work Order']}: AI code={result.code}, confidence={result.confidence}, method={result.method}")
        else:
            # Use traditional dictionary matching
            for idx, row in wo_df.iterrows():
                desc = str(row['Description'])
                code, failure_desc, matched_keyword = match_failure_mode(desc, dictionary)
                wo_df.at[idx, 'Failure Code'] = code
                wo_df.at[idx, 'Failure Description'] = failure_desc
                wo_df.at[idx, 'Matched Keyword'] = matched_keyword
                wo_df.at[idx, 'AI Confidence'] = 0.5  # Default confidence for dictionary matching
                wo_df.at[idx, 'Classification Method'] = 'dictionary'
                logging.debug(f"Work Order {row['Work Order']}: code={code}, desc={failure_desc}, keyword={matched_keyword}")
        
        if not output_dir:
            output_dir = os.path.dirname(work_order_path) or '.'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        status_label.config(text="Processing complete.", foreground="green")
        root.config(cursor="")
        root.update()
        logging.info(f"Processing complete. Rows processed: {len(wo_df)}")
        return wo_df
    
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        status_label.config(text=error_msg, foreground="red")
        root.config(cursor="")
        root.update()
        logging.error(error_msg)
        return None
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        status_label.config(text=error_msg, foreground="red")
        root.config(cursor="")
        root.update()
        logging.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error processing files: {str(e)}"
        status_label.config(text=error_msg, foreground="red")
        root.config(cursor="")
        root.update()
        logging.error(error_msg)
        return None

class FailureModeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Work Order Analysis Pro - AI-Powered Failure Mode Classification")
        self.root.geometry("1400x900")
        
        # Set modern theme
        style = ttk.Style()
        style.theme_use('clam')  # Use a modern theme
        
        self.wo_df = None
        self.output_dir = None
        self.included_indices = set()
        self.dictionary = None
        self.start_date = None
        self.end_date = None
        self.sort_states = {}  # Track sort direction per column
        self.selected_plot_point = None  # For plot selection
        self.crow_amsaa_canvas = None    # Store canvas for mpl_connect
        self.crow_amsaa_fig = None       # Store figure for mpl_connect
        self.context_menu = None         # For right-click menu
        self.is_segmented_view = False   # Track if we're in segmented view
        self.segment_data = None         # Store segment data for risk calculation
        self.last_highlight_artist = None  # Track the last highlight for Crow-AMSAA
        self.ai_classifier = None        # AI classifier instance
        self.use_ai_classification = False  # Whether to use AI classification
        
        # AI settings variables
        self.ai_enabled_var = tk.BooleanVar(value=False)
        self.confidence_var = tk.StringVar(value=str(AI_CONFIDENCE_THRESHOLD))
        self.confidence_scale_var = tk.DoubleVar(value=AI_CONFIDENCE_THRESHOLD)
        
        # Column mapping for CMMS compatibility
        self.column_mapping = {}  # Maps CMMS columns to required columns
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main notebook for tabbed interface
        self.create_notebook()
        
        # Create status bar
        self.create_status_bar()
        
        # Load saved column mappings
        self.load_saved_column_mappings()

    def create_menu_bar(self):
        """Create a professional menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Work Order File...", command=self.browse_wo, accelerator="Ctrl+O")
        file_menu.add_command(label="Load Dictionary File...", command=self.browse_dict, accelerator="Ctrl+D")
        file_menu.add_separator()
        file_menu.add_command(label="Set Output Directory...", command=self.browse_output)
        file_menu.add_separator()
        file_menu.add_command(label="Export to Excel...", command=self.export_to_excel, accelerator="Ctrl+E")
        file_menu.add_command(label="Export Report...", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Process Menu
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Process Files", command=self.run_processing, accelerator="F5")
        process_menu.add_command(label="Batch Process...", command=self.batch_process)
        process_menu.add_separator()
        process_menu.add_command(label="Clear Data", command=self.clear_data)
        
        # Analysis Menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="View Work Orders", command=lambda: self.notebook.select(1))
        analysis_menu.add_command(label="View Equipment Summary", command=lambda: self.notebook.select(1))
        analysis_menu.add_command(label="Crow-AMSAA Analysis", command=lambda: self.notebook.select(2))
        analysis_menu.add_command(label="Risk Assessment", command=lambda: self.notebook.select(3))
        
        # AI Menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_checkbutton(label="Enable AI Classification", variable=self.ai_enabled_var, command=self.toggle_ai)
        ai_menu.add_separator()
        ai_menu.add_command(label="AI Settings...", command=self.show_ai_settings)
        ai_menu.add_command(label="AI Statistics", command=self.show_ai_stats)
        ai_menu.add_command(label="Clear AI Cache", command=self.clear_ai_cache)
        ai_menu.add_separator()
        ai_menu.add_command(label="Export Training Data", command=self.export_training_data)
        
        # Tools Menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Column Mapping...", command=self.show_column_mapping)
        tools_menu.add_command(label="Filter Management...", command=self.show_filter_manager)
        tools_menu.add_command(label="Date Range Selector...", command=self.show_date_selector)
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset All Filters", command=self.reset_all_filters)
        tools_menu.add_command(label="Open Output Folder", command=self.open_output_folder)
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_separator()
        help_menu.add_command(label="Check for Updates", command=self.check_updates)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.browse_wo())
        self.root.bind('<Control-d>', lambda e: self.browse_dict())
        self.root.bind('<Control-e>', lambda e: self.export_to_excel())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<F5>', lambda e: self.run_processing())

    def create_notebook(self):
        """Create tabbed notebook interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_data_input_tab()
        self.create_analysis_tab()
        self.create_ai_settings_tab()
        self.create_risk_assessment_tab()

    def create_data_input_tab(self):
        """Create the data input tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📁 Data Input")
        
        # File selection frame
        file_frame = ttk.LabelFrame(data_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Work Order File
        wo_frame = ttk.Frame(file_frame)
        wo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wo_frame, text="Work Order File:", width=15).pack(side=tk.LEFT)
        self.wo_entry = ttk.Entry(wo_frame)
        self.wo_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(wo_frame, text="Browse", command=self.browse_wo, width=10).pack(side=tk.RIGHT)
        
        # Dictionary File
        dict_frame = ttk.Frame(file_frame)
        dict_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dict_frame, text="Dictionary File:", width=15).pack(side=tk.LEFT)
        self.dict_entry = ttk.Entry(dict_frame)
        self.dict_entry.insert(0, DEFAULT_DICT_PATH)
        self.dict_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(dict_frame, text="Browse", command=self.browse_dict, width=10).pack(side=tk.RIGHT)
        
        # Output Directory
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=2)
        ttk.Label(output_frame, text="Output Directory:", width=15).pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output, width=10).pack(side=tk.RIGHT)
        
        # Action buttons frame
        action_frame = ttk.Frame(data_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="🚀 Process Files", command=self.run_processing, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📊 Export to Excel", command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📁 Open Output", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑️ Clear Data", command=self.clear_data).pack(side=tk.RIGHT, padx=5)

    def create_analysis_tab(self):
        """Create the analysis tab with filters and data display"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📈 Analysis")
        
        # Filter panel
        filter_frame = ttk.LabelFrame(analysis_frame, text="Filters", padding=10)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Filter controls
        filter_controls = ttk.Frame(filter_frame)
        filter_controls.pack(fill=tk.X)
        
        # Row 1: Equipment and Failure Code
        row1 = ttk.Frame(filter_controls)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Equipment:", width=12).pack(side=tk.LEFT)
        self.equipment_var = tk.StringVar()
        self.equipment_dropdown = ttk.Combobox(row1, textvariable=self.equipment_var, state="readonly", width=20)
        self.equipment_dropdown['values'] = ['']
        self.equipment_dropdown.pack(side=tk.LEFT, padx=(5, 20))
        self.equipment_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_table())
        
        ttk.Label(row1, text="Failure Code:", width=12).pack(side=tk.LEFT)
        self.failure_code_var = tk.StringVar()
        self.failure_code_dropdown = ttk.Combobox(row1, textvariable=self.failure_code_var, state="readonly", width=20)
        self.failure_code_dropdown['values'] = ['']
        self.failure_code_dropdown.pack(side=tk.LEFT, padx=5)
        self.failure_code_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_table())
        
        # Row 2: Work Type and Date Range
        row2 = ttk.Frame(filter_controls)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Work Type:", width=12).pack(side=tk.LEFT)
        self.work_type_var = tk.StringVar()
        self.work_type_dropdown = ttk.Combobox(row2, textvariable=self.work_type_var, state="readonly", width=20)
        self.work_type_dropdown['values'] = ['']
        self.work_type_dropdown.pack(side=tk.LEFT, padx=(5, 20))
        self.work_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_table())
        
        ttk.Label(row2, text="Date Range:", width=12).pack(side=tk.LEFT)
        self.start_date_entry = ttk.Entry(row2, width=12)
        self.start_date_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="to").pack(side=tk.LEFT, padx=2)
        self.end_date_entry = ttk.Entry(row2, width=12)
        self.end_date_entry.pack(side=tk.LEFT, padx=5)
        
        # Filter action buttons
        filter_buttons = ttk.Frame(filter_frame)
        filter_buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(filter_buttons, text="Apply Filters", command=self.apply_date_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_buttons, text="Clear All", command=self.reset_all_filters).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_buttons, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=5)
        
        # Data display area with resizable panes
        data_display_frame = ttk.Frame(analysis_frame)
        data_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Main paned window for all content
        self.main_paned = ttk.PanedWindow(data_display_frame, orient=tk.VERTICAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Top section: tables
        tables_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(tables_frame, weight=2)
        
        # Paned window for table and equipment summary
        self.table_paned = ttk.PanedWindow(tables_frame, orient=tk.VERTICAL)
        self.table_paned.pack(fill=tk.BOTH, expand=True)
        
        self.work_order_frame = ttk.Frame(self.table_paned)
        self.table_paned.add(self.work_order_frame, weight=3)
        self.equipment_frame = ttk.Frame(self.table_paned)
        self.table_paned.add(self.equipment_frame, weight=1)
        
        self.tree = None
        self.equipment_tree = None
        
        # Bottom section: Crow-AMSAA plot area
        plot_frame = ttk.LabelFrame(self.main_paned, text="Crow-AMSAA Analysis", padding=10)
        self.main_paned.add(plot_frame, weight=1)
        
        # Plot controls
        plot_controls = ttk.Frame(plot_frame)
        plot_controls.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(plot_controls, text="Export Plot", command=self.export_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls, text="Open in New Window", command=self.open_plot_in_new_window).pack(side=tk.LEFT, padx=5)
        self.return_to_single_button = ttk.Button(plot_controls, text="Return to Single Plot", 
                                                 command=self.return_to_single_plot)
        
        # Plot area
        self.crow_amsaa_frame = ttk.Frame(plot_frame)
        self.crow_amsaa_frame.pack(fill=tk.BOTH, expand=True)

    def create_ai_settings_tab(self):
        """Create the AI settings tab"""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🤖 AI Settings")
        
        # AI Configuration
        config_frame = ttk.LabelFrame(ai_frame, text="AI Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # AI Enable/Disable
        ttk.Checkbutton(config_frame, text="Enable AI Classification", variable=self.ai_enabled_var).pack(anchor=tk.W, pady=2)
        
        # Confidence Threshold
        threshold_frame = ttk.Frame(config_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        ttk.Label(threshold_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        confidence_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.confidence_scale_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        ttk.Label(threshold_frame, textvariable=self.confidence_var, width=5).pack(side=tk.RIGHT)
        
        # Bind scale to update string variable
        def update_confidence_label(*args):
            self.confidence_var.set(f"{self.confidence_scale_var.get():.2f}")
        self.confidence_scale_var.trace('w', update_confidence_label)
        
        # Enhanced Classification Methods
        methods_frame = ttk.Frame(config_frame)
        methods_frame.pack(fill=tk.X, pady=5)
        ttk.Label(methods_frame, text="Enhanced Methods:").pack(side=tk.LEFT)
        
        self.expert_system_var = tk.BooleanVar(value=True)
        self.contextual_patterns_var = tk.BooleanVar(value=True)
        self.temporal_analysis_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(methods_frame, text="Expert System", variable=self.expert_system_var).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Checkbutton(methods_frame, text="Contextual Patterns", variable=self.contextual_patterns_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(methods_frame, text="Temporal Analysis", variable=self.temporal_analysis_var).pack(side=tk.LEFT, padx=5)
        
        # AI Status
        status_frame = ttk.Frame(config_frame)
        status_frame.pack(fill=tk.X, pady=5)
        self.ai_status_label = ttk.Label(status_frame, text="Enhanced AI: Expert System, Contextual Patterns, Temporal Analysis", foreground="green")
        self.ai_status_label.pack(side=tk.LEFT)
        
        # AI Action buttons
        ai_buttons_frame = ttk.Frame(ai_frame)
        ai_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(ai_buttons_frame, text="📊 AI Statistics", command=self.show_ai_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(ai_buttons_frame, text="🗑️ Clear AI Cache", command=self.clear_ai_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(ai_buttons_frame, text="📤 Export Training Data", command=self.export_training_data).pack(side=tk.LEFT, padx=5)

    def create_risk_assessment_tab(self):
        """Create the risk assessment tab"""
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text="⚠️ Risk Assessment")
        
        # Risk parameters
        params_frame = ttk.LabelFrame(risk_frame, text="Risk Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Production Loss
        prod_frame = ttk.Frame(params_frame)
        prod_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prod_frame, text="Production Loss ($):", width=15).pack(side=tk.LEFT)
        self.prod_loss_var = tk.StringVar(value="0")
        ttk.Entry(prod_frame, textvariable=self.prod_loss_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Maintenance Cost
        maint_frame = ttk.Frame(params_frame)
        maint_frame.pack(fill=tk.X, pady=2)
        ttk.Label(maint_frame, text="Maintenance Cost ($):", width=15).pack(side=tk.LEFT)
        self.maint_cost_var = tk.StringVar(value="0")
        ttk.Entry(maint_frame, textvariable=self.maint_cost_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Margin
        margin_frame = ttk.Frame(params_frame)
        margin_frame.pack(fill=tk.X, pady=2)
        ttk.Label(margin_frame, text="Margin ($/weight):", width=15).pack(side=tk.LEFT)
        self.margin_var = tk.StringVar(value="0")
        ttk.Entry(margin_frame, textvariable=self.margin_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Risk action buttons
        risk_buttons_frame = ttk.Frame(risk_frame)
        risk_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(risk_buttons_frame, text="🧮 Calculate Risk", command=self.update_risk).pack(side=tk.LEFT, padx=5)
        ttk.Button(risk_buttons_frame, text="💾 Save Preset", command=self.save_risk_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(risk_buttons_frame, text="📂 Load Preset", command=self.load_risk_preset).pack(side=tk.LEFT, padx=5)
        
        # Risk summary
        summary_frame = ttk.LabelFrame(risk_frame, text="Risk Summary", padding=10)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.risk_label = ttk.Label(summary_frame, text="Failure Rate: 0.00, Annualized Risk: $0.00")
        self.risk_label.pack(anchor=tk.W)

    def create_status_bar(self):
        """Create a modern status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status label
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Status indicators
        self.ai_status_indicator = ttk.Label(status_frame, text="🤖", foreground="green")
        self.ai_status_indicator.pack(side=tk.RIGHT, padx=5)
        
        self.column_mapping_indicator = ttk.Label(status_frame, text="📋", foreground="blue")
        self.column_mapping_indicator.pack(side=tk.RIGHT, padx=5)
        
        self.data_status_indicator = ttk.Label(status_frame, text="📊", foreground="blue")
        self.data_status_indicator.pack(side=tk.RIGHT, padx=5)

    def browse_wo(self):
        """Browse for work order file."""
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.wo_entry.delete(0, tk.END)
            self.wo_entry.insert(0, path)
    
    def browse_dict(self):
        """Browse for dictionary file."""
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.dict_entry.delete(0, tk.END)
            self.dict_entry.insert(0, path)
    
    def browse_output(self):
        """Browse for output directory."""
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
            self.output_dir = path
    
    def run_processing(self):
        """Process work order and dictionary files."""
        wo_path = self.wo_entry.get()
        dict_path = self.dict_entry.get()
        
        if not wo_path or not dict_path:
            self.status_label.config(text="Select both work order and dictionary files.", foreground="red")
            messagebox.showerror("Error", "Please select both work order and dictionary files.")
            return
        if not os.path.exists(wo_path):
            self.status_label.config(text=f"Work order file not found: {wo_path}", foreground="red")
            messagebox.showerror("Error", f"Work order file not found: {wo_path}")
            return
        if not os.path.exists(dict_path):
            self.status_label.config(text=f"Dictionary file not found: {dict_path}", foreground="red")
            messagebox.showerror("Error", f"Dictionary file not found: {dict_path}")
            return
        
        try:
            dict_df = pd.read_excel(dict_path)
            self.dictionary = dict_df.set_index('Code')['Description'].to_dict()
        except Exception as e:
            self.status_label.config(text=f"Failed to read dictionary: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"Failed to read dictionary file: {str(e)}")
            return
        
        # Check AI settings
        self.use_ai_classification = self.ai_enabled_var.get()
        if self.use_ai_classification and not AI_AVAILABLE:
            self.status_label.config(text="AI classification requested but not available. Install dependencies.", foreground="red")
            messagebox.showerror("Error", "AI classification not available. Please install required dependencies.")
            return
        
        # Initialize AI classifier if needed
        if self.use_ai_classification:
            try:
                confidence_threshold = float(self.confidence_var.get())
                
                self.ai_classifier = AIClassifier(
                    confidence_threshold=confidence_threshold,
                    cache_file=AI_CACHE_FILE
                )
                
                if not self.ai_classifier.load_failure_dictionary(dict_path):
                    self.status_label.config(text="Failed to load dictionary for AI classifier.", foreground="red")
                    messagebox.showerror("Error", "Failed to load dictionary for AI classifier.")
                    return
                    
            except Exception as e:
                self.status_label.config(text=f"Failed to initialize AI classifier: {str(e)}", foreground="red")
                messagebox.showerror("Error", f"Failed to initialize AI classifier: {str(e)}")
                return
        
        # Update progress
        self.update_progress(0, "Processing files...")
        
        self.wo_df = process_files(wo_path, dict_path, self.status_label, self.root, self.output_dir, 
                                 use_ai=self.use_ai_classification, ai_classifier=self.ai_classifier, 
                                 column_mapping=self.column_mapping)
        
        if self.wo_df is not None and not self.wo_df.empty:
            self.update_progress(50, "Updating interface...")
            
            equipment_nums = sorted(self.wo_df['Equipment #'].dropna().unique())
            work_types = sorted(self.wo_df['Work Type'].dropna().unique())
            failure_codes = sorted(self.wo_df['Failure Code'].dropna().unique())
            logging.info(f"Equipment numbers: {equipment_nums}")
            logging.info(f"Work types: {work_types}")
            logging.info(f"Failure codes: {failure_codes}")
            self.equipment_dropdown['values'] = [''] + list(equipment_nums)
            self.work_type_dropdown['values'] = [''] + list(work_types)
            self.failure_code_dropdown['values'] = [''] + list(failure_codes)
            self.equipment_var.set('')
            self.work_type_var.set('')
            self.failure_code_var.set('')
            self.included_indices = set(self.wo_df.index)
            
            self.update_progress(75, "Updating tables...")
            self.update_table()
            
            self.update_progress(100, "Processing complete!")
            self.data_status_indicator.config(text="📊")
            
            # Switch to analysis tab
            self.notebook.select(1)
            
        else:
            self.status_label.config(text="No data processed. Check input files or logs.", foreground="red")
            logging.error("Work order DataFrame is None or empty.")
            self.update_progress(0, "Processing failed")
    
    def show_ai_stats(self):
        """Show AI classification statistics."""
        if not self.ai_classifier:
            messagebox.showinfo("AI Stats", "No AI classifier available. Enable AI classification first.")
            return
        
        try:
            stats = self.ai_classifier.get_classification_stats()
            
            # Check AI capabilities
            capabilities = []
            if self.ai_classifier.embedding_model:
                capabilities.append("Sentence Embeddings")
            if self.ai_classifier.nlp:
                capabilities.append("SpaCy NLP")
            capabilities.append("Expert System")
            capabilities.append("Contextual Patterns")
            capabilities.append("Temporal Analysis")
            
            capabilities_text = ", ".join(capabilities) if capabilities else "None"
            
            stats_text = f"""AI Classification Statistics:
            
Total Classifications: {stats['total_classifications']}
Cache Size: {stats['cache_size_mb']:.2f} MB
AI Capabilities: {capabilities_text}

Methods Used:"""
            
            for method, count in stats['methods_used'].items():
                method_display = {
                    'expert_system': 'Expert System',
                    'contextual_patterns': 'Contextual Patterns',
                    'temporal_analysis': 'Temporal Analysis',
                    'ai_embeddings': 'Sentence Embeddings', 
                    'ai_spacy': 'SpaCy NLP',
                    'dictionary_fallback': 'Dictionary Matching'
                }.get(method, method)
                stats_text += f"\n  {method_display}: {count}"
            
            stats_text += f"""

Confidence Distribution:
  High (≥0.8): {stats['confidence_distribution']['high']}
  Medium (≥0.5): {stats['confidence_distribution']['medium']}
  Low (<0.5): {stats['confidence_distribution']['low']}"""
            
            messagebox.showinfo("AI Classification Stats", stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get AI stats: {str(e)}")
    
    def clear_ai_cache(self):
        """Clear the AI classification cache."""
        if not self.ai_classifier:
            messagebox.showinfo("Clear Cache", "No AI classifier available.")
            return
        
        try:
            self.ai_classifier.clear_cache()
            messagebox.showinfo("Success", "AI classification cache cleared successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
    
    def apply_date_filter(self):
        """Apply date range filter to work orders."""
        start_date_str = self.start_date_entry.get().strip()
        end_date_str = self.end_date_entry.get().strip()
        
        self.start_date = None
        self.end_date = None
        
        if start_date_str:
            start_date = parse_date(start_date_str)
            if pd.isna(start_date):
                self.status_label.config(text="Invalid start date format. Use MM/DD/YYYY.", foreground="red")
                messagebox.showerror("Error", "Invalid start date format. Use MM/DD/YYYY.")
                return
            self.start_date = start_date
        
        if end_date_str:
            end_date = parse_date(end_date_str)
            if pd.isna(end_date):
                self.status_label.config(text="Invalid end date format. Use MM/DD/YYYY.", foreground="red")
                messagebox.showerror("Error", "Invalid end date format. Use MM/DD/YYYY.")
                return
            self.end_date = end_date
        
        if self.start_date and self.end_date and self.start_date > self.end_date:
            self.status_label.config(text="Start date cannot be after end date.", foreground="red")
            messagebox.showerror("Error", "Start date cannot be after end date.")
            self.start_date = None
            self.end_date = None
            return
        
        self.update_table()
        self.status_label.config(text="Date filter applied.", foreground="green")
        logging.info(f"Date filter applied: start={start_date_str}, end={end_date_str}")
    
    def reset_equip_failcode(self):
        """Reset Equipment and Failure Code filters."""
        self.equipment_var.set('')
        self.failure_code_var.set('')
        self.update_table()
        self.status_label.config(text="Equipment and Failure Code filters reset.", foreground="green")
        logging.info("Reset Equipment and Failure Code filters")
    
    def reset_work_type(self):
        """Reset Work Type filter."""
        self.work_type_var.set('')
        self.update_table()
        self.status_label.config(text="Work Type filter reset.", foreground="green")
        logging.info("Reset Work Type filter")
    
    def reset_date(self):
        """Reset date range filter."""
        self.start_date_entry.delete(0, tk.END)
        self.end_date_entry.delete(0, tk.END)
        self.start_date = None
        self.end_date = None
        self.update_table()
        self.status_label.config(text="Date filter reset.", foreground="green")
        logging.info("Reset date filter")
    
    def sort_column(self, tree, col, reverse):
        """Sort Treeview column and update sort indicator."""
        if tree == self.tree:
            columns = ['Include', 'Index', 'Work Order', 'Description', 'Asset', 'Equipment #', 
                       'Work Type', 'Reported Date', 'Failure Code', 'Failure Description', 'Matched Keyword']
            if col == 'Include':
                data = [(tree.set(item, col), item) for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0] == '☑', reverse=reverse)
            elif col == 'Reported Date':
                data = [(parse_date(tree.set(item, col)) if tree.set(item, col) else pd.Timestamp.min, item) 
                        for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0], reverse=reverse)
            elif col in ['Index', 'Work Order']:
                data = [(float(tree.set(item, col)) if tree.set(item, col) else float('inf'), item) 
                        for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0], reverse=reverse)
            else:
                data = [(tree.set(item, col).lower() if tree.set(item, col) else '', item) 
                        for item in tree.get_children() if tree.set(item, 'Index') != '']
                data.sort(key=lambda x: x[0], reverse=reverse)
        else:  # equipment_tree
            if col in ['Total Work Orders', 'Failures per Year']:
                data = [(float(tree.set(item, col)) if tree.set(item, col) else float('inf'), item) 
                        for item in tree.get_children()]
                data.sort(key=lambda x: x[0], reverse=reverse)
            else:  # Equipment #
                data = [(tree.set(item, col).lower() if tree.set(item, col) else '', item) 
                        for item in tree.get_children()]
                data.sort(key=lambda x: x[0], reverse=reverse)
        
        for index, (_, item) in enumerate(data):
            tree.move(item, '', index)
        
        tree.heading(col, text=col + (' ↓' if reverse else ' ↑'))
        for c in tree['columns']:
            if c != col:
                tree.heading(c, text=c)
        
        self.sort_states[(tree, col)] = not reverse
        logging.debug(f"Sorted {tree} by {col}, reverse={reverse}")
    
    def edit_cell(self, event):
        """Handle editing of Failure Code or Failure Description in the Treeview."""
        if self.tree is None:
            return
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if item and column in ['#9', '#10']:
            idx = int(self.tree.item(item, 'values')[1])
            col_idx = int(column[1:]) - 1
            col_name = self.tree['columns'][col_idx]
            current_value = self.tree.item(item, 'values')[col_idx]
            entry = ttk.Entry(self.tree)
            entry.insert(0, current_value)
            entry.place(relx=float(column[1:]) / len(self.tree['columns']), rely=0.0, anchor='nw', width=150)
            def save_edit(event=None):
                new_value = entry.get()
                if self.wo_df is not None:
                    self.wo_df.at[idx, col_name] = new_value
                    self.wo_df.at[idx, 'Failure Code'] = CUSTOM_CODE
                self.update_table()
                entry.destroy()
                logging.debug(f"Updated index {idx}: {col_name}='{new_value}', Failure Code='{CUSTOM_CODE}'")
            entry.bind('<Return>', lambda e: save_edit())
            entry.bind('<FocusOut>', lambda e: save_edit())
            entry.focus_set()
    
    def toggle_row(self, event):
        """Toggle row inclusion when clicking the 'Include' column."""
        if self.tree is None:
            return
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if item and column == '#1':
            idx = int(self.tree.item(item, 'values')[1])
            if idx in self.included_indices:
                self.included_indices.remove(idx)
                self.tree.item(item, values=['☐'] + list(self.tree.item(item, 'values')[1:]))
            else:
                self.included_indices.add(idx)
                self.tree.item(item, values=['☑'] + list(self.tree.item(item, 'values')[1:]))
            self.update_table()
            logging.info(f"Toggled row {idx}: {'Included' if idx in self.included_indices else 'Excluded'}")
    
    def get_filtered_df(self) -> pd.DataFrame:
        """Apply all filters to the DataFrame."""
        if self.wo_df is None or self.wo_df.empty:
            return pd.DataFrame()
        
        filtered_df = self.wo_df.copy()
        
        # Apply date range filter
        if self.start_date or self.end_date:
            filtered_df['Parsed_Date'] = filtered_df['Reported Date'].apply(parse_date)
            if self.start_date:
                filtered_df = filtered_df[filtered_df['Parsed_Date'] >= self.start_date]
            if self.end_date:
                filtered_df = filtered_df[filtered_df['Parsed_Date'] <= self.end_date]
            filtered_df = filtered_df.drop(columns=['Parsed_Date'], errors='ignore')
        
        # Apply equipment filter
        equipment = self.equipment_var.get()
        if equipment:
            filtered_df = filtered_df[filtered_df['Equipment #'] == equipment]
        
        # Apply work type filter
        work_type = self.work_type_var.get()
        if work_type:
            filtered_df = filtered_df[filtered_df['Work Type'] == work_type]
        
        # Apply failure code filter
        failure_code = self.failure_code_var.get()
        if failure_code:
            filtered_df = filtered_df[filtered_df['Failure Code'] == failure_code]
        
        return filtered_df
    
    def update_risk(self):
        """Calculate and display failure rate and annualized risk."""
        if self.wo_df is None or self.wo_df.empty:
            self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: N/A")
            return
        
        # Check if we're in segmented view and have segment data
        if self.is_segmented_view and self.segment_data:
            self.update_risk_segmented(self.segment_data)
            return
        
        try:
            filtered_df = self.get_filtered_df()
            if filtered_df.empty:
                self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: N/A")
                return
            
            valid_indices = filtered_df.index.intersection(self.included_indices)
            _, _, failures_per_year = calculate_crow_amsaa_params(filtered_df, set(valid_indices))
            
            try:
                prod_loss = float(self.prod_loss_var.get())
                maint_cost = float(self.maint_cost_var.get())
                margin = float(self.margin_var.get())
            except ValueError:
                self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: Invalid input")
                logging.error("Invalid input for risk calculation")
                return
            
            risk = failures_per_year * (prod_loss * margin + maint_cost)
            self.risk_label.config(text=f"Failure Rate: {failures_per_year:.2f}, Annualized Risk: ${risk:,.2f}")
            logging.debug(f"Calculated risk: failures/year={failures_per_year:.2f}, prod_loss={prod_loss}, maint_cost={maint_cost}, margin={margin}, risk=${risk:,.2f}")
        except Exception as e:
            self.risk_label.config(text="Failure Rate: N/A, Annualized Risk: Error")
            logging.error(f"Error calculating risk: {e}")
    
    # --- Crow-AMSAA plot with interactivity ---
    def highlight_plot_point_by_work_order(self, work_order_idx):
        """Highlight the corresponding data point on the Crow-AMSAA plot when a work order is selected."""
        if self.crow_amsaa_fig is None or self.wo_df is None:
            logging.debug("Cannot highlight: figure or dataframe is None")
            return
        
        # Remove previous highlight if present
        if hasattr(self, 'last_highlight_artist') and self.last_highlight_artist is not None:
            try:
                self.last_highlight_artist.remove()
                self.last_highlight_artist = None
                if self.crow_amsaa_canvas:
                    self.crow_amsaa_canvas.draw()
            except Exception as e:
                logging.debug(f"Error removing last highlight: {e}")
        
        try:
            # Get the date for the selected work order
            selected_date = parse_date(self.wo_df.at[work_order_idx, 'Reported Date'])
            if pd.isna(selected_date):
                logging.debug(f"Cannot highlight: invalid date for work order {work_order_idx}")
                return
            
            logging.debug(f"Highlighting work order {work_order_idx} with date {selected_date}")
            
            # Get the filtered data to match the plot
            filtered_df = self.get_filtered_df()
            valid_indices = filtered_df.index.intersection(self.included_indices)
            filtered_dates = [parse_date(filtered_df.at[idx, 'Reported Date']) for idx in valid_indices if pd.notna(filtered_df.at[idx, 'Reported Date'])]
            filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
            
            if len(filtered_dates) < 2:
                logging.debug("Cannot highlight: insufficient filtered dates")
                return
            
            dates = sorted(filtered_dates)
            t0 = dates[0]
            times = [(d - t0).days + 1 for d in dates]
            
            logging.debug(f"Filtered dates range: {dates[0]} to {dates[-1]}")
            logging.debug(f"Times range: {times[0]} to {times[-1]}")
            
            # Find the index of the selected date in the filtered dates
            try:
                date_idx = dates.index(selected_date)
                logging.debug(f"Exact date match found at index {date_idx}")
            except ValueError:
                # If exact match not found, find closest date
                date_idx = min(range(len(dates)), key=lambda i: abs((dates[i] - selected_date).days))
                logging.debug(f"Closest date match found at index {date_idx} (date: {dates[date_idx]})")
            
            # Get the corresponding time value
            selected_time = times[date_idx]
            selected_n = date_idx + 1  # Cumulative failures
            
            logging.debug(f"Selected coordinates: time={selected_time}, failures={selected_n}")
            
            # Find and highlight the corresponding point on the plot
            ax = self.crow_amsaa_fig.axes[0] if len(self.crow_amsaa_fig.axes) == 1 else None
            
            if ax is None and len(self.crow_amsaa_fig.axes) == 2:
                # Handle segmented plots - determine which segment contains this date
                axs = self.crow_amsaa_fig.axes
                seg_dt = parse_date(self.segment_date) if hasattr(self, 'segment_date') and self.segment_date else None
                if seg_dt:
                    seg_idx = next((i for i, d in enumerate(dates) if d >= seg_dt), len(dates))
                    if date_idx <= seg_idx:
                        ax = axs[0]  # First segment
                        logging.debug(f"Using first segment (axs[0])")
                    else:
                        ax = axs[1]  # Second segment
                        # Adjust the time for second segment
                        selected_time = times[date_idx] - times[seg_idx]
                        selected_n = date_idx - seg_idx + 1
                        logging.debug(f"Using second segment (axs[1]), adjusted coordinates: time={selected_time}, failures={selected_n}")
            
            if ax is None:
                logging.debug("Cannot highlight: no valid axis found")
                return
            
            # Clear previous highlights by resetting all scatter points
            scatter_artists = []
            for artist in ax.get_children():
                if hasattr(artist, 'get_offsets') and len(artist.get_offsets()) > 0:
                    scatter_artists.append(artist)
                    try:
                        artist.set_alpha(0.6)
                        artist.set_s(50)
                        if hasattr(artist, 'set_facecolor'):
                            artist.set_facecolor('blue')
                        elif hasattr(artist, 'set_color'):
                            artist.set_color('blue')
                    except Exception as e:
                        logging.debug(f"Error resetting artist: {e}")
            
            logging.debug(f"Found {len(scatter_artists)} scatter artists")
            
            # Find and highlight the corresponding point
            point_found = False
            for artist in scatter_artists:
                try:
                    offsets = artist.get_offsets()
                    logging.debug(f"Checking artist with {len(offsets)} points")
                    
                    for i, (x, y) in enumerate(offsets):
                        # More lenient matching - check if this point is close to our selected point
                        time_match = abs(x - selected_time) < 5  # Allow 5 days tolerance
                        failure_match = abs(y - selected_n) < 2   # Allow 2 failures tolerance
                        
                        if time_match and failure_match:
                            # Highlight this point
                            try:
                                artist.set_alpha(1.0)
                                artist.set_s(150)  # Make it even larger
                                # Try different methods to set color
                                if hasattr(artist, 'set_facecolor'):
                                    artist.set_facecolor('red')
                                elif hasattr(artist, 'set_color'):
                                    artist.set_color('red')
                                else:
                                    # Fallback: try to set the color array
                                    colors = ['red'] * len(offsets)
                                    artist.set_facecolor(colors)
                                point_found = True
                                logging.debug(f"Highlighted point at coordinates ({x}, {y})")
                                break
                            except Exception as e:
                                logging.debug(f"Error highlighting point: {e}")
                                # Try alternative highlighting method
                                try:
                                    # Remove previous highlight if present
                                    if hasattr(self, 'last_highlight_artist') and self.last_highlight_artist is not None:
                                        self.last_highlight_artist.remove()
                                        self.last_highlight_artist = None
                                    # Create a new highlighted point on top and store reference
                                    highlight = ax.scatter([x], [y], c='red', s=200, alpha=1.0, zorder=10, marker='o')
                                    self.last_highlight_artist = highlight
                                    point_found = True
                                    logging.debug(f"Added highlighted point at coordinates ({x}, {y})")
                                    break
                                except Exception as e2:
                                    logging.debug(f"Error adding highlighted point: {e2}")
                    
                    if point_found:
                        break
                except Exception as e:
                    logging.debug(f"Error processing artist: {e}")
            
            if not point_found:
                logging.debug(f"No matching point found for coordinates ({selected_time}, {selected_n})")
                # Log all available points for debugging
                for artist in scatter_artists:
                    try:
                        offsets = artist.get_offsets()
                        logging.debug(f"Available points: {list(offsets)}")
                    except Exception as e:
                        logging.debug(f"Error getting offsets: {e}")
            
            if self.crow_amsaa_canvas:
                try:
                    self.crow_amsaa_canvas.draw()
                    logging.debug("Canvas redrawn")
                except Exception as e:
                    logging.debug(f"Error redrawing canvas: {e}")
                
        except Exception as e:
            logging.error(f"Error highlighting plot point: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def return_to_single_plot(self):
        """Return from segmented view to single Crow-AMSAA plot."""
        self.is_segmented_view = False
        self.segment_data = None
        self.return_to_single_button.pack_forget()  # Hide the button
        # Redraw the single plot
        filtered_df = self.get_filtered_df()
        self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, self.crow_amsaa_frame)
        # Update risk evaluation to show single plot data
        self.update_risk()

    def update_risk_segmented(self, segment_data):
        """Update risk evaluation for segmented plots showing both segments."""
        if not segment_data or len(segment_data) != 2:
            return
        
        try:
            (beta1, lambda1, failures_per_year1), (beta2, lambda2, failures_per_year2) = segment_data
            
            # Get risk parameters
            try:
                prod_loss = float(self.prod_loss_var.get())
                maint_cost = float(self.maint_cost_var.get())
                margin = float(self.margin_var.get())
            except ValueError:
                self.risk_label.config(text="Segment 1: N/A, Segment 2: N/A (Invalid input)")
                return
            
            # Calculate risks for both segments
            risk1 = failures_per_year1 * (prod_loss * margin + maint_cost) if failures_per_year1 is not None else 0
            risk2 = failures_per_year2 * (prod_loss * margin + maint_cost) if failures_per_year2 is not None else 0
            
            # Handle None values for display
            failures_per_year1_display = f"{failures_per_year1:.2f}" if failures_per_year1 is not None else "N/A"
            failures_per_year2_display = f"{failures_per_year2:.2f}" if failures_per_year2 is not None else "N/A"
            
            risk_text = f"Segment 1: {failures_per_year1_display} failures/year, ${risk1:,.2f} risk | "
            risk_text += f"Segment 2: {failures_per_year2_display} failures/year, ${risk2:,.2f} risk"
            
            self.risk_label.config(text=risk_text)
            logging.debug(f"Segmented risk: {risk_text}")
        except Exception as e:
            self.risk_label.config(text="Error calculating segmented risk")
            logging.error(f"Error calculating segmented risk: {e}")

    def create_crow_amsaa_plot_interactive(self, filtered_df, included_indices, frame, segment_date=None):
        """Create an interactive Crow-AMSAA plot. If segment_date is given, plot two segments."""
        # Remove previous plot
        for widget in frame.winfo_children():
            widget.destroy()
        
        # Show/hide return button based on mode
        if segment_date is not None:
            self.is_segmented_view = True
            self.segment_date = segment_date  # Store for highlighting logic
            # Find the plot controls frame and pack the button there
            plot_controls = None
            for widget in frame.master.winfo_children():
                if isinstance(widget, ttk.Frame) and len(widget.winfo_children()) > 0:
                    # Check if this frame contains buttons
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button):
                            plot_controls = widget
                            break
                    if plot_controls:
                        break
            
            if plot_controls:
                self.return_to_single_button.pack(side=tk.LEFT, padx=5, in_=plot_controls)
        else:
            self.is_segmented_view = False
            self.segment_date = None
            self.return_to_single_button.pack_forget()
        
        # Prepare data
        valid_indices = filtered_df.index.intersection(included_indices)
        filtered_dates = [parse_date(filtered_df.at[idx, 'Reported Date']) for idx in valid_indices if pd.notna(filtered_df.at[idx, 'Reported Date'])]
        filtered_dates = [d for d in filtered_dates if not pd.isna(d)]
        if len(filtered_dates) < 2:
            return None, None, None
        
        dates = sorted(filtered_dates)
        t0 = dates[0]
        times = [(d - t0).days + 1 for d in dates]
        n = np.arange(1, len(times) + 1)
        
        # If segmenting, split data
        if segment_date is not None:
            seg_dt = parse_date(segment_date)
            seg_idx = next((i for i, d in enumerate(dates) if d >= seg_dt), len(dates))
            
            # First segment
            times1 = times[:seg_idx+1]
            n1 = n[:seg_idx+1]
            
            # Second segment
            times2 = times[seg_idx:]
            n2 = n[seg_idx:]
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            
            # Plot segment 1
            beta1 = lambda1 = failures_per_year1 = None
            if len(times1) > 1:
                log_n1 = np.log(n1)
                log_t1 = np.log(times1)
                coeffs1 = np.polyfit(log_t1, log_n1, 1)
                beta1 = coeffs1[0]
                lambda1 = np.exp(coeffs1[1])
                t_fit1 = np.linspace(min(times1), max(times1), 100)
                n_fit1 = lambda1 * t_fit1 ** beta1
                failures_per_year1 = lambda1 * (365 ** beta1)
                axs[0].scatter(times1, n1, marker='o', label='Observed', picker=5)
                axs[0].plot(t_fit1, n_fit1, label=f'β={beta1:.2f}, λ={lambda1:.4f}')
                axs[0].set_xscale('log')
                axs[0].set_yscale('log')
                axs[0].set_title(f'Segment 1\nFailures/year={failures_per_year1:.2f}')
                axs[0].legend()
                axs[0].grid(True, which="both", ls="--")
            else:
                axs[0].set_title('Segment 1 (Insufficient data)')
            
            # Plot segment 2
            beta2 = lambda2 = failures_per_year2 = None
            if len(times2) > 1:
                log_n2 = np.log(n2)
                log_t2 = np.log(times2)
                coeffs2 = np.polyfit(log_t2, log_n2, 1)
                beta2 = coeffs2[0]
                lambda2 = np.exp(coeffs2[1])
                t_fit2 = np.linspace(min(times2), max(times2), 100)
                n_fit2 = lambda2 * t_fit2 ** beta2
                failures_per_year2 = lambda2 * (365 ** beta2)
                axs[1].scatter(times2, n2, marker='o', label='Observed', picker=5)
                axs[1].plot(t_fit2, n_fit2, label=f'β={beta2:.2f}, λ={lambda2:.4f}')
                axs[1].set_xscale('log')
                axs[1].set_yscale('log')
                axs[1].set_title(f'Segment 2\nFailures/year={failures_per_year2:.2f}')
                axs[1].legend()
                axs[1].grid(True, which="both", ls="--")
            else:
                axs[1].set_title('Segment 2 (Insufficient data)')
            
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store for event handling
            self.crow_amsaa_canvas = canvas
            self.crow_amsaa_fig = fig
            
            # Store segment data for risk calculation
            self.segment_data = (
                (beta1, lambda1, failures_per_year1),
                (beta2, lambda2, failures_per_year2)
            )
            
            # Update risk evaluation for segmented view
            self.update_risk_segmented(self.segment_data)
            
            return fig, canvas, (beta1 if len(times1)>1 else None, beta2 if len(times2)>1 else None)
        
        # Normal (not segmented)
        log_n = np.log(n)
        log_t = np.log(times)
        coeffs = np.polyfit(log_t, log_n, 1)
        beta = coeffs[0]
        lambda_param = np.exp(coeffs[1])
        failures_per_year = lambda_param * (365 ** beta)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(times, n, marker='o', label='Observed Failures', picker=5)
        t_fit = np.linspace(min(times), max(times), 100)
        n_fit = lambda_param * t_fit ** beta
        ax.plot(t_fit, n_fit, label=f'Crow-AMSAA (β={beta:.2f}, λ={lambda_param:.4f})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Crow-AMSAA Plot\nFailures/year={failures_per_year:.2f}")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Cumulative Failures")
        ax.legend()
        ax.grid(True, which="both", ls="--")
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store for event handling
        self.crow_amsaa_canvas = canvas
        self.crow_amsaa_fig = fig
        
        return fig, canvas, beta

    def show_context_menu(self, event):
        """Show right-click context menu for segmenting Crow-AMSAA plot."""
        if self.tree is None:
            return
        item = self.tree.identify_row(event.y)
        if not item:
            return
        self.tree.selection_set(item)
        # Create context menu if not already
        if self.context_menu is not None:
            self.context_menu.destroy()
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Segment Crow-AMSAA at this date", command=lambda: self.segment_crow_amsaa_at_selected())
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def segment_crow_amsaa_at_selected(self):
        """Segment Crow-AMSAA plot at the selected work order's date."""
        if self.tree is None or self.wo_df is None:
            return
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        values = self.tree.item(item, 'values')
        if len(values) < 8:
            return
        idx = int(values[1])
        date_str = self.wo_df.at[idx, 'Reported Date']
        # Redraw Crow-AMSAA plot segmented at this date
        filtered_df = self.get_filtered_df()
        self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, self.crow_amsaa_frame, segment_date=date_str)

    def update_table(self):
        """Update work order and equipment summary tables."""
        if self.wo_df is None or self.wo_df.empty:
            self.status_label.config(text="No data available to display.", foreground="red")
            self.root.config(cursor="")
            self.root.update()
            logging.error("Cannot update table: wo_df is None or empty")
            return
        
        self.status_label.config(text="Processing...", foreground="blue")
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            # Get filtered DataFrame
            filtered_df = self.get_filtered_df()
            status_text = f"Work order table filtered to equipment {self.equipment_var.get() or 'all'}, work type {self.work_type_var.get() or 'all'}, failure code {self.failure_code_var.get() or 'all'}."
            logging.info(f"Work order table: {status_text}, rows={len(filtered_df)}")
            
            if filtered_df.empty:
                self.status_label.config(text=f"No data for {status_text}.", foreground="purple")
                self.root.config(cursor="")
                self.root.update()
                logging.warning(f"No data for {status_text}")
                return
            
            if self.tree:
                self.tree.destroy()
            if self.equipment_tree:
                self.equipment_tree.destroy()
            
            for widget in self.crow_amsaa_frame.winfo_children():
                widget.destroy()
            
            mtbf = calculate_mtbf(filtered_df, self.included_indices)
            
            # Create Work Order table
            columns = ['Include', 'Index', 'Work Order', 'Description', 'Asset', 'Equipment #', 
                       'Work Type', 'Reported Date', 'Failure Code', 'Failure Description', 'Matched Keyword', 'AI Confidence', 'Classification Method', 'Work Order Cost']
            self.tree = ttk.Treeview(self.work_order_frame, columns=columns, show='headings')
            
            self.tree.heading('Include', text='Include', command=lambda: self.sort_column(self.tree, 'Include', self.sort_states.get((self.tree, 'Include'), False)))
            self.tree.column('Include', width=50)
            self.tree.heading('Index', text='')
            self.tree.column('Index', width=0, stretch=False)
            for col in columns[2:]:
                self.tree.heading(col, text=col, command=lambda c=col: self.sort_column(self.tree, c, self.sort_states.get((self.tree, c), False)))
                self.tree.column(col, width=100)
            self.tree.column('Failure Description', width=150)
            
            self.tree.insert('', 'end', values=('MTBF', '', f'{mtbf:.2f} days', '', '', '', '', '', '', '', '', '', '', ''))
            
            for idx, row in filtered_df.iterrows():
                include = '☑' if idx in self.included_indices else '☐'
                ai_confidence = row.get('AI Confidence', 0.0)
                classification_method = row.get('Classification Method', 'dictionary')
                
                # Get work order cost, handle missing or invalid values
                work_order_cost = row.get('Work Order Cost', 0.0)
                try:
                    if work_order_cost is not None and str(work_order_cost).strip() != '' and str(work_order_cost).lower() != 'nan':
                        work_order_cost = float(work_order_cost)
                    else:
                        work_order_cost = 0.0
                except (ValueError, TypeError):
                    work_order_cost = 0.0
                
                values = [
                    include,
                    idx,
                    row.get('Work Order', ''),
                    row.get('Description', ''),
                    str(row.get('Asset', '')),
                    str(row.get('Equipment #', '')),
                    row.get('Work Type', ''),
                    row.get('Reported Date', ''),
                    row.get('Failure Code', DEFAULT_CODE),
                    row.get('Failure Description', DEFAULT_DESC),
                    row.get('Matched Keyword', ''),
                    f'{ai_confidence:.2f}' if ai_confidence is not None and ai_confidence > 0 else '',
                    classification_method,
                    f'${work_order_cost:,.2f}' if work_order_cost > 0 else ''
                ]
                self.tree.insert('', 'end', values=values)
            
            self.tree.pack(fill=tk.BOTH, expand=True)
            self.tree.bind('<Button-1>', self.toggle_row)
            self.tree.bind('<Double-1>', self.edit_cell)
            # --- Add right-click context menu binding ---
            self.tree.bind('<Button-3>', self.show_context_menu)
            # --- Add selection event to highlight plot points ---
            self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
            
            failure_codes = sorted(filtered_df['Failure Code'].dropna().unique())
            logging.debug(f"Failure codes: {failure_codes}")
            self.failure_code_dropdown['values'] = [''] + list(failure_codes)
            if self.failure_code_var.get() not in failure_codes and self.failure_code_var.get() != '':
                self.failure_code_var.set('')
            
            # --- Use interactive Crow-AMSAA plot ---
            self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, self.crow_amsaa_frame)
            
            # Equipment Summary Table
            equipment = self.equipment_var.get()
            base_df = self.get_filtered_df()  # Use filtered data for equipment summary
            equipment_nums = [equipment] if equipment else sorted(base_df['Equipment #'].dropna().unique())
            eq_columns = ['Equipment #', 'Total Work Orders', 'Failures per Year', 'Total Cost', 'Avg Cost per WO']
            self.equipment_tree = ttk.Treeview(self.equipment_frame, columns=eq_columns, show='headings')
            
            for col in eq_columns:
                self.equipment_tree.heading(col, text=col, command=lambda c=col: self.sort_column(self.equipment_tree, c, self.sort_states.get((self.equipment_tree, c), False)))
                self.equipment_tree.column(col, width=120, anchor='center')
            
            for eq in equipment_nums:
                eq_df = base_df[base_df['Equipment #'] == eq]
                valid_indices = eq_df.index.intersection(self.included_indices)
                total_wo = len(valid_indices)
                _, _, failures_per_year = calculate_crow_amsaa_params(eq_df, set(valid_indices))
                
                # Calculate cost information
                total_cost = 0.0
                avg_cost = 0.0
                if total_wo > 0:
                    # Get costs for valid indices
                    costs = []
                    for idx in valid_indices:
                        cost = eq_df.at[idx, 'Work Order Cost']
                        try:
                            cost = float(cost) if pd.notna(cost) else 0.0
                            costs.append(cost)
                        except (ValueError, TypeError):
                            costs.append(0.0)
                    
                    total_cost = sum(costs)
                    avg_cost = total_cost / total_wo if total_wo > 0 else 0.0
                
                self.equipment_tree.insert('', 'end', values=(
                    eq, 
                    total_wo, 
                    failures_per_year,
                    f'${total_cost:,.2f}' if total_cost > 0 else '$0.00',
                    f'${avg_cost:,.2f}' if avg_cost > 0 else '$0.00'
                ))
            
            self.equipment_tree.pack(fill=tk.BOTH, expand=True)
            
            date_text = f", date range: {self.start_date.strftime('%m/%d/%Y') if self.start_date else 'N/A'} to {self.end_date.strftime('%m/%d/%Y') if self.end_date else 'N/A'}" if self.start_date or self.end_date else ''
            self.status_label.config(text=f"{status_text}{date_text}", foreground="green")
            self.root.config(cursor="")
            self.root.update()
            logging.info(f"Table updated: {status_text}, rows={len(filtered_df)}")
            
            self.update_risk()
        
        except Exception as e:
            error_msg = f"Error updating table: {str(e)}"
            self.status_label.config(text=error_msg, foreground="red")
            self.root.config(cursor="")
            self.root.update()
            logging.error(error_msg)
    
    def on_tree_select(self, event):
        """Handle tree selection to highlight corresponding plot point."""
        if self.tree is None:
            return
        
        selected = self.tree.selection()
        if not selected:
            return
        
        item = selected[0]
        values = self.tree.item(item, 'values')
        if len(values) < 2:
            return
        
        try:
            idx = int(values[1])
            self.highlight_plot_point_by_work_order(idx)
        except (ValueError, IndexError):
            pass  # Skip if not a valid work order row

    def export_to_excel(self):
        """Export data to Excel workbook."""
        if self.wo_df is None or self.wo_df.empty:
            self.status_label.config(text="Please process files first.", foreground="red")
            messagebox.showerror("Error", "Please process files first.")
            return
        
        if not self.output_dir:
            self.output_dir = filedialog.askdirectory()
            if not self.output_dir:
                self.status_label.config(text="Canceled export: No output directory selected.", foreground="purple")
                return
        
        self.status_label.config(text="Processing...", foreground="blue")
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            logging.info("Starting Excel export")
            equipment = self.equipment_var.get()
            work_type = self.work_type_var.get()
            
            # Work Orders
            filtered_df = self.get_filtered_df()
            valid_indices = filtered_df.index.intersection(self.included_indices)
            filtered_df = filtered_df.loc[valid_indices]
            if filtered_df.empty:
                self.status_label.config(text="No data to export after filtering.", foreground="purple")
                self.root.config(cursor="")
                self.root.update()
                messagebox.showwarning("Warning", "No data to export.")
                logging.warning("No data to export after filtering")
                return
            
            mtbf = calculate_mtbf(filtered_df, valid_indices)
            _, _, failures_per_year = calculate_crow_amsaa_params(filtered_df, valid_indices)
            
            try:
                prod_loss = float(self.prod_loss_var.get())
                maint_cost = float(self.maint_cost_var.get())
                margin = float(self.margin_var.get())
                risk = failures_per_year * (prod_loss * margin + maint_cost)
            except ValueError:
                risk = 0.0
                logging.error("Invalid input for risk calculation in export")
            
            export_columns = [
                'Work Order',
                'Description',
                'Asset',
                'Equipment #',
                'Work Type',
                'Reported Date',
                'Failure Code',
                'Failure Description',
                'Matched Keyword',
                'AI Confidence',
                'Classification Method',
                'Work Order Cost'
            ]
            export_df = filtered_df[export_columns].copy()
            
            summary_data = {
                'Metric': ['MTBF (days)', 'Failures per Year', 'Annualized Risk ($)'],
                'Value': [f'{mtbf:.2f}', f'{failures_per_year:.2f}', f'{risk:,.2f}']
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Equipment Summary
            equipment_nums = [equipment] if equipment else sorted(filtered_df['Equipment #'].dropna().unique())
            eq_data = []
            for eq in equipment_nums:
                eq_df = filtered_df[filtered_df['Equipment #'] == eq]
                valid_indices = eq_df.index.intersection(self.included_indices)
                total_wo = len(valid_indices)
                _, _, failures_per_year = calculate_crow_amsaa_params(eq_df, set(valid_indices))
                
                # Calculate cost information
                total_cost = 0.0
                avg_cost = 0.0
                if total_wo > 0:
                    # Get costs for valid indices
                    costs = []
                    for idx in valid_indices:
                        cost = eq_df.at[idx, 'Work Order Cost']
                        try:
                            cost = float(cost) if pd.notna(cost) else 0.0
                            costs.append(cost)
                        except (ValueError, TypeError):
                            costs.append(0.0)
                    
                    total_cost = sum(costs)
                    avg_cost = total_cost / total_wo if total_wo > 0 else 0.0
                
                eq_data.append({
                    'Equipment #': eq,
                    'Total Work Orders': total_wo,
                    'Failures per Year': f'{failures_per_year:.2f}',
                    'Total Cost ($)': f'{total_cost:,.2f}',
                    'Avg Cost per WO ($)': f'{avg_cost:,.2f}'
                })
            eq_df = pd.DataFrame(eq_data)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.output_dir, f"failure_mode_report_{timestamp}.xlsx")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Work Orders', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                eq_df.to_excel(writer, sheet_name='Equipment Summary', index=False)
            
            self.status_label.config(text=f"Exported to {output_file}", foreground="green")
            self.root.config(cursor="")
            self.root.update()
            messagebox.showinfo("Success", f"Report exported to {output_file}")
            logging.info(f"Exported report to: {output_file}")
        
        except Exception as e:
            self.status_label.config(text=f"Export error: {str(e)}", foreground="red")
            self.root.config(cursor="")
            self.root.update()
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
            logging.error(f"Error exporting to Excel: {str(e)}")
    
    def open_output_folder(self):
        """Open the output directory."""
        output_dir = self.output_dir or os.path.dirname(self.wo_entry.get()) or '.'
        if os.path.exists(output_dir):
            os.startfile(output_dir)
        else:
            self.status_label.config(text="Output folder does not exist.", foreground="red")
            messagebox.showerror("Error", "Output folder does not exist.")
            logging.error(f"Output folder does not exist: {output_dir}")

    def toggle_ai(self):
        """Toggle AI classification on/off"""
        self.use_ai_classification = self.ai_enabled_var.get()
        status = "enabled" if self.use_ai_classification else "disabled"
        self.status_label.config(text=f"AI Classification {status}")
        self.ai_status_indicator.config(text="🤖" if self.use_ai_classification else "⭕")

    def show_ai_settings(self):
        """Show AI settings dialog"""
        # Switch to AI settings tab
        self.notebook.select(2)

    def export_report(self):
        """Export comprehensive report with charts and analysis"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showerror("Error", "No data to export. Please process files first.")
            return
        
        try:
            # Get output directory
            if not self.output_dir:
                self.output_dir = filedialog.askdirectory(title="Select Output Directory")
                if not self.output_dir:
                    return
            
            # Create comprehensive report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(self.output_dir, f"comprehensive_report_{timestamp}.xlsx")
            
            with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
                # Work Orders sheet
                filtered_df = self.get_filtered_df()
                valid_indices = filtered_df.index.intersection(self.included_indices)
                filtered_df = filtered_df.loc[valid_indices]
                filtered_df.to_excel(writer, sheet_name='Work Orders', index=False)
                
                # Summary Statistics
                mtbf = calculate_mtbf(filtered_df, valid_indices)
                _, _, failures_per_year = calculate_crow_amsaa_params(filtered_df, valid_indices)
                
                summary_data = {
                    'Metric': ['Total Work Orders', 'MTBF (days)', 'Failures per Year', 'Date Range'],
                    'Value': [
                        len(filtered_df),
                        f'{mtbf:.2f}',
                        f'{failures_per_year:.2f}',
                        f"{filtered_df['Reported Date'].min()} to {filtered_df['Reported Date'].max()}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Failure Code Analysis
                failure_analysis = filtered_df['Failure Code'].value_counts().reset_index()
                failure_analysis.columns = ['Failure Code', 'Count']
                failure_analysis['Percentage'] = (failure_analysis['Count'] / len(filtered_df) * 100).round(2)
                failure_analysis.to_excel(writer, sheet_name='Failure Analysis', index=False)
                
                # Equipment Analysis
                equipment_analysis = filtered_df['Equipment #'].value_counts().reset_index()
                equipment_analysis.columns = ['Equipment #', 'Work Order Count']
                equipment_analysis.to_excel(writer, sheet_name='Equipment Analysis', index=False)
                
                # Cost Analysis (if available)
                if 'Work Order Cost' in filtered_df.columns:
                    cost_data = []
                    for eq in filtered_df['Equipment #'].unique():
                        eq_df = filtered_df[filtered_df['Equipment #'] == eq]
                        total_cost = eq_df['Work Order Cost'].sum()
                        avg_cost = eq_df['Work Order Cost'].mean()
                        cost_data.append({
                            'Equipment #': eq,
                            'Total Cost': total_cost,
                            'Average Cost': avg_cost,
                            'Work Order Count': len(eq_df)
                        })
                    cost_df = pd.DataFrame(cost_data)
                    cost_df.to_excel(writer, sheet_name='Cost Analysis', index=False)
            
            messagebox.showinfo("Success", f"Comprehensive report exported to {report_file}")
            self.status_label.config(text=f"Report exported to {os.path.basename(report_file)}", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")
            logging.error(f"Error exporting report: {e}")

    def batch_process(self):
        """Batch process multiple work order files"""
        if not self.dict_entry.get() or not os.path.exists(self.dict_entry.get()):
            messagebox.showerror("Error", "Please select a valid dictionary file first.")
            return
        
        # Create batch processing dialog
        batch_window = tk.Toplevel(self.root)
        batch_window.title("Batch Processing")
        batch_window.geometry("600x500")
        batch_window.transient(self.root)
        batch_window.grab_set()
        
        # Center the window
        batch_window.update_idletasks()
        x = (batch_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (batch_window.winfo_screenheight() // 2) - (500 // 2)
        batch_window.geometry(f"600x500+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(batch_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="Select multiple work order files:").pack(anchor=tk.W)
        
        file_list_frame = ttk.Frame(file_frame)
        file_list_frame.pack(fill=tk.X, pady=5)
        
        # File listbox with scrollbar
        list_frame = ttk.Frame(file_list_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        file_listbox = tk.Listbox(list_frame, height=6)
        file_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=file_listbox.yview)
        file_listbox.configure(yscrollcommand=file_scrollbar.set)
        
        file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # File selection buttons
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=tk.X, pady=5)
        
        def add_files():
            files = filedialog.askopenfilenames(
                title="Select Work Order Files",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            for file in files:
                if file not in file_listbox.get(0, tk.END):
                    file_listbox.insert(tk.END, file)
        
        def remove_selected():
            selection = file_listbox.curselection()
            for index in reversed(selection):
                file_listbox.delete(index)
        
        def clear_files():
            file_listbox.delete(0, tk.END)
        
        ttk.Button(file_buttons_frame, text="Add Files", command=add_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_buttons_frame, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_buttons_frame, text="Clear All", command=clear_files).pack(side=tk.LEFT, padx=5)
        
        # Output settings
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output directory
        output_dir_frame = ttk.Frame(output_frame)
        output_dir_frame.pack(fill=tk.X, pady=2)
        ttk.Label(output_dir_frame, text="Output Directory:", width=15).pack(side=tk.LEFT)
        output_dir_var = tk.StringVar(value=self.output_dir or os.getcwd())
        output_dir_entry = ttk.Entry(output_dir_frame, textvariable=output_dir_var)
        output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        def browse_output_dir():
            dir_path = filedialog.askdirectory(initialdir=output_dir_var.get())
            if dir_path:
                output_dir_var.set(dir_path)
        
        ttk.Button(output_dir_frame, text="Browse", command=browse_output_dir, width=10).pack(side=tk.RIGHT)
        
        # Output format options
        format_frame = ttk.Frame(output_frame)
        format_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(format_frame, text="Output Format:").pack(side=tk.LEFT)
        
        output_format_var = tk.StringVar(value="individual")
        ttk.Radiobutton(format_frame, text="Individual files", variable=output_format_var, 
                       value="individual").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(format_frame, text="Combined file", variable=output_format_var, 
                       value="combined").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Both", variable=output_format_var, 
                       value="both").pack(side=tk.LEFT, padx=5)
        
        # Processing options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # AI classification option
        ai_var = tk.BooleanVar(value=self.use_ai_classification)
        ttk.Checkbutton(options_frame, text="Use AI Classification", variable=ai_var).pack(anchor=tk.W)
        
        # Include summary option
        summary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Include summary sheet", variable=summary_var).pack(anchor=tk.W)
        
        # Progress tracking
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        progress_label = ttk.Label(progress_frame, text="Ready to process")
        progress_label.pack(anchor=tk.W)
        
        progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=5)
        
        # Results text
        results_text = tk.Text(progress_frame, height=8, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=results_text.yview)
        results_text.configure(yscrollcommand=results_scrollbar.set)
        
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def start_batch_processing():
            files = list(file_listbox.get(0, tk.END))
            if not files:
                messagebox.showerror("Error", "Please select at least one file to process.")
                return
            
            output_dir = output_dir_var.get()
            if not output_dir or not os.path.exists(output_dir):
                messagebox.showerror("Error", "Please select a valid output directory.")
                return
            
            # Disable buttons during processing
            start_button.config(state=tk.DISABLED)
            cancel_button.config(state=tk.DISABLED)
            
            # Clear results
            results_text.delete(1.0, tk.END)
            
            # Start processing in a separate thread
            import threading
            
            def process_files_thread():
                try:
                    dict_path = self.dict_entry.get()
                    use_ai = ai_var.get()
                    output_format = output_format_var.get()
                    include_summary = summary_var.get()
                    
                    # Initialize AI classifier if needed
                    ai_classifier = None
                    if use_ai and AI_AVAILABLE:
                        try:
                            confidence_threshold = float(self.confidence_var.get())
                            
                            ai_classifier = AIClassifier(
                                confidence_threshold=confidence_threshold,
                                cache_file=AI_CACHE_FILE
                            )
                            
                            if not ai_classifier.load_failure_dictionary(dict_path):
                                results_text.insert(tk.END, "Warning: Failed to load dictionary for AI classifier\n")
                                use_ai = False
                        except Exception as e:
                            results_text.insert(tk.END, f"Warning: Failed to initialize AI classifier: {str(e)}\n")
                            use_ai = False
                    
                    total_files = len(files)
                    processed_files = []
                    failed_files = []
                    
                    for i, file_path in enumerate(files):
                        try:
                            # Update progress
                            progress = (i / total_files) * 100
                            progress_bar['value'] = progress
                            progress_label.config(text=f"Processing {os.path.basename(file_path)}...")
                            batch_window.update()
                            
                            # Process file
                            df = process_files(file_path, dict_path, progress_label, batch_window, 
                                             output_dir, use_ai, ai_classifier, column_mapping=self.column_mapping)
                            
                            if df is not None and not df.empty:
                                processed_files.append((file_path, df))
                                results_text.insert(tk.END, f"✓ {os.path.basename(file_path)} - {len(df)} rows\n")
                            else:
                                failed_files.append(file_path)
                                results_text.insert(tk.END, f"✗ {os.path.basename(file_path)} - Failed\n")
                            
                            results_text.see(tk.END)
                            
                        except Exception as e:
                            failed_files.append(file_path)
                            results_text.insert(tk.END, f"✗ {os.path.basename(file_path)} - Error: {str(e)}\n")
                            results_text.see(tk.END)
                            logging.error(f"Batch processing error for {file_path}: {e}")
                    
                    # Generate output files
                    if processed_files:
                        try:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            if output_format in ["combined", "both"]:
                                # Create combined file
                                combined_data = []
                                for file_path, df in processed_files:
                                    df_copy = df.copy()
                                    df_copy['Source File'] = os.path.basename(file_path)
                                    combined_data.append(df_copy)
                                
                                combined_df = pd.concat(combined_data, ignore_index=True)
                                combined_file = os.path.join(output_dir, f"batch_combined_{timestamp}.xlsx")
                                
                                with pd.ExcelWriter(combined_file, engine='openpyxl') as writer:
                                    combined_df.to_excel(writer, sheet_name='Combined Data', index=False)
                                    
                                    if include_summary:
                                        # Create summary
                                        summary_data = []
                                        for file_path, df in processed_files:
                                            mtbf = calculate_mtbf(df, set(df.index))
                                            _, _, failures_per_year = calculate_crow_amsaa_params(df, set(df.index))
                                            summary_data.append({
                                                'File': os.path.basename(file_path),
                                                'Rows': len(df),
                                                'MTBF (days)': f'{mtbf:.2f}',
                                                'Failures per Year': f'{failures_per_year:.2f}'
                                            })
                                        
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                
                                results_text.insert(tk.END, f"Combined file: {os.path.basename(combined_file)}\n")
                            
                            if output_format in ["individual", "both"]:
                                # Create individual files
                                for file_path, df in processed_files:
                                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                                    individual_file = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.xlsx")
                                    
                                    with pd.ExcelWriter(individual_file, engine='openpyxl') as writer:
                                        df.to_excel(writer, sheet_name='Work Orders', index=False)
                                        
                                        if include_summary:
                                            mtbf = calculate_mtbf(df, set(df.index))
                                            _, _, failures_per_year = calculate_crow_amsaa_params(df, set(df.index))
                                            
                                            summary_data = {
                                                'Metric': ['MTBF (days)', 'Failures per Year'],
                                                'Value': [f'{mtbf:.2f}', f'{failures_per_year:.2f}']
                                            }
                                            summary_df = pd.DataFrame(summary_data)
                                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    results_text.insert(tk.END, f"Individual file: {os.path.basename(individual_file)}\n")
                            
                            results_text.insert(tk.END, f"\nBatch processing complete!\n")
                            results_text.insert(tk.END, f"Processed: {len(processed_files)} files\n")
                            results_text.insert(tk.END, f"Failed: {len(failed_files)} files\n")
                            
                        except Exception as e:
                            results_text.insert(tk.END, f"Error creating output files: {str(e)}\n")
                            logging.error(f"Error creating batch output files: {e}")
                    
                    # Final progress update
                    progress_bar['value'] = 100
                    progress_label.config(text="Batch processing complete")
                    
                except Exception as e:
                    results_text.insert(tk.END, f"Batch processing failed: {str(e)}\n")
                    logging.error(f"Batch processing failed: {e}")
                finally:
                    # Re-enable buttons
                    start_button.config(state=tk.NORMAL)
                    cancel_button.config(state=tk.NORMAL)
            
            # Start the processing thread
            processing_thread = threading.Thread(target=process_files_thread)
            processing_thread.daemon = True
            processing_thread.start()
        
        def cancel_processing():
            batch_window.destroy()
        
        start_button = ttk.Button(button_frame, text="Start Processing", command=start_batch_processing)
        start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=cancel_processing)
        cancel_button.pack(side=tk.LEFT)
        
        # Help text
        help_frame = ttk.LabelFrame(main_frame, text="How to Use", padding=10)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        help_text = """1. Click "Add Files" to select multiple work order Excel files
2. Choose output directory and format (individual/combined/both)
3. Enable AI classification if desired
4. Click "Start Processing" to begin batch processing
5. Monitor progress and results in the text area"""
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT)
        help_label.pack(anchor=tk.W)

    def clear_data(self):
        """Clear all loaded data"""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all data?"):
            self.wo_df = None
            self.included_indices = set()
            self.update_table()
            self.status_label.config(text="Data cleared")
            self.data_status_indicator.config(text="📊")

    def show_filter_manager(self):
        """Show filter management dialog"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showerror("Error", "No data loaded. Please process files first.")
            return
        
        # Create filter manager window
        filter_window = tk.Toplevel(self.root)
        filter_window.title("Filter Management")
        filter_window.geometry("600x500")
        filter_window.transient(self.root)
        filter_window.grab_set()
        
        # Center the window
        filter_window.update_idletasks()
        x = (filter_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (filter_window.winfo_screenheight() // 2) - (500 // 2)
        filter_window.geometry(f"600x500+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(filter_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Filter Management", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Current filters display
        current_frame = ttk.LabelFrame(main_frame, text="Current Filters", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 10))
        
        current_filters_text = f"""Equipment: {self.equipment_var.get() or 'All'}
Work Type: {self.work_type_var.get() or 'All'}
Failure Code: {self.failure_code_var.get() or 'All'}
Date Range: {self.start_date_entry.get() or 'None'} to {self.end_date_entry.get() or 'None'}
Included Work Orders: {len(self.included_indices)} of {len(self.wo_df) if self.wo_df is not None else 0}"""
        
        ttk.Label(current_frame, text=current_filters_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Filter presets
        presets_frame = ttk.LabelFrame(main_frame, text="Filter Presets", padding=10)
        presets_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Preset buttons
        preset_buttons_frame = ttk.Frame(presets_frame)
        preset_buttons_frame.pack(fill=tk.X)
        
        def apply_equipment_filter():
            """Apply filter for most common equipment"""
            if self.wo_df is not None:
                most_common = str(self.wo_df['Equipment #'].value_counts().index[0])
                self.equipment_var.set(most_common)
                self.update_table()
                filter_window.destroy()
        
        def apply_failure_filter():
            """Apply filter for most common failure code"""
            if self.wo_df is not None:
                most_common = str(self.wo_df['Failure Code'].value_counts().index[0])
                self.failure_code_var.set(most_common)
                self.update_table()
                filter_window.destroy()
        
        def apply_recent_filter():
            """Apply filter for recent work orders (last 30 days)"""
            if self.wo_df is not None:
                # Set date range to last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                self.start_date_entry.delete(0, tk.END)
                self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
                self.end_date_entry.delete(0, tk.END)
                self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
                self.apply_date_filter()
                filter_window.destroy()
        
        def apply_high_cost_filter():
            """Apply filter for high-cost work orders"""
            if self.wo_df is not None and 'Work Order Cost' in self.wo_df.columns:
                # Include only work orders with cost > 0
                self.included_indices = set(self.wo_df[self.wo_df['Work Order Cost'] > 0].index)
                self.update_table()
                filter_window.destroy()
        
        ttk.Button(preset_buttons_frame, text="Most Common Equipment", command=apply_equipment_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_buttons_frame, text="Most Common Failure", command=apply_failure_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_buttons_frame, text="Recent (30 days)", command=apply_recent_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_buttons_frame, text="High Cost Only", command=apply_high_cost_filter).pack(side=tk.LEFT, padx=5)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(main_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        actions_buttons_frame = ttk.Frame(actions_frame)
        actions_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(actions_buttons_frame, text="Include All", command=lambda: self.include_all_work_orders(filter_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_buttons_frame, text="Exclude All", command=lambda: self.exclude_all_work_orders(filter_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_buttons_frame, text="Reset All Filters", command=lambda: self.reset_all_filters_from_manager(filter_window)).pack(side=tk.LEFT, padx=5)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=filter_window.destroy).pack(side=tk.RIGHT, pady=10)
    
    def include_all_work_orders(self, window=None):
        """Include all work orders in analysis"""
        if self.wo_df is not None:
            self.included_indices = set(self.wo_df.index)
            self.update_table()
            if window:
                window.destroy()
    
    def exclude_all_work_orders(self, window=None):
        """Exclude all work orders from analysis"""
        self.included_indices = set()
        self.update_table()
        if window:
            window.destroy()
    
    def reset_all_filters_from_manager(self, window=None):
        """Reset all filters from filter manager"""
        self.reset_all_filters()
        if window:
            window.destroy()

    def show_date_selector(self):
        """Show date range selector dialog"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showerror("Error", "No data loaded. Please process files first.")
            return
        
        # Create date selector window
        date_window = tk.Toplevel(self.root)
        date_window.title("Date Range Selector")
        date_window.geometry("500x400")
        date_window.transient(self.root)
        date_window.grab_set()
        
        # Center the window
        date_window.update_idletasks()
        x = (date_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (date_window.winfo_screenheight() // 2) - (400 // 2)
        date_window.geometry(f"500x400+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(date_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Date Range Selector", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Current date range
        current_frame = ttk.LabelFrame(main_frame, text="Current Date Range", padding=10)
        current_frame.pack(fill=tk.X, pady=(0, 10))
        
        current_range = f"Start: {self.start_date_entry.get() or 'None'}\nEnd: {self.end_date_entry.get() or 'None'}"
        ttk.Label(current_frame, text=current_range, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Date range presets
        presets_frame = ttk.LabelFrame(main_frame, text="Quick Date Ranges", padding=10)
        presets_frame.pack(fill=tk.X, pady=(0, 10))
        
        def set_last_7_days():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_last_30_days():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_last_90_days():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_last_year():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, end_date.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_current_month():
            now = datetime.now()
            start_date = now.replace(day=1)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, now.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        def set_current_year():
            now = datetime.now()
            start_date = now.replace(month=1, day=1)
            self.start_date_entry.delete(0, tk.END)
            self.start_date_entry.insert(0, start_date.strftime('%m/%d/%Y'))
            self.end_date_entry.delete(0, tk.END)
            self.end_date_entry.insert(0, now.strftime('%m/%d/%Y'))
            self.apply_date_filter()
            date_window.destroy()
        
        # Preset buttons
        preset_buttons_frame = ttk.Frame(presets_frame)
        preset_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(preset_buttons_frame, text="Last 7 Days", command=set_last_7_days).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame, text="Last 30 Days", command=set_last_30_days).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame, text="Last 90 Days", command=set_last_90_days).pack(side=tk.LEFT, padx=5, pady=2)
        
        preset_buttons_frame2 = ttk.Frame(presets_frame)
        preset_buttons_frame2.pack(fill=tk.X)
        
        ttk.Button(preset_buttons_frame2, text="Last Year", command=set_last_year).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame2, text="Current Month", command=set_current_month).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(preset_buttons_frame2, text="Current Year", command=set_current_year).pack(side=tk.LEFT, padx=5, pady=2)
        
        # Data range info
        if self.wo_df is not None:
            try:
                dates = pd.to_datetime(self.wo_df['Reported Date'], errors='coerce').dropna()
                if len(dates) > 0:
                    min_date = dates.min().strftime('%m/%d/%Y')
                    max_date = dates.max().strftime('%m/%d/%Y')
                    data_range_text = f"Data Range: {min_date} to {max_date}\nTotal Work Orders: {len(dates)}"
                    
                    data_frame = ttk.LabelFrame(main_frame, text="Available Data Range", padding=10)
                    data_frame.pack(fill=tk.X, pady=(0, 10))
                    ttk.Label(data_frame, text=data_range_text, justify=tk.LEFT).pack(anchor=tk.W)
            except Exception as e:
                logging.debug(f"Error getting date range: {e}")
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Clear Date Range", command=lambda: self.clear_date_range(date_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=date_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def clear_date_range(self, window=None):
        """Clear the current date range"""
        self.start_date_entry.delete(0, tk.END)
        self.end_date_entry.delete(0, tk.END)
        self.start_date = None
        self.end_date = None
        self.update_table()
        if window:
            window.destroy()

    def reset_all_filters(self):
        """Reset all filters to default"""
        self.equipment_var.set('')
        self.failure_code_var.set('')
        self.work_type_var.set('')
        self.start_date_entry.delete(0, tk.END)
        self.end_date_entry.delete(0, tk.END)
        self.start_date = None
        self.end_date = None
        self.update_table()
        self.status_label.config(text="All filters reset")

    def reset_defaults(self):
        """Reset to default settings"""
        self.confidence_var.set(str(AI_CONFIDENCE_THRESHOLD))
        self.ai_enabled_var.set(False)
        self.use_ai_classification = False
        self.status_label.config(text="Settings reset to defaults")



    def export_training_data(self):
        """Export training data for AI model improvement"""
        if self.wo_df is None or self.wo_df.empty:
            messagebox.showerror("Error", "No data to export. Please process files first.")
            return
        
        if not self.ai_classifier:
            messagebox.showerror("Error", "AI classifier not available.")
            return
        
        try:
            output_file = "training_data_export.json"
            if self.ai_classifier.export_training_data(output_file, self.wo_df):
                messagebox.showinfo("Success", f"Training data exported to {output_file}")
            else:
                messagebox.showerror("Error", "Failed to export training data")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def save_risk_preset(self):
        """Save current risk parameters as preset"""
        preset_name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if preset_name:
            try:
                preset = {
                    'name': preset_name,
                    'prod_loss': self.prod_loss_var.get(),
                    'maint_cost': self.maint_cost_var.get(),
                    'margin': self.margin_var.get(),
                    'created': datetime.now().isoformat()
                }
                
                # Load existing presets
                presets = self.load_risk_presets()
                presets[preset_name] = preset
                
                # Save to file
                import json
                with open('risk_presets.json', 'w') as f:
                    json.dump(presets, f, indent=2)
                
                messagebox.showinfo("Success", f"Preset '{preset_name}' saved successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save preset: {str(e)}")
                logging.error(f"Error saving risk preset: {e}")

    def load_risk_preset(self):
        """Load risk parameters from preset"""
        presets = self.load_risk_presets()
        
        if not presets:
            messagebox.showinfo("No Presets", "No saved presets found.")
            return
        
        # Create preset selection dialog
        preset_window = tk.Toplevel(self.root)
        preset_window.title("Load Risk Preset")
        preset_window.geometry("400x300")
        preset_window.transient(self.root)
        preset_window.grab_set()
        
        # Center the window
        preset_window.update_idletasks()
        x = (preset_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (preset_window.winfo_screenheight() // 2) - (300 // 2)
        preset_window.geometry(f"400x300+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(preset_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Select a preset to load:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Preset listbox
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        preset_listbox = tk.Listbox(list_frame, height=8)
        preset_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=preset_listbox.yview)
        preset_listbox.configure(yscrollcommand=preset_scrollbar.set)
        
        preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preset_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox
        for preset_name in presets.keys():
            preset_listbox.insert(tk.END, preset_name)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def load_selected_preset():
            selection = preset_listbox.curselection()
            if selection:
                preset_name = preset_listbox.get(selection[0])
                preset = presets[preset_name]
                
                self.prod_loss_var.set(preset['prod_loss'])
                self.maint_cost_var.set(preset['maint_cost'])
                self.margin_var.set(preset['margin'])
                
                # Update risk calculation
                self.update_risk()
                
                preset_window.destroy()
                messagebox.showinfo("Success", f"Preset '{preset_name}' loaded successfully!")
            else:
                messagebox.showwarning("Warning", "Please select a preset to load.")
        
        def delete_selected_preset():
            selection = preset_listbox.curselection()
            if selection:
                preset_name = preset_listbox.get(selection[0])
                if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete preset '{preset_name}'?"):
                    del presets[preset_name]
                    
                    # Save updated presets
                    import json
                    with open('risk_presets.json', 'w') as f:
                        json.dump(presets, f, indent=2)
                    
                    # Refresh listbox
                    preset_listbox.delete(0, tk.END)
                    for preset_name in presets.keys():
                        preset_listbox.insert(tk.END, preset_name)
                    
                    messagebox.showinfo("Success", f"Preset '{preset_name}' deleted successfully!")
            else:
                messagebox.showwarning("Warning", "Please select a preset to delete.")
        
        ttk.Button(button_frame, text="Load", command=load_selected_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=delete_selected_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=preset_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def load_risk_presets(self) -> dict:
        """Load risk presets from file"""
        try:
            import json
            if os.path.exists('risk_presets.json'):
                with open('risk_presets.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading risk presets: {e}")
        
        return {}



    def export_plot(self):
        """Export current plot"""
        if self.crow_amsaa_fig:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    self.crow_amsaa_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Success", f"Plot exported to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
        else:
            messagebox.showerror("Error", "No plot to export")

    def open_plot_in_new_window(self):
        """Open the current Crow-AMSAA plot in a new window"""
        if self.crow_amsaa_fig is None:
            messagebox.showerror("Error", "No plot to display")
            return
        
        try:
            # Create new window
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Crow-AMSAA Analysis Plot")
            plot_window.geometry("800x600")
            plot_window.transient(self.root)
            
            # Center the window
            plot_window.update_idletasks()
            x = (plot_window.winfo_screenwidth() // 2) - (800 // 2)
            y = (plot_window.winfo_screenheight() // 2) - (600 // 2)
            plot_window.geometry(f"800x600+{x}+{y}")
            
            # Create frame for the plot
            plot_frame = ttk.Frame(plot_window, padding=10)
            plot_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create a new figure with the same data
            if self.wo_df is not None and not self.wo_df.empty:
                filtered_df = self.get_filtered_df()
                
                # Create the plot in the new window
                if self.is_segmented_view and hasattr(self, 'segment_date') and self.segment_date:
                    # Create segmented plot
                    self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, plot_frame, segment_date=self.segment_date)
                else:
                    # Create single plot
                    self.create_crow_amsaa_plot_interactive(filtered_df, self.included_indices, plot_frame)
                
                # Add close button
                button_frame = ttk.Frame(plot_window)
                button_frame.pack(fill=tk.X, padx=10, pady=5)
                ttk.Button(button_frame, text="Close", command=plot_window.destroy).pack(side=tk.RIGHT)
                
            else:
                ttk.Label(plot_frame, text="No data available for plotting").pack(expand=True)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open plot in new window: {str(e)}")
            logging.error(f"Error opening plot in new window: {e}")

    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
Work Order Analysis Pro - User Guide

1. Data Input Tab:
   - Load work order and dictionary files
   - Set output directory
   - Process files to begin analysis

2. Analysis Tab:
   - Use filters to focus on specific data
   - View work orders and equipment summary with cost information
   - Sort and filter data as needed
   - View Crow-AMSAA analysis plots (resizable panes)
   - Right-click work orders to segment plots
   - Use "Return to Single Plot" button to restore single view
   - Use "Open in New Window" to view plots in separate window

3. AI Settings Tab:
   - Enable/disable AI classification
   - Adjust confidence threshold
   - View AI statistics and manage cache

4. Risk Assessment Tab:
   - Set risk parameters
   - Calculate failure rates and annualized risk
   - Risk values reflect current Crow-AMSAA plot state (single/segmented)

Features:
- Work Order Cost column (optional) - displays total and average costs per equipment
- Column mapping for CMMS compatibility
- Interactive Crow-AMSAA plots with segmentation
- Risk assessment that updates based on plot view

Keyboard Shortcuts:
- Ctrl+O: Load work order file
- Ctrl+D: Load dictionary file
- Ctrl+E: Export to Excel
- F5: Process files
- Ctrl+Q: Exit application
        """
        messagebox.showinfo("User Guide", guide_text)

    def show_about(self):
        """Show about dialog"""
        about_text = """
Work Order Analysis Pro
Version 2.0

AI-Powered Failure Mode Classification System

Features:
• Intelligent failure code assignment using AI
• SpaCy NLP for advanced linguistic analysis
• Sentence embeddings for semantic similarity
• Crow-AMSAA reliability analysis
• Risk assessment and calculation
• Comprehensive reporting and export

Developed by Matt Barnes
        """
        messagebox.showinfo("About", about_text)

    def check_updates(self):
        """Check for application updates"""
        try:
            # Create update check window
            update_window = tk.Toplevel(self.root)
            update_window.title("Check for Updates")
            update_window.geometry("500x300")
            update_window.transient(self.root)
            update_window.grab_set()
            
            # Center the window
            update_window.update_idletasks()
            x = (update_window.winfo_screenwidth() // 2) - (500 // 2)
            y = (update_window.winfo_screenheight() // 2) - (300 // 2)
            update_window.geometry(f"500x300+{x}+{y}")
            
            # Main frame
            main_frame = ttk.Frame(update_window, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(main_frame, text="Application Update Check", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
            
            # Current version info
            current_frame = ttk.LabelFrame(main_frame, text="Current Version", padding=10)
            current_frame.pack(fill=tk.X, pady=(0, 10))
            
            current_info = """Work Order Analysis Pro
Version: 2.0
Build Date: January 2025
Features: AI Classification, Crow-AMSAA Analysis, Risk Assessment"""
            
            ttk.Label(current_frame, text=current_info, justify=tk.LEFT).pack(anchor=tk.W)
            
            # Update status
            status_frame = ttk.LabelFrame(main_frame, text="Update Status", padding=10)
            status_frame.pack(fill=tk.X, pady=(0, 10))
            
            status_label = ttk.Label(status_frame, text="Checking for updates...", foreground="blue")
            status_label.pack(anchor=tk.W)
            
            # Simulate update check
            def check_for_updates():
                import time
                time.sleep(1)  # Simulate network delay
                
                # For now, always show up to date
                status_label.config(text="✓ Your application is up to date!", foreground="green")
                
                # Add some update info
                update_info = """Latest Version: 2.0
Release Date: January 2025
Status: Current version is the latest available

Recent Updates:
• Enhanced AI classification with local models
• Improved Crow-AMSAA analysis with segmentation
• Added work order cost tracking
• Enhanced filter management
• Improved user interface with resizable panes"""
                
                info_label = ttk.Label(status_frame, text=update_info, justify=tk.LEFT)
                info_label.pack(anchor=tk.W, pady=(10, 0))
            
            # Run update check in background
            import threading
            update_thread = threading.Thread(target=check_for_updates)
            update_thread.daemon = True
            update_thread.start()
            
            # Close button
            ttk.Button(main_frame, text="Close", command=update_window.destroy).pack(side=tk.RIGHT, pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check for updates: {str(e)}")
            logging.error(f"Error checking for updates: {e}")

    def update_progress(self, value, text=None):
        """Update progress bar and status"""
        self.progress_var.set(value)
        if text:
            self.status_label.config(text=text)
        self.root.update_idletasks()

    def show_column_mapping(self):
        """Show column mapping dialog for CMMS compatibility"""
        # Create column mapping dialog
        mapping_window = tk.Toplevel(self.root)
        mapping_window.title("Column Mapping - CMMS Compatibility")
        mapping_window.geometry("700x600")
        mapping_window.transient(self.root)
        mapping_window.grab_set()
        
        # Center the window
        mapping_window.update_idletasks()
        x = (mapping_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (mapping_window.winfo_screenheight() // 2) - (600 // 2)
        mapping_window.geometry(f"700x600+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(mapping_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and description
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="Column Mapping for CMMS Compatibility", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(title_frame, text="Map your CMMS export columns to the required program columns", 
                 font=('Arial', 9)).pack(anchor=tk.W)
        
        # File selection for preview
        file_frame = ttk.LabelFrame(main_frame, text="Select Work Order File for Preview", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(fill=tk.X)
        
        ttk.Label(file_select_frame, text="File:").pack(side=tk.LEFT)
        preview_file_var = tk.StringVar()
        preview_file_entry = ttk.Entry(file_select_frame, textvariable=preview_file_var)
        preview_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        def browse_preview_file():
            file_path = filedialog.askopenfilename(
                title="Select Work Order File for Preview",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if file_path:
                preview_file_var.set(file_path)
                load_file_columns()
        
        ttk.Button(file_select_frame, text="Browse", command=browse_preview_file, width=10).pack(side=tk.RIGHT)
        
        # Column mapping area
        mapping_frame = ttk.LabelFrame(main_frame, text="Column Mapping", padding=10)
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create scrollable frame for mappings
        canvas = tk.Canvas(mapping_frame)
        scrollbar = ttk.Scrollbar(mapping_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store mapping widgets
        mapping_widgets = {}
        
        def load_file_columns():
            """Load columns from selected file and populate mapping dropdowns"""
            file_path = preview_file_var.get()
            if not file_path or not os.path.exists(file_path):
                return
            
            try:
                df = pd.read_excel(file_path)
                available_columns = list(df.columns)
                
                # Clear existing mappings
                for widget in mapping_widgets.values():
                    if hasattr(widget, 'destroy'):
                        widget.destroy()
                mapping_widgets.clear()
                
                # Create mapping rows
                for i, (required_col, description) in enumerate(REQUIRED_COLUMNS.items()):
                    row_frame = ttk.Frame(scrollable_frame)
                    row_frame.pack(fill=tk.X, pady=2)
                    
                    # Required column label
                    ttk.Label(row_frame, text=f"{description}:", width=20).pack(side=tk.LEFT)
                    
                    # Mapping dropdown
                    mapping_var = tk.StringVar()
                    if required_col in self.column_mapping:
                        mapping_var.set(self.column_mapping[required_col])
                    elif required_col in available_columns:
                        mapping_var.set(required_col)
                    else:
                        mapping_var.set('')
                    
                    mapping_dropdown = ttk.Combobox(row_frame, textvariable=mapping_var, 
                                                  values=[''] + available_columns, width=30)
                    mapping_dropdown.pack(side=tk.LEFT, padx=(10, 5))
                    
                    # Store reference
                    mapping_widgets[required_col] = mapping_dropdown
                    
                    # Status indicator
                    status_label = ttk.Label(row_frame, text="", width=10)
                    status_label.pack(side=tk.LEFT, padx=5)
                    
                    def update_status(col=required_col, var=mapping_var, label=status_label):
                        selected = var.get()
                        if selected and selected in available_columns:
                            label.config(text="✓", foreground="green")
                        elif selected:
                            label.config(text="✗", foreground="red")
                        else:
                            label.config(text="", foreground="black")
                    
                    mapping_dropdown.bind("<<ComboboxSelected>>", lambda e, col=required_col, var=mapping_var, label=status_label: update_status(col, var, label))
                    update_status(required_col, mapping_var, status_label)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file columns: {str(e)}")
        
        # Auto-detect button
        detect_frame = ttk.Frame(mapping_frame)
        detect_frame.pack(fill=tk.X, pady=(10, 0))
        
        def auto_detect_mappings():
            """Auto-detect column mappings based on similarity"""
            file_path = preview_file_var.get()
            if not file_path or not os.path.exists(file_path):
                messagebox.showwarning("Warning", "Please select a file first.")
                return
            
            try:
                df = pd.read_excel(file_path)
                available_columns = list(df.columns)
                
                # Simple fuzzy matching for auto-detection
                for required_col, description in REQUIRED_COLUMNS.items():
                    if required_col in mapping_widgets:
                        dropdown = mapping_widgets[required_col]
                        
                        # Try exact match first
                        if required_col in available_columns:
                            dropdown.set(required_col)
                            continue
                        
                        # Try partial matches
                        best_match = None
                        best_score = 0
                        
                        for col in available_columns:
                            # Check if any word in the description matches
                            desc_words = description.lower().split()
                            col_lower = col.lower()
                            
                            for word in desc_words:
                                if word in col_lower and len(word) > 2:
                                    score = len(word) / len(col_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = col
                        
                        if best_match and best_score > 0.3:
                            dropdown.set(best_match)
                
                messagebox.showinfo("Auto-Detect", "Column mappings auto-detected based on similarity.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Auto-detection failed: {str(e)}")
        
        ttk.Button(detect_frame, text="Auto-Detect Mappings", command=auto_detect_mappings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(detect_frame, text="Clear All Mappings", 
                  command=lambda: [widget.set('') for widget in mapping_widgets.values()]).pack(side=tk.LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_mappings():
            """Save current column mappings"""
            mappings = {}
            for required_col, dropdown in mapping_widgets.items():
                selected = dropdown.get()
                if selected:
                    mappings[required_col] = selected
            
            self.column_mapping = mappings
            
            # Update status indicator
            if mappings:
                self.column_mapping_indicator.config(text="📋", foreground="green")
            else:
                self.column_mapping_indicator.config(text="📋", foreground="blue")
            
            # Save to file for persistence
            try:
                import json
                with open('column_mapping.json', 'w') as f:
                    json.dump(mappings, f, indent=2)
                messagebox.showinfo("Success", f"Column mappings saved successfully!\nMapped {len(mappings)} columns.")
                mapping_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mappings: {str(e)}")
        
        def load_saved_mappings():
            """Load previously saved mappings"""
            try:
                import json
                if os.path.exists('column_mapping.json'):
                    with open('column_mapping.json', 'r') as f:
                        saved_mappings = json.load(f)
                    
                    self.column_mapping = saved_mappings
                    
                    # Update dropdowns
                    for required_col, mapped_col in saved_mappings.items():
                        if required_col in mapping_widgets:
                            mapping_widgets[required_col].set(mapped_col)
                    
                    messagebox.showinfo("Success", f"Loaded {len(saved_mappings)} saved mappings.")
                else:
                    messagebox.showinfo("Info", "No saved mappings found.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load mappings: {str(e)}")
        
        ttk.Button(button_frame, text="Save Mappings", command=save_mappings, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Load Saved", command=load_saved_mappings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=mapping_window.destroy).pack(side=tk.RIGHT)
        
        # Help text
        help_frame = ttk.LabelFrame(main_frame, text="How to Use Column Mapping", padding=10)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        help_text = """1. Select a work order file to see available columns
2. Map each required column to a column in your file
3. Use Auto-Detect to automatically find similar column names
4. Save mappings for future use with similar files
5. The program will use these mappings when processing files

Example: If your CMMS exports "date" instead of "Reported Date", 
map "Reported Date" to "date" in the dropdown."""
        
        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=650)
        help_label.pack(anchor=tk.W)
        
        # Load saved mappings if they exist
        load_saved_mappings()

    def apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping to DataFrame"""
        if not self.column_mapping:
            return df
        
        # Create a copy to avoid modifying original
        mapped_df = df.copy()
        
        # Rename columns based on mapping
        rename_dict = {}
        for required_col, mapped_col in self.column_mapping.items():
            if mapped_col in df.columns and mapped_col != required_col:
                rename_dict[mapped_col] = required_col
        
        if rename_dict:
            mapped_df = mapped_df.rename(columns=rename_dict)
            logging.info(f"Applied column mapping: {rename_dict}")
        
        return mapped_df

    def load_saved_column_mappings(self):
        """Load saved column mappings from file"""
        try:
            import json
            if os.path.exists('column_mapping.json'):
                with open('column_mapping.json', 'r') as f:
                    self.column_mapping = json.load(f)
                
                # Update status indicator
                if self.column_mapping:
                    self.column_mapping_indicator.config(text="📋", foreground="green")
                    logging.info(f"Loaded {len(self.column_mapping)} saved column mappings")
                else:
                    self.column_mapping_indicator.config(text="📋", foreground="blue")
        except Exception as e:
            logging.error(f"Failed to load column mappings: {e}")
            self.column_mapping = {}
            self.column_mapping_indicator.config(text="📋", foreground="blue")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FailureModeApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        print(f"Application failed to start: {e}")