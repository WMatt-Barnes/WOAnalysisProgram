# -*- mode: python ; coding: utf-8 -*-

# PyInstaller spec file for WO Analysis Program - Optimized for speed
# This spec file creates a directory-based executable for faster startup

import os
import sys
from pathlib import Path

# Get the current directory
current_dir = Path.cwd()

# Define data files to include
data_files = [
    # Configuration files
    ('app_config.json', '.'),
    ('risk_presets.json', '.'),
    
    # Failure mode dictionaries
    ('failure_mode_dictionary_.xlsx', '.'),
    ('failure_mode_dictionary_2.xlsx', '.'),
    ('sample_failure_dictionary.xlsx', '.'),
    
    # Icons directory
    ('icons', 'icons'),
    
    # AI cache file (will be created if not exists)
    ('ai_classification_cache.json', '.'),
]

# Define hidden imports - keeping only essential ones
hidden_imports = [
    # Core Python modules
    'tkinter',
    'tkinter.filedialog',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.simpledialog',
    
    # Data processing
    'pandas',
    'numpy',
    'openpyxl',
    
    # AI and NLP - simplified
    'nltk',
    'nltk.tokenize',
    'nltk.stem',
    'rapidfuzz',
    'rapidfuzz.fuzz',
    
    # Statistical analysis
    'scipy',
    'scipy.stats',
    'scipy.optimize',
    
    # Plotting
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.ticker',
    'matplotlib.lines',
    'matplotlib.patches',
    
    # Custom modules
    'ai_failure_classifier',
    'weibull_analysis',
    'fmea_export',
    'pm_analysis',
    'spares_analysis',
]

# Define binaries to include (if any external DLLs are needed)
binaries = []

# Define excludes to reduce bundle size
excludes = [
    'test',
    'tests',
    'unittest',
    'pytest',
    'doctest',
    'pdb',
    'profile',
    'cProfile',
    'pstats',
    'trace',
    'distutils',
    'setuptools',
    'pip',
    'wheel',
    'virtualenv',
    'venv',
    'jupyter',
    'notebook',
    'ipython',
    'spyder',
    'pylint',
    'flake8',
    'black',
    'mypy',
    # Exclude problematic AI libraries that might cause issues
    'sentence_transformers',
    'spacy',
    'sklearn',
    'torch',
    'transformers',
]

# Create the Analysis object
a = Analysis(
    ['WorkOrderAnalysisCur2.py'],
    pathex=[str(current_dir)],
    binaries=binaries,
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Create the PYZ archive
pyz = PYZ(a.pure)

# Create the executable (directory-based for faster startup)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # This is key for onedir
    name='WOAnalysisProgram',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons/app_icon.ico',
    version_file=None,
)

# Create the collection (directory)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WOAnalysisProgram'
) 