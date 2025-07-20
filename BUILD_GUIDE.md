# WO Analysis Program - Build Guide

This guide will walk you through building a distributable executable for the WO Analysis Program using PyInstaller.

## Prerequisites

Before building, ensure you have:

1. **Python 3.8+** installed on your system
2. **Virtual environment** set up with all dependencies installed
3. **All required files** present in the project directory

## Quick Start

### Option 1: Using PowerShell Script (Recommended)

1. Open PowerShell in the project directory
2. Run the build script:

```powershell
# Basic build
.\build_distributable.ps1

# Clean build (removes previous builds)
.\build_distributable.ps1 -Clean

# Build with console window (for debugging)
.\build_distributable.ps1 -Console

# Build in debug mode
.\build_distributable.ps1 -Debug
```

### Option 2: Using Batch File

1. Open Command Prompt in the project directory
2. Run the batch file:

```cmd
# Basic build
build.bat

# Clean build
build.bat clean
```

### Option 3: Manual Build

1. Activate your virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

2. Install PyInstaller (if not already installed):
```powershell
pip install pyinstaller
```

3. Build using the spec file:
```powershell
pyinstaller WOAnalysisProgram.spec
```

## Build Configuration

The build process uses the `WOAnalysisProgram.spec` file which includes:

### Data Files Included
- `app_config.json` - Application configuration
- `risk_presets.json` - Risk assessment presets
- `failure_mode_dictionary_*.xlsx` - Failure mode dictionaries
- `icons/` - Application icons
- `ai_classification_cache.json` - AI classification cache

### Hidden Imports
The spec file includes all necessary Python modules:
- Core: tkinter, pandas, numpy, matplotlib
- AI/NLP: nltk, rapidfuzz, sentence_transformers, spacy, sklearn
- Statistical: scipy
- Custom modules: ai_failure_classifier, weibull_analysis, etc.

### Build Options
- **Console**: Set to `False` for a GUI-only application
- **Icon**: Uses `icons/app_icon.ico`
- **Optimization**: UPX compression enabled for smaller file size
- **Debug**: Can be enabled for troubleshooting

## Output

After a successful build, you'll find:

1. **`dist/WOAnalysisProgram.exe`** - The main executable
2. **`dist/WOAnalysisProgram_Package/`** - Complete distribution package containing:
   - Executable
   - Documentation files
   - Sample data
   - Test files
   - Run script

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Ensure all dependencies are installed in your virtual environment
   - Check that the spec file includes all required hidden imports

2. **Large executable size**
   - The executable includes all Python dependencies and may be 100-500MB
   - This is normal for Python applications with scientific libraries

3. **Missing data files**
   - Verify all files listed in the spec file's `datas` section exist
   - Check file paths are correct

4. **Build fails with import errors**
   - Try building with `-Console` flag to see error messages
   - Check that all custom modules are in the same directory

### Debug Mode

To build with debugging enabled:

```powershell
.\build_distributable.ps1 -Debug -Console
```

This will:
- Include debug information
- Show console window for error messages
- Create a larger but more debuggable executable

### Performance Optimization

To reduce executable size:

1. **Exclude unnecessary modules** in the spec file
2. **Use UPX compression** (already enabled)
3. **Remove unused dependencies** from requirements.txt

## Distribution

### Single Executable
The `dist/WOAnalysisProgram.exe` file can be distributed as-is. It includes:
- All Python dependencies
- Required data files
- Application icons

### Complete Package
The `dist/WOAnalysisProgram_Package/` folder contains:
- Executable
- Documentation
- Sample data
- Run script for easy execution

### System Requirements
- Windows 10/11 (64-bit)
- No Python installation required on target machine
- Minimum 4GB RAM recommended
- 500MB free disk space

## Advanced Configuration

### Customizing the Spec File

You can modify `WOAnalysisProgram.spec` to:

1. **Add more data files**:
```python
datas=[
    ('your_file.txt', '.'),
    ('your_folder', 'your_folder'),
]
```

2. **Include additional modules**:
```python
hiddenimports=[
    'your_module',
    'your_package.submodule',
]
```

3. **Change build options**:
```python
exe = EXE(
    # ... other options ...
    console=True,  # Show console window
    debug=True,    # Include debug info
    upx=False,     # Disable compression
)
```

### Version Information

To add version information to your executable, create a version file:

```python
# version_info.txt
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'Your Company'),
         StringStruct(u'FileDescription', u'WO Analysis Program'),
         StringStruct(u'FileVersion', u'1.0.0'),
         StringStruct(u'InternalName', u'WOAnalysisProgram'),
         StringStruct(u'LegalCopyright', u'Copyright (c) 2024'),
         StringStruct(u'OriginalFilename', u'WOAnalysisProgram.exe'),
         StringStruct(u'ProductName', u'WO Analysis Program'),
         StringStruct(u'ProductVersion', u'1.0.0')])
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
```

Then add to your spec file:
```python
exe = EXE(
    # ... other options ...
    version_file='version_info.txt',
)
```

## Support

If you encounter issues during the build process:

1. Check the troubleshooting section above
2. Try building with debug mode enabled
3. Verify all dependencies are correctly installed
4. Check that all required files are present in the project directory

For additional help, refer to the PyInstaller documentation or check the project's technical documentation. 