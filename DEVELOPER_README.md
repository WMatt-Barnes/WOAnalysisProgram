# Work Order Analysis Program - Developer Guide

## Overview

The Work Order Analysis Program is a comprehensive Python application designed to analyze maintenance work orders and classify failure modes using both traditional dictionary-based methods and advanced AI-powered classification. The system provides reliability analysis, risk assessment, and detailed reporting capabilities.

## Architecture Overview

```
WOAnalysisProgram/
├── WorkOrderAnalysisCur2.py      # Main GUI application
├── ai_failure_classifier.py      # AI classification engine
├── failure_mode_dictionary_2.xlsx # Failure mode dictionary
├── requirements.txt              # Python dependencies
├── test_files/                   # Test data files
├── venv/                         # Virtual environment
└── Various output files          # Generated during operation
```

## Core Components

### 1. Main Application (`WorkOrderAnalysisCur2.py`)

**Purpose**: The primary GUI application that orchestrates all functionality.

**Key Features**:
- Tkinter-based GUI with tabbed interface
- File processing and data management
- Filtering and analysis capabilities
- Export and reporting functions
- Integration with AI classification

**Architecture**:
```python
class FailureModeApp:
    def __init__(self, root):
        # Initialize GUI components
        # Set up data structures
        # Configure AI integration
```

**Main Methods**:
- `process_files()`: Core file processing logic
- `update_table()`: Refresh data displays
- `export_to_excel()`: Generate reports
- `run_processing()`: Orchestrate the entire workflow

### 2. AI Classification Engine (`ai_failure_classifier.py`)

**Purpose**: Advanced failure mode classification using multiple AI methodologies.

**Classification Methods**:

#### A. Expert System Classifier
- **Method**: Rule-based classification using predefined expert rules
- **Implementation**: `ExpertSystemClassifier` class
- **How it works**:
  ```python
  # Rules are defined with conditions and weights
  rules = [
      {
          'name': 'faulty_signal',
          'conditions': [
              {'type': 'keyword', 'value': 'signal', 'weight': 0.4},
              {'type': 'pattern', 'value': r'signal.*fault', 'weight': 0.5}
          ],
          'failure_code': '11.2',
          'description': 'Faulty Signal/Indication/Alarm'
      }
  ]
  ```

#### B. Contextual Pattern Classifier
- **Method**: Equipment-aware classification using context
- **Implementation**: `ContextualPatternClassifier` class
- **How it works**:
  ```python
  # Equipment contexts define common failures and patterns
  equipment_contexts = {
      'pump': {
          'common_failures': ['seal leak', 'bearing failure', 'cavitation'],
          'context_words': ['pressure', 'flow', 'seal', 'bearing'],
          'failure_patterns': {...}
      }
  }
  ```

#### C. Temporal Analysis Classifier
- **Method**: Time-based pattern recognition
- **Implementation**: `TemporalAnalysisClassifier` class
- **How it works**: Analyzes failure patterns over time to identify trends and seasonal effects

#### D. Hybrid Classification
- **Method**: Combines all classifiers with confidence scoring
- **Implementation**: `AIClassifier.classify_hybrid()`
- **Process**:
  1. Try Expert System first (highest confidence)
  2. Try Contextual Patterns
  3. Try Temporal Analysis
  4. Fall back to dictionary matching

### 3. Traditional Dictionary Matching

**Purpose**: Fallback classification using keyword matching.

**Methods**:
- **Exact Matching**: Direct keyword matches
- **Fuzzy Matching**: Using rapidfuzz library for approximate matches
- **Stemmed Matching**: Using NLTK SnowballStemmer for word variations
- **Regex Matching**: Pattern-based matching

**Implementation**:
```python
def match_failure_mode(description: str, dictionary: list) -> tuple:
    # 1. Normalize text
    # 2. Try exact matches
    # 3. Try fuzzy matches
    # 4. Try stemmed matches
    # 5. Try regex patterns
```

## Data Flow

### 1. File Processing Pipeline

```
Input Files → Validation → Column Mapping → Data Loading → Classification → Analysis → Export
```

**Step-by-step**:
1. **File Validation**: Check file existence and format
2. **Column Mapping**: Map CMMS columns to required format
3. **Data Loading**: Read Excel files into pandas DataFrames
4. **Classification**: Apply AI and/or dictionary classification
5. **Analysis**: Calculate MTBF, Crow-AMSAA parameters, risk metrics
6. **Export**: Generate Excel reports with multiple sheets

### 2. Classification Flow

```
Work Order Description → AI Classifier → Expert System → Contextual Patterns → Temporal Analysis → Dictionary Fallback → Result
```

**Confidence Scoring**:
- Expert System: 0.5-1.0 (highest confidence)
- Contextual Patterns: 0.3-0.8
- Temporal Analysis: 0.2-0.7
- Dictionary: 0.5 (default)

## Key Methodologies

### 1. Crow-AMSAA Analysis

**Purpose**: Reliability growth modeling to analyze failure patterns over time.

**Implementation**:
```python
def calculate_crow_amsaa_params(filtered_df, included_indices):
    # 1. Extract failure dates
    # 2. Calculate cumulative times
    # 3. Fit log-linear model: log(N) = log(λ) + β*log(T)
    # 4. Return λ (scale parameter) and β (shape parameter)
```

**Parameters**:
- **λ (lambda)**: Scale parameter indicating failure intensity
- **β (beta)**: Shape parameter indicating reliability trend
  - β < 1: Reliability growth (improving)
  - β = 1: Constant failure rate
  - β > 1: Reliability decay (worsening)

### 2. Risk Assessment

**Purpose**: Calculate financial impact of failures.

**Formula**:
```
Annualized Risk = Failures per Year × (Production Loss × Margin + Maintenance Cost)
```

**Implementation**:
```python
def update_risk(self):
    failures_per_year = calculate_crow_amsaa_params(...)[2]
    risk = failures_per_year * (prod_loss * margin + maint_cost)
```

### 3. MTBF Calculation

**Purpose**: Calculate Mean Time Between Failures.

**Implementation**:
```python
def calculate_mtbf(filtered_df, included_indices):
    # 1. Sort failure dates
    # 2. Calculate time differences
    # 3. Return average time between failures
```

## File Formats

### 1. Work Order File (Excel)
**Required Columns**:
- `Work Order`: Work order number
- `Description`: Work description
- `Asset`: Asset name
- `Equipment #`: Equipment identifier
- `Work Type`: Type of work
- `Reported Date`: Date work was reported

### 2. Dictionary File (Excel)
**Required Columns**:
- `Keyword`: Keywords to match
- `Code`: Failure code
- `Description`: Failure description

### 3. Output Files
- **Excel Report**: Multiple sheets with analysis results
- **Training Data Export**: JSON format for AI model improvement
- **Column Mapping**: JSON format for CMMS compatibility

## Configuration and Settings

### 1. AI Configuration
```python
AI_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for AI classification
AI_CACHE_FILE = "ai_classification_cache.json"  # Cache file for performance
```

### 2. Classification Thresholds
```python
THRESHOLD = 75  # Fuzzy matching threshold (0-100)
```

### 3. Column Mapping
- Supports CMMS compatibility
- Auto-detection of similar column names
- Persistent mapping storage

## Testing and Validation

### 1. Test Files
Located in `test_files/`:
- `cmms_format1.xlsx`: CMMS format example
- `cmms_format2.xlsx`: Alternative CMMS format
- `standard_format.xlsx`: Standard format example

### 2. Test Scripts
- `test_improved_classifier.py`: Test AI classification methods
- `debug_expert_system.py`: Debug expert system scoring
- `analyze_training_data.py`: Analyze training data patterns

### 3. Validation Methods
- **Cross-validation**: Compare AI vs dictionary results
- **Confidence scoring**: Evaluate classification confidence
- **Pattern analysis**: Identify classification patterns

## Performance Considerations

### 1. Caching
- AI classification results cached in JSON file
- Reduces processing time for repeated classifications
- Cache can be cleared via GUI

### 2. Batch Processing
- Support for processing multiple files
- Progress tracking and error handling
- Configurable output formats

### 3. Memory Management
- Large datasets processed in chunks
- Efficient pandas operations
- Proper cleanup of matplotlib figures

## Error Handling

### 1. File Validation
- Check file existence and format
- Validate required columns
- Handle missing or corrupted data

### 2. Classification Errors
- Fallback to dictionary matching
- Log classification failures
- Provide user feedback

### 3. Data Processing Errors
- Graceful handling of date parsing errors
- Skip invalid rows with logging
- Continue processing with valid data

## Development Guidelines

### 1. Code Structure
- **Modular design**: Separate concerns into different classes
- **Type hints**: Use Python type hints for better code documentation
- **Error handling**: Comprehensive try-catch blocks
- **Logging**: Detailed logging for debugging

### 2. Adding New Features
1. **AI Classifiers**: Extend base classifier classes
2. **Analysis Methods**: Add new calculation functions
3. **Export Formats**: Implement new export methods
4. **GUI Features**: Add new tabs or dialogs

### 3. Testing New Features
1. Create test cases in appropriate test files
2. Validate with different data formats
3. Test error conditions
4. Update documentation

## Common Development Tasks

### 1. Adding New Equipment Types
```python
# In ai_failure_classifier.py, add to equipment_contexts
'new_equipment': {
    'common_failures': ['failure1', 'failure2'],
    'context_words': ['keyword1', 'keyword2'],
    'failure_patterns': {
        'failure1': [r'pattern1', r'pattern2']
    }
}
```

### 2. Adding New Expert Rules
```python
# In ExpertSystemClassifier._initialize_rules()
{
    'name': 'new_failure_mode',
    'conditions': [
        {'type': 'keyword', 'value': 'keyword', 'weight': 0.4},
        {'type': 'pattern', 'value': r'regex_pattern', 'weight': 0.5}
    ],
    'failure_code': 'XX.X',
    'description': 'New Failure Mode Description'
}
```

### 3. Modifying Classification Logic
- Update confidence thresholds
- Modify classifier order in hybrid classification
- Adjust scoring algorithms

## Troubleshooting

### 1. Common Issues
- **Import errors**: Check virtual environment and dependencies
- **File format errors**: Validate Excel file structure
- **Memory issues**: Process smaller datasets or optimize code
- **GUI freezing**: Use threading for long operations

### 2. Debugging
- Check log files (`matching_log.txt`)
- Use debug scripts for specific components
- Enable verbose logging for detailed output

### 3. Performance Issues
- Clear AI cache if corrupted
- Check for memory leaks in matplotlib
- Optimize pandas operations for large datasets

## Future Enhancements

### 1. Planned Features
- Machine learning model training
- Advanced visualization options
- Real-time data integration
- Mobile application

### 2. Scalability Improvements
- Database integration
- Cloud deployment options
- API endpoints for external integration
- Multi-user support

## Getting Started for New Developers

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd WOAnalysisProgram

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. First Run
```bash
# Run the application
python WorkOrderAnalysisCur2.py

# Test with sample data
# 1. Load test_files/standard_format.xlsx
# 2. Load failure_mode_dictionary_2.xlsx
# 3. Enable AI classification
# 4. Process files
```

### 3. Understanding the Code
1. Start with `WorkOrderAnalysisCur2.py` main class
2. Review `ai_failure_classifier.py` for AI logic
3. Examine test files for data format examples
4. Run test scripts to understand functionality

### 4. Making Changes
1. Create feature branch
2. Implement changes with proper error handling
3. Add tests for new functionality
4. Update documentation
5. Test with various data formats
6. Submit pull request

## Support and Resources

### 1. Documentation
- This developer guide
- Code comments and docstrings
- User guide in application

### 2. Logs and Debugging
- `matching_log.txt`: Application logs
- Console output for real-time debugging
- Test script outputs for validation

### 3. Data Files
- Sample data in `test_files/`
- Dictionary file for reference
- Training data exports for analysis

This developer guide provides a comprehensive overview of the Work Order Analysis Program. For specific implementation details, refer to the code comments and docstrings within each file. 