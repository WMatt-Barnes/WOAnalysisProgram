# Software User Guide - Work Order Analysis Pro

## Table of Contents
1. [Getting Started](#getting-started)
2. [Menu Navigation](#menu-navigation)
3. [Tab Interface](#tab-interface)
4. [File Operations](#file-operations)
5. [Processing Functions](#processing-functions)
6. [Analysis Features](#analysis-features)
7. [AI Features](#ai-features)
8. [Tools and Utilities](#tools-and-utilities)
9. [Export and Reporting](#export-and-reporting)
10. [Keyboard Shortcuts](#keyboard-shortcuts)

## Getting Started

### Launching the Application
- Start the application by running `python WorkOrderAnalysisCur2.py`
- The main window will open with a tabbed interface
- The application uses a modern theme with resizable panes

### Initial Setup
1. **Load Work Order File**: Click "Browse" next to "Work Order File" or use File → Load Work Order File
2. **Load Dictionary File**: The default dictionary path is pre-filled, or browse to select a custom dictionary
3. **Set Output Directory**: Choose where to save analysis results and reports
4. **Process Files**: Click "Process Files" to begin analysis

## Menu Navigation

### File Menu
- **Load Work Order File...** (Ctrl+O): Browse and select Excel files containing work order data
- **Load Dictionary File...** (Ctrl+D): Select the failure mode dictionary file
- **Set Output Directory...**: Choose where to save exported files and reports
- **Export to Excel...** (Ctrl+E): Export current analysis results to Excel format
- **Export Report...**: Generate comprehensive analysis reports
- **Exit** (Ctrl+Q): Close the application

### Process Menu
- **Process Files** (F5): Execute the main analysis workflow
- **Batch Process...**: Process multiple work order files simultaneously
- **Clear Data**: Remove all loaded data and reset the interface

### Analysis Menu
- **View Work Orders**: Switch to the Analysis tab showing work order data
- **View Equipment Summary**: Display equipment-level summary statistics
- **Crow-AMSAA Analysis**: Show reliability analysis plots
- **Risk Assessment**: Access risk calculation and assessment tools

### AI Menu
- **Enable AI Classification**: Toggle AI-powered failure mode classification on/off
- **AI Settings...**: Configure AI parameters and confidence thresholds
- **AI Statistics**: View performance metrics and classification statistics
- **Clear AI Cache**: Remove cached AI classification results
- **Export Training Data**: Export classified data for model improvement

### Tools Menu
- **Column Mapping...**: Map CMMS export columns to required program columns
- **Filter Management...**: Access advanced filtering options and presets
- **Date Range Selector...**: Quick date range selection tools
- **Reset All Filters**: Clear all applied filters
- **Open Output Folder**: Open the output directory in file explorer
- **View FMEA Export Data...**: Review data prepared for FMEA export

### Help Menu
- **Software User Guide**: This comprehensive GUI navigation guide
- **Technical Application Guide**: Detailed technical explanations of analyses
- **About**: Application version and developer information
- **Check for Updates**: Verify if newer versions are available

## Tab Interface

### 📁 Data Input Tab
**Purpose**: Load and configure data sources

**Components**:
- **Work Order File**: Select Excel file containing work order data
- **Dictionary File**: Choose failure mode classification dictionary
- **Output Directory**: Set location for exported files
- **Action Buttons**:
  - 🚀 Process Files: Execute analysis
  - 📊 Export to Excel: Export current data
  - 📁 Open Output: Open output folder
  - 🗑️ Clear Data: Reset all data

**Usage**:
1. Browse to select work order file (Excel format)
2. Verify dictionary file path (default provided)
3. Set output directory for results
4. Click "Process Files" to begin analysis

### 📈 Analysis Tab
**Purpose**: View and filter work order data with interactive analysis

**Components**:
- **Filters Panel**:
  - Equipment dropdown: Filter by specific equipment
  - Failure Code Source: Choose between AI/Dictionary or User classifications
  - Failure Code dropdown: Filter by failure mode
  - Work Type dropdown: Filter by work order type
  - Date Range: Set start and end dates
  - Filter buttons: Apply, Clear All, Reset Defaults

- **Data Display**:
  - Work Orders table: Detailed work order information with checkboxes
  - Equipment Summary table: Equipment-level statistics
  - Crow-AMSAA Plot: Interactive reliability analysis chart

**Interactive Features**:
- **Row Selection**: Click checkboxes to include/exclude work orders from analysis
- **Column Sorting**: Click column headers to sort data
- **Plot Interaction**: Right-click on plot points to segment analysis
- **Context Menu**: Right-click work orders for additional options

### 🤖 AI Settings Tab
**Purpose**: Configure AI classification parameters

**Components**:
- **Enable AI Classification**: Master toggle for AI features
- **Confidence Threshold**: Adjustable slider (0.0-1.0) for classification confidence
- **AI Statistics**: View classification performance metrics
- **Cache Management**: Clear AI classification cache

**Configuration**:
1. Check "Enable AI Classification" to activate AI features
2. Adjust confidence threshold based on desired accuracy
3. Monitor AI statistics for performance insights
4. Clear cache if needed for fresh classifications

### ⚠️ Risk Assessment Tab
**Purpose**: Calculate and visualize risk metrics

**Components**:
- **Risk Parameters**:
  - Annual Operating Hours: Equipment operating time per year
  - Cost per Failure: Average cost of each failure
  - Risk Threshold: Maximum acceptable risk level
- **Risk Calculation**: Automatic calculation based on current data
- **Risk Plot**: Visual representation of risk distribution
- **Preset Management**: Save and load risk parameter configurations

**Usage**:
1. Set operating parameters (hours, costs, thresholds)
2. Review calculated risk metrics
3. Adjust parameters as needed
4. Save configurations for future use

### 📊 Weibull Analysis Tab
**Purpose**: Perform Weibull reliability analysis

**Components**:
- **Equipment Selection**: Choose equipment for analysis
- **Failure Mode Selection**: Select specific failure modes
- **Weibull Plot**: Visual representation of failure distribution
- **Parameter Display**: Beta (shape) and Eta (scale) parameters
- **Work Order Table**: Detailed failure time data

**Features**:
- Interactive plot with confidence bounds
- Parameter estimation and goodness-of-fit testing
- Export capabilities for plots and results

### 🔧 PM Analysis Tab
**Purpose**: Preventive maintenance optimization

**Components**:
- **Equipment Selection**: Choose equipment for PM analysis
- **Analysis Results**: PM frequency recommendations
- **Comparison Plots**: Visual comparison of different PM strategies
- **Cost Analysis**: Economic evaluation of PM options

**Outputs**:
- Recommended PM frequencies
- Cost-benefit analysis
- Failure rate comparisons
- Exportable reports and plots

### 📦 Spares Analysis Tab
**Purpose**: Spare parts demand forecasting and stocking recommendations

**Components**:
- **Equipment Selection**: Choose equipment for spares analysis
- **Demand Analysis**: Historical spares demand patterns
- **Forecasting**: Monte Carlo simulation results
- **Stocking Recommendations**: Service level-based recommendations

**Features**:
- Multi-year demand forecasting
- Service level sensitivity analysis
- Cost optimization recommendations
- Interactive plots and reports

## File Operations

### Loading Work Order Files
1. **Supported Formats**: Excel (.xlsx) files
2. **Required Columns**: Work order data with descriptions, dates, equipment info
3. **Column Mapping**: Use Tools → Column Mapping for CMMS compatibility
4. **File Validation**: Automatic validation of required data fields

### Loading Dictionary Files
1. **Format**: Excel file with failure mode definitions
2. **Structure**: Keywords, failure codes, descriptions
3. **Default Dictionary**: Pre-configured dictionary provided
4. **Custom Dictionaries**: Support for user-defined failure modes

### Output Directory Management
1. **Default Location**: Application directory
2. **Custom Paths**: User-selectable output locations
3. **File Organization**: Automatic organization of exported files
4. **Quick Access**: Tools → Open Output Folder

## Processing Functions

### Single File Processing
1. **Data Validation**: Automatic checking of file format and content
2. **AI Classification**: Intelligent failure mode assignment
3. **Dictionary Matching**: Traditional keyword-based classification
4. **Progress Tracking**: Real-time progress updates
5. **Error Handling**: Comprehensive error reporting

### Batch Processing
1. **Multiple Files**: Process multiple work order files simultaneously
2. **File Selection**: Add/remove files from batch queue
3. **Output Management**: Organized output for each file
4. **Progress Monitoring**: Overall and per-file progress tracking
5. **Error Recovery**: Continue processing despite individual file errors

## Analysis Features

### Data Filtering
- **Equipment Filter**: Focus on specific equipment types
- **Failure Mode Filter**: Analyze specific failure patterns
- **Date Range Filter**: Time-based data selection
- **Work Type Filter**: Filter by maintenance activity type
- **Combined Filters**: Multiple filter criteria simultaneously

### Interactive Tables
- **Sortable Columns**: Click headers to sort data
- **Row Selection**: Checkbox-based work order inclusion
- **Cell Editing**: Double-click to edit failure codes
- **Context Menus**: Right-click for additional options
- **Export Options**: Direct export from table views

### Plot Interactions
- **Crow-AMSAA Plots**: Interactive reliability analysis
- **Point Selection**: Click to highlight specific work orders
- **Segmentation**: Right-click to create time-based segments
- **Zoom and Pan**: Interactive plot navigation
- **Export Options**: Save plots in various formats

## AI Features

### Classification Methods
1. **AI Classification**: Intelligent semantic analysis
2. **Dictionary Matching**: Traditional keyword-based approach
3. **Hybrid System**: Combines AI and dictionary methods
4. **Confidence Scoring**: Reliability indicators for each classification

### Configuration Options
- **Confidence Threshold**: Adjustable classification sensitivity
- **Model Selection**: Choose between different AI models
- **Cache Management**: Performance optimization through caching
- **Training Data**: Export for model improvement

### Performance Monitoring
- **Statistics Dashboard**: Classification performance metrics
- **Method Distribution**: Breakdown of classification methods used
- **Confidence Distribution**: Distribution of confidence scores
- **Cache Performance**: Cache hit rates and performance

## Tools and Utilities

### Column Mapping
1. **CMMS Compatibility**: Map various CMMS export formats
2. **Auto-Detection**: Automatic column identification
3. **Manual Mapping**: Custom column assignments
4. **Mapping Storage**: Save and reuse column mappings

### Filter Management
1. **Preset Filters**: Save commonly used filter combinations
2. **Quick Filters**: Pre-defined filter sets
3. **Filter Export**: Share filter configurations
4. **Filter Reset**: Quick return to default state

### Date Range Selection
1. **Quick Ranges**: Last 7/30/90 days, current month/year
2. **Custom Ranges**: User-defined date periods
3. **Range Clearing**: Remove date restrictions
4. **Range Validation**: Automatic date format checking

## Export and Reporting

### Excel Export
- **Complete Data**: All work order and analysis data
- **Formatted Output**: Professional formatting and styling
- **Multiple Sheets**: Organized data across multiple worksheets
- **Calculations**: Embedded formulas and summaries

### Report Generation
- **Comprehensive Reports**: Detailed analysis summaries
- **Customizable Content**: Select report sections
- **Professional Formatting**: Ready for presentation
- **Multiple Formats**: Excel, PDF, and text formats

### Plot Export
- **High Resolution**: Publication-quality images
- **Multiple Formats**: PNG, JPG, PDF, SVG
- **Custom Sizing**: Adjustable output dimensions
- **Metadata**: Include analysis parameters and timestamps

## Keyboard Shortcuts

### File Operations
- **Ctrl+O**: Load work order file
- **Ctrl+D**: Load dictionary file
- **Ctrl+E**: Export to Excel
- **Ctrl+Q**: Exit application

### Processing
- **F5**: Process files
- **F6**: Batch process (if available)

### Navigation
- **Tab**: Navigate between interface elements
- **Enter**: Activate buttons and confirm selections
- **Escape**: Cancel dialogs and close windows

### Table Operations
- **Space**: Toggle row selection (checkboxes)
- **Enter**: Edit selected cell
- **Arrow Keys**: Navigate table cells
- **Ctrl+A**: Select all rows

### Plot Interactions
- **Left Click**: Select plot points
- **Right Click**: Context menu
- **Mouse Wheel**: Zoom in/out
- **Drag**: Pan around plot

## Troubleshooting

### Common Issues
1. **File Loading Errors**: Check file format and column structure
2. **Processing Failures**: Verify data quality and required fields
3. **AI Classification Issues**: Check confidence thresholds and model availability
4. **Export Problems**: Ensure output directory permissions

### Performance Optimization
1. **Large Files**: Use batch processing for multiple files
2. **AI Processing**: Adjust confidence thresholds for speed/accuracy balance
3. **Memory Usage**: Clear data between large file processing
4. **Cache Management**: Clear AI cache if experiencing issues

### Data Quality
1. **Required Fields**: Ensure all mandatory columns are present
2. **Date Formats**: Use consistent date formatting
3. **Equipment Names**: Maintain consistent equipment naming
4. **Failure Descriptions**: Provide detailed, consistent descriptions

## Support and Resources

### Built-in Help
- **User Guide**: This comprehensive navigation guide
- **Technical Guide**: Detailed analysis methodology explanations
- **About Dialog**: Version information and credits

### External Resources
- **Documentation**: Check for additional documentation files
- **Sample Data**: Use provided sample files for testing
- **Configuration Files**: Review and modify application settings

### Best Practices
1. **Regular Backups**: Save important configurations and mappings
2. **Data Validation**: Verify data quality before processing
3. **Incremental Analysis**: Start with small datasets for testing
4. **Documentation**: Keep records of analysis parameters and results 