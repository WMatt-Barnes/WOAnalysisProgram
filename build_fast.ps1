# PowerShell script to build WO Analysis Program with optimized startup
# This script creates a directory-based build for faster startup

param(
    [switch]$Clean,
    [switch]$Debug,
    [switch]$Console
)

Write-Host "=== WO Analysis Program Fast Build Script ===" -ForegroundColor Green
Write-Host ""

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to activate virtual environment
function Activate-VirtualEnvironment {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
        Write-Host "Virtual environment activated successfully." -ForegroundColor Green
    } else {
        Write-Host "ERROR: Virtual environment not found at venv\Scripts\Activate.ps1" -ForegroundColor Red
        Write-Host "Please ensure you have created a virtual environment first." -ForegroundColor Red
        exit 1
    }
}

# Function to clean previous builds
function Clean-Build {
    Write-Host "Cleaning previous build artifacts..." -ForegroundColor Yellow
    
    $foldersToClean = @("build", "dist", "__pycache__")
    
    foreach ($folder in $foldersToClean) {
        if (Test-Path $folder) {
            Remove-Item -Path $folder -Recurse -Force
            Write-Host "Removed $folder" -ForegroundColor Green
        }
    }
    
    # Clean .pyc files
    Get-ChildItem -Path . -Filter "*.pyc" -Recurse | Remove-Item -Force
    Write-Host "Cleaned .pyc files" -ForegroundColor Green
}

# Function to check dependencies
function Test-Dependencies {
    Write-Host "Checking dependencies..." -ForegroundColor Yellow
    
    # Check if PyInstaller is installed
    try {
        $pyinstallerVersion = python -c "import PyInstaller; print(PyInstaller.__version__)"
        Write-Host "PyInstaller version: $pyinstallerVersion" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: PyInstaller not found. Installing..." -ForegroundColor Red
        pip install pyinstaller
    }
    
    # Check if main application file exists
    if (-not (Test-Path "WorkOrderAnalysisCur2_optimized.py")) {
        Write-Host "ERROR: Optimized application file WorkOrderAnalysisCur2_optimized.py not found!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "All dependencies checked successfully." -ForegroundColor Green
}

# Function to build the executable
function Build-Executable {
    Write-Host "Building optimized executable (directory-based for faster startup)..." -ForegroundColor Yellow
    
    # Determine console setting
    $consoleFlag = ""
    if ($Console) {
        $consoleFlag = "--console"
        Write-Host "Building with console window enabled" -ForegroundColor Yellow
    } else {
        $consoleFlag = "--windowed"
        Write-Host "Building in windowed mode" -ForegroundColor Yellow
    }
    
    # Build command for directory-based build (faster startup)
    $buildCommand = "pyinstaller --onedir $consoleFlag --icon=icons/app_icon.ico --add-data `"app_config.json;.`" --add-data `"risk_presets.json;.`" --add-data `"failure_mode_dictionary_.xlsx;.`" --add-data `"failure_mode_dictionary_2.xlsx;.`" --add-data `"sample_failure_dictionary.xlsx;.`" --add-data `"icons;icons`" --add-data `"ai_classification_cache.json;.`" --name WOAnalysisProgram_Fast WorkOrderAnalysisCur2_optimized.py"
    
    Write-Host "Running: $buildCommand" -ForegroundColor Cyan
    Write-Host ""
    
    # Execute the build
    try {
        Invoke-Expression $buildCommand
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "=== BUILD SUCCESSFUL ===" -ForegroundColor Green
            Write-Host "Executable created in: dist\WOAnalysisProgram_Fast\WOAnalysisProgram_Fast.exe" -ForegroundColor Green
            
            # Check if executable was created
            if (Test-Path "dist\WOAnalysisProgram_Fast\WOAnalysisProgram_Fast.exe") {
                $fileSize = (Get-Item "dist\WOAnalysisProgram_Fast\WOAnalysisProgram_Fast.exe").Length
                $fileSizeKB = [math]::Round($fileSize / 1KB, 2)
                Write-Host "Executable size: $fileSizeKB KB" -ForegroundColor Green
                
                # Calculate total directory size
                $totalSize = (Get-ChildItem "dist\WOAnalysisProgram_Fast" -Recurse | Measure-Object -Property Length -Sum).Sum
                $totalSizeMB = [math]::Round($totalSize / 1MB, 2)
                Write-Host "Total package size: $totalSizeMB MB" -ForegroundColor Green
            }
        } else {
            Write-Host "ERROR: Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "ERROR: Build failed with exception: $_" -ForegroundColor Red
        exit 1
    }
}

# Function to create distribution package
function Create-DistributionPackage {
    Write-Host "Creating distribution package..." -ForegroundColor Yellow
    
    $sourceFolder = "dist\WOAnalysisProgram_Fast"
    $packageFolder = "dist\WOAnalysisProgram_Fast_Package"
    
    # Create package directory
    if (Test-Path $packageFolder) {
        Remove-Item -Path $packageFolder -Recurse -Force
    }
    New-Item -ItemType Directory -Path $packageFolder | Out-Null
    
    # Copy entire directory
    Copy-Item -Path $sourceFolder -Destination $packageFolder -Recurse -Force
    
    # Copy documentation
    $docs = @("README.md", "SOFTWARE_USER_GUIDE.md", "TECHNICAL_APPLICATION_GUIDE.md", "BUILD_GUIDE.md")
    foreach ($doc in $docs) {
        if (Test-Path $doc) {
            Copy-Item -Path $doc -Destination $packageFolder -Force
        }
    }
    
    # Copy sample data
    if (Test-Path "sample_data") {
        Copy-Item -Path "sample_data" -Destination $packageFolder -Recurse -Force
    }
    
    # Copy test files
    if (Test-Path "test_files") {
        Copy-Item -Path "test_files" -Destination $packageFolder -Recurse -Force
    }
    
    # Create a simple batch file to run the program
    $batchContent = @"
@echo off
echo Starting WO Analysis Program (Fast Version)...
cd "%~dp0"
"WOAnalysisProgram_Fast.exe"
pause
"@
    $batchContent | Out-File -FilePath "$packageFolder\Run_WOAnalysisProgram_Fast.bat" -Encoding ASCII
    
    # Create a README for the package
    $readmeContent = @"
# WO Analysis Program - Fast Version

This is the optimized version of the WO Analysis Program with faster startup times.

## How to Run

1. Double-click `Run_WOAnalysisProgram_Fast.bat` to start the program
2. Or double-click `WOAnalysisProgram_Fast.exe` directly

## Benefits of This Version

- **Faster Startup**: Uses directory-based packaging instead of single-file
- **Lower Memory Usage**: No extraction required at startup
- **Better Performance**: Optimized imports and deferred loading

## System Requirements

- Windows 10/11 (64-bit)
- No Python installation required
- Minimum 4GB RAM recommended
- 500MB free disk space

## Troubleshooting

If the program doesn't start:
1. Try running `WOAnalysisProgram_Fast.exe` from the command line to see error messages
2. Ensure all files in this directory are present
3. Check that your antivirus isn't blocking the executable

## Support

For additional help, refer to the documentation files included in this package.
"@
    $readmeContent | Out-File -FilePath "$packageFolder\README_FAST_VERSION.md" -Encoding UTF8
    
    Write-Host "Distribution package created at: $packageFolder" -ForegroundColor Green
}

# Function to show performance comparison
function Show-PerformanceComparison {
    Write-Host ""
    Write-Host "=== PERFORMANCE COMPARISON ===" -ForegroundColor Cyan
    Write-Host "Single-file build (--onefile):" -ForegroundColor Yellow
    Write-Host "  - Slower startup (extraction required)" -ForegroundColor White
    Write-Host "  - Higher memory usage" -ForegroundColor White
    Write-Host "  - Single executable file" -ForegroundColor White
    Write-Host ""
    Write-Host "Directory-based build (--onedir):" -ForegroundColor Yellow
    Write-Host "  - Faster startup (no extraction)" -ForegroundColor White
    Write-Host "  - Lower memory usage" -ForegroundColor White
    Write-Host "  - Multiple files in directory" -ForegroundColor White
    Write-Host ""
    Write-Host "This build uses the directory-based approach for optimal performance." -ForegroundColor Green
}

# Main execution
try {
    # Activate virtual environment
    Activate-VirtualEnvironment
    
    # Clean if requested
    if ($Clean) {
        Clean-Build
    }
    
    # Check dependencies
    Test-Dependencies
    
    # Build executable
    Build-Executable
    
    # Create distribution package
    Create-DistributionPackage
    
    # Show performance comparison
    Show-PerformanceComparison
    
    Write-Host ""
    Write-Host "=== BUILD PROCESS COMPLETED ===" -ForegroundColor Green
    Write-Host "Your optimized distributable is ready in the dist folder!" -ForegroundColor Green
    Write-Host "Expected startup time improvement: 50-80% faster" -ForegroundColor Green
    
} catch {
    Write-Host "ERROR: Build process failed: $_" -ForegroundColor Red
    exit 1
} 