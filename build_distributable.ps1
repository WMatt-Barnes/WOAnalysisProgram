# PowerShell script to build WO Analysis Program distributable
# This script automates the PyInstaller build process

param(
    [switch]$Clean,
    [switch]$Debug,
    [switch]$Console
)

Write-Host "=== WO Analysis Program Build Script ===" -ForegroundColor Green
Write-Host ""

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

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
    if (-not (Test-Path "WorkOrderAnalysisCur2.py")) {
        Write-Host "ERROR: Main application file WorkOrderAnalysisCur2.py not found!" -ForegroundColor Red
        exit 1
    }
    
    # Check if spec file exists
    if (-not (Test-Path "WOAnalysisProgram.spec")) {
        Write-Host "ERROR: Spec file WOAnalysisProgram.spec not found!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "All dependencies checked successfully." -ForegroundColor Green
}

# Function to build the executable
function Build-Executable {
    Write-Host "Building executable..." -ForegroundColor Yellow
    
    # Determine console setting
    $consoleFlag = ""
    if ($Console) {
        $consoleFlag = "--console"
        Write-Host "Building with console window enabled" -ForegroundColor Yellow
    }
    
    # Determine debug setting
    $debugFlag = ""
    if ($Debug) {
        $debugFlag = "--debug"
        Write-Host "Building in debug mode" -ForegroundColor Yellow
    }
    
    # Build command
    $buildCommand = "pyinstaller WOAnalysisProgram.spec $consoleFlag $debugFlag"
    
    Write-Host "Running: $buildCommand" -ForegroundColor Cyan
    Write-Host ""
    
    # Execute the build
    try {
        Invoke-Expression $buildCommand
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "=== BUILD SUCCESSFUL ===" -ForegroundColor Green
            Write-Host "Executable created in: dist\WOAnalysisProgram.exe" -ForegroundColor Green
            
            # Check if executable was created
            if (Test-Path "dist\WOAnalysisProgram.exe") {
                $fileSize = (Get-Item "dist\WOAnalysisProgram.exe").Length
                $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
                Write-Host "Executable size: $fileSizeMB MB" -ForegroundColor Green
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
    
    $distFolder = "dist\WOAnalysisProgram"
    $packageFolder = "dist\WOAnalysisProgram_Package"
    
    # Create package directory
    if (Test-Path $packageFolder) {
        Remove-Item -Path $packageFolder -Recurse -Force
    }
    New-Item -ItemType Directory -Path $packageFolder | Out-Null
    
    # Copy executable and dependencies
    Copy-Item -Path "dist\WOAnalysisProgram.exe" -Destination $packageFolder -Force
    
    # Copy documentation
    $docs = @("README.md", "SOFTWARE_USER_GUIDE.md", "TECHNICAL_APPLICATION_GUIDE.md")
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
echo Starting WO Analysis Program...
"WOAnalysisProgram.exe"
pause
"@
    $batchContent | Out-File -FilePath "$packageFolder\Run_WOAnalysisProgram.bat" -Encoding ASCII
    
    Write-Host "Distribution package created at: $packageFolder" -ForegroundColor Green
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
    
    Write-Host ""
    Write-Host "=== BUILD PROCESS COMPLETED ===" -ForegroundColor Green
    Write-Host "Your distributable is ready in the dist folder!" -ForegroundColor Green
    
} catch {
    Write-Host "ERROR: Build process failed: $_" -ForegroundColor Red
    exit 1
} 