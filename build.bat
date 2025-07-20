@echo off
echo === WO Analysis Program Build Script ===
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
echo Checking PyInstaller...
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous builds if requested
if "%1"=="clean" (
    echo Cleaning previous builds...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist __pycache__ rmdir /s /q __pycache__
    for /r . %%f in (*.pyc) do del "%%f" 2>nul
)

REM Build the executable
echo Building executable...
pyinstaller WOAnalysisProgram.spec

if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo === BUILD SUCCESSFUL ===
echo Executable created in: dist\WOAnalysisProgram.exe
echo.

REM Check file size
if exist "dist\WOAnalysisProgram.exe" (
    for %%A in ("dist\WOAnalysisProgram.exe") do echo File size: %%~zA bytes
)

echo.
echo Build process completed!
pause 