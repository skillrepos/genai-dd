@echo off
REM setup.bat - Setup environment for genai-dd (Windows cmd version)

echo [1/7] Checking Python installation...
where python >nul 2>nul
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.
    exit /b 1
)

echo [2/7] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

echo [3/7] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo [4/7] Upgrading pip...
python -m pip install --upgrade pip

echo [5/7] Installing requirements (excluding torch)...
powershell -Command "Get-Content requirements\requirements.txt | Where-Object {$_ -notmatch '^torch'} | Set-Content tmp_requirements.txt"
pip install -r tmp_requirements.txt

echo [6/7] Pinning numpy to <2
pip install "numpy<2"

echo [7/7] Installing PyTorch...
REM Detect architecture manually (assumes CPU-only torch for simplicity)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

echo.
echo ✅ Setup complete. To activate your environment:
echo    call .venv\Scripts\activate.bat

