@echo off
setlocal

set "VENV_DIR=.venv"
set "REQUIREMENTS=requirements.txt"

echo [1/4] Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python was not found.
    pause
    exit /b 1
)

echo [2/4] Create venv
if not exist "%VENV_DIR%\Scripts\python.exe" (
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
) else (
    echo venv already exists.
)

echo [3/4] Upgrade pip
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate venv.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

echo [4/4] Install requirements
if exist "%REQUIREMENTS%" (
    python -m pip install -r "%REQUIREMENTS%"
    if errorlevel 1 (
        echo ERROR: Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    echo WARNING: requirements.txt was not found.
)

echo.
echo Done.
echo venv: %VENV_DIR%
echo.
pause

endlocal