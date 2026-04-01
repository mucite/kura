@echo off
REM ============================================================
REM  Kura Build Verification Script
REM  Checks that all files are ready for release build
REM ============================================================

echo.
echo ============================================
echo   Kura v2026.3.2 Build Verification
echo ============================================
echo.

cd /d "%~dp0"

SET "ERRORS=0"
SET "WARNINGS=0"

REM ── Check core Python files ────────────────────────────────
echo [1/8] Checking core Python files...
IF NOT EXIST "main_windows.py" (
  echo   ERROR: main_windows.py not found
  SET /A ERRORS+=1
) ELSE (
  findstr /C:"APP_VERSION   = \"2026.3.2\"" main_windows.py >nul
  IF ERRORLEVEL 1 (
    echo   WARNING: APP_VERSION mismatch in main_windows.py
    SET /A WARNINGS+=1
  ) ELSE (
    echo   OK: main_windows.py [v2026.3.2]
  )
)

IF NOT EXIST "physio_scribe_crossplatform.py" (
  echo   ERROR: physio_scribe_crossplatform.py not found
  SET /A ERRORS+=1
) ELSE (
  findstr /C:"n_ctx=2048" physio_scribe_crossplatform.py >nul
  IF ERRORLEVEL 1 (
    echo   WARNING: Context window not set to 2048
    SET /A WARNINGS+=1
  ) ELSE (
    echo   OK: physio_scribe_crossplatform.py [n_ctx=2048]
  )
)

REM ── Check build files ──────────────────────────────────────
echo.
echo [2/8] Checking build configuration...
IF NOT EXIST "Kura_windows.spec" (
  echo   ERROR: Kura_windows.spec not found
  SET /A ERRORS+=1
) ELSE (
  findstr /C:"shared.billing_engine" Kura_windows.spec >nul
  IF ERRORLEVEL 1 (
    echo   WARNING: billing_engine not in hidden imports
    SET /A WARNINGS+=1
  ) ELSE (
    echo   OK: Kura_windows.spec [billing_engine included]
  )
)

IF NOT EXIST "Kura.iss" (
  echo   ERROR: Kura.iss not found
  SET /A ERRORS+=1
) ELSE (
  findstr /C:"AppVersion \"2026.3.2\"" Kura.iss >nul
  IF ERRORLEVEL 1 (
    echo   WARNING: Version mismatch in Kura.iss
    SET /A WARNINGS+=1
  ) ELSE (
    echo   OK: Kura.iss [v2026.3.2]
  )
)

REM ── Check models ───────────────────────────────────────────
echo.
echo [3/8] Checking AI models...
IF NOT EXIST "..\models\Llama-3.2-3B-Instruct-4bit-GGUF\Llama-3.2-3B-Instruct-Q4_K_M.gguf" (
  echo   ERROR: LLM model not found
  SET /A ERRORS+=1
) ELSE (
  echo   OK: Llama-3.2-3B-Instruct-Q4_K_M.gguf found
)

IF NOT EXIST "..\models\whisper\large-v3.pt" (
  echo   ERROR: Whisper model not found
  SET /A ERRORS+=1
) ELSE (
  echo   OK: whisper/large-v3.pt found
)

REM ── Check shared modules ───────────────────────────────────
echo.
echo [4/8] Checking shared modules...
IF NOT EXIST "..\shared\billing_engine.py" (
  echo   ERROR: billing_engine.py not found
  SET /A ERRORS+=1
) ELSE (
  echo   OK: billing_engine.py
)

IF NOT EXIST "..\shared\config_manager.py" (
  echo   ERROR: config_manager.py not found
  SET /A ERRORS+=1
) ELSE (
  echo   OK: config_manager.py
)

IF NOT EXIST "..\shared\license_manager.py" (
  echo   ERROR: license_manager.py not found
  SET /A ERRORS+=1
) ELSE (
  echo   OK: license_manager.py
)

REM ── Check Python dependencies ──────────────────────────────
echo.
echo [5/8] Checking Python dependencies...
python -c "import llama_cpp" 2>nul
IF ERRORLEVEL 1 (
  echo   ERROR: llama-cpp-python not installed
  SET /A ERRORS+=1
) ELSE (
  echo   OK: llama-cpp-python
)

python -c "import whisper" 2>nul
IF ERRORLEVEL 1 (
  echo   ERROR: openai-whisper not installed
  SET /A ERRORS+=1
) ELSE (
  echo   OK: openai-whisper
)

python -c "import PySimpleGUI" 2>nul
IF ERRORLEVEL 1 (
  echo   ERROR: PySimpleGUI not installed
  SET /A ERRORS+=1
) ELSE (
  echo   OK: PySimpleGUI
)

REM ── Check build tools ──────────────────────────────────────
echo.
echo [6/8] Checking build tools...
where pyinstaller >nul 2>&1
IF ERRORLEVEL 1 (
  echo   ERROR: pyinstaller not found
  echo          Install: pip install pyinstaller
  SET /A ERRORS+=1
) ELSE (
  echo   OK: pyinstaller
)

where iscc >nul 2>&1
IF NOT ERRORLEVEL 1 (
  echo   OK: Inno Setup
) ELSE IF EXIST "%ProgramFiles(x86)%\Inno Setup 6\iscc.exe" (
  echo   OK: Inno Setup [Program Files x86]
) ELSE IF EXIST "%ProgramFiles%\Inno Setup 6\iscc.exe" (
  echo   OK: Inno Setup [Program Files]
) ELSE (
  echo   WARNING: Inno Setup not found
  echo            Install from https://jrsoftware.org/isdl.php
  SET /A WARNINGS+=1
)

REM ── Check code signing ─────────────────────────────────────
echo.
echo [7/8] Checking code signing configuration...
IF DEFINED CERT_PFX (
  IF EXIST "%CERT_PFX%" (
    echo   OK: Certificate found [%CERT_PFX%]
    IF DEFINED CERT_PASS (
      echo   OK: Certificate password configured
    ) ELSE (
      echo   WARNING: CERT_PASS not set
      SET /A WARNINGS+=1
    )
  ) ELSE (
    echo   WARNING: CERT_PFX points to non-existent file
    SET /A WARNINGS+=1
  )
) ELSE (
  echo   INFO: Code signing not configured [build will be unsigned]
)

REM ── Syntax check ───────────────────────────────────────────
echo.
echo [8/8] Running syntax checks...
python -m py_compile main_windows.py 2>nul
IF ERRORLEVEL 1 (
  echo   ERROR: main_windows.py has syntax errors
  SET /A ERRORS+=1
) ELSE (
  echo   OK: main_windows.py syntax
)

python -m py_compile physio_scribe_crossplatform.py 2>nul
IF ERRORLEVEL 1 (
  echo   ERROR: physio_scribe_crossplatform.py has syntax errors
  SET /A ERRORS+=1
) ELSE (
  echo   OK: physio_scribe_crossplatform.py syntax
)

REM ── Summary ────────────────────────────────────────────────
echo.
echo ============================================
echo   Verification Summary
echo ============================================
echo.

IF %ERRORS% EQU 0 (
  IF %WARNINGS% EQU 0 (
    echo   Status: READY TO BUILD
    echo   Errors: 0
    echo   Warnings: 0
    echo.
    echo   Next steps:
    echo     1. Build: build_release.bat v2026.3.2
    echo     2. Test the installer on a clean Windows machine
    echo     3. Submit to Microsoft SmartScreen if signed
    echo.
    exit /b 0
  ) ELSE (
    echo   Status: READY WITH WARNINGS
    echo   Errors: 0
    echo   Warnings: %WARNINGS%
    echo.
    echo   You can proceed, but review warnings above.
    echo.
    exit /b 0
  )
) ELSE (
  echo   Status: NOT READY
  echo   Errors: %ERRORS%
  echo   Warnings: %WARNINGS%
  echo.
  echo   Fix errors above before building.
  echo.
  exit /b 1
)

