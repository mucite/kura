@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Kura Medical — Windows Release Builder
REM
REM  Produces a signed Inno Setup installer (.exe).
REM  No Microsoft account or paid certificate required.
REM
REM  First-time setup (run once):
REM    PowerShell -ExecutionPolicy Bypass -File setup_codesign.ps1
REM    setx CERT_PFX "%cd%\kura_codesign.pfx"
REM    setx CERT_PASS "your-pfx-password"
REM
REM  Then build every release with:
REM    build_release.bat v2026.1
REM
REM  After each build, whitelist with Microsoft (free, 1-3 days):
REM    submit_to_microsoft.bat v2026.1
REM ============================================================

SET VERSION=%~1
IF "%VERSION%"=="" SET VERSION=v2026

REM Strip leading "v" for numeric version
SET NUMERIC_VERSION=%VERSION:v=%

echo.
echo ============================================
echo   Kura Windows Release %VERSION%
echo ============================================
echo.

cd /d "%~dp0"

REM ── Dependency checks ────────────────────────────────────────────────────
where pyinstaller >nul 2>&1
if errorlevel 1 (
  echo ERROR: pyinstaller not found.
  echo   Run: pip install pyinstaller
  exit /b 1
)

SET "ISCC="
where iscc >nul 2>&1
if not errorlevel 1 (
  SET "ISCC=iscc"
) else if exist "%ProgramFiles(x86)%\Inno Setup 6\iscc.exe" (
  SET "ISCC=%ProgramFiles(x86)%\Inno Setup 6\iscc.exe"
) else if exist "%ProgramFiles%\Inno Setup 6\iscc.exe" (
  SET "ISCC=%ProgramFiles%\Inno Setup 6\iscc.exe"
) else (
  echo ERROR: Inno Setup not found.
  echo   Install free from https://jrsoftware.org/isdl.php
  exit /b 1
)

REM ── Check signing configuration ───────────────────────────────────────────
SET "SIGNING_ENABLED=0"
SET "SIGN_ARGS="

IF DEFINED CERT_PFX IF DEFINED CERT_PASS (
  IF EXIST "%CERT_PFX%" (
    SET "SIGNING_ENABLED=1"

    REM Locate signtool.exe (ships with Windows SDK / Visual Studio)
    SET "SIGNTOOL="
    FOR /F "delims=" %%P IN ('where signtool 2^>nul') DO SET "SIGNTOOL=%%P"
    IF NOT DEFINED SIGNTOOL (
      REM Search common SDK paths
      FOR %%D IN (
        "%ProgramFiles(x86)%\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"
        "%ProgramFiles(x86)%\Windows Kits\10\bin\10.0.19041.0\x64\signtool.exe"
        "%ProgramFiles(x86)%\Windows Kits\10\bin\x64\signtool.exe"
      ) DO IF EXIST "%%~D" SET "SIGNTOOL=%%~D"
    )

    IF NOT DEFINED SIGNTOOL (
      echo WARNING: signtool.exe not found. Signing disabled.
      echo   Install Windows SDK (free): https://developer.microsoft.com/windows/downloads/windows-sdk/
      SET "SIGNING_ENABLED=0"
    ) ELSE (
      REM Build the SignTool command Inno Setup will call for each file ($f)
      REM /tr = RFC 3161 timestamp server (free), ensures signature survives cert expiry
      SET "SIGN_ARGS=/DSIGN_TOOL_CMD=""%SIGNTOOL%"" sign /f ""%CERT_PFX%"" /p ""%CERT_PASS%"" /fd SHA256 /td SHA256 /tr http://timestamp.sectigo.com /v"""
    )
  ) ELSE (
    echo WARNING: CERT_PFX file not found: %CERT_PFX%
    echo   Run setup_codesign.ps1 first. Building unsigned.
  )
)

IF "%SIGNING_ENABLED%"=="1" (
  echo Signing:  ENABLED  [%CERT_PFX%]
) ELSE (
  echo Signing:  DISABLED  ^(set CERT_PFX + CERT_PASS to enable^)
)
echo.

REM ── Step 1/4: Build Kura.exe ──────────────────────────────────────────────
echo Step 1/4 - Building Kura.exe with PyInstaller...
call build_windows.bat
if not exist "dist\Kura\Kura.exe" (
  echo ERROR: PyInstaller build failed.
  exit /b 1
)
echo    OK: dist\Kura\Kura.exe

REM ── Step 2/4: Sign Kura.exe (before packaging into installer) ────────────
IF "%SIGNING_ENABLED%"=="1" (
  echo.
  echo Step 2/4 - Signing Kura.exe...
  "%SIGNTOOL%" sign /f "%CERT_PFX%" /p "%CERT_PASS%" /fd SHA256 /td SHA256 /tr http://timestamp.sectigo.com /v "dist\Kura\Kura.exe"
  if errorlevel 1 (
    echo ERROR: Signing Kura.exe failed. Aborting.
    exit /b 1
  )
  echo    OK
) ELSE (
  echo.
  echo Step 2/4 - Signing Kura.exe... SKIPPED ^(no certificate^)
)

REM ── Step 3/4: Compile Inno Setup installer ────────────────────────────────
echo.
echo Step 3/4 - Compiling Inno Setup installer...

IF "%SIGNING_ENABLED%"=="1" (
  REM Register the sign tool with Inno Setup so it signs the installer .exe too
  SET "ISS_SIGNTOOL=/DSIGN_TOOL_NAME=kura_sign /DSIGN_CMD=""%SIGNTOOL%"" sign /f ""%CERT_PFX%"" /p ""%CERT_PASS%"" /fd SHA256 /td SHA256 /tr http://timestamp.sectigo.com /v $f"""
)

"%ISCC%" ^
  /DAppVersion=%NUMERIC_VERSION% ^
  /DCERT_PFX="%CERT_PFX%" ^
  /O"dist" ^
  /F"Kura_Setup_%NUMERIC_VERSION%" ^
  "Kura.iss"

if not exist "dist\Kura_Setup_%NUMERIC_VERSION%.exe" (
  echo.
  echo ERROR: Inno Setup compilation failed.
  exit /b 1
)
echo    OK: dist\Kura_Setup_%NUMERIC_VERSION%.exe

REM ── Step 3b: Sign the installer .exe itself ───────────────────────────────
IF "%SIGNING_ENABLED%"=="1" (
  echo    Signing installer .exe...
  "%SIGNTOOL%" sign /f "%CERT_PFX%" /p "%CERT_PASS%" /fd SHA256 /td SHA256 /tr http://timestamp.sectigo.com /v "dist\Kura_Setup_%NUMERIC_VERSION%.exe"
  if errorlevel 1 (
    echo WARNING: Signing installer failed. Continuing without signature.
  ) ELSE (
    echo    OK
  )
)

REM ── Step 4/4: SHA-256 checksum ────────────────────────────────────────────
echo.
echo Step 4/4 - SHA-256 checksum...
cd dist
powershell -NoProfile -NonInteractive -Command ^
  "$h = (Get-FileHash -Algorithm SHA256 'Kura_Setup_%NUMERIC_VERSION%.exe').Hash.ToLower(); $h + '  Kura_Setup_%NUMERIC_VERSION%.exe' | Out-File 'Kura_Setup_%NUMERIC_VERSION%.exe.sha256' -Encoding ASCII -NoNewline"
echo    OK:
type "Kura_Setup_%NUMERIC_VERSION%.exe.sha256"
cd ..

REM ── Summary ───────────────────────────────────────────────────────────────
echo.
echo ============================================
echo   Build complete!
echo ============================================
echo.
echo Output:
echo   dist\Kura_Setup_%NUMERIC_VERSION%.exe
echo   dist\Kura_Setup_%NUMERIC_VERSION%.exe.sha256
echo.
IF "%SIGNING_ENABLED%"=="1" (
  echo Signed with: %CERT_PFX%
  echo SmartScreen: Yellow warning until Microsoft whitelists it ^(1-3 days^)
  echo.
  echo Next step — whitelist with Microsoft for free:
  echo   submit_to_microsoft.bat %VERSION%
) ELSE (
  echo Signed: NO — customers will see SmartScreen red warning.
  echo.
  echo To enable signing ^(free, one-time setup^):
  echo   1. PowerShell -ExecutionPolicy Bypass -File setup_codesign.ps1
  echo   2. setx CERT_PFX "%%cd%%\kura_codesign.pfx"
  echo   3. setx CERT_PASS "your-password"
  echo   4. Reopen terminal and re-run: build_release.bat %VERSION%
)
echo.
endlocal
