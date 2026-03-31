@echo off
REM ============================================================
REM  Kura Medical — Microsoft SmartScreen Whitelist Submission
REM
REM  Run this AFTER building each new release.
REM
REM  What this does:
REM    Generates the SHA-256 hash of your installer and opens
REM    the Microsoft Security Intelligence submission portal.
REM    You paste the hash and submit the file for free review.
REM
REM  Timeline: Microsoft typically reviews within 1-3 business days.
REM  Result:   SmartScreen stops warning about that specific binary.
REM
REM  You must resubmit for each new release version.
REM ============================================================

setlocal

SET VERSION=%~1
IF "%VERSION%"=="" (
    REM Try to find the latest installer in dist\
    FOR /F "delims=" %%F IN ('dir /B /O-D "dist\Kura_Setup_*.exe" 2^>nul') DO (
        SET "INSTALLER=dist\%%F"
        GOTO :found
    )
    echo ERROR: No installer found in dist\.
    echo Run build_release.bat first, or pass the version: submit_to_microsoft.bat v2026.1
    exit /b 1
    :found
) ELSE (
    SET "INSTALLER=dist\Kura_Setup_%VERSION:v=%.exe"
)

IF NOT EXIST "%INSTALLER%" (
    echo ERROR: Installer not found: %INSTALLER%
    exit /b 1
)

echo.
echo ============================================
echo   SmartScreen Whitelist Submission
echo ============================================
echo.
echo Installer: %INSTALLER%
echo.

REM ── Compute SHA-256 ───────────────────────────────────────────────────────
echo Computing SHA-256...
FOR /F "tokens=*" %%H IN ('powershell -NoProfile -Command "Get-FileHash -Algorithm SHA256 '%INSTALLER%' | Select-Object -ExpandProperty Hash"') DO SET "HASH=%%H"

echo.
echo   SHA-256: %HASH%
echo.

REM ── Copy hash to clipboard ────────────────────────────────────────────────
echo %HASH% | clip
echo   (Hash copied to clipboard)
echo.

REM ── Instructions ─────────────────────────────────────────────────────────
echo ============================================
echo   Submission instructions
echo ============================================
echo.
echo  1. The Microsoft submission portal is about to open in your browser.
echo.
echo  2. On the portal:
echo       - Click "Submit a file"
echo       - Select "I believe this file is safe"
echo       - File submission type: "Incorrectly detected as malware/malicious"
echo       - Attach:  %INSTALLER%
echo       - Describe: "Kura Medical - AI documentation for physiotherapy.
echo                    100%% local processing, no cloud. GDPR-compliant."
echo.
echo  3. Note your submission ID — Microsoft emails you the result.
echo.
echo  4. Approval usually takes 1-3 business days.
echo     After approval: SmartScreen will no longer warn about this installer.
echo.
echo  5. Repeat for EACH new release version.
echo.
echo  IMPORTANT: You also need to submit Kura.exe (the main app),
echo  not just the installer. Do both.
echo.
pause

REM ── Open Microsoft portal ─────────────────────────────────────────────────
start "" "https://www.microsoft.com/en-us/wdsi/filesubmission"

echo.
echo Portal opened. Good luck!
echo.
endlocal
