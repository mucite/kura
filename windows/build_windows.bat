@echo off
REM Build Kura.exe for Windows using spec file.
REM No Apple Developer account or code signing required.
REM
REM Usage:
REM   build_windows.bat

cd /d "%~dp0"

echo Cleaning old builds...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo.
echo Building Kura.exe with PyInstaller...
echo.

py -3.12 -m PyInstaller Kura_windows.spec --noconfirm

if not exist "dist\Kura\Kura.exe" (
  echo.
  echo ERROR: Build failed — dist\Kura\Kura.exe not found.
  exit /b 1
)

echo.
echo Build complete: dist\Kura\Kura.exe