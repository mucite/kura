param([string]$Version = "2026.4.0")

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura - Build (No Setup Required)" -ForegroundColor Green
Write-Host "  Version: $Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$distPath = Join-Path $scriptDir "dist\Kura"
$outputDir = Join-Path $scriptDir "dist"

Write-Host "[1/3] Building PyInstaller..." -ForegroundColor Yellow
Push-Location $scriptDir
pyinstaller Kura_windows.spec -y
$success = $LASTEXITCODE -eq 0
Pop-Location

if (-not $success) {
    Write-Host "ERROR: PyInstaller failed" -ForegroundColor Red
    Write-Host "Run the command manually to see detailed errors:" -ForegroundColor Yellow
    Write-Host "  cd windows" -ForegroundColor Gray
    Write-Host "  pyinstaller Kura_windows.spec" -ForegroundColor Gray
    exit 1
}

# Copy models folder to dist
Write-Host "  Copying AI models..." -ForegroundColor Gray
$modelsSource = Join-Path $scriptDir "..\models"
$modelsDest = Join-Path $distPath "models"
if (Test-Path $modelsSource) {
    Copy-Item $modelsSource $modelsDest -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  Models copied" -ForegroundColor Gray
}

Write-Host "[OK] PyInstaller complete" -ForegroundColor Green

Write-Host ""
Write-Host "[2/3] Preparing files..." -ForegroundColor Yellow

if (-not (Test-Path $distPath)) {
    Write-Host "ERROR: dist\Kura not found" -ForegroundColor Red
    exit 1
}

# Use absolute paths
$installerSrc = Join-Path $scriptDir "Kura_Installer.ps1"
$uninstallerSrc = Join-Path $scriptDir "Kura_Uninstaller.ps1"
$installerDst = Join-Path $distPath "Kura_Installer.ps1"
$uninstallerDst = Join-Path $distPath "Kura_Uninstaller.ps1"

Copy-Item $installerSrc $installerDst -Force -ErrorAction Stop
Copy-Item $uninstallerSrc $uninstallerDst -Force -ErrorAction Stop

$batchPath = Join-Path $distPath "Install.bat"
$batch = "@echo off`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0Kura_Installer.ps1`""
$batch | Out-File $batchPath -Encoding ASCII -Force

Write-Host "[OK] Files ready" -ForegroundColor Green

Write-Host ""
Write-Host "[3/3] Creating ZIP..." -ForegroundColor Yellow

$zipPath = Join-Path $outputDir "Kura_Setup_$Version.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Add-Type -AssemblyName System.IO.Compression.FileSystem
try {
    [System.IO.Compression.ZipFile]::CreateFromDirectory($distPath, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
    Write-Host "[OK] ZIP created" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to create ZIP" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""

$size = (Get-Item $zipPath).Length
$sizeGB = [math]::Round($size / 1GB, 2)
$sizeMB = [math]::Round($size / 1MB, 1)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SUCCESS!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output: $zipPath" -ForegroundColor Yellow
if ($sizeGB -ge 1) { Write-Host "Size: $sizeGB GB" } else { Write-Host "Size: $sizeMB MB" }
Write-Host ""
Write-Host "Share with users:" -ForegroundColor Yellow
Write-Host "  1. Extract ZIP" -ForegroundColor Gray
Write-Host "  2. Run Install.bat" -ForegroundColor Gray
Write-Host "  3. Done!" -ForegroundColor Gray
Write-Host ""

