param([string]$Version = "2026.3.0")

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura - Optimized Build (5-6 GB)" -ForegroundColor Green
Write-Host "  Version: $Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$distPath = Join-Path $scriptDir "dist\Kura"
$outputDir = Join-Path $scriptDir "dist"

# Clean everything first
Write-Host "[1/5] Cleaning..." -ForegroundColor Yellow
if (Test-Path $outputDir) { Remove-Item $outputDir -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

Write-Host "[OK] Cleaned"
Write-Host ""

# Build PyInstaller
Write-Host "[2/5] Building PyInstaller..." -ForegroundColor Yellow
Push-Location $scriptDir
pyinstaller Kura_windows.spec -y
Pop-Location

if (-not (Test-Path $distPath)) {
    Write-Host "ERROR: PyInstaller failed" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] PyInstaller complete"
Write-Host ""

# Copy installer scripts
Write-Host "[3/5] Adding installer scripts..." -ForegroundColor Yellow
Copy-Item (Join-Path $scriptDir "Kura_Installer.ps1") $distPath -Force
Copy-Item (Join-Path $scriptDir "Kura_Uninstaller.ps1") $distPath -Force
"@echo off`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0Kura_Installer.ps1`"" | Out-File (Join-Path $distPath "Install.bat") -Encoding ASCII -Force
Write-Host "[OK] Scripts added"
Write-Host ""

# Add models
Write-Host "[4/5] Adding AI models..." -ForegroundColor Yellow
$modelsSource = Join-Path $scriptDir "..\models"
$modelsDest = Join-Path $distPath "models"
if (Test-Path $modelsSource) {
    Copy-Item $modelsSource $modelsDest -Recurse -Force
}
Write-Host "[OK] Models added"
Write-Host ""

# Create ZIP
Write-Host "[5/5] Creating ZIP..." -ForegroundColor Yellow
$zipPath = Join-Path $outputDir "Kura_Setup_$Version.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($distPath, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
Write-Host "[OK] ZIP created"
Write-Host ""

# Show size
if (Test-Path $zipPath) {
    $size = (Get-Item $zipPath).Length
    $sizeGB = [math]::Round($size / 1GB, 2)
    $sizeMB = [math]::Round($size / 1MB, 1)

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  BUILD COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "ZIP File: $zipPath" -ForegroundColor Yellow
    if ($sizeGB -ge 1) {
        Write-Host "Size: $sizeGB GB" -ForegroundColor Green
    } else {
        Write-Host "Size: $sizeMB MB" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "Ready to share with users!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "ERROR: ZIP file not created" -ForegroundColor Red
    exit 1
}

