param([string]$Version = "2026.4.1")

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura - Slim Build (300 MB)" -ForegroundColor Green
Write-Host "  Models download on first launch" -ForegroundColor Yellow
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

# Skip models - they will download on first launch
Write-Host "[4/5] Checking for models..." -ForegroundColor Yellow
$modelsSource = Join-Path $scriptDir "..\models"
if (Test-Path $modelsSource) {
    Write-Host "   Models folder found - SKIPPING (slim build)" -ForegroundColor Yellow
    Write-Host "   Models will auto-download on first app launch" -ForegroundColor Cyan
} else {
    Write-Host "   No models folder - will download on first launch" -ForegroundColor Cyan
}
Write-Host "[OK] Slim build (no models included)"
Write-Host ""

# Create ZIP
Write-Host "[5/5] Creating ZIP..." -ForegroundColor Yellow
$zipPath = Join-Path $outputDir "Kura_Windows_v${Version}.zip"
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
    # SHA-256 checksum
    $hash = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLower()
    "$hash  Kura_Windows_v${Version}.zip" | Out-File "$zipPath.sha256" -Encoding ASCII -NoNewline
    Write-Host ""
    Write-Host "SHA-256: $hash" -ForegroundColor Cyan
    Write-Host "Checksum: $zipPath.sha256" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Models auto-download on first launch" -ForegroundColor Yellow
    Write-Host "Ready to share with users!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "ERROR: ZIP file not created" -ForegroundColor Red
    exit 1
}

