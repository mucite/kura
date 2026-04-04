param([string]$Version = "2026.3.0")

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura - Add Models & Create ZIP" -ForegroundColor Green
Write-Host "  Version: $Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$distPath = Join-Path $scriptDir "dist\Kura"
$outputDir = Join-Path $scriptDir "dist"
$modelsSource = Join-Path $scriptDir "..\models"

Write-Host "[1/3] Checking files..." -ForegroundColor Yellow

if (-not (Test-Path $distPath)) {
    Write-Host "ERROR: dist\Kura not found. Run build.ps1 first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $modelsSource)) {
    Write-Host "ERROR: models folder not found at $modelsSource" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Files found" -ForegroundColor Green

Write-Host ""
Write-Host "[2/3] Adding AI models..." -ForegroundColor Yellow

$modelsDest = Join-Path $distPath "models"
Remove-Item $modelsDest -Recurse -Force -ErrorAction SilentlyContinue

# Copy each model individually
Write-Host "  Copying Llama-3.2 model..." -ForegroundColor Gray
Copy-Item (Join-Path $modelsSource "Llama-3.2-3B-Instruct-4bit-GGUF") (Join-Path $modelsDest "Llama-3.2-3B-Instruct-4bit-GGUF") -Recurse -Force

Write-Host "  Copying Whisper model..." -ForegroundColor Gray
Copy-Item (Join-Path $modelsSource "whisper") (Join-Path $modelsDest "whisper") -Recurse -Force

Write-Host "[OK] Models added" -ForegroundColor Green

Write-Host ""
Write-Host "[3/3] Creating ZIP..." -ForegroundColor Yellow

$zipPath = Join-Path $outputDir "Kura_Setup_$Version.zip"
if (Test-Path $zipPath) {
    Write-Host "  Removing old ZIP..." -ForegroundColor Gray
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
try {
    Write-Host "  Compressing files..." -ForegroundColor Gray
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

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SUCCESS!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output: $zipPath" -ForegroundColor Yellow
Write-Host "Size: $sizeGB GB" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ready to share with users!" -ForegroundColor Green
Write-Host ""

