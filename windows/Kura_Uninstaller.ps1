param(
    [string]$InstallPath = "$env:LOCALAPPDATA\Programs\Kura Medical"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura Medical - Uninstallation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Confirm uninstall
Write-Host "This will completely remove Kura Medical from your computer." -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Type 'yes' to uninstall, or press Enter to cancel"
if ($confirm -ne "yes") {
    Write-Host "Uninstallation cancelled." -ForegroundColor Gray
    exit 0
}

Write-Host ""

# Step 1: Close any running Kura instance
Write-Host "[1/4] Closing Kura Medical..." -ForegroundColor Yellow
$kurasRunning = @(Get-Process Kura -ErrorAction SilentlyContinue)
if ($kurasRunning.Count -gt 0) {
    Write-Host "  Terminating running instances..." -ForegroundColor Gray
    Stop-Process -Name Kura -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
Write-Host "[OK] Application closed" -ForegroundColor Green
Write-Host ""

# Step 2: Remove installation folder
Write-Host "[2/4] Removing application files..." -ForegroundColor Yellow
if (Test-Path $InstallPath) {
    Remove-Item $InstallPath -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[OK] Application files removed" -ForegroundColor Green
} else {
    Write-Host "[OK] Application folder not found (already removed)" -ForegroundColor Green
}
Write-Host ""

# Step 3: Remove shortcuts
Write-Host "[3/4] Removing shortcuts..." -ForegroundColor Yellow

# Remove Start Menu shortcut
$startMenuPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Kura Medical"
if (Test-Path $startMenuPath) {
    Remove-Item $startMenuPath -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  Removed: Start Menu folder" -ForegroundColor Gray
}

# Remove Desktop shortcut
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcut = "$desktopPath\Kura Medical.lnk"
if (Test-Path $desktopShortcut) {
    Remove-Item $desktopShortcut -Force -ErrorAction SilentlyContinue
    Write-Host "  Removed: Desktop shortcut" -ForegroundColor Gray
}

Write-Host "[OK] Shortcuts removed" -ForegroundColor Green
Write-Host ""

# Step 4: Clean cache and data
Write-Host "[4/4] Cleaning cache and data..." -ForegroundColor Yellow

# Remove cache
$cacheDir = "$env:LOCALAPPDATA\Kura"
if (Test-Path $cacheDir) {
    Remove-Item $cacheDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "  Removed: Cache and logs" -ForegroundColor Gray
}

# Ask about user data
$docsPath = "$env:USERPROFILE\Documents\Kura"
if (Test-Path $docsPath) {
    Write-Host ""
    Write-Host "User reports and settings found in: $docsPath" -ForegroundColor Yellow
    $deleteData = Read-Host "Delete user reports and settings? (type 'yes' to delete, or press Enter to keep)"
    if ($deleteData -eq "yes") {
        Remove-Item $docsPath -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Removed: User reports and settings" -ForegroundColor Gray
    } else {
        Write-Host "  Kept: User reports and settings" -ForegroundColor Gray
    }
}

Write-Host "[OK] Cache cleaned" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Uninstallation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Kura Medical has been completely removed from your computer." -ForegroundColor Green
Write-Host ""
Write-Host "Thank you for using Kura Medical!" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 2

