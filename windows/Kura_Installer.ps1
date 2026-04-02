param(
    [string]$InstallPath = "$env:LOCALAPPDATA\Programs\Kura Medical"
)

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePath = $scriptPath

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura Medical - Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin (not required, but helpful)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

Write-Host "Installation path: $InstallPath" -ForegroundColor Yellow
Write-Host ""

# Step 1: Close any running Kura instance
Write-Host "[1/4] Checking for running Kura instances..." -ForegroundColor Yellow
$kurasRunning = @(Get-Process Kura -ErrorAction SilentlyContinue)
if ($kurasRunning.Count -gt 0) {
    Write-Host "  Closing running Kura instances..." -ForegroundColor Gray
    Stop-Process -Name Kura -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
Write-Host "[OK] Ready to install" -ForegroundColor Green
Write-Host ""

# Step 2: Create installation directory
Write-Host "[2/4] Creating installation directory..." -ForegroundColor Yellow
if (Test-Path $InstallPath) {
    Write-Host "  Removing old installation..." -ForegroundColor Gray
    Remove-Item $InstallPath -Recurse -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}
New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
Write-Host "[OK] Installation directory ready" -ForegroundColor Green
Write-Host ""

# Step 3: Copy files
Write-Host "[3/4] Copying application files..." -ForegroundColor Yellow
$itemsToCopy = @()

# Get all items in source except this script
Get-ChildItem $sourcePath -Exclude "Kura_Installer.ps1", "*.md", "*.bat" | ForEach-Object {
    if ($_.PSIsContainer) {
        Copy-Item $_.FullName "$InstallPath\$($_.Name)" -Recurse -Force
        Write-Host "  Copied: $($_.Name)\" -ForegroundColor Gray
    } else {
        Copy-Item $_.FullName $InstallPath -Force
        Write-Host "  Copied: $($_.Name)" -ForegroundColor Gray
    }
}
Write-Host "[OK] Files copied" -ForegroundColor Green
Write-Host ""

# Step 4: Create shortcuts
Write-Host "[4/4] Creating shortcuts..." -ForegroundColor Yellow

# Create Start Menu shortcut
$startMenuPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
$kuraMenuPath = "$startMenuPath\Kura Medical"
New-Item -ItemType Directory -Path $kuraMenuPath -Force | Out-Null

$shortcutPath = "$kuraMenuPath\Kura Medical.lnk"
$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortCut($shortcutPath)
$shortcut.TargetPath = "$InstallPath\Kura.exe"
$shortcut.WorkingDirectory = $InstallPath
$shortcut.Description = "Kura Medical - AI-powered physiotherapy documentation"
$shortcut.Save()
Write-Host "  Created: Start Menu shortcut" -ForegroundColor Gray

# Create Desktop shortcut (optional)
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcut = "$desktopPath\Kura Medical.lnk"
$desktopLink = $WshShell.CreateShortCut($desktopShortcut)
$desktopLink.TargetPath = "$InstallPath\Kura.exe"
$desktopLink.WorkingDirectory = $InstallPath
$desktopLink.Description = "Kura Medical - AI-powered physiotherapy documentation"
$desktopLink.Save()
Write-Host "  Created: Desktop shortcut" -ForegroundColor Gray

# Create Uninstaller shortcut
$uninstallerPath = "$kuraMenuPath\Uninstall Kura Medical.lnk"
$uninstallerLink = $WshShell.CreateShortCut($uninstallerPath)
$uninstallerLink.TargetPath = "powershell.exe"
$uninstallerLink.Arguments = "-NoProfile -ExecutionPolicy Bypass -Command `"& '$PSScriptRoot\Kura_Uninstaller.ps1'`""
$uninstallerLink.WorkingDirectory = $InstallPath
$uninstallerLink.Description = "Uninstall Kura Medical"
$uninstallerLink.Save()
Write-Host "  Created: Uninstaller shortcut" -ForegroundColor Gray

Write-Host "[OK] Shortcuts created" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Kura Medical has been installed successfully." -ForegroundColor Green
Write-Host ""
Write-Host "Start Kura from:" -ForegroundColor Yellow
Write-Host "  • Start Menu → Kura Medical" -ForegroundColor Gray
Write-Host "  • Desktop → Kura Medical icon" -ForegroundColor Gray
Write-Host ""
Write-Host "To uninstall:" -ForegroundColor Yellow
Write-Host "  • Start Menu → Kura Medical → Uninstall" -ForegroundColor Gray
Write-Host ""

# Ask to launch
Write-Host "Launch Kura now?" -ForegroundColor Yellow
$response = Read-Host "Enter 'yes' to launch, or press Enter to exit"
if ($response -eq "yes") {
    & "$InstallPath\Kura.exe"
}

Write-Host "Thank you for using Kura Medical!" -ForegroundColor Cyan

