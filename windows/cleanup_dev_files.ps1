# Kura Development Files Cleanup Script
# Removes unnecessary files before building distribution

param(
    [switch]$DryRun = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura - Cleanup Development Files" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN MODE] - No files will be deleted" -ForegroundColor Magenta
    Write-Host ""
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Files and directories to remove
$itemsToRemove = @(
    # Build artifacts
    "$projectRoot\windows\build",
    "$projectRoot\windows\dist",
    "$projectRoot\windows\__pycache__",
    "$projectRoot\macos\build",
    "$projectRoot\macos\__pycache__",
    "$projectRoot\shared\__pycache__",

    # Python cache
    "$projectRoot\.venv",
    "$projectRoot\*.pyc",
    "$projectRoot\**\*.pyc",
    "$projectRoot\**\__pycache__",

    # IDE files
    "$projectRoot\.idea",
    "$projectRoot\.vscode",
    "$projectRoot\.DS_Store",
    "$projectRoot\**\.DS_Store",

    # Git (keep .gitignore but remove .git if distributing source)
    # "$projectRoot\.git",  # Uncomment to remove Git history

    # Documentation files (optional - keep for developers)
    # "$projectRoot\FIXES_APPLIED.md",
    # "$projectRoot\WINDOWS_BOOT_FIX.md",
    # "$projectRoot\FINAL_BOOT_FIX_COMPLETE.md",

    # macOS specific (if building Windows-only)
    # "$projectRoot\macos",  # Uncomment to remove macOS files

    # Website files (if not needed in distribution)
    # "$projectRoot\website",  # Uncomment to remove website files

    # Test/example files
    "$projectRoot\copy_whisper_model.py",

    # Old setup files
    "$projectRoot\Kura_Setup_*.zip"
)

$totalSize = 0
$itemCount = 0

foreach ($item in $itemsToRemove) {
    if (Test-Path $item) {
        $itemSize = 0

        if (Test-Path $item -PathType Container) {
            $itemSize = (Get-ChildItem $item -Recurse | Measure-Object -Property Length -Sum).Sum
        } else {
            $files = Get-Item $item -ErrorAction SilentlyContinue
            if ($files) {
                $itemSize = ($files | Measure-Object -Property Length -Sum).Sum
            }
        }

        $sizeMB = [math]::Round($itemSize / 1MB, 2)
        $itemName = Split-Path $item -Leaf

        Write-Host "  [-] $itemName" -ForegroundColor Red -NoNewline
        if ($sizeMB -gt 0) {
            Write-Host " ($sizeMB MB)" -ForegroundColor Gray
        } else {
            Write-Host ""
        }

        if (-not $DryRun) {
            Remove-Item $item -Recurse -Force -ErrorAction SilentlyContinue
        }

        $totalSize += $itemSize
        $itemCount++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "  WOULD REMOVE:" -ForegroundColor Yellow
} else {
    Write-Host "  REMOVED:" -ForegroundColor Green
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Items: $itemCount" -ForegroundColor White
Write-Host "Space: $([math]::Round($totalSize / 1MB, 2)) MB" -ForegroundColor White
Write-Host ""

if ($DryRun) {
    Write-Host "Run without -DryRun to actually delete files:" -ForegroundColor Yellow
    Write-Host "  .\cleanup_dev_files.ps1" -ForegroundColor Gray
    Write-Host ""
}

