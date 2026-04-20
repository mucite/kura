KURA MEDICAL — MSIX Store Assets
==================================
Replace every PNG in this folder with real artwork before Store submission.
Microsoft Partner Center will also let you upload assets during the submission
wizard, but having them in the package is preferred for offline/sideload use.

Required files and exact pixel dimensions
------------------------------------------
StoreLogo.png           50 x 50      (Store listing logo)
Square44x44Logo.png     44 x 44      (taskbar / small icon)
Square150x150Logo.png  150 x 150     (Start menu medium tile)
Wide310x150Logo.png    310 x 150     (Start menu wide tile)
SmallTile.png           71 x 71      (Start menu small tile)
LargeTile.png          310 x 310     (Start menu large tile)
SplashScreen.png       620 x 300     (launch splash)
BadgeLogo.png           24 x 24      (lock-screen badge, white on transparent)

Scaling variants (optional but recommended for HiDPI)
------------------------------------------------------
For each asset you can supply scale-100/125/150/200/400 variants using the
naming convention:
    Square150x150Logo.scale-200.png  (300 x 300)

See: https://learn.microsoft.com/windows/apps/design/style/iconography/app-icon-construction

Quick placeholder generation
-----------------------------
Run this PowerShell snippet to generate solid-colour placeholders that let
build_msix.ps1 succeed locally before you have real artwork:

    $sizes = @{
        "StoreLogo.png"          = "50x50"
        "Square44x44Logo.png"    = "44x44"
        "Square150x150Logo.png"  = "150x150"
        "Wide310x150Logo.png"    = "310x150"
        "SmallTile.png"          = "71x71"
        "LargeTile.png"          = "310x310"
        "SplashScreen.png"       = "620x300"
        "BadgeLogo.png"          = "24x24"
    }
    Add-Type -AssemblyName System.Drawing
    foreach ($name in $sizes.Keys) {
        $wh  = $sizes[$name] -split "x"
        $bmp = New-Object System.Drawing.Bitmap([int]$wh[0],[int]$wh[1])
        $g   = [System.Drawing.Graphics]::FromImage($bmp)
        $g.Clear([System.Drawing.Color]::FromArgb(10,14,26))
        $bmp.Save("$PSScriptRoot\$name", [System.Drawing.Imaging.ImageFormat]::Png)
        $g.Dispose(); $bmp.Dispose()
    }
    Write-Host "Placeholder assets created."