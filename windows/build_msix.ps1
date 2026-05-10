param(
    [string]$Version      = "2026.4.2",
    [string]$Publisher    = "CN=14D11CC0-7C61-448C-BA4A-261CEA23CAC7",
    [string]$CertPfx      = "",
    [string]$CertPassword = "",
    [switch]$SkipPyInstaller
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$distKura    = Join-Path $scriptDir "dist\Kura"
$layoutDir   = Join-Path $scriptDir "dist\MsixLayout"
$outputDir   = Join-Path $scriptDir "dist"
$msixPath    = Join-Path $outputDir "Kura_$Version.msix"
$manifestSrc = Join-Path $scriptDir "msix\AppxManifest.xml"
$assetsSrc   = Join-Path $scriptDir "msix\Assets"

# Locate a Windows SDK tool by name (checks PATH then common SDK install locations)
function Find-SdkTool([string]$ToolName) {
    $inPath = Get-Command $ToolName -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }

    $sdkRoots = @(
        "C:\Program Files (x86)\Windows Kits\10\bin",
        "C:\Program Files\Windows Kits\10\bin"
    )
    foreach ($root in $sdkRoots) {
        if (Test-Path $root) {
            $hit = Get-ChildItem $root -Recurse -Filter $ToolName -ErrorAction SilentlyContinue |
                   Where-Object { $_.FullName -match "x64" } |
                   Sort-Object LastWriteTime -Descending |
                   Select-Object -First 1
            if ($hit) { return $hit.FullName }
        }
    }
    return $null
}

$makeappx = Find-SdkTool "makeappx.exe"
$signtool = Find-SdkTool "signtool.exe"

if (-not $makeappx) {
    Write-Host "ERROR: makeappx.exe not found." -ForegroundColor Red
    Write-Host "Install the Windows SDK from:" -ForegroundColor Red
    Write-Host "  https://developer.microsoft.com/windows/downloads/windows-sdk/" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kura Pro - MSIX Package Build"         -ForegroundColor Green
Write-Host "  Version: $Version"                      -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ----- Step 1: PyInstaller --------------------------------------------------
if (-not $SkipPyInstaller) {
    Write-Host "[1/5] Building PyInstaller bundle..." -ForegroundColor Yellow
    Push-Location $scriptDir
    pyinstaller Kura_windows.spec -y
    Pop-Location
    if (-not (Test-Path $distKura)) {
        Write-Host "ERROR: PyInstaller failed - dist\Kura not found." -ForegroundColor Red
        exit 1
    }
    Write-Host "[1/5] PyInstaller complete." -ForegroundColor Green
} else {
    Write-Host "[1/5] Skipping PyInstaller (using existing dist\Kura)." -ForegroundColor DarkYellow
    if (-not (Test-Path $distKura)) {
        Write-Host "ERROR: dist\Kura not found. Run without -SkipPyInstaller first." -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# ----- Step 2: Assemble MSIX layout -----------------------------------------
Write-Host "[2/5] Assembling package layout..." -ForegroundColor Yellow

if (Test-Path $layoutDir) { Remove-Item $layoutDir -Recurse -Force }
New-Item -ItemType Directory -Path $layoutDir | Out-Null

Copy-Item "$distKura\*" $layoutDir -Recurse -Force

$assetsTarget = Join-Path $layoutDir "Assets"
if (-not (Test-Path $assetsTarget)) {
    New-Item -ItemType Directory -Path $assetsTarget | Out-Null
}

$requiredAssets = @(
    "StoreLogo.png", "Square44x44Logo.png", "Square150x150Logo.png",
    "Wide310x150Logo.png", "SmallTile.png", "LargeTile.png",
    "SplashScreen.png", "BadgeLogo.png"
)

$missingAssets = @($requiredAssets | Where-Object { -not (Test-Path (Join-Path $assetsSrc $_)) })

if ($missingAssets.Count -gt 0) {
    Write-Host "WARNING: Generating placeholder PNGs for missing assets:" -ForegroundColor Yellow
    foreach ($a in $missingAssets) { Write-Host "   $a" -ForegroundColor DarkYellow }

    Add-Type -AssemblyName System.Drawing
    $assetSizes = @{
        "StoreLogo.png"          = @(50,  50)
        "Square44x44Logo.png"    = @(44,  44)
        "Square150x150Logo.png"  = @(150, 150)
        "Wide310x150Logo.png"    = @(310, 150)
        "SmallTile.png"          = @(71,  71)
        "LargeTile.png"          = @(310, 310)
        "SplashScreen.png"       = @(620, 300)
        "BadgeLogo.png"          = @(24,  24)
    }
    foreach ($assetName in $missingAssets) {
        $wh  = $assetSizes[$assetName]
        $bmp = New-Object System.Drawing.Bitmap($wh[0], $wh[1])
        $g   = [System.Drawing.Graphics]::FromImage($bmp)
        $g.Clear([System.Drawing.Color]::FromArgb(10, 14, 26))
        $bmp.Save((Join-Path $assetsSrc $assetName), [System.Drawing.Imaging.ImageFormat]::Png)
        $g.Dispose()
        $bmp.Dispose()
    }
}

Copy-Item "$assetsSrc\*.png" $assetsTarget -Force -ErrorAction SilentlyContinue

# Write manifest directly — avoids all file encoding/BOM issues with template patching
$msixVersion = (($Version -split "\." | Select-Object -First 3) -join ".") + ".0"
$manifestOut  = Join-Path $layoutDir "AppxManifest.xml"
$manifestXml  = '<?xml version="1.0" encoding="utf-8"?>' + "`r`n"
$manifestXml += '<Package' + "`r`n"
$manifestXml += '  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"' + "`r`n"
$manifestXml += '  xmlns:mp="http://schemas.microsoft.com/appx/2014/phone/manifest"' + "`r`n"
$manifestXml += '  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"' + "`r`n"
$manifestXml += '  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"' + "`r`n"
$manifestXml += '  xmlns:desktop="http://schemas.microsoft.com/appx/manifest/desktop/windows10"' + "`r`n"
$manifestXml += '  IgnorableNamespaces="uap mp rescap desktop">' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '  <Identity' + "`r`n"
$manifestXml += '    Name="KuraMedical.KuraPro"' + "`r`n"
$manifestXml += '    Publisher="' + $Publisher + '"' + "`r`n"
$manifestXml += '    Version="' + $msixVersion + '"' + "`r`n"
$manifestXml += '    ProcessorArchitecture="x64" />' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '  <mp:PhoneIdentity PhoneProductId="1909c148-1082-435a-a051-516b8f0e3a19" PhonePublisherId="2a34eaa7-1326-4299-a7d7-00f990ea629a" />' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '  <Properties>' + "`r`n"
$manifestXml += '    <DisplayName>Kura Pro</DisplayName>' + "`r`n"
$manifestXml += '    <PublisherDisplayName>Kura Medical</PublisherDisplayName>' + "`r`n"
$manifestXml += '    <Logo>Assets\StoreLogo.png</Logo>' + "`r`n"
$manifestXml += '    <Description>KI-Dokumentation fuer Physiotherapie - 100% lokal, DSGVO-konform</Description>' + "`r`n"
$manifestXml += '  </Properties>' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '  <Dependencies>' + "`r`n"
$manifestXml += '    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.22621.0" />' + "`r`n"
$manifestXml += '  </Dependencies>' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '  <Resources>' + "`r`n"
$manifestXml += '    <Resource Language="de-DE" />' + "`r`n"
$manifestXml += '    <Resource Language="en-US" />' + "`r`n"
$manifestXml += '  </Resources>' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '  <Applications>' + "`r`n"
$manifestXml += '    <Application Id="Kura" Executable="Kura.exe" EntryPoint="Windows.FullTrustApplication">' + "`r`n"
$manifestXml += '      <uap:VisualElements' + "`r`n"
$manifestXml += '        DisplayName="Kura Pro"' + "`r`n"
$manifestXml += '        Description="KI-Dokumentation fuer Physiotherapie"' + "`r`n"
$manifestXml += '        BackgroundColor="transparent"' + "`r`n"
$manifestXml += '        Square150x150Logo="Assets\Square150x150Logo.png"' + "`r`n"
$manifestXml += '        Square44x44Logo="Assets\Square44x44Logo.png">' + "`r`n"
$manifestXml += '        <uap:DefaultTile Wide310x150Logo="Assets\Wide310x150Logo.png" Square71x71Logo="Assets\SmallTile.png" Square310x310Logo="Assets\LargeTile.png" ShortName="Kura Pro" />' + "`r`n"
$manifestXml += '        <uap:SplashScreen Image="Assets\SplashScreen.png" BackgroundColor="#0a0e1a" />' + "`r`n"
$manifestXml += '        <uap:LockScreen Notification="badge" BadgeLogo="Assets\BadgeLogo.png" />' + "`r`n"
$manifestXml += '      </uap:VisualElements>' + "`r`n"
$manifestXml += '      <Extensions>' + "`r`n"
$manifestXml += '        <desktop:Extension Category="windows.fullTrustProcess" Executable="Kura.exe" />' + "`r`n"
$manifestXml += '      </Extensions>' + "`r`n"
$manifestXml += '    </Application>' + "`r`n"
$manifestXml += '  </Applications>' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '  <Capabilities>' + "`r`n"
$manifestXml += '    <rescap:Capability Name="runFullTrust" />' + "`r`n"
$manifestXml += '    <DeviceCapability Name="microphone" />' + "`r`n"
$manifestXml += '  </Capabilities>' + "`r`n"
$manifestXml += "`r`n"
$manifestXml += '</Package>' + "`r`n"

[System.IO.File]::WriteAllText($manifestOut, $manifestXml, [System.Text.UTF8Encoding]::new($false))

Write-Host "[2/5] Layout assembled at: $layoutDir" -ForegroundColor Green
Write-Host ""

# ----- Step 3: Pack MSIX ----------------------------------------------------
Write-Host "[3/5] Packing MSIX..." -ForegroundColor Yellow

if (Test-Path $msixPath) { Remove-Item $msixPath -Force }

& $makeappx pack /d $layoutDir /p $msixPath /nv /o
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: makeappx failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

Write-Host "[3/5] MSIX created: $msixPath" -ForegroundColor Green
Write-Host ""

# ----- Step 4: Sign (optional, for sideload testing) ------------------------
Write-Host "[4/5] Code signing..." -ForegroundColor Yellow

if ($CertPfx -and (Test-Path $CertPfx)) {
    if (-not $signtool) {
        Write-Host "WARNING: signtool.exe not found - skipping signing." -ForegroundColor Yellow
    } else {
        $signArgs = @(
            "sign", "/fd", "SHA256",
            "/a", "/f", $CertPfx,
            "/p", $CertPassword,
            "/tr", "http://timestamp.digicert.com",
            "/td", "SHA256",
            $msixPath
        )
        & $signtool @signArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: signtool failed (exit code $LASTEXITCODE)." -ForegroundColor Red
            exit 1
        }
        Write-Host "[4/5] Package signed with: $CertPfx" -ForegroundColor Green
    }
} else {
    Write-Host "   No certificate - package is unsigned." -ForegroundColor DarkYellow
    Write-Host "   Store submission: Microsoft signs during ingestion (no cert needed)." -ForegroundColor DarkYellow
    Write-Host "   Sideload testing: pass -CertPfx and -CertPassword to sign locally." -ForegroundColor DarkYellow
    Write-Host "[4/5] Signing skipped." -ForegroundColor Green
}
Write-Host ""

# ----- Step 5: Summary ------------------------------------------------------
Write-Host "[5/5] Done." -ForegroundColor Yellow

$size   = (Get-Item $msixPath).Length
$sizeMB = [math]::Round($size / 1MB, 1)
$sizeGB = [math]::Round($size / 1GB, 2)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BUILD COMPLETE"                         -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "MSIX : $msixPath" -ForegroundColor Yellow

if ($sizeGB -ge 1) {
    Write-Host "Size : $sizeGB GB" -ForegroundColor Green
} else {
    Write-Host "Size : $sizeMB MB" -ForegroundColor Green
}
