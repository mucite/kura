# ============================================================
#  Kura Medical — Free Code Signing Certificate Setup
#  Run ONCE on your build machine.
#
#  Usage:
#    PowerShell -ExecutionPolicy Bypass -File setup_codesign.ps1
#
#  What this does:
#    1. Creates a self-signed Authenticode certificate in your
#       Windows Certificate Store (free, no CA required).
#    2. Exports it as kura_codesign.pfx  (keep this SECRET — it
#       is your private signing key).
#    3. Exports the public cert as kura_codesign.cer  (safe to
#       share — customers can install it to trust your software).
#
#  After running this script:
#    • Set environment variable CERT_PASS to the password you chose.
#    • run_release.bat will auto-sign every build.
#    • See submit_to_microsoft.bat to get the binary whitelisted
#      by Microsoft SmartScreen (free, 1-3 business days).
# ============================================================

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$PfxPath    = Join-Path $ScriptDir "kura_codesign.pfx"
$CerPath    = Join-Path $ScriptDir "kura_codesign.cer"
$Subject    = "CN=Kura Medical, O=Kura Medical, L=Germany, C=DE"
$FriendlyName = "Kura Medical Code Signing"

Write-Host ""
Write-Host "============================================"
Write-Host "  Kura Medical — Certificate Setup"
Write-Host "============================================"
Write-Host ""

# ── Warn if certificate already exists ─────────────────────────────────────
$existing = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -like "*Kura Medical*" }
if ($existing) {
    Write-Host "⚠  A Kura Medical certificate already exists in your store."
    Write-Host "   Thumbprint: $($existing.Thumbprint)"
    $overwrite = Read-Host "   Create a new one anyway? (Y/N)"
    if ($overwrite -ne "Y") {
        Write-Host "Aborted."
        exit 0
    }
}

# ── Password for PFX ──────────────────────────────────────────────────────
Write-Host "Choose a password to protect the certificate file (kura_codesign.pfx)."
Write-Host "You will need this password every time you build a release."
Write-Host "Store it safely — losing it means creating a new certificate."
Write-Host ""
$password = Read-Host "Password" -AsSecureString
$passwordConfirm = Read-Host "Confirm password" -AsSecureString

$plain1 = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password))
$plain2 = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($passwordConfirm))

if ($plain1 -ne $plain2) {
    Write-Host "❌ Passwords do not match. Aborted."
    exit 1
}

# ── Create certificate ────────────────────────────────────────────────────
Write-Host ""
Write-Host "Creating self-signed code signing certificate..."

$cert = New-SelfSignedCertificate `
    -Type          CodeSigningCert `
    -Subject       $Subject `
    -KeyUsage      DigitalSignature `
    -FriendlyName  $FriendlyName `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -HashAlgorithm SHA256 `
    -KeyLength     4096 `
    -NotAfter      (Get-Date).AddYears(10)

Write-Host "   OK  Thumbprint: $($cert.Thumbprint)"

# ── Export PFX (private key + cert) ──────────────────────────────────────
Export-PfxCertificate -Cert $cert -FilePath $PfxPath -Password $password | Out-Null
Write-Host "   OK  Private key: $PfxPath"

# ── Export CER (public cert only, safe to share) ─────────────────────────
Export-Certificate -Cert $cert -FilePath $CerPath -Type CERT | Out-Null
Write-Host "   OK  Public cert: $CerPath"

# ── Print next steps ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================"
Write-Host "  Certificate created!"
Write-Host "============================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "  1. Set the certificate password as an environment variable"
Write-Host "     so build_release.bat can use it:"
Write-Host ""
Write-Host "       setx CERT_PASS ""your-password-here"""
Write-Host ""
Write-Host "     Then close and reopen your terminal."
Write-Host ""
Write-Host "  2. Build a signed release:"
Write-Host ""
Write-Host "       build_release.bat v2026.1"
Write-Host ""
Write-Host "  3. After building, run submit_to_microsoft.bat to"
Write-Host "     whitelist the installer with Microsoft SmartScreen (free)."
Write-Host ""
Write-Host "  KEEP kura_codesign.pfx SECRET — add it to .gitignore."
Write-Host ""
