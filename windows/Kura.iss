; ============================================================
;  Kura Medical — Inno Setup Installer
;  Free installer: https://jrsoftware.org/isdl.php
;
;  Build manually:
;    iscc Kura.iss
;
;  Build with custom version:
;    iscc /DAppVersion=2026.1.0 Kura.iss
;
;  Build via pipeline:
;    build_release.bat v2026.1
; ============================================================

; ── Version (override with /DAppVersion=x.x.x on command line) ──────────────
#ifndef AppVersion
  #define AppVersion "2026.3.2"
#endif

; ── Constants ────────────────────────────────────────────────────────────────
#define AppName        "Kura Medical"
#define AppShortName   "Kura"
#define AppPublisher   "Kura Medical"
#define AppURL         "https://kura-medical.de"
#define AppExeName     "Kura.exe"
#define AppDescription "KI-Dokumentation für Physiotherapie"
#define SourceDir      "dist\Kura"
#define AppCopyright   "© 2026 Kura Medical"

; ============================================================
[Setup]
; ── Identity ─────────────────────────────────────────────────────────────────
; IMPORTANT: Keep this GUID identical across all versions — it ties upgrades together.
AppId={{A7F2D3B8-4C1E-4F9A-8B2D-5E6F7A8C9D0E}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
AppCopyright={#AppCopyright}

; ── Installation target ───────────────────────────────────────────────────────
; Per-user install — no admin / UAC prompt required.
; User can override to machine-wide via the dialog below.
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; ── Output ────────────────────────────────────────────────────────────────────
OutputDir=dist
OutputBaseFilename=Kura_Setup_{#AppVersion}
; Uncomment when you have a .ico file:
; SetupIconFile=..\assets\kura.ico

; ── Code signing (optional — activates when CERT_PFX and CERT_PASS are set) ──
; Inno Setup signs the compiled installer .exe automatically.
; Set up signing once with: PowerShell -File setup_codesign.ps1
; Then set env vars:  CERT_PFX = path to kura_codesign.pfx
;                     CERT_PASS = pfx password
#ifdef CERT_PFX
SignTool=kura_sign $f
#endif

; ── Compression ───────────────────────────────────────────────────────────────
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes

; ── Installer UI ──────────────────────────────────────────────────────────────
WizardStyle=modern
WizardResizable=no
ShowLanguageDialog=auto
DisableWelcomePage=no
DisableReadyMemo=no

; ── Version metadata (appears in Properties → Details of the setup .exe) ─────
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppDescription}
VersionInfoCopyright={#AppCopyright}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}

; ── Uninstaller ───────────────────────────────────────────────────────────────
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
CreateUninstallRegKey=yes

; ── System requirement ────────────────────────────────────────────────────────
MinVersion=10.0

; ============================================================
[Languages]
Name: "german";  MessagesFile: "compiler:Languages\German.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

; ============================================================
[CustomMessages]

; Welcome page body
german.AppWelcomeBody=Kura Medical installiert KI-gestützte Dokumentation auf Ihrem PC.%n%n%n  • 100 %% lokale Verarbeitung%n  • Keine Cloud-Verbindung erforderlich%n  • DSGVO-konform nach § 125 SGB V%n%n%nBitte schließen Sie alle anderen Programme, bevor Sie fortfahren.
english.AppWelcomeBody=Kura Medical installs AI-powered physiotherapy documentation on your PC.%n%n%n  • 100 %% local processing%n  • No cloud connection required%n  • GDPR-compliant (§ 125 SGB V)%n%n%nPlease close all other programs before continuing.

; Finish page
german.AppFinishedBody=Kura Medical wurde erfolgreich installiert.%n%nBeim ersten Start fragt Windows nach dem Mikrofon-Zugriff. Klicken Sie auf "Zulassen".%n%nIhre Berichte werden in Dokumente\Kura\ gespeichert.
english.AppFinishedBody=Kura Medical has been installed successfully.%n%nOn first launch, Windows will ask for microphone access. Click "Allow".%n%nYour reports are saved in Documents\Kura\.

; ============================================================
[Tasks]
Name: "desktopicon"; \
  Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"

; ============================================================
[Files]
; All application files from the PyInstaller output directory
Source: "{#SourceDir}\*"; \
  DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

; ============================================================
[Icons]
; Start Menu
Name: "{group}\{#AppName}"; \
  Filename: "{app}\{#AppExeName}"; \
  Comment: "{#AppDescription}"

Name: "{group}\{cm:UninstallProgram,{#AppName}}"; \
  Filename: "{uninstallexe}"

; Desktop shortcut (only if task ticked)
Name: "{userdesktop}\{#AppName}"; \
  Filename: "{app}\{#AppExeName}"; \
  Comment: "{#AppDescription}"; \
  Tasks: desktopicon

; ============================================================
[Run]
; "Launch Kura Medical" checkbox on the final page
Filename: "{app}\{#AppExeName}"; \
  Description: "{cm:LaunchProgram,{#AppName}}"; \
  Flags: nowait postinstall skipifsilent

; ============================================================
[UninstallDelete]
; App cache / logs under %LOCALAPPDATA%\Kura — always removed automatically.
; Reports under Documents\Kura\ — user is asked in CurUninstallStepChanged.
Type: filesandordirs; Name: "{localappdata}\Kura"

; ============================================================
[Code]

// ── Helpers ────────────────────────────────────────────────────────────────

function WizardPage(): TWizardPage;
begin
  Result := nil; // placeholder
end;

// Kill a running Kura instance before upgrade/install
procedure KillRunningApp();
var
  ResultCode: Integer;
begin
  Exec(
    ExpandConstant('{sys}\taskkill.exe'),
    '/F /IM Kura.exe /T',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode
  );
  Sleep(800);
end;

// Remove Zone.Identifier (MOTW) from all installed files.
// Files written by the installer normally carry no MOTW, but any file that
// was copied into the build output while it itself was downloaded may be
// marked. Unblock-File strips the alternate data stream unconditionally.
procedure UnblockInstalledFiles();
var
  ResultCode: Integer;
  AppPath, PSCommand: String;
begin
  AppPath    := ExpandConstant('{app}');
  PSCommand  := '-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ' +
                '"Get-ChildItem -Path ''' + AppPath + ''' -Recurse -File ' +
                '| ForEach-Object { try { Unblock-File -Path $_.FullName } catch {} }"';
  Exec(
    ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    PSCommand,
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode
  );
end;

// ── Wizard page customisation ──────────────────────────────────────────────

procedure InitializeWizard();
begin
  // Override the generic welcome text with our own
  WizardForm.WelcomeLabel2.Caption := CustomMessage('AppWelcomeBody');
  WizardForm.FinishedLabel.Caption  := CustomMessage('AppFinishedBody');
end;

// ── Pre-install: close running instance ───────────────────────────────────

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  NeedsRestart := False;
  KillRunningApp();
end;

// ── Post-install: unblock files ────────────────────────────────────────────

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    UnblockInstalledFiles();
end;

// ── Pre-uninstall: close running instance ─────────────────────────────────

function InitializeUninstall(): Boolean;
begin
  KillRunningApp();
  Result := True;
end;

// ── Post-uninstall: clean up all resources ────────────────────────────────
// %LOCALAPPDATA%\Kura  → removed automatically via [UninstallDelete].
// Documents\Kura\      → user is asked; default is NO to protect reports.

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DocsDir: String;
begin
  if CurUninstallStep <> usPostUninstall then Exit;

  DocsDir := ExpandConstant('{userdocs}\Kura');
  if not DirExists(DocsDir) then Exit;

  if MsgBox(
    'Sollen Ihre Berichte und Einstellungen ebenfalls gelöscht werden?' + #13#10 +
    #13#10 +
    DocsDir + #13#10 +
    #13#10 +
    'Wählen Sie "Nein", um die Berichte zu behalten.',
    mbConfirmation, MB_YESNO or MB_DEFBUTTON2
  ) = IDYES then
    DelTree(DocsDir, True, True, True);
end;