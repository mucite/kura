import json
import multiprocessing
import os
import subprocess
import threading
import time
import sys
import traceback
from datetime import datetime

import rumps
from fpdf import FPDF
from dotenv import load_dotenv

# --- Crash Logging Setup ---
def setup_crash_logging():
    """Setup crash logging to help diagnose issues"""
    import faulthandler
    log_dir = os.path.expanduser("~/Library/Logs/Kura")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # faulthandler catches C-level crashes (SIGSEGV, SIGABRT, Metal assertion failures)
    # and writes a native Python traceback — bypasses sys.excepthook which only catches Python exceptions
    fault_file = os.path.join(log_dir, "fault.log")
    _fh = open(fault_file, 'a')
    faulthandler.enable(file=_fh)

    def log_crash(exc_type, exc_value, exc_traceback):
        with open(log_file, 'w') as f:
            f.write(f"Kura Crash Report - {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            f.write("Exception:\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write("\n" + "="*60 + "\n")
            f.write("Environment:\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"Base path: {os.path.dirname(os.path.realpath(__file__))}\n")
            if getattr(sys, 'frozen', False):
                _mpath = os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)), 'models')
            else:
                _mpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models')
            f.write(f"Models path: {_mpath}\n")
            f.write(f"Models exist: {os.path.exists(_mpath)}\n")

        # Show user-friendly error
        os.system(f'osascript -e \'display alert "Kura Fehler" message "App konnte nicht starten. Log: {log_file}" buttons {{"OK"}} default button "OK"\'')
        sys.exit(1)

    sys.excepthook = log_crash
    return log_file

# Setup crash logging before anything else
CRASH_LOG = setup_crash_logging()

# Add parent directory to path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.license_manager import LicenseManager
# KuraEngine is imported lazily inside boot() so scipy/mlx_whisper don't block startup

# --- Load environment variables from .env file ---
# SECURITY: Load .env from user's Documents folder, NOT from bundle
user_env_file = os.path.expanduser("~/Documents/Kura/.env")
user_env_dir = os.path.dirname(user_env_file)

# Create user config directory if it doesn't exist
os.makedirs(user_env_dir, exist_ok=True)

# If user .env doesn't exist, create it from bundled example or template
if not os.path.exists(user_env_file):
    # Try to find bundled .env.example
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            exe_dir = os.path.dirname(sys.executable)
            base_path = os.path.join(exe_dir, '..', 'Resources')
        example_file = os.path.join(base_path, '.env.example')
    else:
        example_file = os.path.join(os.path.dirname(__file__), '..', '.env.example')
    
    # Copy example if it exists, otherwise create template
    if os.path.exists(example_file):
        import shutil
        shutil.copy(example_file, user_env_file)
        print(f"📝 Created user .env from example: {user_env_file}")
    else:
        # Create basic .env template
        with open(user_env_file, 'w') as f:
            f.write("# Kura Configuration\n")
            f.write("# Get HF_TOKEN at: https://huggingface.co/settings/tokens\n")
            f.write("HF_TOKEN=your_token_here\n\n")
            f.write("# Lemon Squeezy API (pre-configured)\n")
            f.write("LEMON_SQUEEZY_API_URL=https://api.lemonsqueezy.com/v1/licenses/activate\n")
            f.write("LEMON_SQUEEZY_API_KEY=\n")
        print(f"📝 Created template .env: {user_env_file}")

# Load from user Documents folder
load_dotenv(user_env_file)
print(f"✅ Loading .env from: {user_env_file}")

# Check if HF_TOKEN is set, if not prompt user
if not os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN") == "your_token_here":
    print("⚠️ HF_TOKEN not configured - will prompt on first use")
    # Set a flag to show setup wizard later
    needs_setup = True
else:
    needs_setup = False
    print(f"✅ HF_TOKEN loaded")

# --- Essential for macOS bundling ---
multiprocessing.freeze_support()
# DO NOT use fork on macOS - causes CoreFoundation crashes
# macOS will use 'spawn' by default which is safe
print("🔧 Multiprocessing: Using default (spawn) for macOS safety")

# Load HF_TOKEN from environment (set in .env or system environment)
if "HF_TOKEN" not in os.environ:
    print("⚠️ Warning: HF_TOKEN not found in environment variables. Set it in .env file or system environment.")


# --- Live Audio Visualizer ---


# --- App version (must match Gist version when releasing) ---
APP_VERSION = "2026.3.2"
_VERSION_URL  = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/version.json"
_DOWNLOAD_URL = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/Kura_macOS_v2026.dmg"


# --- Main App ---
def _asset(name):
    """Resolve icon path whether running as bundle or from source."""
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'assets', name)


class KuraApp(rumps.App):
    def __init__(self):
        # Image-based icon: fixed pixel width → never pushes other menu bar items off screen
        super().__init__("", icon=_asset("icon_idle.png"), template=True, quit_button=None)

        # --- Folder Initialization ---
        # IMPORTANT: Store reports in user's Documents (app bundle is read-only on DMG)
        user_data_dir = os.path.expanduser("~/Documents/Kura")
        self.report_dir = os.path.join(user_data_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        print(f"📁 Reports directory: {self.report_dir}")

        # --- License Manager ---
        self.license_mgr = LicenseManager()
        status = self.license_mgr.verify_locally()

        # --- Menu items (stored as attrs for dynamic updates) ---
        is_pro = (status is True)

        self.status_item  = rumps.MenuItem("⏳ Modelle laden...", callback=None)
        self._item_start  = rumps.MenuItem("Neue Sitzung", callback=None)
        self._item_stop   = rumps.MenuItem("Stoppen & Auswerten", callback=None)
        self._item_arch   = rumps.MenuItem("Archiv", callback=self.open_archive)
        self._item_config = rumps.MenuItem(
            "Praxis-Einstellungen",
            callback=self.open_practice_config if is_pro else self._config_locked,
        )
        self._item_gist_override = rumps.MenuItem(
            "Konfiguration anpassen",
            callback=self.open_gist_override if is_pro else self._config_locked,
        )
        self._item_lic    = rumps.MenuItem("", callback=self.activate_license)
        self._item_deact  = rumps.MenuItem("Lizenz deaktivieren", callback=self.deactivate_license)
        self._item_update = rumps.MenuItem("Aktualisierungen prüfen", callback=self.check_for_update)

        self.menu = [
            self.status_item,
            None,
            self._item_start,
            self._item_stop,
            None,
            self._item_arch,
            None,
            self._item_config,
            self._item_gist_override,
            None,
            self._item_lic,
            self._item_deact,
            None,
            self._item_update,
            rumps.MenuItem("Beenden", callback=self._quit),
        ]
        self._refresh_menu_state(status)

        # --- Internal State ---
        self.engine = None
        self.recording = False
        self.patient_name = "Unbekannt"
        self.insurance_type = None  # set during patient intake; InsuranceType.GKV/PKV/BG
        _kura_data_dir = os.path.expanduser("~/Library/Application Support/Kura")
        os.makedirs(_kura_data_dir, exist_ok=True)
        self.temp_audio = os.path.join(_kura_data_dir, "session.wav")
        self.last_report = None
        self.last_billing_result = None

        # ── Microphone ownership ──────────────────────────────────────────────
        # ffmpeg is started in its own process group (os.setsid) so we can
        # kill it atomically. A PID file lets the next launch clean up after
        # a SIGKILL / force-quit where atexit/SIGTERM handlers never ran.
        self._ffmpeg_pid_file = os.path.join(_kura_data_dir, "ffmpeg.pid")
        self._kill_stale_ffmpeg()

        # atexit fires on normal exit and Cmd+Q (not SIGKILL — that's what
        # the PID file covers on next launch).
        # Do NOT set a SIGTERM handler: macOS sends SIGTERM to apps during
        # Gatekeeper/launch validation, and intercepting it would kill the app
        # before it starts. rumps/NSApplication manages graceful shutdown.
        import atexit
        atexit.register(self._kill_stale_ffmpeg)
        atexit.register(self._quit)
        # --- NEW: Timer State ---
        self.seconds_elapsed = 0
        self.timer = rumps.Timer(self.update_timer, 1)  # Ticks every 1 second

        self.pending_ai_result = None
        self._review_in_progress = False
        self._ui_queue = __import__('queue').Queue()  # thread-safe UI update queue
        self.gui_monitor = rumps.Timer(self.check_for_ai_result, 0.5)
        self.gui_monitor.start()

        # --- Boot AI Engine in Background ---
        threading.Thread(target=self.boot, daemon=True).start()

    def _on_main(self, fn):
        """Schedule a UI call to run on the main thread via the gui_monitor."""
        self._ui_queue.put(fn)

    def _refresh_menu_state(self, license_status=None):
        """Rebuild menu labels and callbacks to match current license + recording state."""
        if license_status is None:
            license_status = self.license_mgr.verify_locally()
        is_pro = (license_status is True)

        # License item
        if is_pro:
            self._item_lic.title = "Abo: Aktiv"
            self._item_lic.set_callback(None)          # not clickable when active
            self._item_deact.set_callback(self.deactivate_license)
        elif license_status == "TRIAL":
            count = self.license_mgr.get_trial_count()
            rem   = self.license_mgr.max_trials - count
            self._item_lic.title = f"Kura Pro aktivieren  ({rem} Testberichte verbleibend)"
            self._item_lic.set_callback(self.activate_license)
            self._item_deact.set_callback(None)
        else:
            self._item_lic.title = "Kura Pro aktivieren"
            self._item_lic.set_callback(self.activate_license)
            self._item_deact.set_callback(None)

        # Practice config + Gist override — Pro only
        self._item_config.set_callback(
            self.open_practice_config if is_pro else self._config_locked
        )
        self._item_gist_override.set_callback(
            self.open_gist_override if is_pro else self._config_locked
        )


    def _set_recording_state(self, recording: bool):
        self._item_start.set_callback(None if recording else self.start)
        self._item_stop.set_callback(self.stop if recording else None)

    def _quit(self, _=None):
        """Graceful shutdown: stop recording, free GPU/Metal memory, then exit."""
        # Stop any active recording first
        if getattr(self, 'recording', False):
            self._kill_stale_ffmpeg()

        # Release LLM weights and flush Metal cache
        engine = getattr(self, 'engine', None)
        if engine is not None:
            try:
                engine.cleanup()
            except Exception:
                pass

        rumps.quit_application()

    def _kill_stale_ffmpeg(self):
        """
        Kill any ffmpeg process we own, using the process group so that even
        child processes of ffmpeg (e.g. codec helpers) are taken down.
        Also handles PID-file recovery after a SIGKILL / force-quit.
        """
        import signal as _signal

        # 1. Live process object from this session
        proc = getattr(self, 'proc', None)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self.proc = None

        # 2. PID file left by a previous crash / force-quit
        pid_file = getattr(self, '_ffmpeg_pid_file', None)
        if pid_file and os.path.exists(pid_file):
            try:
                with open(pid_file) as f:
                    old_pid = int(f.read().strip())
                try:
                    os.killpg(os.getpgid(old_pid), _signal.SIGTERM)
                except ProcessLookupError:
                    pass  # already gone — that's fine
            except Exception:
                pass
            try:
                os.remove(pid_file)
            except Exception:
                pass

    def check_for_ai_result(self, _):
        """Drains the UI queue and opens the review window on the main thread."""
        while not self._ui_queue.empty():
            try:
                self._ui_queue.get_nowait()()
            except Exception:
                pass

        if self.pending_ai_result and not self._review_in_progress:
            res = self.pending_ai_result
            self.pending_ai_result = None
            self._set_icon("idle")
            self._review_in_progress = True
            # afplay is non-blocking — fires immediately, no UI freeze
            subprocess.Popen(['afplay', '/System/Library/Sounds/Glass.aiff'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            soap = res.get('soap', {})
            br = res.get('billing_result')

            if br:
                billing_line = br.format_billing_line()
                total_items  = len([a for a in br.audit_items if a.status != "PASS"])
                pass_items   = len([a for a in br.audit_items if a.status == "PASS"])
                if br.audit_status == "PASS":
                    audit_summary = f"✅ AUDIT BESTANDEN — alle {pass_items} Prüfpunkte erfüllt"
                elif br.audit_status == "BLOCK":
                    audit_summary = f"🔴 ABRECHNUNG GESPERRT — ärztliche Abklärung erforderlich"
                else:
                    audit_summary = f"⚠️  PRÜFUNG ERFORDERLICH — {total_items} Hinweise (vor Abrechnung prüfen)"
                flagged = [str(i) for i in br.audit_items if i.status in ("WARN", "FAIL", "BLOCK")]
                audit_notes = "\n".join(flagged) if flagged else ""
            else:
                billing_line = res.get('billing_suggestion', '–')
                audit_summary = "⚠️  Abrechnungsprüfung nicht verfügbar"
                warnings = res.get('compliance_check', [])
                audit_notes = "\n".join(warnings) if warnings else ""

            icd           = res.get('icd10', '–')
            date_str      = time.strftime("%d.%m.%Y")
            patient_display = self.patient_name.replace('_', ' ')
            profile_label = res.get('profile_label', '')

            footer = f"{billing_line}  |  ICD-10: {icd}\n{audit_summary}"
            if audit_notes:
                footer += f"\n{audit_notes}"

            profile_line = f"Profil: {profile_label}\n" if profile_label else ""
            initial_text = (
                f"KURA — {patient_display}  |  {date_str}\n"
                f"{profile_line}"
                f"{'─'*52}\n\n"
                f"SUBJEKTIV\n{soap.get('S', '')}\n\n"
                f"OBJEKTIV\n{soap.get('O', '')}\n\n"
                f"ASSESSMENT\n{soap.get('A', '')}\n\n"
                f"PLAN\n{soap.get('P', '')}\n\n"
                f"{'─'*52}\n"
                f"{footer}"
            )

            window = rumps.Window(
                message="Prüfen und in Abrechnung übernehmen:",
                title="KURA v2026.3.2",
                default_text=initial_text,
                ok="KOPIEREN & PDF",
                cancel="Abbrechen"
            )
            response = window.run()
            self._review_in_progress = False

            if response.clicked:
                threading.Thread(target=self.finalize_from_simple_text,
                                 args=(response.text, res), daemon=True).start()

    def finalize_from_simple_text(self, edited_text, res=None):
        """THE PAYWALL GATEKEEPER & LEARNING ENGINE: Handles License, Logic, and PDF."""
        # 1. CHECK LICENSE STATUS
        status = self.license_mgr.verify_locally()

        if status is False:
            rumps.alert(
                title="Kura Testphase beendet",
                message="Sie haben das Limit von 5 kostenlosen Berichten erreicht.\n\n"
                        "Bitte aktivieren Sie Kura Pro, um Ihre Arbeit zu speichern.",
                ok="Jetzt aktivieren"
            )
            self.activate_license(None)
            return

        # 2. IF IN TRIAL: Handle metadata
        if status == "TRIAL":
            current_usage = self.license_mgr.get_trial_count()
            remaining = self.license_mgr.max_trials - (current_usage + 1)
            self.license_mgr.increment_trial()
            rumps.notification(
                title="Kura Testphase",
                subtitle=f"Bericht {current_usage + 1} von {self.license_mgr.max_trials}",
                message=f"Noch {remaining} kostenlose Berichte verbleibend."
            )

        # 3. PROCEED WITH THE ACTUAL WORK
        try:
            import re

            # --- NEW: LEARNING ENGINE LOGIC ---
            # A. Extract the ICD code the user (potentially) edited in the window
            # This looks for "ICD-10: M41.2" or "ICD-10: G81.1"
            icd_match = re.search(r"ICD-10:\s*([A-Z][0-9][0-9]\.[0-9])", edited_text)

            # Define user_icd defensively so it can be referenced later even if
            # the AI result is missing (avoid UnboundLocalError).
            user_icd = icd_match.group(1) if icd_match else None

            if user_icd and res:
                ai_icd = res.get('icd10')
                transcript = res.get('transcript', "")

                # If the therapist manually fixed a hallucination (e.g., M37.0 -> M41.2)
                if user_icd != ai_icd:
                    # Assuming you've initialized self.engine.learning_mgr
                    try:
                        self.engine.learning_mgr.log_correction(transcript, ai_icd, user_icd)
                        print(f"🧠 Sharpener: Learned {user_icd} for this context.")
                    except Exception:
                        # Learning should not block finalization; ignore failures
                        print("⚠️ Learning manager failed to log correction.")

            # --- B. CLIPBOARD LOGIC ---
            # Strip Kura header and billing footer — paste only clean SOAP into practice software
            soap_only = re.sub(r'^KURA[^\n]*\n[-─]+\n\n?', '', edited_text)  # remove title bar
            soap_only = re.sub(r'\n[-─]{3,}.*', '', soap_only, flags=re.DOTALL)  # remove footer
            soap_only = soap_only.strip()
            process = subprocess.Popen('pbcopy', stdin=subprocess.PIPE)
            process.communicate(soap_only.encode('utf-8'))

            # --- C. STATE & PDF & ARCHIVE ---
            self.last_report = edited_text
            self.last_billing_result = res.get('billing_result') if res else None
            self.save_pdf(None)

            timestamp = time.strftime("%Y%m%d-%H%M")
            patient_dir = os.path.join(self.report_dir, self.patient_name)
            os.makedirs(patient_dir, exist_ok=True)

            # Save full data for future "Reflective Learning"
            with open(os.path.join(patient_dir, f"{timestamp}.json"), 'w', encoding='utf-8') as f:
                json.dump({
                    "text": edited_text,
                    "patient": self.patient_name,
                    "icd10": user_icd if icd_match else "Unknown",
                    "timestamp": timestamp
                }, f, ensure_ascii=False, indent=4)

            self.update_license_display()

        except Exception as e:
            print(f"❌ Finalize Error: {e}")
            rumps.alert("Systemfehler", f"Konnte Daten nicht verarbeiten: {e}")

    def update_license_display(self):
        self._refresh_menu_state()

    # ── Update check ──────────────────────────────────────────────────────────

    def check_for_update(self, _=None):
        """Check Gist version against APP_VERSION. Called on boot and from tray."""
        def _run():
            try:
                import requests as _req
                r = _req.get(_VERSION_URL, timeout=6)
                if r.status_code != 200:
                    self._on_main(lambda: self._item_update.set_callback(self.check_for_update))
                    return
                gist_ver = r.json().get("version", "")
                if self._version_gt(gist_ver, APP_VERSION):
                    # New version available
                    self._on_main(lambda v=gist_ver: (
                        setattr(self._item_update, 'title',
                                f"Update verfuegbar: v{v} (jetzt herunterladen)"),
                        self._item_update.set_callback(self._open_update_page),
                        rumps.notification(
                            "Kura Update verfuegbar",
                            f"Version {v} ist bereit",
                            "Klicken Sie auf 'Update verfuegbar' im Tray und beenden Sie Kura vor der Installation.",
                        ),
                    ))
                else:
                    self._on_main(lambda: (
                        setattr(self._item_update, 'title', "Aktualisierungen pruefen"),
                        self._item_update.set_callback(self.check_for_update),
                        rumps.notification("Kura", "Kein Update", f"Sie verwenden die aktuelle Version ({APP_VERSION})."),
                    ))
            except Exception as e:
                print(f"Update-Prüfung fehlgeschlagen: {e}")
                self._on_main(lambda: rumps.notification(
                    "Kura", "Update-Prüfung fehlgeschlagen",
                    "Keine Verbindung. Bitte später erneut versuchen."
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _open_update_page(self, _=None):
        rumps.alert(
            title="Kura beenden vor Update",
            message=(
                f"Neue Version verfuegbar.\n\n"
                "Wichtig: Beenden Sie Kura zuerst ueber 'Beenden' im Tray,\n"
                "bevor Sie die neue Version installieren.\n\n"
                "GitHub-Releases wird jetzt geoeffnet."
            ),
            ok="Herunterladen"
        )
        subprocess.Popen(["open", _DOWNLOAD_URL])

    @staticmethod
    def _version_gt(a: str, b: str) -> bool:
        """Return True if version string a is strictly greater than b (e.g. '2026.4.0' > '2026.3.2')."""
        try:
            return tuple(int(x) for x in a.split(".")) > tuple(int(x) for x in b.split("."))
        except Exception:
            return False

    def update_timer(self, _):
        """Updates the menu bar title with recording duration. Detects ffmpeg death."""
        if self.recording:
            # Detect if ffmpeg died unexpectedly (crash, OOM, killed externally)
            proc = getattr(self, 'proc', None)
            if proc is not None and proc.poll() is not None:
                # ffmpeg exited on its own — stop cleanly
                import logging as _logging
                _logging.getLogger("kura").warning(
                    "ffmpeg exited unexpectedly (rc=%d) after %ds — auto-stopping",
                    proc.returncode, self.seconds_elapsed
                )
                self.recording = False
                self.timer.stop()
                self._set_recording_state(False)
                self._set_icon("idle")
                self.status_item.title = "✅ Kura Bereit (Lokal & DSGVO)"
                self.seconds_elapsed = 0
                try:
                    if os.path.exists(self._ffmpeg_pid_file):
                        os.remove(self._ffmpeg_pid_file)
                except Exception:
                    pass
                rumps.notification(
                    "Kura – Aufnahme unterbrochen",
                    "Mikrofon-Aufnahme unerwartet beendet",
                    "Bitte neue Sitzung starten."
                )
                return

            self.seconds_elapsed += 1
            mins, secs = divmod(self.seconds_elapsed, 60)
            self._on_main(lambda m=f"⏺ {mins}:{secs:02d} – Aufnahme läuft":
                          setattr(self.status_item, 'title', m))

    # --- Boot Engine ---
    def _ffmpeg_path(self):
        """Return the ffmpeg binary path: bundled first, then Homebrew fallbacks."""
        import shutil
        candidates = []
        if getattr(sys, 'frozen', False):
            base = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
            candidates.append(os.path.join(base, 'ffmpeg'))
        candidates += ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']
        for p in candidates:
            if os.path.isfile(p):
                return p
        found = shutil.which('ffmpeg')
        if found:
            return found
        raise FileNotFoundError(
            "ffmpeg nicht gefunden. Installieren Sie es mit: brew install ffmpeg"
        )

    def _set_icon(self, state: str, label: str = ""):
        """Thread-safe icon + title update. state: 'idle' | 'record' | 'ai'"""
        icon = _asset(f"icon_{state}.png")
        # idle uses template=True (black stethoscope adapts to dark/light menu bar)
        # record/ai use template=False to preserve red/blue colour
        tmpl = (state == "idle")
        self._on_main(lambda i=icon, t=label, tm=tmpl: (
            setattr(self, 'template', tm),
            setattr(self, 'icon', i),
            setattr(self, 'title', t),
        ))

    def _set_status(self, text):
        """Thread-safe status update — always runs on the main thread via gui_monitor."""
        self._on_main(lambda t=text: setattr(self.status_item, 'title', t))

    def boot(self):
        """Initialize AI engine with proper error handling"""
        try:
            self._set_status("⏳ Modelle laden... (0%)")

            # Resolve correct models path first (bundle vs source)
            if getattr(sys, 'frozen', False):
                bundle_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
                models_path = os.path.join(bundle_dir, 'models')
                print(f"Running as bundle, models path: {models_path}")
            else:
                models_path = os.path.join(os.path.dirname(__file__), '..', 'models')

            if not os.path.exists(models_path):
                raise FileNotFoundError(f"Models directory not found: {models_path}")

            self._set_status("⏳ Modelle laden... (30%)")

            # Import here so scipy/mlx_whisper don't block the app startup
            from physio_scribe import KuraEngine
            self.engine = KuraEngine()

            self._set_status("⏳ Initialisierung... (90%)")
            time.sleep(0.3)

            self._on_main(lambda: (
                setattr(self.status_item, 'title', "✅ Kura Bereit (Lokal & DSGVO)"),
                setattr(self, 'template', True),
                setattr(self, 'icon', _asset("icon_idle.png")),
                self._item_start.set_callback(self.start),
                rumps.notification("Kura", "Bereit", "KI-Modelle geladen. Kura ist einsatzbereit.")
            ))
            # Silent background update check — runs after boot, no UI block
            self.check_for_update()

        except MemoryError as e:
            error_msg = f"Speicher-Fehler: {e}"
            print(f"MEMORY ERROR: {error_msg}")
            self._set_status("❌ Zu wenig RAM")
            self._on_main(lambda: rumps.notification(
                "Kura Speicherfehler",
                "Nicht genug RAM",
                "Bitte schließen Sie andere Apps und starten Sie Kura neu."
            ))

        except RuntimeError as e:
            error_msg = str(e)
            if "GPU" in error_msg or "Metal" in error_msg:
                print(f"GPU ERROR: {error_msg}")
                self._set_status("❌ GPU-Fehler")
                self._on_main(lambda msg=error_msg: (
                    rumps.notification(
                        "Kura GPU-Fehler",
                        "Metal-Speicher konnte nicht alloziert werden",
                        "Lösung: Mac neu starten oder andere GPU-Apps schließen"
                    ),
                    rumps.alert(
                        "GPU-Speicherfehler (IOGPUDeviceShmem)",
                        "Kura konnte keinen GPU-Speicher allozieren.\n\n"
                        "Häufige Ursachen:\n"
                        "• Andere Apps belegen GPU (Chrome, Photoshop, etc.)\n"
                        "• Vorherige Kura-Instanz läuft noch\n"
                        "• System-RAM ist voll\n\n"
                        "Lösung: Mac neu starten oder andere GPU-Apps schließen."
                    )
                ))
            log_dir = os.path.expanduser("~/Library/Logs/Kura")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"boot_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            with open(log_file, 'w') as f:
                f.write(error_msg + "\n\n")
                f.write(traceback.format_exc())
            self._on_main(lambda lf=log_file: rumps.notification(
                "Kura Start-Fehler", "App konnte nicht starten", f"Log: {lf}"
            ))

        except FileNotFoundError as e:
            print(f"MODELS NOT FOUND: {e}")
            self._set_status("❌ Modelle fehlen")
            self._on_main(lambda msg=str(e): rumps.alert(
                "KI-Modelle nicht gefunden",
                f"Die Modelldateien fehlen:\n{msg}\n\n"
                "Bitte stellen Sie sicher, dass die Modelle im 'models/'-Ordner vorhanden sind "
                "und die App neu starten.",
                ok="OK"
            ))

        except Exception as e:
            print(f"BOOT ERROR: {e}\n{traceback.format_exc()}")
            self._set_status("❌ Start-Fehler")
            log_dir = os.path.expanduser("~/Library/Logs/Kura")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"boot_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            with open(log_file, 'w') as f:
                f.write(str(e) + "\n\n")
                f.write(traceback.format_exc())
            self._on_main(lambda lf=log_file: rumps.notification(
                "Kura Start-Fehler", "Unbekannter Fehler beim Start", f"Log: {lf}"
            ))

    def check_microphone_permission(self):
        """Check mic permission via AVFoundation — no audio capture, no orange indicator."""
        try:
            from AVFoundation import (
                AVCaptureDevice, AVMediaTypeAudio,
                AVAuthorizationStatusAuthorized, AVAuthorizationStatusNotDetermined,
            )
            status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
            if status == AVAuthorizationStatusAuthorized:
                return True
            if status == AVAuthorizationStatusNotDetermined:
                # Request access — macOS shows the system dialog
                granted = [False]
                done = __import__('threading').Event()
                def handler(ok):
                    granted[0] = ok
                    done.set()
                AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                    AVMediaTypeAudio, handler
                )
                done.wait(timeout=30)
                return granted[0]
            return False
        except Exception as e:
            print(f"⚠️ AVFoundation permission check failed: {e}, assuming granted")
            # pyobjc-framework-AVFoundation not installed — can't check status.
            # If the user has already granted mic permission in System Settings,
            # ffmpeg will work fine. Returning True avoids opening a probe stream
            # that triggers the orange indicator and can conflict with ffmpeg.
            return True

    # --- Open Reports Folder ---
    def open_archive(self, _):
        subprocess.run(["open", self.report_dir])

    # --- Practice Config ---
    def _config_locked(self, _):
        rumps.alert(
            title="Kura Pro erforderlich",
            message="Praxis-Einstellungen sind nur mit einem aktiven Kura Pro Abo verfuegbar.\n\n"
                    "Aktivieren Sie Ihr Abo, um Praxisname, Betriebsstaettennummer\n"
                    "und individuelle Abrechnungsregeln zu konfigurieren.",
            ok="Abo aktivieren",
        )
        self.activate_license(None)

    def open_practice_config(self, _):
        cfg_path = os.path.expanduser("~/.kura_practice.json")

        # Step 1 — Praxis name
        w1 = rumps.Window(
            message="Praxisname (z.B. Physiotherapie Mustermann):",
            title="Praxis-Einstellungen 1/3",
            default_text=self._pc_get("practice", "name", cfg_path) or "",
            ok="Weiter",
            cancel="Abbrechen",
            dimensions=(320, 24),
        )
        r1 = w1.run()
        if not r1.clicked:
            return
        name = r1.text.strip() or "Meine Praxis"

        # Step 2 — Betriebsstaettennummer
        w2 = rumps.Window(
            message="Betriebsstaettennummer (BSNR, 9-stellig):",
            title="Praxis-Einstellungen 2/3",
            default_text=self._pc_get("practice", "license_number", cfg_path) or "",
            ok="Weiter",
            cancel="Abbrechen",
            dimensions=(320, 24),
        )
        r2 = w2.run()
        if not r2.clicked:
            return
        bsnr = r2.text.strip()

        # Step 3 — Location
        w3 = rumps.Window(
            message="Standort (Stadt / Adresse):",
            title="Praxis-Einstellungen 3/3",
            default_text=self._pc_get("practice", "location", cfg_path) or "",
            ok="Speichern",
            cancel="Abbrechen",
            dimensions=(320, 24),
        )
        r3 = w3.run()
        if not r3.clicked:
            return

        # Save
        try:
            from shared.practice_config import PracticeConfig
            pc = PracticeConfig(practice_file=cfg_path)
            pc.config["practice"]["name"]           = name
            pc.config["practice"]["license_number"] = bsnr
            pc.config["practice"]["location"]       = r3.text.strip()
            pc.save()
            rumps.notification(
                "Kura", "Einstellungen gespeichert",
                f"{name} — BSNR {bsnr}",
            )
            # Offer advanced edit
            if rumps.alert(
                title="Erweiterte Konfiguration",
                message="Moechten Sie die vollstaendige Konfigurationsdatei oeffnen?\n\n"
                        "(JSON-Editor — fuer ICD-10-Regeln, Abrechnungscodes, Audit-Schwellwerte)",
                ok="Oeffnen",
                cancel="Fertig",
            ) == 1:
                subprocess.run(["open", cfg_path])
        except Exception as e:
            rumps.alert("Fehler", f"Konnte Einstellungen nicht speichern:\n{e}")

    def open_gist_override(self, _):
        """
        Pull the latest Gist, create a local override file pre-filled with those
        values (if it doesn't already exist), then open it in the system editor.
        Customer edits are applied locally on next startup — never sent anywhere.
        """
        if not self.engine:
            rumps.alert("Nicht bereit", "KI-Engine noch nicht geladen. Bitte warten.")
            return

        cfg = self.engine.config
        override_path = cfg.local_override_path

        # On first use: create the template from current (Gist) values
        created = not os.path.exists(override_path)
        try:
            cfg.create_override_template()
        except Exception as e:
            rumps.alert("Fehler", f"Konnte Konfigurationsdatei nicht erstellen:\n{e}")
            return

        if created:
            rumps.alert(
                title="Konfiguration anpassen",
                message=(
                    "Eine Konfigurationsdatei wurde aus den aktuellen Kura-Werten erstellt:\n\n"
                    f"{override_path}\n\n"
                    "Aendern Sie nur die Werte, die Sie anpassen moechten.\n"
                    "Die Datei wird beim naechsten Start von Kura automatisch geladen.\n\n"
                    "Die Datei verbleibt ausschliesslich auf Ihrem Geraet."
                ),
            )
        subprocess.run(["open", override_path])

    def _pc_get(self, section: str, key: str, cfg_path: str) -> str:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f).get(section, {}).get(key, "")
        except Exception:
            return ""

    # --- Start Session ---
    def start(self, _):
        if self.recording:
            rumps.alert("Fehler", "Aufnahme läuft bereits.")
            return
            
        if not self.engine:
            rumps.alert("Fehler", "KI-Engine nicht bereit.\n\nBitte warten Sie, bis 'KI-Modelle laden...' abgeschlossen ist.\n\nFalls das Problem weiterhin besteht, starten Sie Kura neu.")
            return

        # Input Window: Standardized for Name and Birthdate
        window = rumps.Window(
            "Eingabe: Name_DDMMYYYY (z.B. Weber_15031964)",
            "Kura - Patienten-Identifikation",
            "Mustermann_01011980"
        )
        response = window.run()

        if response.clicked:
            raw_input = response.text.strip().replace(" ", "_")
            self.patient_name = raw_input

            # Insurance type dialog
            from shared.billing_engine import InsuranceType
            ins_window = rumps.Window(
                "Kassentyp eingeben: GKV, PKV oder BG",
                "Kura - Versicherungstyp",
                "GKV"
            )
            ins_response = ins_window.run()
            ins_text = ins_response.text.strip().upper() if ins_response.clicked else "GKV"
            self.insurance_type = {
                "PKV": InsuranceType.PKV,
                "BG":  InsuranceType.BG,
            }.get(ins_text, InsuranceType.GKV)

            # Check microphone permission at first use, not at startup
            if not self.check_microphone_permission():
                rumps.alert(
                    "Mikrofon-Berechtigung erforderlich",
                    "Kura benötigt Zugriff auf das Mikrofon.\n\n"
                    "Systemeinstellungen → Datenschutz & Sicherheit → Mikrofon → Kura aktivieren\n\n"
                    "Danach Kura neu starten.",
                    ok="OK"
                )
                return

            self.seconds_elapsed = 0
            self.timer.start()
            self.recording = True
            self._set_recording_state(True)
            self._set_icon("record")

            try:
                # Record as 16kHz mono PCM — exactly what Whisper needs.
                # os.setsid puts ffmpeg in its own process group so we can
                # kill it atomically (os.killpg) even after a crash.
                self.proc = subprocess.Popen(
                    [self._ffmpeg_path(), '-y',
                     '-f', 'avfoundation', '-i', ':0',
                     '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le',
                     '-t', '5400',   # hard cap: 90 min max (physiotherapy session limit)
                     self.temp_audio],
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
                # Write PID so next launch can clean up after SIGKILL / force-quit
                try:
                    with open(self._ffmpeg_pid_file, 'w') as _f:
                        _f.write(str(self.proc.pid))
                except Exception:
                    pass
                rumps.notification("Kura", "Aufnahme gestartet", f"Patient: {self.patient_name}")
            except Exception as e:
                self.recording = False
                self.timer.stop()
                self._set_recording_state(False)
                rumps.alert("Aufnahmefehler", f"Konnte Aufnahme nicht starten: {e}\n\nPrüfen Sie Mikrofon-Berechtigung in Systemeinstellungen.")

    def stop(self, _):
        if not self.recording:
            return

        self.timer.stop()
        self.recording = False
        self._set_recording_state(False)
        self._set_icon("idle")
        self.status_item.title = "✅ Kura Bereit (Lokal & DSGVO)"

        self._kill_stale_ffmpeg()
        # PID file no longer needed — recording ended cleanly
        try:
            if os.path.exists(self._ffmpeg_pid_file):
                os.remove(self._ffmpeg_pid_file)
        except Exception:
            pass

        if self.seconds_elapsed < 10:
            rumps.alert("Aufnahme zu kurz",
                        f"Nur {self.seconds_elapsed}s aufgenommen.\n"
                        "Bitte mindestens 10 Sekunden sprechen.")
            self.seconds_elapsed = 0
            return

        if not self._audio_has_speech(self.temp_audio):
            rumps.alert("Kein Ton erkannt",
                        "Die Aufnahme enthält kein hörbares Sprachsignal.\n"
                        "Prüfen Sie das Mikrofon und versuchen Sie es erneut.")
            try:
                os.remove(self.temp_audio)
            except Exception:
                pass
            self.seconds_elapsed = 0
            return

        self.seconds_elapsed = 0

        status = self.license_mgr.verify_locally()
        if status is True or status == "TRIAL":
            self._set_icon("ai")
            threading.Thread(target=self.run_ai).start()
        else:
            self.show_upgrade_dialog()

    def _audio_has_speech(self, path: str, silence_db: float = -60.0) -> bool:
        """Returns True if the audio file contains signal above silence_db (dBFS)."""
        try:
            import re
            result = subprocess.run(
                [self._ffmpeg_path(), '-i', path,
                 '-af', 'volumedetect', '-f', 'null', '/dev/null'],
                capture_output=True, text=True, timeout=15
            )
            match = re.search(r'mean_volume:\s*([-\d.]+)\s*dB', result.stderr)
            if match:
                return float(match.group(1)) > silence_db
            return True  # can't determine — let Whisper try
        except Exception:
            return True

    # --- AI Engine Workflow ---
    def run_ai(self):
        try:
            def update_status(msg):
                # Show status text next to the AI icon in the menu bar
                self._set_icon("ai", msg)

            # Execute AI Engine - this is the heavy part
            res = self.engine.run_full_flow(
                self.temp_audio,
                status_callback=update_status,
                insurance_type=self.insurance_type,
            )

            # Short pause to let the user see the "Prüfung" status
            time.sleep(0.8)

            # DSGVO COMPLIANCE: Delete audio recording immediately after processing
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
                    print(f"✅ Audio file deleted: {self.temp_audio} (DSGVO compliance)")
            except Exception as e:
                print(f"⚠️ Could not delete audio file: {e}")

            # Send to the main thread listener
            self.pending_ai_result = res

        except BaseException as e:
            import traceback
            tb = traceback.format_exc()
            print(f"❌ AI ERROR: {e}\n{tb}")
            try:
                import logging
                logging.getLogger("kura").error("run_ai crash: %s\n%s", e, tb)
            except Exception:
                pass
            self._set_icon("idle")
            self.pending_ai_result = None

            # Delete audio file even on error (DSGVO compliance)
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception:
                pass

            if not isinstance(e, (KeyboardInterrupt, SystemExit)):
                rumps.notification("Kura Fehler", "KI-Abbruch", str(e)[:200])

    def finalize_report(self, edited_res):
        """This runs AFTER the therapist clicks 'Save' in the window."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M")
            display_name = self.patient_name.replace("_", " ")

            # 1. Build the Report Body from EDITED data
            soap = edited_res.get('soap', {})
            soap_text = f"S: {soap.get('S')}\nO: {soap.get('O')}\nA: {soap.get('A')}\nP: {soap.get('P')}"

            # 2. Re-construct self.last_report for PDF/Clipboard
            version = self.engine.config.version
            self.last_report = (
                f"--- KURA PHYSIO-PROTOKOLL v{version} ---\n"
                f"ID: {timestamp} | PATIENT: {display_name}\n"
                f"ICD-10: {edited_res.get('icd10')}\n\n"
                f"SOAP-BEFUND:\n{soap_text}\n\n"
                f"ABRECHNUNG: {edited_res.get('billing_suggestion', '20701')}"
            )

            # 3. Save JSON Archive
            patient_dir = os.path.join(self.report_dir, self.patient_name)
            os.makedirs(patient_dir, exist_ok=True)
            with open(os.path.join(patient_dir, f"{timestamp}.json"), 'w', encoding='utf-8') as f:
                json.dump(edited_res, f, ensure_ascii=False, indent=4)

            # 4. Save PDF & Notify
            self.save_pdf(None)
            rumps.notification("Kura", "Erfolg", f"Bericht für {display_name} gespeichert.")

        except Exception as e:
            print(f"❌ Finalize Error: {e}")

    # --- License Upgrade ---
    def show_upgrade_dialog(self):
        import webbrowser
        msg = "Ihre Testphase ist beendet.\nAktivieren Sie Kura Pro fuer unbegrenzte Berichte."
        if rumps.alert("Kura Pro", msg, ok="Abo starten", cancel="Spaeter") == 1:
            webbrowser.open("https://kura.lemonsqueezy.com/checkout/buy/2400563b-a13a-4e42-b734-d79122e7ec92")
            self.activate_license(None)

    # --- Activate License ---
    def activate_license(self, _):
        win = rumps.Window(
            message=(
                "Geben Sie Ihren Kura Pro Lizenzschluessel ein:\n\n"
                "Format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX\n"
                "(aus Ihrer Kaufbestaetigung kopieren)"
            ),
            title="Kura Pro aktivieren",
            default_text="",
            ok="Aktivieren",
            cancel="Abbrechen",
            dimensions=(340, 24),
        )
        win.add_button("Jetzt kaufen")
        res = win.run()

        if res.clicked == 1:  # Aktivieren
            ok, msg = self.license_mgr.activate(res.text.strip())
            if ok:
                rumps.alert("Aktivierung erfolgreich", msg)
                self.update_license_display()
            else:
                rumps.alert("Aktivierung fehlgeschlagen", msg)
        elif res.clicked == 2:  # Jetzt kaufen
            import webbrowser
            webbrowser.open("https://kura-medical.de/#pricing")

    def deactivate_license(self, _):
        if rumps.alert(
            title="Lizenz deaktivieren?",
            message=(
                "Dies entfernt Kura Pro von diesem Geraet.\n\n"
                "Der Schluessel kann danach auf einem anderen Geraet aktiviert werden.\n\n"
                "Internetverbindung erforderlich."
            ),
            ok="Deaktivieren",
            cancel="Abbrechen",
        ) == 1:
            ok, msg = self.license_mgr.deactivate()
            rumps.alert("Erledigt" if ok else "Fehler", msg)
            self.update_license_display()

    def save_pdf(self, _):
        if not self.last_report:
            return

        # 1. Define Paths
        safe_name = self.patient_name.replace(' ', '_')
        desktop_path = os.path.expanduser(f"~/Desktop/Kura_{safe_name}.pdf")

        timestamp_file = time.strftime("%Y%m%d-%H%M")
        patient_folder = os.path.join(self.report_dir, safe_name)
        os.makedirs(patient_folder, exist_ok=True)
        archive_path = os.path.join(patient_folder, f"Bericht_{timestamp_file}.pdf")

        try:
            import re as _re

            # ── Character safety ──────────────────────────────────────────
            _EMOJI_MAP = [
                ('\u2705', '[PASS]'), ('\u26a0\ufe0f', '[WARN]'), ('\u26a0', '[WARN]'),
                ('\U0001f534', '[STOP]'), ('\U0001f4cb', '[FEHLT]'), ('\u2753', '[?]'),
                ('\u274c', '[STOP]'), ('\u2714', '[PASS]'), ('\u2713', '[PASS]'),
            ]
            _CHAR_MAP = str.maketrans({
                '\u20ac': 'EUR',            # €
                '\u2013': '-', '\u2014': '-',   # – —
                '\u2192': '->', '\u2190': '<-', '\u2194': '<->',  # → ← ↔
                '\u2265': '>=', '\u2264': '<=', '\u2260': '!=',   # ≥ ≤ ≠
                '\u00b1': '+/-',            # ±
                '\u201e': '"', '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
                '\u2502': '|', '\u2500': '-', '\u2501': '-', '\u2550': '=',
                '\u2588': '#', '\u25a0': '#',
                '\xb0': 'Grad',
            })

            def s(text: str) -> str:
                if not isinstance(text, str):
                    text = str(text)
                for em, rep in _EMOJI_MAP:
                    text = text.replace(em, rep)
                text = text.translate(_CHAR_MAP)
                return text.encode('latin-1', 'replace').decode('latin-1')

            # ── Parse SOAP from review text ───────────────────────────────
            text = str(self.last_report)

            def _field(label):
                m = _re.search(rf'{label}\n(.*?)(?=\n[A-Z]{{2,}}|\n-{{3,}}|\Z)', text, _re.S)
                return m.group(1).strip() if m else ""

            soap_s = _field("SUBJEKTIV")
            soap_o = _field("OBJEKTIV")
            soap_a = _field("ASSESSMENT")
            soap_p = _field("PLAN")

            footer_m = _re.search(r'-{3,}\n(.*)', text, _re.S)
            footer_raw = footer_m.group(1).strip() if footer_m else ""

            # Split billing line from audit notes
            footer_lines = footer_raw.splitlines()
            billing_line = s(footer_lines[0]) if footer_lines else ""
            audit_lines  = [s(l) for l in footer_lines[1:] if l.strip()]

            # ── Layout constants ──────────────────────────────────────────
            now            = datetime.now()
            patient_display = self.patient_name.replace("_", " ")
            date_str       = now.strftime("%d.%m.%Y")
            time_str       = now.strftime("%H:%M")
            W              = 180   # usable width (A4 210 - 15 margins x2)
            COL            = 90    # half width

            DARK  = (25,  25,  25)
            MID   = (90,  90,  90)
            LIGHT = (140, 140, 140)
            WHITE = (255, 255, 255)
            BODY  = (20,  20,  20)
            RULE  = (210, 210, 210)
            SHADE = (248, 248, 248)

            # ── Build PDF ─────────────────────────────────────────────────
            pdf = FPDF()
            pdf.set_margins(15, 15, 15)
            pdf.add_page()

            # ── 1. Header ─────────────────────────────────────────────────
            pdf.set_fill_color(*DARK)
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(W, 12, s("  KURA  Physiotherapeutischer Befund"),
                     new_x="LMARGIN", new_y="NEXT", align="L", fill=True)

            # ── 2. Patient info bar ───────────────────────────────────────
            pdf.set_fill_color(*SHADE)
            pdf.set_text_color(*MID)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(COL, 7, s(f"  Patient: {patient_display}"),
                     new_x="RIGHT", new_y="TOP", fill=True)
            pdf.cell(COL, 7, s(f"Datum: {date_str}   {time_str} Uhr  "),
                     new_x="LMARGIN", new_y="NEXT", align="R", fill=True)
            pdf.ln(6)

            # ── 3. SOAP sections ──────────────────────────────────────────
            SECTIONS = [
                ("SUBJEKTIV",  soap_s),
                ("OBJEKTIV",   soap_o),
                ("ASSESSMENT", soap_a),
                ("PLAN",       soap_p),
            ]

            for title, body in SECTIONS:
                if not body:
                    continue
                # Section label row
                pdf.set_fill_color(*SHADE)
                pdf.set_draw_color(*RULE)
                pdf.set_text_color(*MID)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(W, 6, s(f"  {title}"),
                         new_x="LMARGIN", new_y="NEXT", fill=True, border="B")
                # Body
                pdf.set_text_color(*BODY)
                pdf.set_font("Helvetica", "", 10)
                pdf.ln(2)
                pdf.multi_cell(W, 5.5, s(body), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

            # ── 4. Abrechnung & Audit ─────────────────────────────────────
            br_obj = getattr(self, 'last_billing_result', None)

            # Section header
            pdf.set_fill_color(*SHADE)
            pdf.set_draw_color(*RULE)
            pdf.set_text_color(*MID)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(W, 6, "  ABRECHNUNG & AUDIT",
                     new_x="LMARGIN", new_y="NEXT", fill=True, border=1)
            pdf.ln(2)

            if br_obj:
                # ── Abrechnungszeile ──────────────────────────────────────
                pdf.set_text_color(*BODY)
                pdf.set_font("Helvetica", "B", 9)
                pdf.multi_cell(W, 5.5, s(br_obj.format_billing_line()),
                               new_x="LMARGIN", new_y="NEXT")

                # Diagnosegruppe + Rechtsgrundlage
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*MID)
                dg_line = (f"Diagnosegruppe: {br_obj.diagnosegruppe} — "
                           f"{br_obj.diagnosegruppe_desc} | {br_obj.legal_basis}")
                pdf.multi_cell(W, 4.5, s(dg_line), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)

                # ── Audit-Status Badge ────────────────────────────────────
                STATUS_COLORS = {
                    "PASS":   (34,  139, 34),   # Grün — Audit bestanden
                    "REVIEW": (200, 100,  0),   # Orange — Hinweise vorhanden
                    "BLOCK":  (180,  20, 20),   # Rot — Abrechnung gesperrt
                }
                STATUS_LABELS = {
                    "PASS":   "AUDIT BESTANDEN",
                    "REVIEW": "PRUEFUNG ERFORDERLICH",
                    "BLOCK":  "ABRECHNUNG GESPERRT",
                }
                badge_col = STATUS_COLORS.get(br_obj.audit_status, (90, 90, 90))
                badge_txt = STATUS_LABELS.get(br_obj.audit_status, br_obj.audit_status)

                pass_count   = sum(1 for a in br_obj.audit_items if a.status == "PASS")
                total_count  = len(br_obj.audit_items)
                open_count   = total_count - pass_count

                if br_obj.audit_status == "PASS":
                    badge_detail = f"Alle {total_count} Pruefpunkte erfuellt"
                elif br_obj.audit_status == "BLOCK":
                    badge_detail = "Aerztliche Abklaerung vor Therapiefortsetzung erforderlich"
                else:
                    badge_detail = f"{open_count} von {total_count} Hinweisen offen"

                pdf.set_fill_color(*badge_col)
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(W, 7, s(f"  {badge_txt}  —  {badge_detail}"),
                         new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.ln(2)

                # ── Einzelne Audit-Punkte ─────────────────────────────────
                # Bei PASS: nur kritische Warnungen (FAIL/BLOCK) zeigen falls vorhanden
                # Bei REVIEW/BLOCK: alle nicht-PASS Punkte zeigen
                if br_obj.audit_status == "PASS":
                    show_items = [a for a in br_obj.audit_items
                                  if a.status in ("FAIL", "BLOCK")]
                else:
                    show_items = [a for a in br_obj.audit_items
                                  if a.status in ("WARN", "FAIL", "BLOCK")]

                if show_items:
                    pdf.set_font("Helvetica", "B", 7.5)
                    pdf.set_text_color(*MID)
                    pdf.cell(W, 5, "  Offene Pruefpunkte:",
                             new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "", 8)
                    for item in show_items:
                        item_col = {
                            "FAIL":  (180, 20, 20),
                            "BLOCK": (180, 20, 20),
                            "WARN":  (180, 100, 0),
                        }.get(item.status, BODY)
                        pdf.set_text_color(*item_col)
                        pdf.multi_cell(W, 4.5, s(f"  {item.icon} {item.label}"
                                                 + (f": {item.detail}" if item.detail else "")),
                                       new_x="LMARGIN", new_y="NEXT")

            else:
                # Fallback: plain text rendering (no billing_result object)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*BODY)
                if billing_line:
                    pdf.multi_cell(W, 5, billing_line, new_x="LMARGIN", new_y="NEXT")
                for al in audit_lines:
                    pdf.multi_cell(W, 5, al, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(1)
            pdf.set_draw_color(*RULE)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())

            # ── 5. Footer watermark ───────────────────────────────────────
            pdf.set_y(-12)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*LIGHT)
            pdf.cell(W, 5,
                     s(f"Kura v2026 | Lokale KI-Verarbeitung | DSGVO-konform | {date_str}"),
                     align="C")

            pdf.output(archive_path)
            pdf.output(desktop_path)
            rumps.notification("Kura", "PDF gespeichert",
                               s(f"{patient_display} | {date_str}"))

        except Exception as e:
            print(f"PDF Error: {e}")
            rumps.notification("Kura Fehler", "PDF konnte nicht erstellt werden", str(e))


if __name__ == "__main__":
    KuraApp().run()
