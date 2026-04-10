import json
import logging
import multiprocessing
import os
import re
import subprocess
import threading
import time
import sys
import traceback
from datetime import datetime

app_logger = logging.getLogger("kura.app")

import rumps
from fpdf import FPDF
from dotenv import load_dotenv
import AppKit as _AK

def _app_activate():
    """Bring app windows to front so dialogs receive keyboard focus.
    Never changes activation policy — toggling between Regular/Accessory
    breaks the NSStatusItem and hides the tray menu."""
    _AK.NSApp.activateIgnoringOtherApps_(True)

def _app_deactivate():
    """No-op: activation policy is never changed, so nothing to restore."""
    pass

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
    try:
        _fh = open(fault_file, 'a')
        if _fh is not None:
            faulthandler.enable(file=_fh)
    except Exception:
        # If fault handler setup fails, continue without it
        pass

    def log_crash(exc_type, exc_value, exc_traceback):
        try:
            with open(log_file, 'w') as f:
                if f is not None:
                    f.write(f"Kura Crash Report - {datetime.now()}\n")
                    f.write("="*60 + "\n\n")
                    f.write("Exception:\n")
                    if exc_type is not None:
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
        except Exception as log_err:
            print(f"Error writing crash log: {log_err}")

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
    # Try to find bundled .env.dist first, then .env.example
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            exe_dir = os.path.dirname(sys.executable)
            base_path = os.path.join(exe_dir, '..', 'Resources')
        dist_env = os.path.join(base_path, '.env.dist')
        example_file = os.path.join(base_path, '.env.example')
    else:
        base_path = os.path.join(os.path.dirname(__file__), '..')
        dist_env = os.path.join(base_path, '.env.dist')
        example_file = os.path.join(base_path, '.env.example')

    # Priority: .env.dist (bundled config) > .env.example > create template
    if os.path.exists(dist_env):
        try:
            import shutil
            shutil.copy(dist_env, user_env_file)
            print(f"✅ Copied bundled .env configuration (includes HF_TOKEN for fast downloads)")
        except Exception as copy_err:
            print(f"Error copying .env.dist: {copy_err}")
    elif os.path.exists(example_file):
        try:
            import shutil
            shutil.copy(example_file, user_env_file)
            print(f"📝 Created user .env from example: {user_env_file}")
            print(f"⚠️  Configure HF_TOKEN for faster downloads")
        except Exception as copy_err:
            print(f"Error copying .env.example: {copy_err}")
    else:
        # Create basic .env template
        try:
            with open(user_env_file, 'w') as f:
                if f is not None:
                    f.write("# Kura Configuration\n")
                    f.write("# Get HF_TOKEN at: https://huggingface.co/settings/tokens\n")
                    f.write("HF_TOKEN=your_token_here\n\n")
                    f.write("# Digistore24 License API\n")
                    f.write("DS24_API_KEY=\n")
                    f.write("DS24_PRODUCT_ID=\n")
            print(f"📝 Created template .env: {user_env_file}")
            print(f"⚠️  Configure HF_TOKEN for faster downloads")
        except Exception as env_err:
            print(f"Error creating .env file: {env_err}")

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

# MLX/Metal GPU environment setup - BEFORE any MLX imports
# These help prevent Metal/GPU initialization issues on Python 3.13
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['MLX_METAL_DEBUG'] = '0'  # Disable Metal debug mode
os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'  # Use safer Metal device wrapper

# Set up cleanup handler for multiprocessing resources
import atexit

def cleanup_multiprocessing_resources():
    """Clean up any leaked multiprocessing resources on exit."""
    try:
        # Force cleanup of resource tracker to prevent semaphore leak warnings
        from multiprocessing import resource_tracker
        if hasattr(resource_tracker, '_resource_tracker'):
            tracker = resource_tracker._resource_tracker
            if tracker is not None:
                # Clear any tracked resources before shutdown
                if hasattr(tracker, '_lock'):
                    try:
                        tracker._lock.acquire(timeout=1.0)
                        tracker._lock.release()
                    except:
                        pass
    except:
        pass  # Silently fail - cleanup is best effort

# Register cleanup handler
atexit.register(cleanup_multiprocessing_resources)

# Load HF_TOKEN from environment (set in .env or system environment)
if "HF_TOKEN" not in os.environ:
    print("⚠️ Warning: HF_TOKEN not found in environment variables. Set it in .env file or system environment.")


# --- Live Audio Visualizer ---


# --- App version — single source of truth: version.json at project root ---
from shared.version import APP_VERSION, VERSION_URL as _VERSION_URL
_DOWNLOAD_URL = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/Kura_macOS_v2026.4.1.dmg"


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
        app_logger.info(f"Reports directory: {self.report_dir}")

        # --- License Manager ---
        self.license_mgr = LicenseManager()
        status = self.license_mgr.verify_locally()

        # --- Menu items (stored as attrs for dynamic updates) ---
        is_pro = (status is True)

        self.status_item  = rumps.MenuItem("⏳  Modelle werden geladen …", callback=None)
        self._item_start  = rumps.MenuItem("▶  Neue Sitzung", callback=None)
        self._item_stop   = rumps.MenuItem("⏹  Stoppen & Auswerten", callback=None)
        self._item_arch   = rumps.MenuItem("📁  Archiv", callback=self.open_archive)
        self._item_config = rumps.MenuItem(
            "⚙️  Praxis-Einstellungen",
            callback=self.open_practice_config if is_pro else self._config_locked,
        )
        self._item_gist_override = rumps.MenuItem(
            "✏️  Konfiguration anpassen",
            callback=self.open_gist_override if is_pro else self._config_locked,
        )
        self._item_lic    = rumps.MenuItem("", callback=self.activate_license)
        self._item_deact  = rumps.MenuItem("🔓  Lizenz deaktivieren", callback=self.deactivate_license)
        self._item_update = rumps.MenuItem("🔄  Nach Updates suchen", callback=self.check_for_update)

        _is_dev = not getattr(sys, "frozen", False)
        if _is_dev:
            self._item_dev_reset = rumps.MenuItem("[DEV] Reset Trial", callback=self._dev_reset_trial)

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
            *([self._item_dev_reset] if _is_dev else []),
            rumps.MenuItem("⏻  Beenden", callback=self._quit),
        ]
        self._refresh_menu_state(status)

        # Boot-time notification — low trial count warning only (non-blocking)
        # Hard block happens at recording start, not at launch, to avoid false positives
        if status == "TRIAL":
            rem = self.license_mgr.max_trials - self.license_mgr.get_trial_count()
            if rem <= 2:
                threading.Timer(1.2, lambda r=rem: self._on_main(
                    lambda: rumps.notification(
                        "Kura Testphase",
                        f"Noch {r} Testbericht{'e' if r != 1 else ''} verbleibend",
                        "Upgrade auf Kura Pro für unbegrenzte Nutzung.",
                    )
                )).start()

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
            self._item_lic.title = "✅  Kura Pro — Lizenz aktiv"
            self._item_lic.set_callback(None)          # not clickable when active
            self._item_deact.set_callback(self.deactivate_license)
        elif license_status == "TRIAL":
            count = self.license_mgr.get_trial_count()
            rem   = self.license_mgr.max_trials - count
            self._item_lic.title = f"🔑  Kura Pro aktivieren  ({rem} von {self.license_mgr.max_trials} Testberichten verbleibend)"
            self._item_lic.set_callback(self.activate_license)
            self._item_deact.set_callback(None)
        else:
            self._item_lic.title = "🔑  Kura Pro aktivieren"
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
                critical_items = [a for a in br.audit_items if a.status in ("FAIL", "BLOCK")]
                pass_items     = len([a for a in br.audit_items if a.status == "PASS"])
                if br.audit_status == "PASS":
                    audit_summary = f"✅ AUDIT BESTANDEN — alle {pass_items} Prüfpunkte erfüllt"
                elif br.audit_status == "BLOCK":
                    audit_summary = f"🔴 ABRECHNUNG GESPERRT — ärztliche Abklärung erforderlich"
                else:
                    n = len(critical_items)
                    audit_summary = (
                        f"⚠️  {n} PFLICHTFELD{'ER' if n != 1 else ''} FEHLT — vor Abrechnung ergänzen"
                        if n else "⚠️  PRÜFUNG EMPFOHLEN"
                    )
                # Show only FAIL/BLOCK items — WARN items are informational hints, not blockers
                audit_notes = "\n".join(str(i) for i in critical_items) if critical_items else ""
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
                f"╔══ KURA — {patient_display}  │  {date_str} ══╗\n"
                f"{profile_line}"
                f"\n"
                f"── SUBJEKTIV ──────────────────────────────────────────\n"
                f"{soap.get('S', '')}\n\n"
                f"── OBJEKTIV ───────────────────────────────────────────\n"
                f"{soap.get('O', '')}\n\n"
                f"── ASSESSMENT ─────────────────────────────────────────\n"
                f"{soap.get('A', '')}\n\n"
                f"── PLAN ───────────────────────────────────────────────\n"
                f"{soap.get('P', '')}\n\n"
                f"───────────────────────────────────────────────────────\n"
                f"{footer}"
            )

            _app_activate()
            try:
                window = rumps.Window(
                    message="Bericht prüfen — bei Bedarf bearbeiten, dann speichern:",
                    title="Kura — Bericht",
                    default_text=initial_text,
                    ok="✓  Speichern & PDF",
                    cancel="Verwerfen",
                    dimensions=(680, 460),
                )
                response = window.run()
                if response.clicked:
                    threading.Thread(target=self.finalize_from_simple_text,
                                     args=(response.text, res), daemon=True).start()
            finally:
                _app_deactivate()
                self._review_in_progress = False

    def finalize_from_simple_text(self, edited_text, res=None):
        """THE PAYWALL GATEKEEPER & LEARNING ENGINE: Handles License, Logic, and PDF."""
        # 1. CHECK LICENSE STATUS
        status = self.license_mgr.verify_locally()

        if status is False:
            title, msg = self._license_block_message()
            rumps.alert(title=title, message=msg, ok="Lizenz aktivieren")
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

            if res:
                ai_icd = res.get('icd10')
                transcript = res.get('transcript', "")
                final_icd = user_icd or ai_icd or "M99.9"
                was_corrected = bool(user_icd and user_icd != ai_icd)

                if was_corrected:
                    try:
                        self.engine.learning_mgr.log_correction(transcript, ai_icd, user_icd)
                        print(f"🧠 Sharpener: Learned {user_icd} for this context.")
                    except Exception:
                        print("⚠️ Learning manager failed to log correction.")

                # Always log the accepted session for few-shot learning
                try:
                    soap = res.get('soap', {})
                    profile_id = res.get('profile_id', 'KG')
                    self.engine.learning_mgr.log_session(
                        transcript, soap, final_icd, profile_id, was_corrected
                    )
                    stats = self.engine.learning_mgr.stats()
                    print(f"🧠 Learning: {stats['total_sessions']} sessions stored "
                          f"({stats['corrected_sessions']} corrected)")
                except Exception as e:
                    print(f"⚠️ Learning log error: {e}")

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

            # Save JSON to date-based folder: archive/YYYY-MM-DD/HHMMSS_PatientName.json
            now = datetime.now()
            date_folder = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H%M%S")  # Include seconds for collision-proof naming
            day_folder = os.path.join(self.report_dir, date_folder)
            os.makedirs(day_folder, exist_ok=True)

            # Save full data for future "Reflective Learning"
            json_path = os.path.join(day_folder, f"{time_str}_{self.patient_name}.json")
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    if f is not None:
                        json.dump({
                            "text": edited_text,
                            "patient": self.patient_name,
                            "icd10": user_icd if icd_match else "Unknown",
                            "timestamp": now.strftime("%Y%m%d-%H%M"),
                            "date": date_folder,
                        }, f, ensure_ascii=False, indent=4)
            except Exception as json_err:
                print(f"JSON archive save error: {json_err}")

            self.update_license_display()

        except Exception as e:
            print(f"❌ Finalize Error: {e}")
            rumps.alert("Systemfehler", f"Konnte Daten nicht verarbeiten: {e}")

    def update_license_display(self):
        self._refresh_menu_state()

    # ── Update check ──────────────────────────────────────────────────────────

    def check_for_update(self, _=None, silent=False):
        """Check remote version.json against APP_VERSION.
        silent=True (boot): no notification on failure or 'already up to date'.
        silent=False (manual): always notify the result.
        """
        def _run():
            try:
                import requests as _req
                r = _req.get(_VERSION_URL, timeout=6)
                if r.status_code != 200:
                    if not silent:
                        self._on_main(lambda: rumps.notification(
                            "Kura", "Update-Prüfung fehlgeschlagen",
                            "Server nicht erreichbar. Bitte später erneut versuchen."
                        ))
                    return
                remote_ver = r.json().get("version", "")
                if self._version_gt(remote_ver, APP_VERSION):
                    self._on_main(lambda v=remote_ver: (
                        setattr(self._item_update, 'title',
                                f"Update verfuegbar: v{v} (jetzt herunterladen)"),
                        self._item_update.set_callback(self._open_update_page),
                        rumps.notification(
                            "Kura Update verfuegbar",
                            f"Version {v} ist bereit",
                            "Klicken Sie auf 'Update verfuegbar' im Tray.",
                        ),
                    ))
                elif not silent:
                    self._on_main(lambda: (
                        setattr(self._item_update, 'title', "Aktualisierungen pruefen"),
                        self._item_update.set_callback(self.check_for_update),
                        rumps.notification("Kura", "Kein Update",
                                           f"Sie verwenden die aktuelle Version ({APP_VERSION})."),
                    ))
            except Exception as e:
                print(f"Update-Prüfung fehlgeschlagen: {e}")
                if not silent:
                    self._on_main(lambda: rumps.notification(
                        "Kura", "Kein Internet",
                        "Update-Prüfung nicht möglich. App funktioniert weiterhin offline."
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
        from shared.version import version_gt
        return version_gt(a, b)

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
            self._set_status("⏳ Modelle prüfen...")

            # ── First-launch model download (one-time, ~5.7 GB) ───────────────
            # Fast path: if models are already on disk this returns instantly.
            # Slow path (first launch): shows a GUI progress dialog, then continues.
            try:
                import sys as _sys, os as _os
                _core = _os.path.join(_os.path.dirname(__file__), '..', 'core')
                if _core not in _sys.path:
                    _sys.path.insert(0, _core)
                from core.model_download_dialog import show_download_dialog_if_needed_macos
                models_ready = show_download_dialog_if_needed_macos()
                if not models_ready:
                    self._set_status("❌ Modelle fehlen")
                    self._on_main(lambda: rumps.alert(
                        "Kura – Setup abgebrochen",
                        "Die KI-Modelle wurden nicht heruntergeladen.\n\n"
                        "Bitte stellen Sie eine Internetverbindung her\n"
                        "und starten Sie Kura neu.",
                        ok="OK"
                    ))
                    return
            except Exception as _dl_err:
                print(f"[download-dialog] {_dl_err} — continuing anyway")

            self._set_status("⏳ Modelle laden... (0%)")

            # Resolve correct models path
            # For bundled apps, use persistent user directory (survives app updates)
            if getattr(sys, 'frozen', False):
                user_app_support = os.path.expanduser("~/Library/Application Support/Kura")
                os.makedirs(user_app_support, exist_ok=True)
                models_path = os.path.join(user_app_support, 'models')
                print(f"[Bundle mode] Models path: {models_path}")
            else:
                # Running from source
                models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
                print(f"[Source mode] Models path: {models_path}")

            # Models will be downloaded on first launch if missing
            # This check is now handled inside KuraEngine.__init__
            if not os.path.exists(models_path):
                print(f"⚠️  Models directory doesn't exist yet: {models_path}")
                print(f"    Will be created during first-launch model download")

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
            # Silent background update check — no notification if offline or already up to date
            self.check_for_update(silent=True)

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
        _app_activate()
        rumps.alert(
            title="Kura Pro erforderlich",
            message="Praxis-Einstellungen sind nur mit einem aktiven Kura Pro Abo verfuegbar.\n\n"
                    "Aktivieren Sie Ihr Abo, um Praxisname, Betriebsstaettennummer\n"
                    "und individuelle Abrechnungsregeln zu konfigurieren.",
            ok="Abo aktivieren",
        )
        _app_deactivate()
        self.activate_license(None)

    def open_practice_config(self, _):
        _app_activate()
        cfg_path = os.path.expanduser("~/.kura_practice.json")

        # Step 1 — Praxis name
        w1 = rumps.Window(
            message="Praxisname  (z. B. Physiotherapie Mustermann):",
            title="Praxis-Einstellungen  1 / 3",
            default_text=self._pc_get("practice", "name", cfg_path) or "",
            ok="Weiter →",
            cancel="Abbrechen",
            dimensions=(400, 24),
        )
        r1 = w1.run()
        if not r1.clicked:
            _app_deactivate()
            return
        name = r1.text.strip() or "Meine Praxis"

        # Step 2 — Betriebsstaettennummer
        _app_activate()
        w2 = rumps.Window(
            message="Betriebsstättennummer  (BSNR, 9-stellig):",
            title="Praxis-Einstellungen  2 / 3",
            default_text=self._pc_get("practice", "license_number", cfg_path) or "",
            ok="Weiter →",
            cancel="Abbrechen",
            dimensions=(400, 24),
        )
        r2 = w2.run()
        if not r2.clicked:
            _app_deactivate()
            return
        bsnr = r2.text.strip()

        # Step 3 — Location
        _app_activate()
        w3 = rumps.Window(
            message="Standort  (Stadt oder Adresse):",
            title="Praxis-Einstellungen  3 / 3",
            default_text=self._pc_get("practice", "location", cfg_path) or "",
            ok="✓  Speichern",
            cancel="Abbrechen",
            dimensions=(400, 24),
        )
        r3 = w3.run()
        if not r3.clicked:
            _app_deactivate()
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
            _app_activate()
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
        finally:
            _app_deactivate()

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

    # --- License block helper ---
    def _boot_license_alert(self):
        """Blocking alert shown on app startup when license is invalid/expired."""
        title, msg = self._license_block_message()
        rumps.alert(title=title, message=msg, ok="Lizenz aktivieren")
        self.activate_license(None)

    def _license_block_message(self) -> tuple[str, str]:
        """Return (title, message) based on block_reason."""
        reason = self.license_mgr.block_reason
        if reason == "trial_expired":
            return (
                "Testphase abgelaufen",
                "Ihre 5 kostenlosen Berichte wurden verwendet.\n\n"
                "Aktivieren Sie Kura Pro, um unbegrenzt weiter zu arbeiten.\n"
                "Eine Internetverbindung ist für die Aktivierung erforderlich."
            )
        if reason == "offline_grace_expired":
            return (
                "Kura Pro — Keine Verbindung",
                "Ihr Abonnement konnte 3 Tage lang nicht geprüft werden.\n\n"
                "Bitte stellen Sie eine Internetverbindung her, damit Kura Ihre "
                "Lizenz mit Digistore24 abgleichen kann.\n\n"
                "Sobald Sie wieder online sind, startet Kura automatisch."
            )
        # subscription_expired (DS24 rejected) or any other reason
        return (
            "Abonnement abgelaufen",
            "Ihr Kura Pro Abonnement ist abgelaufen oder wurde storniert.\n\n"
            "Bitte erneuern Sie Ihr Abonnement auf Digistore24 und aktivieren "
            "Sie Ihren neuen Lizenzschlüssel.\n\n"
            "Eine Internetverbindung ist für die Aktivierung erforderlich."
        )

    # --- Start Session ---
    def start(self, _):
        if self.recording:
            _app_activate()
            rumps.alert("Fehler", "Aufnahme läuft bereits.")
            _app_deactivate()
            return

        # ── License gate — check BEFORE allowing any recording ───────────────
        status = self.license_mgr.verify_locally()
        if status is False:
            _app_activate()
            title, msg = self._license_block_message()
            rumps.alert(title=title, message=msg, ok="Lizenz aktivieren")
            _app_deactivate()
            self.activate_license(None)
            return
        if status is True and self.license_mgr.grace_days_remaining > 0:
            days = self.license_mgr.grace_days_remaining
            rumps.notification(
                "Kura Pro — Offline-Modus",
                f"Lizenzprüfung fehlgeschlagen — noch {days} Tag{'e' if days != 1 else ''} Offline-Gnadenfrist.",
                "Bitte bald Internetverbindung herstellen.",
            )

        if not self.engine:
            _app_activate()
            rumps.alert("Fehler", "KI-Engine nicht bereit.\n\nBitte warten Sie, bis 'KI-Modelle laden...' abgeschlossen ist.\n\nFalls das Problem weiterhin besteht, starten Sie Kura neu.")
            _app_deactivate()
            return

        # Simple patient name input
        _start_recording = False
        _app_activate()
        try:
            window = rumps.Window(
                message="Vorname Nachname (z. B. Müller, Schäfer, Voß):",
                title="Neue Sitzung",
                default_text="",
                ok="Weiter",
                cancel="Abbrechen",
                dimensions=(320, 24),
            )
            response = window.run()

            if not response.clicked:
                return

            if response.clicked:
                import unicodedata
                raw = response.text.strip()
                # NFC-normalise so ä/ö/ü/ß typed via dead keys are one codepoint
                raw = unicodedata.normalize("NFC", raw)
                # Keep all letters (including ä ö ü Ä Ö Ü ß) and digits; replace only
                # filesystem-unsafe characters (/ \ : * ? " < > |) with nothing
                raw = re.sub(r'[/\\:*?"<>|]', '', raw)
                raw_input = raw.strip().replace(" ", "_")
                self.patient_name = raw_input if raw_input else "Patient"

                from shared.billing_engine import InsuranceType
                ins_win = rumps.Window(
                    message="Welche Versicherung hat der Patient?\n\n"
                            "GKV — Gesetzlich (§125 SGB V)   →  OK\n"
                            "PKV — Privat (GebüTh)              →  PKV\n"
                            "BG  — Berufsgenossenschaft         →  BG",
                    title="Versicherungstyp",
                    default_text="",
                    ok="GKV",
                    cancel="PKV",
                    dimensions=(340, 1),
                )
                # Swap cancel label to show BG option via the text field hint
                # Use a second pass: first ask GKV vs other, then distinguish PKV/BG
                ins_resp = ins_win.run()
                from shared.billing_engine import InsuranceType
                if ins_resp.clicked == 1:
                    # "GKV" button
                    self.insurance_type = InsuranceType.GKV
                else:
                    # "PKV" button — ask if actually BG
                    bg_choice = rumps.alert(
                        title="PKV oder BG?",
                        message="Berufsgenossenschaft (BG / DGUV)?",
                        ok="BG",
                        cancel="PKV",
                    )
                    # rumps.alert returns raw NSAlert code: 1000 = ok, 1001 = cancel
                    self.insurance_type = (
                        InsuranceType.BG if bg_choice == 1000 else InsuranceType.PKV
                    )

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

                _start_recording = True
        finally:
            _app_deactivate()   # always restore LSUIElement policy, even on exception

        if _start_recording:
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

    # --- Session Delta Tracking ---
    def _inject_session_delta(self, res: dict) -> dict:
        """
        Load the most recent archived session for this patient and inject a VAS
        delta comparison into S so auditors see measurable progress per session.

        Example injection: "VAS 5/10" → "VAS 5/10 (Vorsitzung: 8/10, Δ: ↓3)"
        """
        import re as _re
        import glob as _glob

        patient_dir = os.path.join(self.report_dir, self.patient_name)
        if not os.path.isdir(patient_dir):
            return res

        archives = sorted(_glob.glob(os.path.join(patient_dir, "*.json")), reverse=True)
        if not archives:
            return res

        prev_soap = None
        for arch in archives[:5]:
            try:
                with open(arch, "r", encoding="utf-8") as _f:
                    prev = json.load(_f)
                candidate = prev.get("soap", {})
                if candidate.get("S"):
                    prev_soap = candidate
                    break
            except Exception:
                continue

        if not prev_soap:
            return res

        prev_s = prev_soap.get("S", "")
        prev_match = _re.search(r"VAS\s*(\d+(?:[.,]\d+)?)/10", prev_s)
        if not prev_match:
            return res
        prev_val = float(prev_match.group(1).replace(",", "."))

        curr_soap = res.get("soap", {})
        curr_s = curr_soap.get("S", "")
        curr_match = _re.search(r"VAS\s*(\d+(?:[.,]\d+)?)/10", curr_s)
        if not curr_match:
            return res
        curr_val = float(curr_match.group(1).replace(",", "."))

        if curr_val == prev_val:
            return res

        delta = curr_val - prev_val
        arrow = "↓" if delta < 0 else "↑"
        abs_d = abs(delta)
        delta_str = f"{arrow}{abs_d:.0f}" if abs_d == int(abs_d) else f"{arrow}{abs_d:.1f}"
        delta_note = f"(Vorsitzung: {prev_val:.0f}/10, Δ: {delta_str})"

        new_s = _re.sub(
            r"(VAS\s*\d+(?:[.,]\d+)?/10)",
            rf"\1 {delta_note}",
            curr_s, count=1
        )
        res["soap"]["S"] = new_s
        return res

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

            # Inject session delta (VAS comparison to most recent archived session)
            try:
                res = self._inject_session_delta(res)
            except Exception:
                pass

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
            br = edited_res.get('billing_result')
            if br:
                billing_line = br.format_billing_line()
            else:
                billing_line = edited_res.get('billing_suggestion', '20501')
            self.last_report = (
                f"--- KURA PHYSIO-PROTOKOLL v{version} ---\n"
                f"ID: {timestamp} | PATIENT: {display_name}\n"
                f"ICD-10: {edited_res.get('icd10')}\n\n"
                f"SOAP-BEFUND:\n{soap_text}\n\n"
                f"ABRECHNUNG: {billing_line}"
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
            webbrowser.open("https://kura-medical.de/#pricing")
            self.activate_license(None)

    # --- Activate License ---
    def activate_license(self, _):
        _app_activate()
        try:
            win = rumps.Window(
                message=(
                    "Lizenzschlüssel eingeben:\n\n"
                    "Format:  XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX\n"
                    "Den Schlüssel direkt aus der Kaufbestätigung kopieren."
                ),
                title="Kura Pro — Aktivierung",
                default_text="",
                ok="✓  Aktivieren",
                cancel="Abbrechen",
                dimensions=(500, 24),
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
                webbrowser.open("https://www.checkout-ds24.com/product/681469")
        finally:
            _app_deactivate()

    def deactivate_license(self, _):
        _app_activate()
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
            _app_activate()
            rumps.alert("Erledigt" if ok else "Fehler", msg)
            self.update_license_display()
        _app_deactivate()

    def _dev_reset_trial(self, _):
        """DEV ONLY — reset trial count and license so the app starts fresh."""
        self.license_mgr.dev_reset_trial()
        self._refresh_menu_state()
        rumps.notification("Dev", "Trial reset", "Trial and license data cleared.")

    def save_pdf(self, _):
        if not self.last_report:
            return

        # Organize by date since patient names can repeat: archive/YYYY-MM-DD/HHMMSS_PatientName.pdf
        import unicodedata as _ud
        safe_name = _ud.normalize("NFC", self.patient_name.replace(' ', '_'))
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")  # ISO format for sorting
        time_str = now.strftime("%H%M%S")  # Include seconds for collision-proof naming

        # Create date-based folder structure
        day_folder = os.path.join(self.report_dir, date_folder)
        os.makedirs(day_folder, exist_ok=True)

        # Single organized location: archive/YYYY-MM-DD/HHMMSS_PatientName.pdf
        pdf_path = os.path.join(day_folder, f"{time_str}_{safe_name}.pdf")

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
                # Matches: ── LABEL ──────\n<content>
                # Stops at: next ── SECTION ── header OR long separator line (footer)
                m = _re.search(
                    rf'[─\-]{{1,4}}\s+{label}[^\n]*\n(.*?)'
                    rf'(?=\n[─\-]{{1,4}}\s+[A-Z]|\n[─\-]{{30,}}|\Z)',
                    text, _re.S
                )
                return m.group(1).strip() if m else ""

            soap_s = _field("SUBJEKTIV")
            soap_o = _field("OBJEKTIV")
            soap_a = _field("ASSESSMENT")
            soap_p = _field("PLAN")

            # Footer: pure line of 30+ dashes (no letters) marks the billing/audit block
            footer_m = _re.search(r'(?m)^[─\-]{30,}$\n(.*)', text, _re.S)
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

            # ── 2b. Practice info (from ~/.kura_practice.json) ────────────
            try:
                _cfg_path = os.path.expanduser("~/.kura_practice.json")
                if os.path.exists(_cfg_path):
                    with open(_cfg_path, "r", encoding="utf-8") as _fp:
                        _pc = json.load(_fp)
                    _pname = _pc.get("practice", {}).get("name", "")
                    _bsnr  = _pc.get("practice", {}).get("license_number", "")
                    _loc   = _pc.get("practice", {}).get("location", "")
                    if _pname or _bsnr:
                        _pline = s(_pname)
                        if _bsnr:
                            _pline += s(f"  |  BSNR: {_bsnr}")
                        if _loc:
                            _pline += s(f"  |  {_loc}")
                        pdf.set_fill_color(*SHADE)
                        pdf.set_text_color(*LIGHT)
                        pdf.set_font("Helvetica", "", 7)
                        pdf.cell(W, 5, f"  {_pline}",
                                 new_x="LMARGIN", new_y="NEXT", fill=True)
            except Exception:
                pass

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

                critical_count = sum(1 for a in br_obj.audit_items if a.status in ("FAIL", "BLOCK"))

                if br_obj.audit_status == "PASS":
                    badge_detail = "Alle Pflichtfelder erfuellt"
                elif br_obj.audit_status == "BLOCK":
                    badge_detail = "Aerztliche Abklaerung vor Therapiefortsetzung erforderlich"
                else:
                    badge_detail = (
                        f"{critical_count} Pflichtfeld{'er' if critical_count != 1 else ''} fehlt"
                        if critical_count else "Hinweise vorhanden — Abrechnung moeglich"
                    )

                pdf.set_fill_color(*badge_col)
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(W, 7, s(f"  {badge_txt}  —  {badge_detail}"),
                         new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.ln(2)

                # ── Einzelne Audit-Punkte ─────────────────────────────────
                # Nur FAIL/BLOCK anzeigen — WARN-Hinweise sind informativ, kein Abrechnungsblocker
                show_items = [a for a in br_obj.audit_items
                              if a.status in ("FAIL", "BLOCK")]

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
                     s(f"Kura v2026.4.1 | Lokale KI-Verarbeitung | DSGVO-konform | {date_str}"),
                     align="C")

            pdf.output(pdf_path)

            patient_display = self.patient_name.replace("_", " ")
            date_str = now.strftime("%d.%m.%Y")
            rumps.notification("Kura", "PDF gespeichert",
                               s(f"{patient_display} | {date_str} | {day_folder}"))

        except Exception as e:
            print(f"PDF Error: {e}")
            rumps.notification("Kura Fehler", "PDF konnte nicht erstellt werden", str(e))


if __name__ == "__main__":
    # Kill any existing Kura instance before starting
    current_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "main.py"],
            capture_output=True, text=True
        )
        for pid_str in result.stdout.strip().splitlines():
            pid = int(pid_str)
            if pid != current_pid:
                os.kill(pid, 15)  # SIGTERM
    except Exception:
        pass

    KuraApp().run()
