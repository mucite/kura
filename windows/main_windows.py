"""
Kura v2026 — Windows
CustomTkinter GUI (FREE, MIT License) - Modern, professional medical interface
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import wave
import webbrowser
from datetime import datetime
from tkinter import messagebox
import tkinter as tk

# ── Fix Windows encoding issues ────────────────────────────────────────────────
# Windows console uses cp1252 by default which can't handle Unicode/emojis
# Force UTF-8 encoding for all I/O operations
if sys.platform == 'win32':
    # Set UTF-8 for stdout/stderr
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    elif hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')
    elif hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Set console code page to UTF-8 if running in console
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page
    except Exception:
        pass

import customtkinter as ctk
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from fpdf import FPDF


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.license_manager import LicenseManager

# ── Version — single source of truth: version.json at project root ────────────
from shared.version import APP_VERSION, VERSION_URL as _VERSION_URL
_DOWNLOAD_URL = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/Kura_Windows_v2026.exe"

# ── CustomTkinter theme ────────────────────────────────────────────────────────

ctk.set_appearance_mode("dark")  # "dark" or "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# ── Crash logging ──────────────────────────────────────────────────────────────

def setup_crash_logging():
    log_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Kura", "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def log_crash(exc_type, exc_value, exc_traceback):
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                if f is not None:
                    f.write(f"Kura Crash Report — {datetime.now()}\n{'='*60}\n\n")
                    if exc_type is not None:
                        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
                    f.write(f"\nPython: {sys.version}\nBase path: {os.path.dirname(os.path.realpath(__file__))}\n")
        except Exception as log_err:
            app_logger.error(f"Error writing crash log: {log_err}")
        messagebox.showerror("Kura Fehler", f"Kura ist abgestürzt.\nLog-Datei: {log_file}")
        sys.exit(1)

    sys.excepthook = log_crash
    return log_file

CRASH_LOG = setup_crash_logging()

# ── User data directory & .env ─────────────────────────────────────────────────

USER_DATA_DIR = os.path.expanduser("~/Documents/Kura")
os.makedirs(USER_DATA_DIR, exist_ok=True)

user_env_file = os.path.join(USER_DATA_DIR, ".env")
if not os.path.exists(user_env_file):
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.join(os.path.dirname(__file__), "..")
    example = os.path.join(base, ".env.example")
    if os.path.exists(example):
        try:
            shutil.copy(example, user_env_file)
        except Exception as copy_err:
            print(f"Error copying .env.example: {copy_err}")
    else:
        try:
            with open(user_env_file, "w", encoding="utf-8") as f:
                if f is not None:
                    f.write("# Kura Configuration\n")
                    f.write("HF_TOKEN=your_token_here\n")
                    f.write("DS24_API_KEY=\n")
                    f.write("DS24_PRODUCT_ID=\n")
        except Exception as env_err:
            print(f"Error creating .env file: {env_err}")

load_dotenv(user_env_file)

# ── Audio recording ────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000  # Whisper optimal

# ── Main Application ───────────────────────────────────────────────────────────

class KuraApp:
    def __init__(self):
        self.report_dir = os.path.join(USER_DATA_DIR, "reports")
        os.makedirs(self.report_dir, exist_ok=True)

        self.license_mgr = LicenseManager()
        self.engine = None
        self.recording = False
        self.patient_name = "Unbekannt"
        self.insurance_type = None
        self.temp_audio = os.path.join(USER_DATA_DIR, "session.wav")
        self.last_report = None
        self.last_billing_result = None
        self.seconds_elapsed = 0
        self._record_thread = None
        self._audio_chunks = []

        # Boot engine in background
        threading.Thread(target=self._boot, daemon=True).start()

    # ── Boot ──────────────────────────────────────────────────────────────────

    def _boot(self):
        try:
            self._post_event("status", "⏳ Modelle laden... bitte warten")
            from physio_scribe_crossplatform import KuraEngine
            self.engine = KuraEngine()
            self._post_event("status", f"✅ Kura Bereit (Lokal & DSGVO) — v{APP_VERSION}")
            self._post_event("boot_done", None)
            threading.Thread(target=self._check_update_background, daemon=True).start()
        except MemoryError:
            self._post_event("status", "❌ Zu wenig RAM")
            self._post_event("error", "Nicht genug Arbeitsspeicher. Bitte schließen Sie andere Programme.")
        except RuntimeError as e:
            self._post_event("status", "❌ Modellfehler")
            self._post_event("error", str(e))
        except Exception as e:
            self._post_event("status", f"❌ Startfehler: {e}")
            self._post_event("error", str(e))

    def _post_event(self, event_type, value):
        """Thread-safe event posting to the main window."""
        if hasattr(self, "root") and self.root:
            self.root.after(0, lambda: self._handle_event(event_type, value))

    def _handle_event(self, event_type, value):
        """Handle events in the main thread."""
        if event_type == "status":
            self.status_label.configure(text=value)
        elif event_type == "progress":
            current = self.output_text.get("1.0", "end-1c")
            lines = current.split('\n')
            if len(lines) > 10:
                lines = lines[-10:]
            new_text = '\n'.join(lines) + '\n' + value
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", new_text)
        elif event_type == "boot_done":
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", f"✅ KI-Modelle geladen. Kura v{APP_VERSION} ist einsatzbereit.\n")
            # Boot-time notification — low trial count warning only (non-blocking)
            # Hard block happens at recording start, not at launch, to avoid false positives
            _boot_status = self.license_mgr.verify_locally()
            if _boot_status == "TRIAL":
                _rem = self.license_mgr.max_trials - self.license_mgr.get_trial_count()

                # Show trial comparison on first launch or when running low
                if _rem == 5 or _rem <= 2:
                    self._show_trial_comparison_dialog(_rem)
                elif _rem <= 2:
                    messagebox.showinfo(
                        "Kura Testphase",
                        f"Noch {_rem} Testbericht{'e' if _rem != 1 else ''} verbleibend.\n\n"
                        "Aktivieren Sie Kura Pro für unbegrenzte Nutzung."
                    )
        elif event_type == "ai_done":
            self._on_ai_done(value)
        elif event_type == "ai_error":
            self.status_label.configure(text="❌ KI-Fehler")
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", f"❌ Fehler: {value}")
        elif event_type == "error":
            messagebox.showerror("Kura Fehler", value)
        elif event_type == "timer":
            self.status_label.configure(text=value)
        elif event_type == "update_available":
            self.status_label.configure(text=f"🆕 Update v{value} verfügbar")
        elif event_type == "update_license":
            self.license_label.configure(text=self._license_text())

    # ── License display ────────────────────────────────────────────────────────

    def _license_text(self):
        status = self.license_mgr.verify_locally()
        if status is True:
            return "✅ Kura Pro: Aktiv"
        elif status == "TRIAL":
            count = self.license_mgr.get_trial_count()
            remaining = self.license_mgr.max_trials - count
            icon = "🟢" if remaining > 2 else ("🟡" if remaining > 0 else "🔴")
            return f"{icon} Testphase: {remaining}/{self.license_mgr.max_trials} verbleibend"
        return "❌ Testphase beendet — Upgrade erforderlich"

    def _is_pro(self):
        return self.license_mgr.verify_locally() is True

    def _create_feature_card(self, parent, feature: dict):
        """Create a clean feature card with icon, title, and description"""
        card_frame = ctk.CTkFrame(parent, fg_color="#2b2b2b", corner_radius=8)
        card_frame.pack(fill="x", pady=5)
        
        content_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        content_frame.pack(fill="x", padx=15, pady=12)
        
        # Icon and title in same row
        header_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        header_frame.pack(fill="x", anchor="w")
        
        icon_label = ctk.CTkLabel(
            header_frame,
            text=feature["icon"],
            font=ctk.CTkFont(size=20),
            width=30
        )
        icon_label.pack(side="left", padx=(0, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text=feature["title"],
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        title_label.pack(side="left", fill="x", expand=True)
        
        # Description
        desc_label = ctk.CTkLabel(
            content_frame,
            text=feature["description"],
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w"
        )
        desc_label.pack(anchor="w", padx=(40, 0), pady=(5, 0))

    def _show_trial_comparison_dialog(self, remaining_trials):
        """Show clean, modern trial upgrade dialog"""
        try:
            from shared.trial_status import TrialStatus

            # Create dialog window
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Kura Testphase")
            dialog.geometry("700x650")
            dialog.resizable(False, False)
            
            # Make it modal
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Main container with padding
            main_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            main_frame.pack(fill="both", expand=True, padx=30, pady=30)

            # Header with icon and trial status
            header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
            header_frame.pack(fill="x", pady=(0, 20))

            status_icon = "🟢" if remaining_trials > 2 else ("🟡" if remaining_trials > 0 else "🔴")
            header_label = ctk.CTkLabel(
                header_frame,
                text=f"{status_icon} Testphase",
                font=ctk.CTkFont(size=24, weight="bold")
            )
            header_label.pack()

            remaining_label = ctk.CTkLabel(
                header_frame,
                text=f"Noch {remaining_trials} von {self.license_mgr.max_trials} Testberichten verfügbar",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            remaining_label.pack(pady=(5, 0))

            # Scrollable content area
            scroll_frame = ctk.CTkScrollableFrame(main_frame, height=300)
            scroll_frame.pack(fill="both", pady=(0, 15))

            # Info message
            info_label = ctk.CTkLabel(
                scroll_frame,
                text="Mit Kura Pro erhalten Sie:",
                font=ctk.CTkFont(size=16, weight="bold"),
                anchor="w"
            )
            info_label.pack(anchor="w", pady=(0, 15))

            # Feature cards
            features = TrialStatus.get_pro_features()
            for feature in features:
                self._create_feature_card(scroll_frame, feature)

            # Upgrade CTA section
            cta_frame = ctk.CTkFrame(main_frame, fg_color="#1a472a", corner_radius=10)
            cta_frame.pack(fill="x", pady=(0, 15))

            cta_content = ctk.CTkFrame(cta_frame, fg_color="transparent")
            cta_content.pack(padx=20, pady=15)

            price_label = ctk.CTkLabel(
                cta_content,
                text="€49/Monat · Jederzeit kündbar",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="white"
            )
            price_label.pack()

            benefit_label = ctk.CTkLabel(
                cta_content,
                text="Unbegrenzte Berichte · Alle Features · Premium Support",
                font=ctk.CTkFont(size=12),
                text_color="#b8e6cc"
            )
            benefit_label.pack(pady=(5, 0))

            # Buttons
            button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
            button_frame.pack(fill="x", pady=(0, 0))
            
            # Configure grid columns to be equal width
            button_frame.grid_columnconfigure(0, weight=1)
            button_frame.grid_columnconfigure(1, weight=1)

            def open_upgrade():
                webbrowser.open("https://kura-medical.de/#preis")
                dialog.destroy()
            
            upgrade_btn = ctk.CTkButton(
                button_frame,
                text="🚀 Jetzt upgraden",
                command=open_upgrade,
                fg_color="#28a745",
                hover_color="#218838",
                font=ctk.CTkFont(size=15, weight="bold"),
                height=45
            )
            upgrade_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
            
            continue_btn = ctk.CTkButton(
                button_frame,
                text="Testphase fortsetzen",
                command=dialog.destroy,
                fg_color="#3b3b3b",
                hover_color="#4a4a4a",
                border_width=2,
                border_color="#606060",
                text_color="#ffffff",
                font=ctk.CTkFont(size=15),
                height=45
            )
            continue_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
        except Exception as e:
            # Fallback to simple messagebox if dialog fails
            messagebox.showinfo(
                "Kura Testphase",
                f"Noch {remaining_trials} Testberichte verbleibend.\n\n"
                "Upgrade zu Kura Pro:\n"
                "• 79 Abrechnungsziffern\n"
                "• ICD-spezifische Regeln\n"
                "• Automatische Updates\n\n"
                "https://kura-medical.de/#preis"
            )

    # ── Recording ─────────────────────────────────────────────────────────────

    def _start_recording(self):
        # ── License gate — block before any recording starts ─────────────────
        status = self.license_mgr.verify_locally()
        if status is False:
            self._show_upgrade_dialog()
            return
        if status is True and self.license_mgr.grace_days_remaining > 0:
            days = self.license_mgr.grace_days_remaining
            messagebox.showwarning(
                "Kura Pro — Offline-Modus",
                f"Lizenzprüfung fehlgeschlagen.\n"
                f"Noch {days} Tag{'e' if days != 1 else ''} Offline-Gnadenfrist verbleibend.\n\n"
                "Bitte bald Internetverbindung herstellen."
            )

        if not self.engine:
            messagebox.showerror("Kura", "KI-Modelle werden noch geladen. Bitte warten.")
            return

        patient = self.patient_entry.get().strip()
        if not patient:
            messagebox.showerror("Kura", "Bitte geben Sie einen Patientennamen ein.")
            return

        import unicodedata
        # NFC-normalise so ä/ö/ü/ß are single codepoints (important for file paths)
        patient = unicodedata.normalize("NFC", patient)
        # Remove only filesystem-unsafe characters; German letters (äöüÄÖÜß) are kept
        patient = re.sub(r'[/\\:*?"<>|]', '', patient)
        self.patient_name = patient.replace(" ", "_") or "Patient"
        self.recording = True
        self.seconds_elapsed = 0
        self._audio_chunks = []

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self._record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._record_thread.start()
        threading.Thread(target=self._timer_thread, daemon=True).start()

    def _record_audio(self):
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=1024) as stream:
                while self.recording:
                    data, _ = stream.read(1024)
                    self._audio_chunks.append(data.copy())
        except Exception as e:
            print(f"Recording error: {e}")

        if self._audio_chunks:
            audio = np.concatenate(self._audio_chunks, axis=0)
            with wave.open(self.temp_audio, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

    def _timer_thread(self):
        while self.recording:
            time.sleep(1)
            self.seconds_elapsed += 1
            mins, secs = divmod(self.seconds_elapsed, 60)
            self._post_event("timer", f"🔴 {mins:02d}:{secs:02d}")

    def _stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        if self._record_thread:
            self._record_thread.join(timeout=3)

        self.stop_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")

        # Minimum recording guard
        if self.seconds_elapsed < 10:
            mins, secs = divmod(self.seconds_elapsed, 60)
            duration = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}s"
            self.status_label.configure(text=f"✅ Aufnahme entfernt ({duration})")
            messagebox.showerror(
                "Kura",
                f"Aufnahme zu kurz ({self.seconds_elapsed}s).\n\n"
                "Bitte mindestens 10 Sekunden sprechen."
            )
            self.seconds_elapsed = 0
            self.status_label.configure(text="🩺 Bereit")
            return

        # Speech detection
        if not self._audio_has_speech():
            mins, secs = divmod(self.seconds_elapsed, 60)
            duration = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}s"
            self.status_label.configure(text=f"✅ Stille Aufnahme entfernt ({duration})")
            messagebox.showerror(
                "Kura",
                "Kein Ton erkannt.\n\n"
                "Die Aufnahme enthält kein hörbares Sprachsignal.\n"
                "Prüfen Sie das Mikrofon und versuchen Sie es erneut."
            )
            self.seconds_elapsed = 0
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception:
                pass
            self.status_label.configure(text="🩺 Bereit")
            return

        # Show duration before processing
        mins, secs = divmod(self.seconds_elapsed, 60)
        duration = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}s"
        self.status_label.configure(text=f"✅ Aufnahme: {duration} — KI-Analyse läuft...")

        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", f"🎙️ Aufnahme beendet: {duration}\n⏳ KI-Verarbeitung startet...\n")

        status = self.license_mgr.verify_locally()
        if status is True or status == "TRIAL":
            threading.Thread(target=self._run_ai, daemon=True).start()
        else:
            self.seconds_elapsed = 0
            self.status_label.configure(text="🩺 Bereit")
            self._show_upgrade_dialog()

    def _audio_has_speech(self, silence_threshold: int = 100) -> bool:
        """Return True if recorded audio chunks contain signal above silence_threshold (RMS)."""
        if not self._audio_chunks:
            return False
        try:
            audio = np.concatenate(self._audio_chunks, axis=0).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))

            if os.path.exists(self.temp_audio):
                file_size = os.path.getsize(self.temp_audio)
                if file_size < 1000:
                    return False

            if peak < 500:
                return False

            return rms > silence_threshold
        except Exception:
            return True

    # ── Session delta tracking ────────────────────────────────────────────────

    def _inject_session_delta(self, res: dict) -> dict:
        """
        Load the most recent archived session for this patient and inject a VAS
        delta comparison into S so auditors see measurable progress per session.

        Example: "VAS 5/10" → "VAS 5/10 (Vorsitzung: 8/10, Δ: ↓3)"
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

    # ── AI pipeline ───────────────────────────────────────────────────────────

    def _run_ai(self):
        try:
            def update_status(msg):
                self._post_event("status", msg)
                self._post_event("progress", msg)

            # Get insurance type from radio buttons
            from shared.billing_engine import InsuranceType
            if self.insurance_pkv_var.get():
                self.insurance_type = InsuranceType.PKV
            elif self.insurance_bg_var.get():
                self.insurance_type = InsuranceType.BG
            else:
                self.insurance_type = InsuranceType.GKV

            res = self.engine.run_full_flow(
                self.temp_audio,
                status_callback=update_status,
                insurance_type=self.insurance_type,
            )

            # DSGVO: delete audio immediately
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception as e:
                print(f"⚠️ Could not delete audio: {e}")

            # Inject session delta (VAS comparison to most recent archived session)
            try:
                res = self._inject_session_delta(res)
            except Exception:
                pass

            self.seconds_elapsed = 0
            self._post_event("ai_done", res)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ AI Error: {error_msg}")
            self.seconds_elapsed = 0
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception:
                pass
            self._post_event("ai_error", str(e))

    def _on_ai_done(self, res):
        """Handle AI completion in main thread."""
        self.status_label.configure(text="✅ Analyse abgeschlossen — Bericht prüfen")
        soap = res.get("soap", {})
        br = res.get("billing_result")
        billing_line = br.format_billing_line() if br else res.get("billing_suggestion", "")
        audit_status = f"AUDIT: {br.audit_status}" if br else ""
        profile_label = res.get("profile_label", "")
        summary = (
            f"ICD-10: {res.get('icd10')}  |  {billing_line}\n"
            f"{audit_status}\n"
            f"PROFIL: {profile_label}\n\n"
            f"S: {soap.get('S')}\n\nO: {soap.get('O')}\n\n"
            f"A: {soap.get('A')}\n\nP: {soap.get('P')}"
        )
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", summary)

        self._show_review_window(res)

    # ── Result review window ───────────────────────────────────────────────────

    def _show_review_window(self, res):
        """Editable review window."""
        soap = res.get("soap", {})
        br = res.get("billing_result")

        if br:
            billing_line = br.format_billing_line()
            critical_items = [a for a in br.audit_items if a.status in ("FAIL", "BLOCK")]
            pass_items = len([a for a in br.audit_items if a.status == "PASS"])
            if br.audit_status == "PASS":
                audit_summary = f"✅ AUDIT BESTANDEN — alle {pass_items} Prüfpunkte erfüllt"
            elif br.audit_status == "BLOCK":
                audit_summary = "🔴 ABRECHNUNG GESPERRT — ärztliche Abklärung erforderlich"
            else:
                n = len(critical_items)
                audit_summary = (
                    f"⚠️ {n} PFLICHTFELD{'ER' if n != 1 else ''} FEHLT — vor Abrechnung ergänzen"
                    if n else "⚠️ PRÜFUNG EMPFOHLEN"
                )
            # Show only FAIL/BLOCK items — WARN items are informational hints, not blockers
            flagged = [str(i) for i in critical_items]
            audit_block = "\n".join(flagged) if flagged else ""
            if br.optimization_hints:
                audit_block += ("\n\nHINWEISE:\n" + "\n".join(br.optimization_hints)) if audit_block \
                    else ("HINWEISE:\n" + "\n".join(br.optimization_hints))
        else:
            billing_line = f"POSITION: {res.get('billing_suggestion', '?')}"
            ins_label = "GKV"
            warnings = res.get("compliance_check", [])
            audit_summary = f"✅ Dokumentation {ins_label}-konform." if not warnings else "⚠️ Prüfung erforderlich"
            audit_block = "\n".join(f"-> {w}" for w in warnings)

        icd = res.get("icd10", "–")
        profile_label = res.get("profile_label", "")
        date_str = datetime.now().strftime("%d.%m.%Y")
        patient_display = self.patient_name.replace("_", " ")
        profile_line = f"Profil: {profile_label}\n" if profile_label else ""

        footer = f"{billing_line}  |  ICD-10: {icd}\n{audit_summary}"
        if audit_block:
            footer += f"\n{audit_block}"

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

        # Create review window
        review_win = ctk.CTkToplevel(self.root)
        review_win.title(f"KURA v{APP_VERSION} — Befund-Revision")
        review_win.geometry("800x700")
        review_win.transient(self.root)
        review_win.grab_set()

        # Label
        ctk.CTkLabel(review_win, text="Prüfen und in Abrechnung übernehmen:",
                    font=("Arial", 14, "bold")).pack(pady=10)

        # Text area
        text_widget = ctk.CTkTextbox(review_win, width=760, height=500, font=("Courier New", 10))
        text_widget.pack(padx=20, pady=10)
        text_widget.insert("1.0", initial_text)

        # Buttons
        button_frame = ctk.CTkFrame(review_win)
        button_frame.pack(pady=10)

        result = {"edited": None}

        def on_save():
            result["edited"] = text_widget.get("1.0", "end-1c")
            review_win.destroy()

        def on_cancel():
            review_win.destroy()

        ctk.CTkButton(button_frame, text="✅ KOPIEREN & PDF", command=on_save,
                     fg_color="#1976d2", width=200).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Abbrechen", command=on_cancel,
                     fg_color="#757575", width=120).pack(side="left", padx=5)

        review_win.wait_window()

        if result["edited"]:
            # Update the main output text with the edited report
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", result["edited"])
            self._finalize(result["edited"], res)
        else:
            self.status_label.configure(text="🩺 Bereit")

    # ── Finalize ───────────────────────────────────────────────────────────────

    def _finalize(self, edited_text: str, res: dict):
        status = self.license_mgr.verify_locally()

        if status is False:
            title, msg = self._license_block_message()
            messagebox.showerror(title, msg)
            self._show_upgrade_dialog()
            return

        # Learning engine
        icd_match = re.search(r"ICD-10:\s*([A-Z][0-9][0-9]\.[0-9])", edited_text)
        user_icd = icd_match.group(1) if icd_match else None
        if self.engine:
            ai_icd = res.get("icd10")
            transcript = res.get("transcript", "")
            final_icd = user_icd or ai_icd or "M99.9"
            was_corrected = bool(user_icd and user_icd != ai_icd)
            if was_corrected:
                try:
                    self.engine.learning_mgr.log_correction(transcript, ai_icd, user_icd)
                    print(f"🧠 Learning: recorded correction {ai_icd} → {user_icd}")
                except Exception:
                    pass
            # Always log the accepted session for few-shot learning
            try:
                soap = res.get("soap", {})
                profile_id = res.get("profile_id", "KG")
                self.engine.learning_mgr.log_session(
                    transcript, soap, final_icd, profile_id, was_corrected
                )
                stats = self.engine.learning_mgr.stats()
                print(f"🧠 Learning: {stats['total_sessions']} sessions stored "
                      f"({stats['corrected_sessions']} corrected)")
            except Exception as e:
                print(f"Learning log error: {e}")

        # Clipboard
        soap_only = re.sub(r'^KURA[^\n]*\n[-─]+\n\n?', '', edited_text)
        soap_only = re.sub(r'\n[-─]{3,}.*', '', soap_only, flags=re.DOTALL).strip()
        try:
            process = subprocess.Popen("clip", stdin=subprocess.PIPE, shell=True)
            process.communicate(soap_only.encode("utf-8"))
        except Exception as e:
            print(f"Clipboard error: {e}")

        # Archive
        self.last_report = edited_text
        self.last_billing_result = res.get("billing_result")

        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        day_folder = os.path.join(self.report_dir, date_folder)
        os.makedirs(day_folder, exist_ok=True)

        json_path = os.path.join(day_folder, f"{time_str}_{self.patient_name}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                if f is not None:
                    json.dump({"text": edited_text, "patient": self.patient_name,
                               "icd10": user_icd or res.get("icd10"),
                               "timestamp": now.strftime("%Y%m%d-%H%M"),
                               "date": date_folder},
                              f, ensure_ascii=False, indent=4)
        except Exception as json_err:
            print(f"JSON archive save error: {json_err}")

        self._save_pdf_to_disk()

        # Trial increment
        if status == "TRIAL":
            count = self.license_mgr.get_trial_count()
            remaining = self.license_mgr.max_trials - (count + 1)
            self.license_mgr.increment_trial()
            self._post_event("update_license", None)
            if remaining <= 0:
                messagebox.showinfo("Kura Testphase",
                    "Bericht gespeichert.\nTestphase beendet — bitte aktivieren Sie Kura Pro.")
            else:
                self.status_label.configure(text=f"✅ Gespeichert — noch {remaining} kostenlose Berichte")
        else:
            self.status_label.configure(text="✅ Bericht gespeichert")

    # ── PDF export ─────────────────────────────────────────────────────────────
    def _save_pdf_to_disk(self):
        """
        Generate PDF from the current report.

        This is a READ-ONLY operation that generates PDF from:
        1. Current content in output_text (if it contains a valid report), OR
        2. The last saved report (self.last_report) as fallback

        This design ensures:
        - PDF export doesn't modify application state (no side effects)
        - Therapists can preview edits in PDF before saving
        - Multiple PDF exports of same content are safe and idempotent
        """
        # Check if output_text has current editable content
        output_content = self.output_text.get("1.0", "end-1c").strip()

        # Use output_text if it contains a valid report, otherwise fall back to last_report
        if output_content and ("SUBJEKTIV" in output_content or "KURA" in output_content):
            report_text = output_content
        elif self.last_report:
            report_text = self.last_report
        else:
            messagebox.showerror("Kura", "Kein Bericht zum Speichern vorhanden.")
            return

        import unicodedata as _ud
        safe_name = _ud.normalize("NFC", self.patient_name.replace(" ", "_"))
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")

        day_folder = os.path.join(self.report_dir, date_folder)
        os.makedirs(day_folder, exist_ok=True)

        pdf_path = os.path.join(day_folder, f"{time_str}_{safe_name}.pdf")

        try:
            # Character safety maps (same as original)
            _EMOJI_MAP = [
                ('\u2705', '[PASS]'), ('\u26a0\ufe0f', '[WARN]'), ('\u26a0', '[WARN]'),
                ('\U0001f534', '[STOP]'), ('\U0001f4cb', '[FEHLT]'), ('\u2753', '[?]'),
                ('\u274c', '[STOP]'), ('\u2714', '[PASS]'), ('\u2713', '[PASS]'),
            ]
            _CHAR_MAP = str.maketrans({
                '\u20ac': 'EUR', '\u2013': '-', '\u2014': '-',
                '\u2192': '->', '\u2190': '<-', '\u2194': '<->',
                '\u2265': '>=', '\u2264': '<=', '\u2260': '!=',
                '\u00b1': '+/-',
                '\u201e': '"', '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
                '\u2502': '|', '\u2500': '-', '\u2501': '-', '\u2550': '=',
                '\u2588': '#', '\u25a0': '#', '\xb0': 'Grad',
            })

            def s(text: str) -> str:
                if not isinstance(text, str):
                    text = str(text)
                for em, rep in _EMOJI_MAP:
                    text = text.replace(em, rep)
                text = text.translate(_CHAR_MAP)
                return text.encode('latin-1', 'replace').decode('latin-1')

            # Parse SOAP sections
            text = str(report_text)

            def _field(label):
                m = re.search(rf'{label}\n(.*?)(?=\n[A-Z]{{2,}}|\n-{{3,}}|\n─{{3,}}|\Z)', text, re.S)
                return m.group(1).strip() if m else ""

            soap_s = _field("SUBJEKTIV")
            soap_o = _field("OBJEKTIV")
            soap_a = _field("ASSESSMENT")
            soap_p = _field("PLAN")

            footer_m = re.search(r'[─-]{3,}\n(.*)', text, re.S)
            footer_raw = footer_m.group(1).strip() if footer_m else ""
            footer_lines = footer_raw.splitlines()
            billing_line = s(footer_lines[0]) if footer_lines else ""
            audit_lines = [s(l) for l in footer_lines[1:] if l.strip()]

            # Layout constants
            patient_display = self.patient_name.replace("_", " ")
            date_str = now.strftime("%d.%m.%Y")
            time_str_display = now.strftime("%H:%M")
            W = 180
            COL = 90

            DARK = (25, 25, 25)
            MID = (90, 90, 90)
            LIGHT = (140, 140, 140)
            WHITE = (255, 255, 255)
            BODY = (20, 20, 20)
            RULE = (210, 210, 210)
            SHADE = (248, 248, 248)

            # Build PDF
            pdf = FPDF()
            pdf.set_margins(15, 15, 15)
            pdf.add_page()

            # Header
            pdf.set_fill_color(*DARK)
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(W, 12, s("  KURA  Physiotherapeutischer Befund"),
                     new_x="LMARGIN", new_y="NEXT", align="L", fill=True)

            # Patient info bar
            pdf.set_fill_color(*SHADE)
            pdf.set_text_color(*MID)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(COL, 7, s(f"  Patient: {patient_display}"),
                     new_x="RIGHT", new_y="TOP", fill=True)
            pdf.cell(COL, 7, s(f"Datum: {date_str}   {time_str_display} Uhr  "),
                     new_x="LMARGIN", new_y="NEXT", align="R", fill=True)

            # Practice info (from ~/.kura_practice.json)
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

            # SOAP sections
            for section_title, body in [
                ("SUBJEKTIV", soap_s),
                ("OBJEKTIV", soap_o),
                ("ASSESSMENT", soap_a),
                ("PLAN", soap_p),
            ]:
                if not body:
                    continue
                pdf.set_fill_color(*SHADE)
                pdf.set_draw_color(*RULE)
                pdf.set_text_color(*MID)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(W, 6, s(f"  {section_title}"),
                         new_x="LMARGIN", new_y="NEXT", fill=True, border="B")
                pdf.set_text_color(*BODY)
                pdf.set_font("Helvetica", "", 10)
                pdf.ln(2)
                pdf.multi_cell(W, 5.5, s(body), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

            # Abrechnung & Audit section
            pdf.set_fill_color(*SHADE)
            pdf.set_draw_color(*RULE)
            pdf.set_text_color(*MID)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(W, 6, "  ABRECHNUNG & AUDIT",
                     new_x="LMARGIN", new_y="NEXT", fill=True, border=1)
            pdf.ln(2)

            br_obj = self.last_billing_result
            if br_obj:
                pdf.set_text_color(*BODY)
                pdf.set_font("Helvetica", "B", 9)
                pdf.multi_cell(W, 5.5, s(br_obj.format_billing_line()),
                               new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*MID)
                dg_line = (f"Diagnosegruppe: {br_obj.diagnosegruppe} — "
                           f"{br_obj.diagnosegruppe_desc} | {br_obj.legal_basis}")
                pdf.multi_cell(W, 4.5, s(dg_line), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)

                STATUS_COLORS = {
                    "PASS": (34, 139, 34),
                    "REVIEW": (200, 100, 0),
                    "BLOCK": (180, 20, 20),
                }
                STATUS_LABELS = {
                    "PASS": "AUDIT BESTANDEN",
                    "REVIEW": "PRUEFUNG ERFORDERLICH",
                    "BLOCK": "ABRECHNUNG GESPERRT",
                }
                badge_col = STATUS_COLORS.get(br_obj.audit_status, (90, 90, 90))
                badge_txt = STATUS_LABELS.get(br_obj.audit_status, br_obj.audit_status)

                pass_count = sum(1 for a in br_obj.audit_items if a.status == "PASS")
                total_count = len(br_obj.audit_items)
                critical_count = sum(1 for a in br_obj.audit_items if a.status in ("FAIL", "BLOCK"))

                if br_obj.audit_status == "PASS":
                    badge_detail = f"Alle {total_count} Pruefpunkte erfuellt"
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

                # Nur FAIL/BLOCK anzeigen — WARN-Hinweise sind informativ, kein Abrechnungsblocker
                show_items = [a for a in br_obj.audit_items if a.status in ("FAIL", "BLOCK")]

                if show_items:
                    pdf.set_font("Helvetica", "B", 7.5)
                    pdf.set_text_color(*MID)
                    pdf.cell(W, 5, "  Offene Pruefpunkte:", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "", 8)
                    for item in show_items:
                        item_col = {
                            "FAIL": (180, 20, 20),
                            "BLOCK": (180, 20, 20),
                            "WARN": (180, 100, 0),
                        }.get(item.status, BODY)
                        pdf.set_text_color(*item_col)
                        pdf.multi_cell(W, 4.5,
                                       s(f"  {item.icon} {item.label}"
                                         + (f": {item.detail}" if item.detail else "")),
                                       new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*BODY)
                if billing_line:
                    pdf.multi_cell(W, 5, billing_line, new_x="LMARGIN", new_y="NEXT")
                for al in audit_lines:
                    pdf.multi_cell(W, 5, al, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(1)
            pdf.set_draw_color(*RULE)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())

            # Footer watermark
            pdf.set_y(-12)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*LIGHT)
            pdf.cell(W, 5,
                     s(f"Kura v{APP_VERSION} | Lokale KI-Verarbeitung | DSGVO-konform | {date_str}"),
                     align="C")

            pdf.output(pdf_path)

            # Notify user
            if messagebox.askyesno(
                "PDF Gespeichert",
                f"PDF erfolgreich gespeichert!\n\n"
                f"Datei: {os.path.basename(pdf_path)}\n"
                f"Pfad: {day_folder}\n\n"
                f"Ordner jetzt öffnen?"
            ):
                os.startfile(day_folder)

        except Exception as e:
            messagebox.showerror("Kura", f"PDF-Fehler: {e}")
            print(f"PDF Error Details: {e}")
            traceback.print_exc()

    # ── Update check ──────────────────────────────────────────────────────────

    @staticmethod
    def _version_gt(a: str, b: str) -> bool:
        from shared.version import version_gt
        return version_gt(a, b)

    def _check_update_background(self):
        """Silent update check on boot."""
        try:
            import requests
            r = requests.get(_VERSION_URL, timeout=6)
            if r.status_code != 200:
                return
            remote_ver = r.json().get("version", "")
            if self._version_gt(remote_ver, APP_VERSION):
                self._post_event("update_available", remote_ver)
        except Exception:
            pass

    def _check_update_manual(self):
        """Manual update check triggered by button."""
        def _run():
            try:
                import requests
                self.status_label.configure(text="🔍 Prüfe auf Updates...")
                r = requests.get(_VERSION_URL, timeout=6)
                if r.status_code != 200:
                    messagebox.showinfo("Kura Update",
                        "Server nicht erreichbar.\nBitte später erneut versuchen.")
                    self.status_label.configure(text="🩺 Bereit")
                    return
                remote_ver = r.json().get("version", "")
                if self._version_gt(remote_ver, APP_VERSION):
                    if messagebox.askyesno(
                        "Kura Update",
                        f"Neue Version verfügbar: v{remote_ver}\n\n"
                        "Wichtig: Beenden Sie Kura zuerst, bevor Sie die neue Version installieren.\n\n"
                        "Jetzt herunterladen?"
                    ):
                        webbrowser.open(_DOWNLOAD_URL)
                else:
                    messagebox.showinfo("Kura Update", f"Kura ist aktuell (v{APP_VERSION}).")
            except Exception:
                messagebox.showinfo("Kura Update",
                    "Kein Internet — Update-Prüfung nicht möglich.\n"
                    "Die App funktioniert weiterhin vollständig offline.")
            finally:
                self.status_label.configure(text="🩺 Bereit")
        threading.Thread(target=_run, daemon=True).start()

    # ── License dialogs ───────────────────────────────────────────────────────

    def _license_block_message(self) -> tuple[str, str]:
        """Return (title, message) based on block_reason."""
        reason = self.license_mgr.block_reason
        if reason == "trial_expired":
            return (
                "Testphase abgelaufen",
                "Ihre 5 kostenlosen Berichte wurden verwendet.\n\n"
                "Aktivieren Sie Kura Pro, um unbegrenzt weiter zu arbeiten.\n\n"
                "Eine Internetverbindung ist für die Aktivierung erforderlich."
            )
        if reason == "offline_grace_expired":
            return (
                "Kura Pro — Keine Verbindung",
                "Ihr Abonnement konnte 3 Tage lang nicht geprüft werden.\n\n"
                "Bitte stellen Sie eine Internetverbindung her, damit Kura "
                "Ihre Lizenz mit Digistore24 abgleichen kann.\n\n"
                "Sobald Sie wieder online sind, startet Kura automatisch."
            )
        return (
            "Abonnement abgelaufen",
            "Ihr Kura Pro Abonnement ist abgelaufen oder wurde storniert.\n\n"
            "Bitte erneuern Sie Ihr Abonnement auf Digistore24 und geben "
            "Sie Ihren neuen Lizenzschlüssel ein.\n\n"
            "Eine Internetverbindung ist für die Aktivierung erforderlich."
        )

    def _show_upgrade_dialog(self):
        title, msg = self._license_block_message()
        if messagebox.askyesno(title, msg + "\n\nJetzt aktivieren?"):
            self._activate_license()

    def _activate_license(self):
        """License activation dialog."""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Kura Pro aktivieren")
        dialog.geometry("560x260")
        dialog.transient(self.root)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Lizenzschlüssel aus der Kaufbestätigung eingeben:",
                    font=("Arial", 12)).pack(pady=(20, 4))
        ctk.CTkLabel(dialog, text="Format: XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX",
                    font=("Arial", 10), text_color="#888888").pack(pady=(0, 6))

        key_var = ctk.StringVar()
        key_entry = ctk.CTkEntry(dialog, width=500, font=("Courier New", 12),
                                 textvariable=key_var, placeholder_text="z.B. RMEL3-3UDDC-YHJHF-C7TH9-QRYJK-FHZSV-KU26F-NS3CC")
        key_entry.pack(pady=6)

        def _on_key_change(*_):
            raw = key_var.get().upper().replace(" ", "")
            # Strip all hyphens, keep only alphanumeric
            chars = [c for c in raw if c.isalnum()][:40]
            # Re-insert hyphens every 5 chars
            groups = [(''.join(chars[i:i+5])) for i in range(0, len(chars), 5)]
            formatted = '-'.join(groups)
            # Only update if changed to avoid recursive loop
            if formatted != key_var.get():
                key_var.set(formatted)
                key_entry.icursor(len(formatted))

        key_var.trace_add("write", _on_key_change)

        result = {"success": False}

        def on_activate():
            key = key_var.get().strip()
            if not key:
                messagebox.showerror("Fehler", "Bitte geben Sie einen Lizenzschlüssel ein.")
                return
            ok, msg = self.license_mgr.activate(key)
            if ok:
                messagebox.showinfo("Aktivierung erfolgreich", f"✅ {msg}")
                result["success"] = True
                dialog.destroy()
            else:
                messagebox.showerror("Aktivierung fehlgeschlagen", msg)

        def on_buy():
            webbrowser.open("https://www.checkout-ds24.com/product/681469")

        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(pady=10)

        ctk.CTkButton(button_frame, text="Aktivieren", command=on_activate,
                     fg_color="#1976d2", width=140).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Jetzt kaufen", command=on_buy,
                     fg_color="#388e3c", width=140).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Abbrechen", command=dialog.destroy,
                     fg_color="#757575", width=120).pack(side="left", padx=5)

        dialog.wait_window()

        if result["success"]:
            self.license_label.configure(text=self._license_text())

    def _deactivate_license(self):
        if messagebox.askyesno(
            "Lizenz deaktivieren?",
            "Dies entfernt Kura Pro von diesem Gerät.\n\n"
            "Der Schlüssel kann danach auf einem anderen Gerät aktiviert werden.\n\n"
            "Internetverbindung erforderlich.\n\nLizenz deaktivieren?"
        ):
            ok, msg = self.license_mgr.deactivate()
            messagebox.showinfo("Erledigt" if ok else "Fehler", msg)
            if ok:
                self.license_label.configure(text=self._license_text())

    def _open_practice_config(self):
        """Practice configuration (Pro only)."""
        if not self._is_pro():
            if messagebox.askyesno(
                "Kura Pro erforderlich",
                "Praxis-Einstellungen sind nur mit einem aktiven Kura Pro Abo verfügbar.\n\n"
                "Aktivieren Sie Ihr Abo, um Praxisname, Betriebsstättennummer\n"
                "und individuelle Abrechnungsregeln zu konfigurieren.\n\n"
                "Jetzt aktivieren?"
            ):
                self._activate_license()
            return

        cfg_path = os.path.expanduser("~/.kura_practice.json")

        # Simple dialogs for practice config
        from tkinter import simpledialog

        def get_value(section, key):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return json.load(f).get(section, {}).get(key, "")
            except Exception:
                return ""

        name = simpledialog.askstring("Praxis-Einstellungen 1/3",
                                     "Praxisname (z.B. Physiotherapie Mustermann):",
                                     initialvalue=get_value("practice", "name") or "")
        if name is None:
            return
        name = name.strip() or "Meine Praxis"

        bsnr = simpledialog.askstring("Praxis-Einstellungen 2/3",
                                     "Betriebsstättennummer (BSNR, 9-stellig):",
                                     initialvalue=get_value("practice", "license_number") or "")
        if bsnr is None:
            return
        bsnr = bsnr.strip()

        location = simpledialog.askstring("Praxis-Einstellungen 3/3",
                                         "Standort (Stadt / Adresse):",
                                         initialvalue=get_value("practice", "location") or "")
        if location is None:
            return

        try:
            from shared.practice_config import PracticeConfig
            pc = PracticeConfig(practice_file=cfg_path)
            pc.config["practice"]["name"] = name
            pc.config["practice"]["license_number"] = bsnr
            pc.config["practice"]["location"] = location.strip()
            pc.save()
            if messagebox.askyesno(
                "Kura Praxis-Einstellungen",
                f"Einstellungen gespeichert: {name} — BSNR {bsnr}\n\n"
                "Erweiterte Konfiguration (ICD-10-Regeln, Abrechnungscodes) jetzt öffnen?"
            ):
                os.startfile(cfg_path)
        except Exception as e:
            messagebox.showerror("Kura Fehler", f"Konnte Einstellungen nicht speichern:\n{e}")

    # ── Main window ───────────────────────────────────────────────────────────

    def run(self):
        self.root = ctk.CTk()
        self.root.title(f"Kura v{APP_VERSION} — Medizinische KI-Dokumentation")
        self.root.geometry("900x750")

        # Header frame
        header_frame = ctk.CTkFrame(self.root)
        header_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(header_frame, text=f"Kura v{APP_VERSION}",
                    font=("Arial", 16, "bold")).pack(side="left", padx=10)

        self.license_label = ctk.CTkLabel(header_frame, text=self._license_text(),
                                         font=("Arial", 10))
        self.license_label.pack(side="right", padx=10)

        # Patient info frame
        patient_frame = ctk.CTkFrame(self.root)
        patient_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(patient_frame, text="Patient:", font=("Arial", 12)).pack(side="left", padx=5)
        self.patient_entry = ctk.CTkEntry(patient_frame, width=250,
                                          font=("Arial", 12),
                                          placeholder_text="Müller, Schäfer, Voß …")
        self.patient_entry.pack(side="left", padx=5)

        # Insurance type frame
        insurance_frame = ctk.CTkFrame(patient_frame)
        insurance_frame.pack(side="left", padx=20)

        ctk.CTkLabel(insurance_frame, text="Versicherung:", font=("Arial", 10)).pack(side="left", padx=5)

        self.insurance_gkv_var = tk.BooleanVar(value=True)
        self.insurance_pkv_var = tk.BooleanVar(value=False)
        self.insurance_bg_var = tk.BooleanVar(value=False)

        def set_gkv():
            self.insurance_gkv_var.set(True)
            self.insurance_pkv_var.set(False)
            self.insurance_bg_var.set(False)

        def set_pkv():
            self.insurance_gkv_var.set(False)
            self.insurance_pkv_var.set(True)
            self.insurance_bg_var.set(False)

        def set_bg():
            self.insurance_gkv_var.set(False)
            self.insurance_pkv_var.set(False)
            self.insurance_bg_var.set(True)

        ctk.CTkRadioButton(insurance_frame, text="GKV", variable=self.insurance_gkv_var,
                          value=True, command=set_gkv).pack(side="left", padx=5)
        ctk.CTkRadioButton(insurance_frame, text="PKV", variable=self.insurance_pkv_var,
                          value=True, command=set_pkv).pack(side="left", padx=5)
        ctk.CTkRadioButton(insurance_frame, text="BG", variable=self.insurance_bg_var,
                          value=True, command=set_bg).pack(side="left", padx=5)

        # Recording buttons
        record_frame = ctk.CTkFrame(self.root)
        record_frame.pack(fill="x", padx=10, pady=10)

        self.start_btn = ctk.CTkButton(record_frame, text="🔴 Sitzung starten",
                                       command=self._start_recording,
                                       width=200, height=40, font=("Arial", 12))
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ctk.CTkButton(record_frame, text="⏹ Stoppen & Verarbeiten",
                                      command=self._stop_recording, state="disabled",
                                      width=220, height=40, font=("Arial", 12))
        self.stop_btn.pack(side="left", padx=5)

        # Action buttons row 1
        action_frame1 = ctk.CTkFrame(self.root)
        action_frame1.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(action_frame1, text="📄 PDF exportieren",
                     command=self._save_pdf_to_disk, width=170).pack(side="left", padx=5)
        ctk.CTkButton(action_frame1, text="📂 Archiv öffnen",
                     command=lambda: os.startfile(self.report_dir), width=160).pack(side="left", padx=5)
        ctk.CTkButton(action_frame1, text="🔑 Lizenz aktivieren",
                     command=self._activate_license, width=170).pack(side="left", padx=5)
        ctk.CTkButton(action_frame1, text="🔓 Deaktivieren",
                     command=self._deactivate_license, width=140).pack(side="left", padx=5)

        # Action buttons row 2
        action_frame2 = ctk.CTkFrame(self.root)
        action_frame2.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(action_frame2, text="🏥 Praxis-Einstellungen",
                     command=self._open_practice_config, width=200).pack(side="left", padx=5)
        ctk.CTkButton(action_frame2, text="🔄 Update prüfen",
                     command=self._check_update_manual, width=150).pack(side="left", padx=5)

        def show_about():
            messagebox.showinfo(
                "Über Kura",
                f"Kura v{APP_VERSION}\n\n"
                "Medizinische KI-Dokumentation\n\n"
                "• 100% Lokale Verarbeitung (DSGVO-sicher)\n"
                "• Professionelle medizinische Dokumentation\n"
                "• § 125 Abs. 1 SGB V konform\n\n"
                "© 2026 Kura Medical"
            )

        ctk.CTkButton(action_frame2, text="ℹ Über Kura",
                     command=show_about, width=130).pack(side="left", padx=5)

        # Output text area
        output_frame = ctk.CTkFrame(self.root)
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.output_text = ctk.CTkTextbox(output_frame, width=860, height=250,
                                         font=("Courier New", 13))
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Status bar
        status_frame = ctk.CTkFrame(self.root)
        status_frame.pack(fill="x", padx=10, pady=10)

        self.status_label = ctk.CTkLabel(status_frame, text="🩺 Bereit",
                                         font=("Arial", 12))
        self.status_label.pack(side="left", padx=10)

        ctk.CTkButton(status_frame, text="Beenden", command=self.root.quit,
                     fg_color="#757575", width=100).pack(side="right", padx=10)

        self.root.mainloop()


if __name__ == "__main__":
    app = KuraApp()
    app.run()

