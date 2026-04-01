"""
Kura v2026 — Windows
PySimpleGUI window application, feature-parity with macOS version.
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

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from fpdf import FPDF
import PySimpleGUI as sg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.license_manager import LicenseManager
# KuraEngine imported lazily inside _boot() so scipy/llama don't block window startup


# ── Version & update URLs ──────────────────────────────────────────────────────

APP_VERSION   = "2026.3.2"
_VERSION_URL  = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/version.json"
_DOWNLOAD_URL = "https://pub-f83ad51a8a6d46859a3b16a78c2b95b3.r2.dev/Kura_Windows_v2026.exe"


# ── Crash logging ──────────────────────────────────────────────────────────────

def setup_crash_logging():
    log_dir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Kura", "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def log_crash(exc_type, exc_value, exc_traceback):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Kura Crash Report — {datetime.now()}\n{'='*60}\n\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write(f"\nPython: {sys.version}\nBase path: {os.path.dirname(os.path.realpath(__file__))}\n")
        sg.popup_error(
            f"Kura ist abgestürzt.\nLog-Datei: {log_file}",
            title="Kura Fehler"
        )
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
        shutil.copy(example, user_env_file)
    else:
        with open(user_env_file, "w") as f:
            f.write("# Kura Configuration\n")
            f.write("HF_TOKEN=your_token_here\n")
            f.write("LEMON_SQUEEZY_API_URL=https://api.lemonsqueezy.com/v1/licenses/activate\n")
            f.write("LEMON_SQUEEZY_API_KEY=\n")

load_dotenv(user_env_file)


# ── GUI theme ──────────────────────────────────────────────────────────────────

sg.theme("DarkBlue3")
sg.set_options(font=("Arial", 10))


# ── Audio recording (sounddevice — no ffmpeg required on Windows) ─────────────

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
        self.insurance_type = None  # InsuranceType.GKV/PKV/BG — set from UI
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
            self._post_event("-STATUS-UPDATE-", "⏳ Modelle laden... bitte warten")
            from physio_scribe_crossplatform import KuraEngine
            self.engine = KuraEngine()
            self._post_event("-STATUS-UPDATE-", f"✅ Kura Bereit (Lokal & DSGVO) — v{APP_VERSION}")
            self._post_event("-BOOT-DONE-", None)
            # Silent background update check — same as macOS
            threading.Thread(target=self._check_update_background, daemon=True).start()
        except MemoryError:
            self._post_event("-STATUS-UPDATE-", "❌ Zu wenig RAM")
            self._post_event("-ERROR-", "Nicht genug Arbeitsspeicher. Bitte schließen Sie andere Programme.")
        except RuntimeError as e:
            self._post_event("-STATUS-UPDATE-", "❌ Modellfehler")
            self._post_event("-ERROR-", str(e))
        except Exception as e:
            self._post_event("-STATUS-UPDATE-", f"❌ Startfehler: {e}")
            self._post_event("-ERROR-", str(e))

    def _post_event(self, key, value):
        """Thread-safe event posting to the main PySimpleGUI window."""
        if hasattr(self, "_window") and self._window:
            self._window.write_event_value(key, value)

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

    # ── Recording ─────────────────────────────────────────────────────────────

    def _start_recording(self, patient_name: str, window):
        if not self.engine:
            sg.popup_error("KI-Modelle werden noch geladen. Bitte warten.")
            return

        self.patient_name = patient_name.strip().replace(" ", "_") or "Patient"
        self.recording = True
        self.seconds_elapsed = 0
        self._audio_chunks = []

        window["-START-"].update(disabled=True)
        window["-STOP-"].update(disabled=False)

        self._record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._record_thread.start()
        threading.Thread(target=self._timer_thread, args=(window,), daemon=True).start()

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

    def _timer_thread(self, window):
        while self.recording:
            time.sleep(1)
            self.seconds_elapsed += 1
            mins, secs = divmod(self.seconds_elapsed, 60)
            window.write_event_value("-TIMER-", f"🔴 {mins:02d}:{secs:02d}")

    def _stop_recording(self, window):
        if not self.recording:
            return

        self.recording = False
        if self._record_thread:
            self._record_thread.join(timeout=3)

        window["-STOP-"].update(disabled=True)
        window["-START-"].update(disabled=False)

        # Minimum recording guard (same as macOS: 10 seconds)
        if self.seconds_elapsed < 10:
            mins, secs = divmod(self.seconds_elapsed, 60)
            duration = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}s"
            window["-STATUS-"].update(f"✅ Aufnahme entfernt ({duration})")
            sg.popup_error(
                f"Aufnahme zu kurz ({self.seconds_elapsed}s).\n\n"
                "Bitte mindestens 10 Sekunden sprechen.",
                title="Kura"
            )
            self.seconds_elapsed = 0
            window["-STATUS-"].update("🩺 Bereit")
            return

        # Speech detection: check audio has actual signal above silence floor
        if not self._audio_has_speech():
            mins, secs = divmod(self.seconds_elapsed, 60)
            duration = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}s"
            window["-STATUS-"].update(f"✅ Stille Aufnahme entfernt ({duration})")
            sg.popup_error(
                "Kein Ton erkannt.\n\n"
                "Die Aufnahme enthält kein hörbares Sprachsignal.\n"
                "Prüfen Sie das Mikrofon und versuchen Sie es erneut.",
                title="Kura"
            )
            self.seconds_elapsed = 0
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception:
                pass
            window["-STATUS-"].update("🩺 Bereit")
            return

        # Show duration with green tick before processing
        mins, secs = divmod(self.seconds_elapsed, 60)
        duration = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}s"
        window["-STATUS-"].update(f"✅ Aufnahme: {duration} — KI-Analyse läuft...")

        # Clear output and show processing start
        window["-OUTPUT-"].update(f"🎙️ Aufnahme beendet: {duration}\n⏳ KI-Verarbeitung startet...\n")

        # Don't reset seconds_elapsed here - reset it after AI completes
        status = self.license_mgr.verify_locally()
        if status is True or status == "TRIAL":
            threading.Thread(target=self._run_ai, args=(window,), daemon=True).start()
        else:
            self.seconds_elapsed = 0
            window["-STATUS-"].update("🩺 Bereit")
            self._show_upgrade_dialog()

    def _audio_has_speech(self, silence_threshold: int = 100) -> bool:
        """Return True if recorded audio chunks contain signal above silence_threshold (RMS)."""
        if not self._audio_chunks:
            return False
        try:
            audio = np.concatenate(self._audio_chunks, axis=0).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            
            # Also calculate peak amplitude to detect any signal
            peak = float(np.max(np.abs(audio)))
            
            # Check if the audio file was actually written and has content
            if os.path.exists(self.temp_audio):
                file_size = os.path.getsize(self.temp_audio)
                if file_size < 1000:  # Less than 1KB is suspicious
                    return False
            
            # Use both RMS and peak to determine if there's actual audio
            # If peak is very low (< 500), it's definitely silence
            if peak < 500:
                return False
            
            return rms > silence_threshold
        except Exception:
            return True  # can't determine — let Whisper try

    # ── AI pipeline ───────────────────────────────────────────────────────────

    def _run_ai(self, window):
        try:
            def update_status(msg):
                # Update status bar
                window.write_event_value("-STATUS-UPDATE-", msg)
                # Also update output box
                window.write_event_value("-PROGRESS-", msg)

            res = self.engine.run_full_flow(
                self.temp_audio,
                status_callback=update_status,
                insurance_type=self.insurance_type,
            )

            # DSGVO: delete audio immediately after processing
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception as e:
                print(f"⚠️ Could not delete audio: {e}")

            # Reset timer after successful processing
            self.seconds_elapsed = 0
            window.write_event_value("-AI-DONE-", res)


        except Exception as e:
            error_msg = str(e)
            print(f"❌ AI Error: {error_msg}")

            # Reset timer on error
            self.seconds_elapsed = 0

            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception:
                pass

            window.write_event_value("-AI-ERROR-", str(e))

    # ── Result review window ───────────────────────────────────────────────────

    def _show_review_window(self, res):
        """
        Editable review window — matches macOS rumps.Window review flow.
        Shows patient, date, audit summary counts, and full SOAP before billing block.
        """
        soap = res.get("soap", {})
        br = res.get("billing_result")

        if br:
            billing_line = br.format_billing_line()
            total_items = len([a for a in br.audit_items if a.status != "PASS"])
            pass_items  = len([a for a in br.audit_items if a.status == "PASS"])
            if br.audit_status == "PASS":
                audit_summary = f"✅ AUDIT BESTANDEN — alle {pass_items} Prüfpunkte erfüllt"
            elif br.audit_status == "BLOCK":
                audit_summary = "🔴 ABRECHNUNG GESPERRT — ärztliche Abklärung erforderlich"
            else:
                audit_summary = f"⚠️ PRÜFUNG ERFORDERLICH — {total_items} Hinweise (vor Abrechnung prüfen)"
            flagged = [str(i) for i in br.audit_items if i.status in ("WARN", "FAIL", "BLOCK")]
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

        icd          = res.get("icd10", "–")
        profile_label = res.get("profile_label", "")
        date_str     = datetime.now().strftime("%d.%m.%Y")
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

        layout = [
            [sg.Text("Prüfen und in Abrechnung übernehmen:", font=("Arial", 10, "bold"))],
            [sg.Multiline(initial_text, size=(76, 26), key="-RESULT-", font=("Courier New", 9))],
            [
                sg.Button("✅ KOPIEREN & PDF", key="-SAVE-", button_color=("white", "#1976d2"), size=(20, 1)),
                sg.Button("Abbrechen", key="-CANCEL-", button_color=("white", "#757575"), size=(12, 1)),
            ],
        ]

        window = sg.Window(
            f"KURA v{APP_VERSION} — Befund-Revision",
            layout,
            finalize=True,
            modal=True,
            resizable=True,
        )

        edited_text = None
        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED, "-CANCEL-"):
                break
            if event == "-SAVE-":
                edited_text = values["-RESULT-"]
                break

        window.close()
        return edited_text

    # ── Finalize (learning engine + clipboard + PDF) ──────────────────────────

    def _finalize(self, edited_text: str, res: dict, window):
        status = self.license_mgr.verify_locally()

        if status is False:
            sg.popup_error(
                "Sie haben das Limit von 5 kostenlosen Berichten erreicht.\n\n"
                "Bitte aktivieren Sie Kura Pro, um weiter zu arbeiten.",
                title="Kura Testphase beendet"
            )
            self._show_upgrade_dialog()
            return

        # Learning engine: detect if therapist corrected ICD code
        icd_match = re.search(r"ICD-10:\s*([A-Z][0-9][0-9]\.[0-9])", edited_text)
        user_icd = icd_match.group(1) if icd_match else None
        if user_icd and self.engine:
            ai_icd = res.get("icd10")
            transcript = res.get("transcript", "")
            if user_icd != ai_icd:
                try:
                    self.engine.learning_mgr.log_correction(transcript, ai_icd, user_icd)
                    print(f"🧠 Learning: recorded correction {ai_icd} → {user_icd}")
                except Exception:
                    pass

        # Clipboard: strip KURA header and billing/audit footer — paste clean SOAP
        soap_only = re.sub(r'^KURA[^\n]*\n[-─]+\n\n?', '', edited_text)
        soap_only = re.sub(r'\n[-─]{3,}.*', '', soap_only, flags=re.DOTALL).strip()
        try:
            process = subprocess.Popen("clip", stdin=subprocess.PIPE, shell=True)
            process.communicate(soap_only.encode("utf-8"))
        except Exception as e:
            print(f"Clipboard error: {e}")

        # Archive JSON + PDF
        self.last_report = edited_text
        self.last_billing_result = res.get("billing_result")
        
        # Save JSON to date-based folder: archive/YYYY-MM-DD/HHMMSS_PatientName.json
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")  # Include seconds for collision-proof naming
        day_folder = os.path.join(self.report_dir, date_folder)
        os.makedirs(day_folder, exist_ok=True)
        
        json_path = os.path.join(day_folder, f"{time_str}_{self.patient_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"text": edited_text, "patient": self.patient_name,
                       "icd10": user_icd or res.get("icd10"), 
                       "timestamp": now.strftime("%Y%m%d-%H%M"),
                       "date": date_folder},
                      f, ensure_ascii=False, indent=4)

        self._save_pdf_to_disk()

        # Trial increment + notification
        if status == "TRIAL":
            count = self.license_mgr.get_trial_count()
            remaining = self.license_mgr.max_trials - (count + 1)
            self.license_mgr.increment_trial()
            window["-LICENSE-"].update(self._license_text())
            if remaining <= 0:
                sg.popup_ok(
                    f"Bericht gespeichert.\nTestphase beendet — bitte aktivieren Sie Kura Pro.",
                    title="Kura Testphase"
                )
            else:
                window["-STATUS-"].update(
                    f"✅ Gespeichert — noch {remaining} kostenlose Berichte"
                )
        else:
            window["-STATUS-"].update("✅ Bericht gespeichert")

        window["-LICENSE-"].update(self._license_text())

    # ── PDF export (matches macOS rich PDF) ──────────────────────────────────

    def _save_pdf_to_disk(self):
        if not self.last_report:
            sg.popup_error("Kein Bericht zum Speichern vorhanden.")
            return

        # Organize by date since patient names can repeat: archive/YYYY-MM-DD/HHMMSS_PatientName.pdf
        safe_name   = self.patient_name.replace(" ", "_")
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")  # ISO format for sorting
        time_str    = now.strftime("%H%M%S")  # Include seconds for collision-proof naming

        # Create date-based folder structure
        day_folder = os.path.join(self.report_dir, date_folder)
        os.makedirs(day_folder, exist_ok=True)
        
        # Single organized location: archive/YYYY-MM-DD/HHMMSS_PatientName.pdf
        pdf_path = os.path.join(day_folder, f"{time_str}_{safe_name}.pdf")

        try:
            # ── Character safety (same table as macOS) ────────────────────────
            _EMOJI_MAP = [
                ('\u2705', '[PASS]'), ('\u26a0\ufe0f', '[WARN]'), ('\u26a0', '[WARN]'),
                ('\U0001f534', '[STOP]'), ('\U0001f4cb', '[FEHLT]'), ('\u2753', '[?]'),
                ('\u274c', '[STOP]'), ('\u2714', '[PASS]'), ('\u2713', '[PASS]'),
            ]
            _CHAR_MAP = str.maketrans({
                '\u20ac': 'EUR',
                '\u2013': '-', '\u2014': '-',
                '\u2192': '->', '\u2190': '<-', '\u2194': '<->',
                '\u2265': '>=', '\u2264': '<=', '\u2260': '!=',
                '\u00b1': '+/-',
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

            # ── Parse SOAP sections from review text ──────────────────────────
            text = str(self.last_report)

            def _field(label):
                m = re.search(rf'{label}\n(.*?)(?=\n[A-Z]{{2,}}|\n-{{3,}}|\n─{{3,}}|\Z)', text, re.S)
                return m.group(1).strip() if m else ""

            soap_s = _field("SUBJEKTIV")
            soap_o = _field("OBJEKTIV")
            soap_a = _field("ASSESSMENT")
            soap_p = _field("PLAN")

            footer_m   = re.search(r'[─-]{3,}\n(.*)', text, re.S)
            footer_raw = footer_m.group(1).strip() if footer_m else ""
            footer_lines  = footer_raw.splitlines()
            billing_line  = s(footer_lines[0]) if footer_lines else ""
            audit_lines   = [s(l) for l in footer_lines[1:] if l.strip()]

            # ── Layout constants ───────────────────────────────────────────────
            now             = datetime.now()
            patient_display = self.patient_name.replace("_", " ")
            date_str        = now.strftime("%d.%m.%Y")
            time_str        = now.strftime("%H:%M")
            W    = 180
            COL  = 90

            DARK  = (25,  25,  25)
            MID   = (90,  90,  90)
            LIGHT = (140, 140, 140)
            WHITE = (255, 255, 255)
            BODY  = (20,  20,  20)
            RULE  = (210, 210, 210)
            SHADE = (248, 248, 248)

            # ── Build PDF ──────────────────────────────────────────────────────
            pdf = FPDF()
            pdf.set_margins(15, 15, 15)
            pdf.add_page()

            # 1. Header
            pdf.set_fill_color(*DARK)
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(W, 12, s("  KURA  Physiotherapeutischer Befund"),
                     new_x="LMARGIN", new_y="NEXT", align="L", fill=True)

            # 2. Patient info bar
            pdf.set_fill_color(*SHADE)
            pdf.set_text_color(*MID)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(COL, 7, s(f"  Patient: {patient_display}"),
                     new_x="RIGHT", new_y="TOP", fill=True)
            pdf.cell(COL, 7, s(f"Datum: {date_str}   {time_str} Uhr  "),
                     new_x="LMARGIN", new_y="NEXT", align="R", fill=True)
            pdf.ln(6)

            # 3. SOAP sections
            for section_title, body in [
                ("SUBJEKTIV",  soap_s),
                ("OBJEKTIV",   soap_o),
                ("ASSESSMENT", soap_a),
                ("PLAN",       soap_p),
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

            # 4. Abrechnung & Audit section
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
                    "PASS":   (34,  139, 34),
                    "REVIEW": (200, 100,  0),
                    "BLOCK":  (180,  20, 20),
                }
                STATUS_LABELS = {
                    "PASS":   "AUDIT BESTANDEN",
                    "REVIEW": "PRUEFUNG ERFORDERLICH",
                    "BLOCK":  "ABRECHNUNG GESPERRT",
                }
                badge_col = STATUS_COLORS.get(br_obj.audit_status, (90, 90, 90))
                badge_txt = STATUS_LABELS.get(br_obj.audit_status, br_obj.audit_status)

                pass_count  = sum(1 for a in br_obj.audit_items if a.status == "PASS")
                total_count = len(br_obj.audit_items)
                open_count  = total_count - pass_count

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

                if br_obj.audit_status == "PASS":
                    show_items = [a for a in br_obj.audit_items if a.status in ("FAIL", "BLOCK")]
                else:
                    show_items = [a for a in br_obj.audit_items if a.status in ("WARN", "FAIL", "BLOCK")]

                if show_items:
                    pdf.set_font("Helvetica", "B", 7.5)
                    pdf.set_text_color(*MID)
                    pdf.cell(W, 5, "  Offene Pruefpunkte:", new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Helvetica", "", 8)
                    for item in show_items:
                        item_col = {
                            "FAIL":  (180, 20, 20),
                            "BLOCK": (180, 20, 20),
                            "WARN":  (180, 100, 0),
                        }.get(item.status, BODY)
                        pdf.set_text_color(*item_col)
                        pdf.multi_cell(W, 4.5,
                                       s(f"  {item.icon} {item.label}"
                                         + (f": {item.detail}" if item.detail else "")),
                                       new_x="LMARGIN", new_y="NEXT")
            else:
                # Fallback: plain text (no billing_result object)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*BODY)
                if billing_line:
                    pdf.multi_cell(W, 5, billing_line, new_x="LMARGIN", new_y="NEXT")
                for al in audit_lines:
                    pdf.multi_cell(W, 5, al, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(1)
            pdf.set_draw_color(*RULE)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())

            # 5. Footer watermark
            pdf.set_y(-12)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*LIGHT)
            pdf.cell(W, 5,
                     s(f"Kura v{APP_VERSION} | Lokale KI-Verarbeitung | DSGVO-konform | {date_str}"),
                     align="C")

            pdf.output(pdf_path)

            # Notify user and offer to open the folder
            result = sg.popup_yes_no(
                f"PDF erfolgreich gespeichert!\n\n"
                f"Datei: {os.path.basename(pdf_path)}\n"
                f"Pfad: {day_folder}\n\n"
                f"Ordner jetzt öffnen?",
                title="PDF Gespeichert"
            )
            if result == "Yes":
                # Open the day's folder
                os.startfile(day_folder)

        except Exception as e:
            sg.popup_error(f"PDF-Fehler: {e}", title="Kura")
            print(f"PDF Error Details: {e}")
            import traceback
            traceback.print_exc()

    # ── Update check ──────────────────────────────────────────────────────────

    @staticmethod
    def _version_gt(a: str, b: str) -> bool:
        try:
            return tuple(int(x) for x in a.split(".")) > tuple(int(x) for x in b.split("."))
        except Exception:
            return False

    def _check_update_background(self):
        """Silent update check on boot — same as macOS."""
        try:
            import requests
            r = requests.get(_VERSION_URL, timeout=6)
            if r.status_code != 200:
                return
            remote_ver = r.json().get("version", "")
            if self._version_gt(remote_ver, APP_VERSION):
                self._post_event("-UPDATE-AVAILABLE-", remote_ver)
        except Exception:
            pass

    def _check_update_manual(self, window):
        """Manual update check triggered by button."""
        def _run():
            try:
                import requests
                r = requests.get(_VERSION_URL, timeout=6)
                if r.status_code != 200:
                    window.write_event_value("-UPDATE-RESULT-", None)
                    return
                remote_ver = r.json().get("version", "")
                window.write_event_value("-UPDATE-RESULT-", remote_ver)
            except Exception:
                window.write_event_value("-UPDATE-RESULT-", "ERROR")
        threading.Thread(target=_run, daemon=True).start()

    def _open_update_page(self):
        result = sg.popup_yes_no(
            f"Neue Version verfügbar!\n\n"
            "Wichtig: Beenden Sie Kura zuerst, bevor Sie die neue Version installieren.\n\n"
            "Jetzt herunterladen?",
            title="Kura Update"
        )
        if result == "Yes":
            webbrowser.open(_DOWNLOAD_URL)

    # ── Practice config (Pro only — matches macOS 3-step dialog) ─────────────

    def _config_locked(self):
        if sg.popup_yes_no(
            "Praxis-Einstellungen sind nur mit einem aktiven Kura Pro Abo verfügbar.\n\n"
            "Aktivieren Sie Ihr Abo, um Praxisname, Betriebsstättennummer\n"
            "und individuelle Abrechnungsregeln zu konfigurieren.\n\n"
            "Jetzt aktivieren?",
            title="Kura Pro erforderlich"
        ) == "Yes":
            self._activate_license()

    def _pc_get(self, section: str, key: str, cfg_path: str) -> str:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f).get(section, {}).get(key, "")
        except Exception:
            return ""

    def _open_practice_config(self):
        if not self._is_pro():
            self._config_locked()
            return

        cfg_path = os.path.expanduser("~/.kura_practice.json")

        # Step 1 — Praxisname
        name = sg.popup_get_text(
            "Praxisname (z.B. Physiotherapie Mustermann):",
            title="Praxis-Einstellungen 1/3",
            default_text=self._pc_get("practice", "name", cfg_path) or "",
        )
        if name is None:
            return
        name = name.strip() or "Meine Praxis"

        # Step 2 — BSNR
        bsnr = sg.popup_get_text(
            "Betriebsstättennummer (BSNR, 9-stellig):",
            title="Praxis-Einstellungen 2/3",
            default_text=self._pc_get("practice", "license_number", cfg_path) or "",
        )
        if bsnr is None:
            return
        bsnr = bsnr.strip()

        # Step 3 — Location
        location = sg.popup_get_text(
            "Standort (Stadt / Adresse):",
            title="Praxis-Einstellungen 3/3",
            default_text=self._pc_get("practice", "location", cfg_path) or "",
        )
        if location is None:
            return

        try:
            from shared.practice_config import PracticeConfig
            pc = PracticeConfig(practice_file=cfg_path)
            pc.config["practice"]["name"]           = name
            pc.config["practice"]["license_number"] = bsnr
            pc.config["practice"]["location"]       = location.strip()
            pc.save()
            if sg.popup_yes_no(
                f"Einstellungen gespeichert: {name} — BSNR {bsnr}\n\n"
                "Erweiterte Konfiguration (ICD-10-Regeln, Abrechnungscodes) jetzt öffnen?",
                title="Kura Praxis-Einstellungen"
            ) == "Yes":
                os.startfile(cfg_path)
        except Exception as e:
            sg.popup_error(f"Konnte Einstellungen nicht speichern:\n{e}", title="Kura Fehler")

    def _open_gist_override(self):
        """Create / open local config override — Pro only."""
        if not self._is_pro():
            self._config_locked()
            return
        if not self.engine:
            sg.popup_error("KI-Engine nicht geladen.")
            return
        try:
            cfg = self.engine.config
            override_path = cfg.local_override_path
            first_time = not os.path.exists(override_path)
            cfg.create_override_template()
            if first_time:
                sg.popup_ok(
                    f"Konfigurationsdatei erstellt:\n{override_path}\n\n"
                    "Bearbeiten Sie nur die Werte, die Sie anpassen möchten.\n"
                    "Alle anderen Werte werden weiterhin aus der Kura-Cloud geladen.\n"
                    "Diese Datei wird NICHT an Kura Medical übertragen.",
                    title="Lokale Konfiguration"
                )
            os.startfile(override_path)
        except Exception as e:
            sg.popup_error(f"Konfiguration konnte nicht geöffnet werden:\n{e}", title="Kura Fehler")

    # ── License dialogs ───────────────────────────────────────────────────────

    def _show_upgrade_dialog(self):
        if sg.popup_yes_no(
            "Testphase beendet. Aktivieren Sie Kura Pro für unbegrenzte Berichte.\n\nJetzt upgraden (€49/Monat)?",
            title="Kura Pro erforderlich"
        ) == "Yes":
            webbrowser.open("https://kura-medical.de/#pricing")

    def _activate_license(self):
        layout = [
            [sg.Text("Lizenzschlüssel eingeben (Format: XXXX-XXXX-XXXX-XXXX):", font=("Arial", 10))],
            [sg.InputText("", key="-KEY-", size=(44, 1), font=("Courier New", 10))],
            [
                sg.Button("Aktivieren", key="-ACTIVATE-", button_color=("white", "#1976d2"), size=(14, 1)),
                sg.Button("Jetzt kaufen", key="-BUY-", button_color=("white", "#388e3c"), size=(14, 1)),
                sg.Button("Abbrechen", key="-CANCEL-", button_color=("white", "#757575"), size=(12, 1)),
            ],
        ]
        win = sg.Window("Kura Pro aktivieren", layout, finalize=True, modal=True)
        result = False
        while True:
            event, values = win.read()
            if event in (sg.WINDOW_CLOSED, "-CANCEL-"):
                break
            if event == "-BUY-":
                webbrowser.open("https://kura-medical.de/#pricing")
            if event == "-ACTIVATE-":
                key = values["-KEY-"].strip()
                if not key:
                    sg.popup_error("Bitte geben Sie einen Lizenzschlüssel ein.")
                    continue
                ok, msg = self.license_mgr.activate(key)
                if ok:
                    sg.popup_ok(f"✅ {msg}", title="Aktivierung erfolgreich")
                    result = True
                    break
                else:
                    sg.popup_error(f"Aktivierung fehlgeschlagen:\n{msg}")
        win.close()
        return result

    def _deactivate_license(self):
        if sg.popup_yes_no(
            "Dies entfernt Kura Pro von diesem Gerät.\n\n"
            "Der Schlüssel kann danach auf einem anderen Gerät aktiviert werden.\n\n"
            "Internetverbindung erforderlich.\n\nLizenz deaktivieren?",
            title="Lizenz deaktivieren?"
        ) == "Yes":
            ok, msg = self.license_mgr.deactivate()
            sg.popup_ok(msg, title="Erledigt" if ok else "Fehler")
            return ok
        return False

    def _reset_trial(self):
        if sg.popup_yes_no(
            "Testzähler auf 0 zurücksetzen?\n\n⚠️ Nur für Entwicklungszwecke.",
            title="Testphase zurücksetzen"
        ) == "Yes":
            try:
                for f in [self.license_mgr.trial_file, self.license_mgr.hardware_id_file]:
                    if os.path.exists(f):
                        os.remove(f)
                sg.popup_ok("Testphase zurückgesetzt. Sie haben wieder 5 kostenlose Berichte.")
            except Exception as e:
                sg.popup_error(f"Fehler: {e}")

    # ── Main window & event loop ──────────────────────────────────────────────

    def run(self):
        layout = [
            [sg.Text(f"Kura v{APP_VERSION}", font=("Arial", 13, "bold")),
             sg.Push(),
             sg.Text(self._license_text(), key="-LICENSE-", font=("Arial", 9))],
            [sg.HSeparator()],

            [sg.Text("Patient:", font=("Arial", 9)),
             sg.InputText("Weber", key="-PATIENT-", size=(30, 1), font=("Arial", 9)),
             sg.Frame("Versicherung", [
                 [sg.Radio("GKV", "INSURANCE", key="-GKV-", default=True, font=("Arial", 9)),
                  sg.Radio("PKV", "INSURANCE", key="-PKV-", font=("Arial", 9)),
                  sg.Radio("BG",  "INSURANCE", key="-BG-",  font=("Arial", 9))],
             ], font=("Arial", 8), pad=(8, 0))],
            [sg.HSeparator()],

            [sg.Button("🔴 Sitzung starten", key="-START-", size=(20, 1), font=("Arial", 9)),
             sg.Button("⏹ Stoppen & Verarbeiten", key="-STOP-", size=(22, 1), font=("Arial", 9), disabled=True)],

            [sg.Button("📄 PDF exportieren",     key="-PDF-",      size=(17, 1), font=("Arial", 9)),
             sg.Button("📂 Archiv öffnen",       key="-ARCHIVE-",  size=(16, 1), font=("Arial", 9)),
             sg.Button("🔑 Lizenz aktivieren",   key="-LIC-ACT-",  size=(17, 1), font=("Arial", 9)),
             sg.Button("🔓 Deaktivieren",        key="-LIC-DEACT-",size=(14, 1), font=("Arial", 9))],

            [sg.Button("🏥 Praxis-Einstellungen", key="-PRACTICE-", size=(20, 1), font=("Arial", 9)),
             sg.Button("⚙️ Konfiguration",       key="-CONFIG-",   size=(16, 1), font=("Arial", 9)),
             sg.Button("🔄 Update prüfen",       key="-UPDATE-",   size=(15, 1), font=("Arial", 9))],

            [sg.Button("ℹ Über Kura",            key="-ABOUT-",    size=(12, 1), font=("Arial", 9)),
             sg.Button("🔧 System Info",          key="-SYSINFO-",  size=(13, 1), font=("Arial", 9)),
             sg.Button("🔄 Trial Reset",          key="-TRIAL-RESET-", size=(13, 1), font=("Arial", 9))],

            [sg.HSeparator()],
            [sg.Multiline(size=(76, 12), key="-OUTPUT-", disabled=True, font=("Courier New", 8),
                          background_color="#1e1e1e", text_color="#d4d4d4")],

            [sg.Text("🩺 Bereit", key="-STATUS-", font=("Arial", 9)),
             sg.Push(),
             sg.Button("Beenden", key="-QUIT-", size=(10, 1),
                       button_color=("white", "#757575"), font=("Arial", 9))],
        ]

        window = sg.Window(
            f"Kura v{APP_VERSION} — Medizinische KI-Dokumentation",
            layout,
            finalize=True,
            size=(760, 600),
            resizable=True,
        )
        self._window = window

        while True:
            event, values = window.read(timeout=200)

            if event in (sg.WINDOW_CLOSED, "-QUIT-"):
                break

            # ── Background / async events ──────────────────────────────────────
            elif event == "-STATUS-UPDATE-":
                window["-STATUS-"].update(values[event])
            
            elif event == "-PROGRESS-":
                # Show progress in output box
                current = window["-OUTPUT-"].get()
                # Keep only the last few progress messages
                lines = current.split('\n')
                if len(lines) > 10:
                    lines = lines[-10:]
                new_text = '\n'.join(lines) + '\n' + values[event]
                window["-OUTPUT-"].update(new_text)

            elif event == "-BOOT-DONE-":
                window["-OUTPUT-"].update(
                    f"✅ KI-Modelle geladen. Kura v{APP_VERSION} ist einsatzbereit.\n"
                )

            elif event == "-TIMER-":
                window["-STATUS-"].update(values[event])

            elif event == "-UPDATE-AVAILABLE-":
                remote_ver = values[event]
                window["-STATUS-"].update(f"🆕 Update v{remote_ver} verfügbar — 'Update prüfen' klicken")

            elif event == "-UPDATE-RESULT-":
                remote_ver = values[event]
                if remote_ver == "ERROR" or remote_ver is None:
                    sg.popup_ok("Update-Prüfung fehlgeschlagen.\nKeine Verbindung — bitte später erneut versuchen.",
                                title="Kura Update")
                elif self._version_gt(remote_ver, APP_VERSION):
                    self._open_update_page()
                else:
                    sg.popup_ok(f"Kura ist aktuell (v{APP_VERSION}).", title="Kura Update")
                window["-STATUS-"].update("🩺 Bereit")

            elif event == "-AI-DONE-":
                res = values[event]
                window["-STATUS-"].update("✅ Analyse abgeschlossen — Bericht prüfen")
                soap = res.get("soap", {})
                br = res.get("billing_result")
                billing_line  = br.format_billing_line() if br else res.get("billing_suggestion", "")
                audit_status  = f"AUDIT: {br.audit_status}" if br else ""
                profile_label = res.get("profile_label", "")
                summary = (
                    f"ICD-10: {res.get('icd10')}  |  {billing_line}\n"
                    f"{audit_status}\n"
                    f"PROFIL: {profile_label}\n\n"
                    f"S: {soap.get('S')}\n\nO: {soap.get('O')}\n\n"
                    f"A: {soap.get('A')}\n\nP: {soap.get('P')}"
                )
                window["-OUTPUT-"].update(summary)

                edited = self._show_review_window(res)
                if edited:
                    self._finalize(edited, res, window)
                else:
                    window["-STATUS-"].update("🩺 Bereit")

            elif event == "-AI-ERROR-":
                window["-STATUS-"].update("❌ KI-Fehler")
                window["-OUTPUT-"].update(f"❌ Fehler: {values[event]}")

            elif event == "-ERROR-":
                sg.popup_error(values[event], title="Kura Fehler")

            # ── User actions ───────────────────────────────────────────────────
            elif event == "-START-":
                patient = values["-PATIENT-"].strip()
                if not patient:
                    sg.popup_error("Bitte geben Sie einen Patientennamen ein.")
                    continue
                from shared.billing_engine import InsuranceType
                self.insurance_type = (
                    InsuranceType.PKV if values.get("-PKV-") else
                    InsuranceType.BG  if values.get("-BG-")  else
                    InsuranceType.GKV
                )
                self._start_recording(patient, window)
                window["-OUTPUT-"].update("🔴 Aufzeichnung läuft...\n")

            elif event == "-STOP-":
                self._stop_recording(window)

            elif event == "-PDF-":
                self._save_pdf_to_disk()

            elif event == "-ARCHIVE-":
                try:
                    os.startfile(self.report_dir)
                except Exception as e:
                    sg.popup_error(f"Ordner konnte nicht geöffnet werden:\n{e}")

            elif event == "-LIC-ACT-":
                if self._activate_license():
                    window["-LICENSE-"].update(self._license_text())

            elif event == "-LIC-DEACT-":
                if self._deactivate_license():
                    window["-LICENSE-"].update(self._license_text())

            elif event == "-PRACTICE-":
                self._open_practice_config()

            elif event == "-CONFIG-":
                self._open_gist_override()

            elif event == "-UPDATE-":
                window["-STATUS-"].update("🔍 Prüfe auf Updates...")
                self._check_update_manual(window)

            elif event == "-ABOUT-":
                sg.popup(
                    f"Kura v{APP_VERSION}\n\n"
                    "Medizinische KI-Dokumentation\n\n"
                    "• 100% Lokale Verarbeitung (DSGVO-sicher)\n"
                    "• Professionelle medizinische Dokumentation\n"
                    "• § 125 Abs. 1 SGB V konform\n\n"
                    "© 2026 Kura Medical",
                    title="Über Kura",
                    button_color=("white", "#1976d2"),
                )

            elif event == "-SYSINFO-":
                trial_count = self.license_mgr.get_trial_count()
                remaining   = self.license_mgr.max_trials - trial_count
                status      = self.license_mgr.verify_locally()
                license_status = (
                    "Kura Pro: Aktiv" if status is True
                    else (f"Testphase: {remaining}/{self.license_mgr.max_trials} verbleibend"
                          if status == "TRIAL" else "Testphase abgelaufen")
                )
                sg.popup(
                    f"System Information\n\n"
                    f"Version:     v{APP_VERSION}\n"
                    f"Lizenz:      {license_status}\n"
                    f"Hardware ID: {self.license_mgr.hardware_id}\n"
                    f"Daten:       {USER_DATA_DIR}\n"
                    f"Log:         {CRASH_LOG}",
                    title="System Info",
                    button_color=("white", "#1976d2"),
                )

            elif event == "-TRIAL-RESET-":
                self._reset_trial()
                window["-LICENSE-"].update(self._license_text())

        window.close()


if __name__ == "__main__":
    app = KuraApp()
    app.run()
