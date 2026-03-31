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
from datetime import datetime

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from fpdf import FPDF
import PySimpleGUI as sg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.license_manager import LicenseManager
# KuraEngine imported lazily inside _boot() so scipy/mlx don't block window startup


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


# ── Audio recording (sounddevice — no ffmpeg required) ───────────────────────

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
        self.seconds_elapsed = 0
        self._record_thread = None

        # Boot engine in background
        threading.Thread(target=self._boot, daemon=True).start()

    # ── Boot ──────────────────────────────────────────────────────────────────

    def _boot(self):
        try:
            self._post_event("-STATUS-UPDATE-", "⏳ Modelle laden... bitte warten")
            from physio_scribe_crossplatform import KuraEngine
            self.engine = KuraEngine()
            self._post_event("-STATUS-UPDATE-", "✅ Kura Bereit (Lokal & DSGVO)")
            self._post_event("-BOOT-DONE-", None)
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

    # ── Recording ─────────────────────────────────────────────────────────────

    def _start_recording(self, patient_name: str, window):
        if not self.engine:
            sg.popup_error("KI-Modelle werden noch geladen. Bitte warten.")
            return

        self.patient_name = patient_name.strip().replace(" ", "_") or "Unbekannt"
        self.recording = True
        self.seconds_elapsed = 0

        window["-START-"].update(disabled=True)
        window["-STOP-"].update(disabled=False)

        # Recording thread
        self._record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._record_thread.start()

        # Timer thread
        threading.Thread(target=self._timer_thread, args=(window,), daemon=True).start()

    def _record_audio(self):
        chunks = []
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=1024) as stream:
                while self.recording:
                    data, _ = stream.read(1024)
                    chunks.append(data.copy())
        except Exception as e:
            print(f"Recording error: {e}")

        if chunks:
            audio = np.concatenate(chunks, axis=0)
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
        window["-STATUS-"].update("🧠 KI-Analyse läuft...")

        status = self.license_mgr.verify_locally()
        if status is True or status == "TRIAL":
            threading.Thread(target=self._run_ai, args=(window,), daemon=True).start()
        else:
            window["-STATUS-"].update("🩺 Bereit")
            self._show_upgrade_dialog()

    # ── AI pipeline ───────────────────────────────────────────────────────────

    def _run_ai(self, window):
        try:
            def update_status(msg):
                window.write_event_value("-STATUS-UPDATE-", msg)

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

            window.write_event_value("-AI-DONE-", res)

        except Exception as e:
            print(f"❌ AI Error: {e}")
            try:
                if os.path.exists(self.temp_audio):
                    os.remove(self.temp_audio)
            except Exception:
                pass
            window.write_event_value("-AI-ERROR-", str(e))

    # ── Result review window ───────────────────────────────────────────────────

    def _show_review_window(self, res):
        """
        Shows the SOAP note in an editable window — therapist can correct before saving.
        Matches the macOS rumps.Window review flow.
        """
        soap = res.get("soap", {})
        warnings = res.get("compliance_check", [])
        warning_str = "\n".join(f"-> {w}" for w in warnings) if warnings else "✅ Dokumentation GKV-konform."

        br = res.get("billing_result")
        if br:
            billing_line = br.format_billing_line()
            audit_block = br.format_audit_report()
            if br.optimization_hints:
                audit_block += "\n\nHINWEISE:\n" + "\n".join(br.optimization_hints)
        else:
            billing_line = f"POSITION: {res.get('billing_suggestion', '?')}"
            audit_block = warning_str

        profile_label = res.get("profile_label", "")
        profile_line = f"PROFIL: {profile_label}\n" if profile_label else ""

        # SOAP first — immediately visible; billing + tick-box audit below
        initial_text = (
            f"{profile_line}"
            f"S: {soap.get('S', '')}\n\n"
            f"O: {soap.get('O', '')}\n\n"
            f"A: {soap.get('A', '')}\n\n"
            f"P: {soap.get('P', '')}\n\n"
            f"{'━'*44}\n"
            f"ABRECHNUNG | ICD-10: {res.get('icd10', '?')}\n"
            f"{billing_line}\n\n"
            f"AUDIT §106b SGB V\n"
            f"{audit_block}\n\n"
            f"PATIENT: {self.patient_name.replace('_', ' ')}"
        )

        layout = [
            [sg.Text("Prüfen und in Abrechnung übernehmen:", font=("Arial", 10, "bold"))],
            [sg.Multiline(initial_text, size=(72, 22), key="-RESULT-", font=("Courier New", 9))],
            [
                sg.Button("✅ KOPIEREN & PDF", key="-SAVE-", button_color=("white", "#1976d2"), size=(18, 1)),
                sg.Button("Abbrechen", key="-CANCEL-", button_color=("white", "#757575"), size=(12, 1)),
            ],
        ]

        window = sg.Window(
            "KURA v2026 — Befund-Revision",
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

        # Clipboard: paste clean text (headers/audit notes stripped)
        clean_text = re.sub(r"---.*?---|->.*?(\n|$)", "", edited_text).strip()
        try:
            process = subprocess.Popen("clip", stdin=subprocess.PIPE, shell=True)
            process.communicate(clean_text.encode("utf-8"))
        except Exception as e:
            print(f"Clipboard error: {e}")

        # Archive JSON
        self.last_report = edited_text
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        patient_dir = os.path.join(self.report_dir, self.patient_name)
        os.makedirs(patient_dir, exist_ok=True)
        with open(os.path.join(patient_dir, f"{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump({"text": edited_text, "patient": self.patient_name,
                       "icd10": user_icd or res.get("icd10"), "timestamp": timestamp},
                      f, ensure_ascii=False, indent=4)

        # PDF
        self._save_pdf_to_disk()

        # Trial increment
        if status == "TRIAL":
            count = self.license_mgr.get_trial_count()
            remaining = self.license_mgr.max_trials - (count + 1)
            self.license_mgr.increment_trial()
            sg.popup_ok(
                f"Bericht {count + 1} von {self.license_mgr.max_trials} gespeichert.\n"
                f"Noch {remaining} kostenlose Berichte verbleibend.",
                title="Kura Testphase"
            )
        else:
            sg.popup_ok("✅ Bericht gespeichert (Desktop + Archiv)", title="Kura")

        window["-LICENSE-"].update(self._license_text())
        window["-STATUS-"].update("🩺 Bereit")

    # ── PDF export ────────────────────────────────────────────────────────────

    def _save_pdf_to_disk(self):
        if not self.last_report:
            sg.popup_error("Kein Bericht zum Speichern vorhanden.")
            return

        safe_name = self.patient_name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        desktop = os.path.join(os.path.expanduser("~/Desktop"), f"Kura_{safe_name}.pdf")
        patient_dir = os.path.join(self.report_dir, safe_name)
        os.makedirs(patient_dir, exist_ok=True)
        archive = os.path.join(patient_dir, f"Bericht_{timestamp}.pdf")

        try:
            pdf = FPDF()
            pdf.set_margins(15, 15, 15)
            pdf.add_page()

            display_name = self.patient_name.replace("_", " ")
            display_ts = datetime.now().strftime("%d.%m.%Y | %H:%M")

            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(180, 10, f"Physiotherapeutischer Befund: {display_name} | {display_ts}",
                     new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(5)

            clean = (self.last_report
                     .replace("–", "-").replace("„", '"').replace("\u201c", '"').replace("°", " Grad")
                     .encode("latin-1", "replace").decode("latin-1"))

            lasegue_pattern = r"(Las.gue[^\.]*?\d+.*?Grad[^\.]*)"
            for line in clean.split("\n"):
                if not line.strip():
                    pdf.ln(5)
                    continue
                if re.search(lasegue_pattern, line, re.IGNORECASE):
                    parts = re.split(f"({lasegue_pattern})", line, flags=re.IGNORECASE)
                    pdf.set_x(15)
                    for part in parts:
                        if re.match(lasegue_pattern, part, re.IGNORECASE):
                            pdf.set_font("Helvetica", "B", 11)
                            pdf.write(8, part)
                        else:
                            pdf.set_font("Helvetica", "", 11)
                            pdf.write(8, part)
                    pdf.ln(8)
                else:
                    pdf.set_font("Helvetica", size=11)
                    pdf.set_x(15)
                    pdf.multi_cell(180, 8, line)

            pdf.output(archive)
            pdf.output(desktop)
        except Exception as e:
            sg.popup_error(f"PDF-Fehler: {e}")

    # ── Dialogs ───────────────────────────────────────────────────────────────

    def _show_upgrade_dialog(self):
        import webbrowser
        if sg.popup_yes_no(
            "Testphase beendet. Aktivieren Sie Kura Pro für unbegrenzte Berichte.\n\nJetzt upgraden (€39/Monat)?",
            title="Kura Pro erforderlich"
        ) == "Yes":
            webbrowser.open("https://kura.lemonsqueezy.com/checkout/buy/2400563b-a13a-4e42-b734-d79122e7ec92")

    def _activate_license(self):
        key = sg.popup_get_text(
            "Geben Sie Ihren Kura Pro Lizenzschlüssel ein:",
            title="Kura Pro Aktivierung"
        )
        if key and key.strip():
            if self.license_mgr.verify_online(key.strip()):
                sg.popup_ok("✅ Kura Pro ist jetzt aktiv!", title="Aktivierung erfolgreich")
                return True
            else:
                sg.popup_error("Ungültiger Lizenzschlüssel.\nBitte überprüfen Sie Ihren Schlüssel.")
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

    def _sync_config(self):
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
            sg.popup_error(f"Konfiguration konnte nicht geöffnet werden:\n{e}")

    # ── Main window & event loop ──────────────────────────────────────────────

    def run(self):
        layout = [
            [sg.Text("Kura v2026.1.0", font=("Arial", 13, "bold")),
             sg.Push(),
             sg.Text(self._license_text(), key="-LICENSE-", font=("Arial", 9))],
            [sg.HSeparator()],

            [sg.Text("Patient:", font=("Arial", 9)),
             sg.InputText("Weber_15031964", key="-PATIENT-", size=(28, 1), font=("Arial", 9)),
             sg.Text("  Kasse:", font=("Arial", 9)),
             sg.Radio("GKV", "INSURANCE", key="-GKV-", default=True, font=("Arial", 9)),
             sg.Radio("PKV", "INSURANCE", key="-PKV-", font=("Arial", 9)),
             sg.Radio("BG",  "INSURANCE", key="-BG-",  font=("Arial", 9))],
            [sg.HSeparator()],

            [sg.Button("🔴 Sitzung starten", key="-START-", size=(20, 1), font=("Arial", 9)),
             sg.Button("⏹ Stoppen & Verarbeiten", key="-STOP-", size=(22, 1), font=("Arial", 9), disabled=True)],

            [sg.Button("📄 PDF exportieren", key="-PDF-", size=(15, 1), font=("Arial", 9)),
             sg.Button("📂 Archiv öffnen", key="-ARCHIVE-", size=(15, 1), font=("Arial", 9)),
             sg.Button("🔑 Lizenz", key="-LICENSE-BTN-", size=(10, 1), font=("Arial", 9)),
             sg.Button("⚙️ Konfig", key="-SYNC-", size=(9, 1), font=("Arial", 9))],

            [sg.Button("ℹ Über Kura", key="-ABOUT-", size=(12, 1), font=("Arial", 9)),
             sg.Button("🔧 System Info", key="-SYSINFO-", size=(13, 1), font=("Arial", 9)),
             sg.Button("🔄 Trial Reset", key="-TRIAL-RESET-", size=(13, 1), font=("Arial", 9))],

            [sg.HSeparator()],
            [sg.Multiline(size=(72, 12), key="-OUTPUT-", disabled=True, font=("Courier New", 8),
                          background_color="#1e1e1e", text_color="#d4d4d4")],

            [sg.Text("🩺 Bereit", key="-STATUS-", font=("Arial", 9)),
             sg.Push(),
             sg.Button("Beenden", key="-QUIT-", size=(10, 1),
                       button_color=("white", "#757575"), font=("Arial", 9))],
        ]

        window = sg.Window(
            "Kura v2026 — Medizinische KI-Dokumentation",
            layout,
            finalize=True,
            size=(720, 560),
            resizable=True,
        )
        self._window = window

        while True:
            event, values = window.read(timeout=200)

            if event in (sg.WINDOW_CLOSED, "-QUIT-"):
                break

            # ── Background events ──────────────────────────────────────────────
            elif event == "-STATUS-UPDATE-":
                window["-STATUS-"].update(values[event])

            elif event == "-BOOT-DONE-":
                window["-OUTPUT-"].update("✅ KI-Modelle geladen. Kura ist einsatzbereit.\n")

            elif event == "-TIMER-":
                window["-STATUS-"].update(values[event])

            elif event == "-AI-DONE-":
                res = values[event]
                window["-STATUS-"].update("✅ Analyse abgeschlossen — Bericht prüfen")
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
                window["-OUTPUT-"].update(summary)

                # Open review/edit window immediately
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

            elif event == "-LICENSE-BTN-":
                if self._activate_license():
                    window["-LICENSE-"].update(self._license_text())

            elif event == "-SYNC-":
                self._sync_config()

            elif event == "-ABOUT-":
                sg.popup(
                    "Kura v2026.1.0\n\n"
                    "Medizinische KI-Dokumentation\n\n"
                    "• 100% Lokale Verarbeitung (DSGVO-sicher)\n"
                    "• Professionelle medizinische Dokumentation\n"
                    "• § 84 Abs. 6/7 SGB V konform\n\n"
                    "© 2026 Kura Medical",
                    title="Über Kura",
                    button_color=("white", "#1976d2"),
                )

            elif event == "-SYSINFO-":
                trial_count = self.license_mgr.get_trial_count()
                remaining = self.license_mgr.max_trials - trial_count
                status = self.license_mgr.verify_locally()
                license_status = (
                    "Kura Pro: Aktiv" if status is True
                    else (f"Testphase: {remaining}/{self.license_mgr.max_trials} verbleibend"
                          if status == "TRIAL" else "Testphase abgelaufen")
                )
                sg.popup(
                    f"System Information\n\n"
                    f"Lizenz: {license_status}\n"
                    f"Hardware ID: {self.license_mgr.hardware_id}\n"
                    f"Version: v2026.1.0\n"
                    f"Daten: {USER_DATA_DIR}\n"
                    f"Log: {CRASH_LOG}",
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