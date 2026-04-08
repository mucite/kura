"""
Model Download Dialog for Windows GUI
Shows progress bar that fills up + download log for user curiosity

Note: macOS uses rumps for menu bar GUI, not tkinter.
      Tkinter is only used on Windows.
"""
import threading
import sys
import os
import platform

# Disable tkinter on macOS - we use rumps for the menu bar GUI instead
_is_macos = platform.system() == "Darwin"

if _is_macos:
    # macOS uses rumps, not tkinter
    print(f"[model_download_dialog] macOS detected - tkinter disabled (using rumps for GUI)")
    TKINTER_AVAILABLE = False
    tk = None
    ttk = None
    messagebox = None
else:
    # Windows: Try to import tkinter for GUI dialogs
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        TKINTER_AVAILABLE = True
    except (ImportError, RuntimeError, Exception) as e:
        # Tkinter not available - fall back to CLI-only mode
        print(f"[model_download_dialog] tkinter unavailable: {e}")
        TKINTER_AVAILABLE = False
        tk = None
        ttk = None
        messagebox = None


class ModelDownloadDialog:
    """Clean GUI dialog with filling progress bar and download log."""

    def __init__(self, parent=None):
        if not TKINTER_AVAILABLE:
            raise RuntimeError("Tkinter is not available - GUI dialog cannot be created")

        self.root = tk.Toplevel() if parent else tk.Tk()
        self.root.title("Kura - Setup")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (300)
        y = (self.root.winfo_screenheight() // 2) - (200)
        self.root.geometry(f"600x400+{x}+{y}")

        # Make it stay on top
        self.root.attributes('-topmost', True)

        # Prevent closing during download
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.success = False
        self.downloading = False
        self.setup_ui()

    def setup_ui(self):
        """Create clean UI with progress and log."""
        # Main frame with padding
        main_frame = tk.Frame(self.root, bg="#ffffff")
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)

        # Icon/Title
        tk.Label(
            main_frame,
            text="🩺 Kura Medical",
            font=("Arial", 18, "bold"),
            bg="#ffffff",
            fg="#000000"
        ).pack(pady=(0, 5))

        tk.Label(
            main_frame,
            text="Downloading AI Models",
            font=("Arial", 12),
            bg="#ffffff",
            fg="#666666"
        ).pack(pady=(0, 15))

        # Status message
        self.status_label = tk.Label(
            main_frame,
            text="Preparing download...",
            font=("Arial", 10),
            bg="#ffffff",
            fg="#333333"
        )
        self.status_label.pack(pady=(0, 5))

        # Progress bar (green, fills up with percentage)
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor='#e0e0e0',
            background='#28a745',
            thickness=25
        )

        self.progress = ttk.Progressbar(
            main_frame,
            style="green.Horizontal.TProgressbar",
            mode='determinate',  # ✅ Fills up, not back-and-forth
            maximum=100,
            length=540
        )
        self.progress.pack(pady=10)

        # Percentage label
        self.percent_label = tk.Label(
            main_frame,
            text="0%",
            font=("Arial", 10, "bold"),
            bg="#ffffff",
            fg="#28a745"
        )
        self.percent_label.pack(pady=(0, 10))

        # Download log area (compact, shows what's happening)
        tk.Label(
            main_frame,
            text="Download Progress:",
            font=("Arial", 9, "bold"),
            bg="#ffffff",
            fg="#333333",
            anchor="w"
        ).pack(fill="x", pady=(5, 3))

        log_scroll = tk.Scrollbar(main_frame)
        log_scroll.pack(side="right", fill="y")

        self.log_text = tk.Text(
            main_frame,
            height=8,
            font=("Consolas", 9),
            bg="#f5f5f5",
            fg="#333333",
            wrap="word",
            yscrollcommand=log_scroll.set
        )
        self.log_text.pack(fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        # ✅ AUTO-START: Begin download immediately
        self.root.after(500, self.start_download)

    def on_close(self):
        """Handle window close button."""
        if self.downloading:
            if messagebox.askyesno("Cancel Setup", "Download in progress.\n\nCancel and exit Kura?"):
                self.success = False
                self.root.destroy()
        else:
            self.success = False
            self.root.destroy()

    def log(self, message):
        """Add message to log."""
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.root.update()

    def update_progress(self, percent, status=None):
        """Update progress bar and percentage."""
        self.progress['value'] = percent
        self.percent_label.config(text=f"{int(percent)}%")
        if status:
            self.status_label.config(text=status)
        self.root.update()

    def start_download(self):
        """Start the download in a background thread."""
        self.downloading = True
        self.update_progress(5, "Starting download...")
        self.log("Checking models...\n")

        # Run download in background thread
        thread = threading.Thread(target=self._download_models, daemon=True)
        thread.start()

    def _download_models(self):
        """Download models (runs in background thread)."""
        try:
            # Load .env file to get HF_TOKEN
            try:
                from dotenv import load_dotenv
                user_data_dir = os.path.expanduser("~/Documents/Kura")
                env_file = os.path.join(user_data_dir, ".env")
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    self.log("✅ Configuration loaded")
            except:
                pass

            # Import model downloader
            from core.model_downloader import ensure_models_available

            self.update_progress(10, "Checking internet...")
            self.log("Connecting to HuggingFace...\n")

            # Redirect stdout to capture and display download progress
            class ProgressWriter:
                """Captures print output and updates progress bar."""
                def __init__(self, dialog):
                    self.dialog = dialog
                    self.buffer = ""
                    self.progress_val = 10

                def write(self, text):
                    self.buffer += text
                    if '\n' in text:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                self.dialog.log(line.strip())

                                # Estimate progress based on log messages
                                if "Downloading" in line and "Llama" in line:
                                    self.progress_val = 15
                                    self.dialog.update_progress(15, "Downloading LLM...")
                                elif "Downloading" in line and "Whisper" in line:
                                    self.progress_val = 70
                                    self.dialog.update_progress(70, "Downloading Whisper...")
                                elif "Downloaded:" in line:
                                    if "Llama" in line or "Meta" in line or "4.9" in line or "4.8" in line:
                                        self.progress_val = 65
                                        self.dialog.update_progress(65, "LLM downloaded ✓")
                                    else:
                                        self.progress_val = 95
                                        self.dialog.update_progress(95, "Whisper downloaded ✓")
                                elif "SETUP COMPLETE" in line or "ready" in line.lower():
                                    self.progress_val = 100
                                    self.dialog.update_progress(100, "Complete!")
                                elif "already" in line.lower() and "installed" in line.lower():
                                    # Models already exist
                                    if "llm" in line.lower():
                                        self.progress_val = 65
                                        self.dialog.update_progress(65, "LLM found ✓")
                                    elif "whisper" in line.lower():
                                        self.progress_val = 95
                                        self.dialog.update_progress(95, "Whisper found ✓")
                        self.buffer = lines[-1]

                def flush(self):
                    pass

            # Replace stdout temporarily
            original_stdout = sys.stdout
            sys.stdout = ProgressWriter(self)

            try:
                result = ensure_models_available()
            finally:
                sys.stdout = original_stdout

            # Handle result
            if result:
                self.success = True
                self.downloading = False
                self.update_progress(100, "✅ Setup complete!")
                self.log("\n✅ All models ready - starting Kura...")

                # Auto-close after 2 seconds
                self.root.after(2000, self.close_success)
            else:
                self.downloading = False
                self.show_error("Download failed. Please check your internet connection.")

        except Exception as e:
            self.downloading = False
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.log(f"\n❌ ERROR: {error_msg}")
            self.show_error(error_msg)

    def show_error(self, error_msg):
        """Show error and offer retry."""
        self.update_progress(0, "❌ Setup failed")

        # Show retry dialog
        if messagebox.askretrycancel(
            "Kura Setup Error",
            f"Model download failed:\n\n{error_msg}\n\n"
            "Please ensure:\n"
            "• Internet connection is active\n"
            "• At least 8 GB free disk space\n"
            "• No firewall blocking huggingface.co\n\n"
            "Retry download?"
        ):
            # Restart download
            self.progress['value'] = 0
            self.percent_label.config(text="0%")
            self.log_text.delete("1.0", "end")
            self.start_download()
        else:
            self.success = False
            self.root.destroy()

    def close_success(self):
        """Close the dialog after successful download."""
        # Unbind close protocol to prevent confirmation dialog
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.destroy()

    def run(self):
        """Run the dialog and return success status."""
        self.root.mainloop()
        return self.success


def show_download_dialog_if_needed():
    """
    Show download dialog if models are missing (Windows / GGUF).
    Returns True if models are ready, False if download failed.
    """
    if not TKINTER_AVAILABLE:
        print("[model_download_dialog] Tkinter not available, falling back to CLI download")
        return False

    try:
        from core.model_downloader import check_model_exists

        llm_exists = check_model_exists("llm")
        whisper_exists = check_model_exists("whisper")

        if llm_exists and whisper_exists:
            return True

        dialog = ModelDownloadDialog()
        return dialog.run()

    except Exception as e:
        if TKINTER_AVAILABLE:
            messagebox.showerror(
                "Kura Error",
                f"Error checking models: {type(e).__name__}: {e}\n\n"
                "Please ensure you have internet connection and restart Kura."
            )
        else:
            print(f"[model_download_dialog] Error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# macOS / MLX variant
# ══════════════════════════════════════════════════════════════════════════════

class ModelDownloadDialogMacOS(ModelDownloadDialog):
    """
    Same UI as the Windows dialog, but downloads MLX safetensors models
    (Llama 3.1-8B + Whisper large-v3-turbo) instead of GGUF.
    """

    def __init__(self, parent=None):
        if not TKINTER_AVAILABLE:
            raise RuntimeError("Tkinter is not available - GUI dialog cannot be created")
        super().__init__(parent)

    def _download_models(self):
        """Download MLX models (runs in background thread)."""
        try:
            # Load .env for HF_TOKEN
            try:
                from dotenv import load_dotenv
                user_data_dir = os.path.expanduser("~/Library/Application Support/Kura")
                env_file = os.path.join(user_data_dir, ".env")
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    self.log("✅ Konfiguration geladen")
            except Exception:
                pass

            from core.model_downloader import ensure_models_available_macos

            self.update_progress(10, "Internetverbindung prüfen...")
            self.log("Verbindung zu HuggingFace...\n")

            class ProgressWriter:
                def __init__(self, dialog):
                    self.dialog = dialog
                    self.buffer = ""
                    self.llm_done = False
                    self.whisper_done = False

                def write(self, text):
                    self.buffer += text
                    if '\n' in text:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                self.dialog.log(line.strip())
                                ll = line.lower()

                                if "llama" in ll and ("download" in ll or "📥" in ll):
                                    self.dialog.update_progress(15, "Downloading Llama 3.1-8B (4.2 GB)...")
                                elif "whisper" in ll and ("download" in ll or "📥" in ll):
                                    self.dialog.update_progress(70, "Downloading Whisper large-v3-turbo (1.5 GB)...")
                                elif "llama" in ll and "ready" in ll:
                                    self.llm_done = True
                                    self.dialog.update_progress(65, "Llama 3.1-8B ✓")
                                elif "whisper" in ll and "ready" in ll:
                                    self.whisper_done = True
                                    self.dialog.update_progress(95, "Whisper ✓")
                                elif "already installed" in ll:
                                    if "llama" in ll or "8b" in ll:
                                        self.dialog.update_progress(65, "Llama 3.1-8B ✓ (bereits vorhanden)")
                                    elif "whisper" in ll:
                                        self.dialog.update_progress(95, "Whisper ✓ (bereits vorhanden)")
                                elif "setup complete" in ll or "kura is ready" in ll:
                                    self.dialog.update_progress(100, "Fertig!")
                        self.buffer = lines[-1]

                def flush(self):
                    pass

            original_stdout = sys.stdout
            sys.stdout = ProgressWriter(self)
            try:
                result = ensure_models_available_macos()
            finally:
                sys.stdout = original_stdout

            if result:
                self.success = True
                self.downloading = False
                self.update_progress(100, "✅ Setup abgeschlossen!")
                self.log("\n✅ Alle Modelle bereit — Kura startet...")
                self.root.after(2000, self.close_success)
            else:
                self.downloading = False
                self.show_error("Download fehlgeschlagen. Bitte Internetverbindung prüfen.")

        except Exception as e:
            self.downloading = False
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.log(f"\n❌ FEHLER: {error_msg}")
            self.show_error(error_msg)

    def show_error(self, error_msg):
        """Show retry dialog in German."""
        self.update_progress(0, "❌ Setup fehlgeschlagen")
        if messagebox.askretrycancel(
            "Kura Setup Fehler",
            f"Modell-Download fehlgeschlagen:\n\n{error_msg}\n\n"
            "Bitte sicherstellen:\n"
            "• Internetverbindung aktiv\n"
            "• Mind. 8 GB freier Speicher\n"
            "• Keine Firewall blockiert huggingface.co\n\n"
            "Erneut versuchen?"
        ):
            self.progress['value'] = 0
            self.percent_label.config(text="0%")
            self.log_text.delete("1.0", "end")
            self.start_download()
        else:
            self.success = False
            self.root.destroy()


def show_download_dialog_if_needed_macos() -> bool:
    """
    macOS: show the MLX model download dialog if models are missing.
    Fast path (models present) returns True instantly with no GUI shown.
    Returns True when models are ready, False if download failed/cancelled.
    """
    try:
        from core.model_downloader import check_macos_model_exists, _MACOS_MODELS

        all_present = all(
            check_macos_model_exists(sub, key, min_sz)
            for _, sub, key, min_sz, _ in _MACOS_MODELS
        )
        if all_present:
            return True

        # If tkinter is not available, fall back to CLI download
        if not TKINTER_AVAILABLE:
            print("[model_download_dialog] Tkinter not available on macOS, using CLI download...")
            from core.model_downloader import ensure_models_available_macos
            return ensure_models_available_macos()

        # Try to create the GUI dialog - may fail due to tkinter compatibility issues
        try:
            # Suppress output from fast-path checks above before showing GUI
            dialog = ModelDownloadDialogMacOS()
            # German UI labels
            dialog.root.title("Kura – Ersteinrichtung")
            return dialog.run()
        except (RuntimeError, Exception) as gui_err:
            # GUI creation failed (e.g., Python 3.13 tkinter incompatibility on macOS)
            # Fall back to CLI download
            print(f"[model_download_dialog] GUI creation failed: {gui_err}")
            print("[model_download_dialog] Falling back to CLI download...")
            from core.model_downloader import ensure_models_available_macos
            return ensure_models_available_macos()

    except Exception as e:
        if TKINTER_AVAILABLE:
            try:
                messagebox.showerror(
                    "Kura Fehler",
                    f"Modell-Prüfung fehlgeschlagen:\n{type(e).__name__}: {e}\n\n"
                    "Bitte Internetverbindung prüfen und Kura neu starten."
                )
            except Exception:
                print(f"[ERROR] Model check failed: {e}")
        else:
            print(f"[ERROR] Model check failed: {e}")
        return False


if __name__ == "__main__":
    """Test the dialog."""
    success = show_download_dialog_if_needed()
    print(f"Download dialog result: {success}")
    sys.exit(0 if success else 1)

