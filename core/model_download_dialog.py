"""
Model Download Dialog for Windows GUI
Shows progress bar that fills up + download log for user curiosity
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os


class ModelDownloadDialog:
    """Clean GUI dialog with filling progress bar and download log."""

    def __init__(self, parent=None):
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
    Show download dialog if models are missing.
    Returns True if models are ready, False if download failed.
    """
    # Check if models exist
    try:
        from core.model_downloader import check_model_exists

        llm_exists = check_model_exists("llm")
        whisper_exists = check_model_exists("whisper")

        if llm_exists and whisper_exists:
            # Models already exist, no need for dialog
            return True

        # Models missing - show dialog
        dialog = ModelDownloadDialog()
        return dialog.run()

    except Exception as e:
        # Error checking models - show error dialog
        messagebox.showerror(
            "Kura Error",
            f"Error checking models: {type(e).__name__}: {e}\n\n"
            "Please ensure you have internet connection and restart Kura."
        )
        return False


if __name__ == "__main__":
    """Test the dialog."""
    success = show_download_dialog_if_needed()
    print(f"Download dialog result: {success}")
    sys.exit(0 if success else 1)

