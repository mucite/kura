"""
Smart Model Downloader for Kura Medical
========================================
Downloads AI models on first launch instead of bundling them in the installer.
This keeps the installer small (~50 MB) while ensuring users get the best model.

Strategy:
1. Installer contains NO models - just the app code
2. On first launch, this script downloads the models
3. Progress bar shows download status
4. Models are cached locally (~6 GB total)

Benefits:
- Installer: 50 MB vs 5 GB (100x smaller!)
- Users always get latest model version
- Bandwidth: Only download once
- Updates: Can update models independently

Authentication:
- Set HF_TOKEN environment variable for faster downloads
- Get free token from https://huggingface.co/settings/tokens
"""
import os
import sys
import signal
import threading
from pathlib import Path

# Import huggingface_hub at module level to avoid threading issues
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None
    snapshot_download = None

# Thread lock for download operations
_download_lock = threading.Lock()

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully without triggering import errors."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\n⚠️  Download cancelled by user - cleaning up...")
    print("Please wait for cleanup to complete...")
    # Don't raise KeyboardInterrupt - just set flag

# Register signal handler (only works in main thread)
try:
    signal.signal(signal.SIGINT, signal_handler)
except ValueError:
    # signal.signal() can only be called from the main thread
    # This is OK - we'll just not have graceful Ctrl+C handling
    pass

def get_model_dir() -> Path:
    """Get the models directory path."""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle - use persistent user directory
        # NOT the app bundle (which gets replaced on updates and is read-only)
        user_app_support = Path(os.path.expanduser("~/Library/Application Support/Kura"))
        user_app_support.mkdir(parents=True, exist_ok=True)
        model_dir = user_app_support / "models"
        print(f"[Bundle mode] Model directory: {model_dir}")
        return model_dir
    else:
        # Running as Python script - go up from core/ to root medic/
        base = Path(__file__).parent.parent
        return base / "models"

def check_model_exists(model_name: str) -> bool:
    """Check if a model file exists and is complete (not corrupted)."""
    model_dir = get_model_dir()

    if model_name == "llm":
        # Check for 8B model in multiple possible locations
        llm_locations = [
            model_dir / "Llama-3.1-8B-Instruct-GGUF" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            model_dir / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        ]
        for path in llm_locations:
            if path.exists():
                # Verify file is not empty and has reasonable size (at least 4 GB)
                file_size = path.stat().st_size
                if file_size > 4_000_000_000:  # At least 4 GB
                    print(f"✅ LLM model already exists: {path.name} ({file_size/1e9:.2f} GB)")
                    return True
                else:
                    print(f"⚠️ Found incomplete LLM model: {path.name} ({file_size/1e9:.2f} GB) - will re-download")
                    return False
        return False

    elif model_name == "whisper":
        whisper_locations = [
            model_dir / "whisper" / "medium.pt",
            model_dir / "whisper" / "large-v3.pt",
            model_dir / "whisper" / "large-v2.pt",
        ]
        for path in whisper_locations:
            if path.exists():
                # Verify file is not empty and has reasonable size (at least 1 GB)
                file_size = path.stat().st_size
                if file_size > 1_000_000_000:  # At least 1 GB
                    print(f"✅ Whisper model already exists: {path.name} ({file_size/1e9:.2f} GB)")
                    return True
                else:
                    print(f"⚠️ Found incomplete Whisper model: {path.name} ({file_size/1e9:.2f} GB) - will re-download")
                    return False
        return False

    return False

def download_model_with_progress(repo_id: str, filename: str, local_dir: Path) -> bool:
    """Download model with progress bar. Thread-safe with lock."""
    # Use lock to prevent concurrent downloads causing import state conflicts
    with _download_lock:
        try:
            # Check if huggingface_hub is available
            if not HF_HUB_AVAILABLE or hf_hub_download is None:
                print("❌ ERROR: huggingface_hub not installed")
                print("   Install with: pip install huggingface_hub")
                return False

            print(f"\n📥 Downloading {filename}...")
            print(f"Source: {repo_id}")
            print(f"Target: {local_dir}")

            # ══════════════════════════════════════════════════════════════
            # CRITICAL FIX: Disable huggingface_hub parallel downloads
            # Parallel downloads use multiprocessing which causes
            # "global import state already initialized" error on Windows
            # ══════════════════════════════════════════════════════════════
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # Keep progress bars
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'     # Disable hf_transfer (uses multiprocessing)

            # Create directory
            local_dir.mkdir(parents=True, exist_ok=True)

            # Get HF_TOKEN from environment (if available)
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

            if hf_token and hf_token != "your_token_here":
                print("✅ Using HF_TOKEN for authenticated download (faster, higher rate limits)")
            else:
                print("\n" + "="*70)
                print("⚠️  WARNING: Unauthenticated HF Hub Request")
                print("="*70)
                print("You are downloading without authentication.")
                print("This will result in:")
                print("  • Slower download speeds")
                print("  • Lower rate limits")
                print("  • Potential download failures during peak times")
                print("\nTo enable faster, authenticated downloads:")
                print("  1. Create free account: https://huggingface.co/join")
                print("  2. Get token: https://huggingface.co/settings/tokens")
                print("  3. Add to .env file in your user directory:")
                if sys.platform == "win32":
                    env_path = Path.home() / "AppData" / "Roaming" / "Kura" / ".env"
                else:
                    env_path = Path.home() / "Library" / "Application Support" / "Kura" / ".env"
                print(f"     {env_path}")
                print("  4. Add line: HF_TOKEN=your_token_here")
                print("="*70 + "\n")
                hf_token = None  # Use anonymous

            # Download with resume capability
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=hf_token,  # ✅ Use token if available
            )

            file_size = Path(model_path).stat().st_size / (1024**3)
            print(f"✅ Downloaded: {filename} ({file_size:.2f} GB)")
            return True

        except KeyboardInterrupt:
            print("\n⚠️  Download cancelled by user")
            return False
        except Exception as e:
            print(f"❌ Download failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def download_llm_model() -> bool:
    """Download Llama-3.1-8B-Instruct model."""
    model_dir = get_model_dir() / "Llama-3.1-8B-Instruct-GGUF"
    model_file = model_dir / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    # Skip if already exists and is complete
    if model_file.exists() and model_file.stat().st_size > 4_000_000_000:
        file_size = model_file.stat().st_size / (1024**3)
        print(f"\n✅ LLM model already downloaded: {model_file.name} ({file_size:.2f} GB)")
        print("   Skipping download.")
        return True

    print("\n" + "="*70)
    print("Downloading Llama-3.1-8B-Instruct (Medical AI Model)")
    print("="*70)
    print("Size: ~4.9 GB (one-time download)")
    print("This will take 5-15 minutes depending on your connection.")
    print("="*70)

    success = download_model_with_progress(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        local_dir=model_dir
    )

    return success

def download_whisper_model() -> bool:
    """Download Whisper medium model."""
    model_dir = get_model_dir() / "whisper"
    model_file = model_dir / "medium.pt"

    # Skip if already exists and is complete
    if model_file.exists() and model_file.stat().st_size > 1_000_000_000:
        file_size = model_file.stat().st_size / (1024**3)
        print(f"\n✅ Whisper model already downloaded: {model_file.name} ({file_size:.2f} GB)")
        print("   Skipping download.")
        return True

    print("\n" + "="*70)
    print("Downloading Whisper Medium (Speech Recognition)")
    print("="*70)
    print("Size: ~1.5 GB")
    print("="*70)

    try:
        import whisper
        model_dir.mkdir(parents=True, exist_ok=True)
        print("\n📥 Downloading Whisper medium model...")
        whisper.load_model("medium", device="cpu", download_root=str(model_dir))
        print("✅ Whisper model downloaded")
        return True
    except Exception as e:
        print(f"❌ Whisper download failed: {e}")
        return False

def ensure_models_available() -> bool:
    """
    Ensure all required models are available.
    Downloads them if missing. Called on first app launch.

    Returns:
        True if all models available, False if download failed or cancelled
    """
    global _shutdown_requested

    try:
        print("\n" + "="*70)
        print("🔍 CHECKING MODEL AVAILABILITY")
        print("="*70)

        model_dir = get_model_dir()
        print(f"Model directory: {model_dir}")
        print(f"Directory exists: {model_dir.exists()}")

        llm_exists = check_model_exists("llm")
        whisper_exists = check_model_exists("whisper")

        print(f"LLM model found: {llm_exists}")
        print(f"Whisper model found: {whisper_exists}")

        if llm_exists and whisper_exists:
            print("✅ All models already installed")
            print("="*70 + "\n")
            return True

        print("\n" + "="*70)
        print("🩺 KURA MEDICAL - FIRST TIME SETUP")
        print("="*70)
        print("Kura needs to download AI models for local processing.")
        print("This is a ONE-TIME download (~6 GB total).")
        print("All patient data stays on your computer - 100% DSGVO compliant.")
        print("="*70)

        # Check internet connection
        print("\n🌐 Checking internet connection...")
        try:
            import socket
            socket.create_connection(("huggingface.co", 443), timeout=5)
            print("✅ Internet connection OK")
        except Exception as e:
            print(f"\n❌ ERROR: No internet connection detected: {e}")
            print("Please connect to the internet and restart Kura.")
            return False

        success = True

        # Download LLM if missing
        if not llm_exists and not _shutdown_requested:
            print("\n📥 LLM model not found - starting download...")
            if not download_llm_model():
                if _shutdown_requested:
                    print("⚠️  LLM download cancelled - will resume next time")
                else:
                    print("❌ LLM download failed!")
                success = False
            else:
                print("✅ LLM download successful!")
        else:
            print("\n✅ LLM model already installed")

        # Download Whisper if missing
        if not whisper_exists and not _shutdown_requested:
            print("\n📥 Whisper model not found - starting download...")
            if not download_whisper_model():
                if _shutdown_requested:
                    print("⚠️  Whisper download cancelled - will resume next time")
                else:
                    print("❌ Whisper download failed!")
                success = False
            else:
                print("✅ Whisper download successful!")
        else:
            print("\n✅ Whisper model already installed")

        if _shutdown_requested:
            print("\n" + "="*70)
            print("⚠️  SETUP CANCELLED")
            print("="*70)
            print("\nDownload cancelled by user.")
            print("Partial downloads will resume next time you start Kura.\n")
            return False

        if success:
            print("\n" + "="*70)
            print("✅ SETUP COMPLETE - KURA IS READY!")
            print("="*70)
            print("\nAll models installed successfully.")
            print("Kura will now start. Future launches will be instant.\n")
        else:
            print("\n" + "="*70)
            print("❌ SETUP INCOMPLETE")
            print("="*70)
            print("\nSome models failed to download.")
            print("Please check your internet connection and try again.\n")

        return success

    except (KeyboardInterrupt, SystemExit):
        # Final catch-all for any cancellation
        print("\n\n" + "="*70)
        print("⚠️  SETUP CANCELLED - CLEANUP COMPLETE")
        print("="*70)
        print("\nDownload cancelled. Partial files saved for next time.")
        print("="*70 + "\n")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# macOS / MLX model support
# Models are in safetensors format from mlx-community, downloaded via
# snapshot_download (entire repo in one call, with resume support).
# ══════════════════════════════════════════════════════════════════════════════

# macOS model specs: (repo_id, local_subdir, min_size_bytes, description)
_MACOS_MODELS = [
    (
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "Meta-Llama-3.1-8B-Instruct-4bit",
        "model.safetensors",
        4_000_000_000,   # at least 4 GB
        "Llama 3.1-8B MLX (Medical AI, ~4.2 GB)",
    ),
    (
        "mlx-community/whisper-large-v3-turbo",
        "whisper-large-v3-turbo",
        "weights.safetensors",
        1_000_000_000,   # at least 1 GB
        "Whisper large-v3-turbo MLX (Speech recognition, ~1.5 GB)",
    ),
]


def check_macos_model_exists(local_subdir: str, key_file: str, min_size: int) -> bool:
    """Return True if the local MLX model directory is complete."""
    model_path = get_model_dir() / local_subdir / key_file
    if model_path.exists():
        size = model_path.stat().st_size
        if size >= min_size:
            print(f"✅ {local_subdir}: already installed ({size/1e9:.2f} GB)")
            return True
        print(f"⚠️  {local_subdir}: incomplete ({size/1e9:.2f} GB) — will re-download")
    return False


def download_macos_model(repo_id: str, local_subdir: str, description: str) -> bool:
    """Download a full HuggingFace repo (MLX safetensors) via snapshot_download."""
    with _download_lock:
        if not HF_HUB_AVAILABLE or snapshot_download is None:
            print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
            return False

        local_dir = get_model_dir() / local_subdir
        local_dir.mkdir(parents=True, exist_ok=True)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token or hf_token == "your_token_here":
            hf_token = None

        print(f"\n📥 Downloading {description}")
        print(f"   Source : {repo_id}")
        print(f"   Target : {local_dir}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=hf_token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "pytorch_model*"],
            )
            print(f"✅ {local_subdir} ready")
            return True
        except KeyboardInterrupt:
            print(f"\n⚠️  Download of {local_subdir} cancelled — will resume next time")
            return False
        except Exception as e:
            print(f"❌ Download failed for {local_subdir}: {e}")
            import traceback
            traceback.print_exc()
            return False


def ensure_models_available_macos() -> bool:
    """
    macOS equivalent of ensure_models_available().
    Downloads MLX LLM + Whisper safetensors on first launch.
    Subsequent launches skip instantly when files are already present.
    """
    global _shutdown_requested

    try:
        print("\n" + "="*70)
        print("🔍 CHECKING MODEL AVAILABILITY (macOS / MLX)")
        print("="*70)

        model_dir = get_model_dir()
        print(f"Model directory: {model_dir}")

        statuses = [
            check_macos_model_exists(sub, key, min_sz)
            for _, sub, key, min_sz, _ in _MACOS_MODELS
        ]

        if all(statuses):
            print("✅ All models already installed")
            print("="*70 + "\n")
            return True

        print("\n" + "="*70)
        print("🩺 KURA MEDICAL — FIRST TIME SETUP (macOS)")
        print("="*70)
        print("Kura needs to download AI models for local processing.")
        print("This is a ONE-TIME download (~5.7 GB total).")
        print("All patient data stays on your Mac — 100% DSGVO compliant.")
        print("="*70)

        # Check internet
        print("\n🌐 Checking internet connection...")
        try:
            import socket
            socket.create_connection(("huggingface.co", 443), timeout=5)
            print("✅ Internet connection OK")
        except Exception as e:
            print(f"\n❌ No internet connection: {e}")
            print("Please connect to the internet and restart Kura.")
            return False

        success = True
        for (repo_id, local_subdir, key_file, min_size, description), already_ok in zip(_MACOS_MODELS, statuses):
            if already_ok or _shutdown_requested:
                if already_ok:
                    print(f"\n✅ {local_subdir} already installed — skipping")
                continue
            print(f"\n📥 {local_subdir} not found — starting download...")
            if not download_macos_model(repo_id, local_subdir, description):
                if _shutdown_requested:
                    print(f"⚠️  {local_subdir} download cancelled — will resume next time")
                else:
                    print(f"❌ {local_subdir} download failed!")
                success = False
            else:
                print(f"✅ {local_subdir} download complete!")

        if _shutdown_requested:
            print("\n⚠️  SETUP CANCELLED — partial downloads saved for next time")
            return False

        if success:
            print("\n" + "="*70)
            print("✅ SETUP COMPLETE — KURA IS READY!")
            print("="*70 + "\n")
        else:
            print("\n❌ SETUP INCOMPLETE — check internet and restart\n")

        return success

    except (KeyboardInterrupt, SystemExit):
        print("\n⚠️  SETUP CANCELLED — partial files saved for next time\n")
        return False


if __name__ == "__main__":
    success = ensure_models_available()
    sys.exit(0 if success else 1)

