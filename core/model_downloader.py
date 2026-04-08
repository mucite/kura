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
"""
import os
import sys
from pathlib import Path

def get_model_dir() -> Path:
    """Get the models directory path."""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        base = Path(os.path.dirname(sys.executable))
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
    """Download model with progress bar."""
    try:
        from huggingface_hub import hf_hub_download
        import requests

        print(f"\n📥 Downloading {filename}...")
        print(f"Source: {repo_id}")
        print(f"Target: {local_dir}")

        # Create directory
        local_dir.mkdir(parents=True, exist_ok=True)

        # Get HF_TOKEN from environment (if available)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        if hf_token and hf_token != "your_token_here":
            print("✅ Using HF_TOKEN for authenticated download (faster)")
        else:
            print("⚠️  No HF_TOKEN - using anonymous download (slower)")
            print("   To speed up: Get free token from https://huggingface.co/settings/tokens")
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

    except Exception as e:
        print(f"❌ Download failed: {e}")
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
        True if all models available, False if download failed
    """
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
    if not llm_exists:
        print("\n📥 LLM model not found - starting download...")
        if not download_llm_model():
            success = False
            print("❌ LLM download failed!")
        else:
            print("✅ LLM download successful!")
    else:
        print("\n✅ LLM model already installed")

    # Download Whisper if missing
    if not whisper_exists:
        print("\n📥 Whisper model not found - starting download...")
        if not download_whisper_model():
            success = False
            print("❌ Whisper download failed!")
        else:
            print("✅ Whisper download successful!")
    else:
        print("\n✅ Whisper model already installed")

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

if __name__ == "__main__":
    success = ensure_models_available()
    sys.exit(0 if success else 1)

