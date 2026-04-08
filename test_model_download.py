"""
Test Model Downloader - Verify auto-download works
Run this to test if the model download logic works correctly
"""
import sys
import os

# Add parent directory to path (same as main app does)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

def test_model_downloader():
    print("\n" + "="*70)
    print("TESTING MODEL DOWNLOADER")
    print("="*70 + "\n")

    # Test 1: Import check
    print("Test 1: Checking if model_downloader can be imported...")
    try:
        from core.model_downloader import ensure_models_available, get_model_dir
        print("✅ PASS: model_downloader imported successfully")
    except ImportError as e:
        print(f"❌ FAIL: Cannot import model_downloader: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Unexpected error importing: {type(e).__name__}: {e}")
        return False

    # Test 2: Model directory check
    print("\nTest 2: Checking model directory...")
    try:
        model_dir = get_model_dir()
        print(f"   Model directory: {model_dir}")
        print(f"   Directory exists: {model_dir.exists()}")
        print("✅ PASS: Model directory accessible")
    except Exception as e:
        print(f"❌ FAIL: Cannot access model directory: {e}")
        return False

    # Test 3: Check model availability (don't download, just check)
    print("\nTest 3: Checking if ensure_models_available can run...")
    try:
        from core.model_downloader import check_model_exists
        llm_exists = check_model_exists("llm")
        whisper_exists = check_model_exists("whisper")

        print(f"   LLM model exists: {llm_exists}")
        print(f"   Whisper model exists: {whisper_exists}")

        if llm_exists and whisper_exists:
            print("✅ PASS: All models already downloaded")
        else:
            print("⚠️  WARNING: Models missing - would trigger download on real launch")
            print("   Missing models:")
            if not llm_exists:
                print("   - LLM (Llama-3.1-8B)")
            if not whisper_exists:
                print("   - Whisper (Speech Recognition)")

        print("✅ PASS: Model check functions work correctly")
    except Exception as e:
        print(f"❌ FAIL: Error checking models: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Simulate what happens in KuraEngine._init_models
    print("\nTest 4: Simulating KuraEngine model loading...")
    try:
        from core.model_downloader import ensure_models_available
        print("   Calling ensure_models_available()...")
        result = ensure_models_available()
        if result:
            print("✅ PASS: ensure_models_available() returned True (models ready)")
        else:
            print("❌ FAIL: ensure_models_available() returned False (download failed)")
            return False
    except Exception as e:
        print(f"❌ FAIL: Error in ensure_models_available: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Model downloader works correctly!")
    print("="*70)
    print("\nThe model auto-download should work when you run the bundled app.")
    print("If it doesn't, check the console output for error messages.\n")
    return True


if __name__ == "__main__":
    print("\nKura Model Downloader Test")
    print(f"Python version: {sys.version}")
    print(f"Script location: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Working directory: {os.getcwd()}")

    success = test_model_downloader()

    if not success:
        print("\n❌ Some tests failed - model auto-download may not work!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed - model auto-download should work!")
        sys.exit(0)

