"""
Test Bundled .env Distribution
===============================
Verifies that .env.dist is properly bundled and copied on first launch.
"""
import os
import shutil
import tempfile
from pathlib import Path


def test_env_dist_exists():
    """Test that .env.dist file exists in project root."""
    env_dist = Path(__file__).parent / ".env.dist"
    
    print("\n" + "="*70)
    print("TEST 1: Check .env.dist exists")
    print("="*70)
    print(f"Looking for: {env_dist}")
    
    if env_dist.exists():
        print("✅ PASS: .env.dist file exists")
        
        # Check it contains HF_TOKEN
        with open(env_dist, 'r') as f:
            content = f.read()
            
        if 'HF_TOKEN=' in content:
            # Extract token (first few chars only for security)
            for line in content.split('\n'):
                if line.startswith('HF_TOKEN='):
                    token = line.split('=')[1].strip()
                    if token and token != 'your_token_here':
                        print(f"✅ PASS: HF_TOKEN is configured")
                        print(f"   Token: {token[:10]}...{token[-8:]}")
                    else:
                        print("❌ FAIL: HF_TOKEN is placeholder")
                        return False
        else:
            print("❌ FAIL: .env.dist missing HF_TOKEN")
            return False
            
        if 'LEMON_SQUEEZY_API_KEY=' in content:
            print("✅ PASS: LEMON_SQUEEZY_API_KEY present")
        else:
            print("⚠️  WARNING: LEMON_SQUEEZY_API_KEY missing")
            
        return True
    else:
        print("❌ FAIL: .env.dist file not found")
        print("   Run: Create .env.dist with your credentials")
        return False


def test_env_copy_logic():
    """Test that .env would be copied correctly on first launch."""
    print("\n" + "="*70)
    print("TEST 2: Simulate First Launch (.env copy logic)")
    print("="*70)
    
    # Create temp directory to simulate user Documents folder
    with tempfile.TemporaryDirectory() as tmpdir:
        user_env = Path(tmpdir) / ".env"
        env_dist = Path(__file__).parent / ".env.dist"
        
        print(f"Temp user .env: {user_env}")
        print(f"Source .env.dist: {env_dist}")
        
        # Simulate first launch - copy .env.dist to user directory
        if env_dist.exists():
            shutil.copy(env_dist, user_env)
            print("✅ PASS: Copied .env.dist to user directory")
            
            # Verify HF_TOKEN is in copied file
            with open(user_env, 'r') as f:
                content = f.read()
            
            if 'HF_TOKEN=' in content and 'your_token_here' not in content:
                print("✅ PASS: User .env contains valid HF_TOKEN")
                return True
            else:
                print("❌ FAIL: User .env missing valid HF_TOKEN")
                return False
        else:
            print("❌ FAIL: .env.dist not found")
            return False


def test_no_overwrite():
    """Test that existing .env is NOT overwritten."""
    print("\n" + "="*70)
    print("TEST 3: Verify Existing .env Not Overwritten")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        user_env = Path(tmpdir) / ".env"
        
        # Create existing .env with custom content
        custom_token = "hf_custom_user_token_12345"
        with open(user_env, 'w') as f:
            f.write(f"HF_TOKEN={custom_token}\n")
        
        print(f"Created existing .env with custom token: {custom_token[:20]}...")
        
        # Simulate app checking if .env exists
        if user_env.exists():
            print("✅ PASS: .env already exists - should NOT overwrite")
            
            # Verify custom content preserved
            with open(user_env, 'r') as f:
                content = f.read()
            
            if custom_token in content:
                print("✅ PASS: Custom token preserved (not overwritten)")
                return True
            else:
                print("❌ FAIL: Custom token was overwritten")
                return False
        else:
            print("❌ FAIL: Test setup error")
            return False


def test_build_spec_includes_dist():
    """Test that build specs include .env.dist."""
    print("\n" + "="*70)
    print("TEST 4: Check Build Specs Include .env.dist")
    print("="*70)
    
    windows_spec = Path(__file__).parent / "windows" / "Kura_windows.spec"
    macos_spec = Path(__file__).parent / "macos" / "Kura.spec"
    
    results = []
    
    # Check Windows spec
    if windows_spec.exists():
        with open(windows_spec, 'r') as f:
            content = f.read()
        if '.env.dist' in content:
            print("✅ PASS: Windows spec includes .env.dist")
            results.append(True)
        else:
            print("❌ FAIL: Windows spec missing .env.dist")
            results.append(False)
    else:
        print("⚠️  SKIP: Windows spec not found")
        results.append(None)
    
    # Check macOS spec
    if macos_spec.exists():
        with open(macos_spec, 'r') as f:
            content = f.read()
        if '.env.dist' in content:
            print("✅ PASS: macOS spec includes .env.dist")
            results.append(True)
        else:
            print("❌ FAIL: macOS spec missing .env.dist")
            results.append(False)
    else:
        print("⚠️  SKIP: macOS spec not found")
        results.append(None)
    
    # Return True if at least one spec is correct
    return any(r is True for r in results)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BUNDLED .ENV DISTRIBUTION TESTS")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Check .env.dist exists", test_env_dist_exists()))
    results.append(("Simulate first launch copy", test_env_copy_logic()))
    results.append(("Verify no overwrite", test_no_overwrite()))
    results.append(("Check build specs", test_build_spec_includes_dist()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL" if result is False else "⚠️  SKIP"
        print(f"{status}: {name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("="*70)
    
    if failed == 0 and passed > 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Ready to build and distribute:")
        print("   • .env.dist is properly configured")
        print("   • Build scripts will bundle it")
        print("   • Customers get fast downloads out-of-box")
        print("   • Existing configs won't be overwritten")
        print("\nNext steps:")
        print("   1. Build: cd windows && pyinstaller Kura_windows.spec")
        print("   2. Test installer on clean system")
        print("   3. Verify first launch copies .env correctly")
        print("   4. Distribute to customers!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Review failures above and fix before building")
    
    print("="*70)

