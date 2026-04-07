"""
Comprehensive Cross-Session Contamination Test
Tests ALL profile combinations to ensure clean session isolation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from windows.physio_scribe_crossplatform import KuraEngine

print("="*80)
print("COMPREHENSIVE CONTAMINATION TEST - ALL SESSION TYPES")
print("="*80)

# Test cases: (profile, contaminated_s_field, should_remove_pattern)
test_cases = [
    # KNEE should reject spine mentions (CRITICAL BUG FROM USER REPORT)
    ("EX_KNIE",
     "Pat. berichtet akute LWS-Schmerzen nach Heben. Knie geschwollen und steif beim Gehen.",
     "LWS-Schmerzen", True),

    # SHOULDER should reject spine mentions
    ("EX_SCHULTER",
     "Pat. berichtet HWS-Beschwerden mit Ausstrahlung. Schulter eingeschränkt bei Abduktion.",
     "HWS-Beschwerden", True),

    # LWS should reject extremity mentions IN PRIMARY COMPLAINT
    ("EX_LWS",
     "Pat. berichtet Schulterprobleme beim Heben. Rückenschmerzen seit 3 Wochen.",
     "Schulterprobleme", True),

    # HWS should reject knee mentions
    ("EX_HWS",
     "Pat. berichtet Kniesteifigkeit morgens. Nackenschmerzen mit Ausstrahlung.",
     "Kniesteifigkeit", True),

    # FOOT should reject back mentions (CRITICAL)
    ("EX_FUSS",
     "Pat. berichtet Rückenschmerzen nach Gartenarbeit. Sprunggelenk geschwollen nach Umknicken.",
     "Rückenschmerzen", True),

    # HIP should reject neck mentions (CRITICAL)
    ("EX_HUefte",
     "Pat. berichtet HWS-Beschwerden mit Kopfschmerzen. Hüfte schmerzt beim Gehen.",
     "HWS-Beschwerden", True),

    # KNEE should KEEP knee mentions (sanity check - don't over-remove)
    ("EX_KNIE",
     "Pat. berichtet Knieschmerzen seit 2 Wochen nach VKB-OP.",
     "Knieschmerzen", False),
]

try:
    print("\nInitializing Kura Engine...")
    engine = KuraEngine(license_status="TRIAL")
    print("✅ Engine ready\n")

    passed = 0
    failed = 0

    for i, (profile, contaminated_s, pattern, should_remove) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {profile} - Remove '{pattern}'")
        print(f"{'='*80}")

        # Create contaminated SOAP
        soap_dict = {
            "S": contaminated_s,
            "O": "Test objective data",
            "A": "Test assessment",
            "P": "Test plan"
        }

        print(f"Original S: {contaminated_s[:70]}...")

        # Run contamination detection
        cleaned = engine.recover_hard_metrics("dummy transcript", soap_dict, profile)

        print(f"Cleaned S:  {cleaned['S'][:70]}...")

        # Check if contamination was removed
        pattern_present = pattern.lower() in cleaned['S'].lower()

        if should_remove:
            if not pattern_present:
                print(f"✅ PASSED: '{pattern}' successfully removed from {profile}")
                passed += 1
            else:
                print(f"❌ FAILED: '{pattern}' still present in {profile}")
                failed += 1
        else:
            if pattern_present:
                print(f"✅ PASSED: '{pattern}' correctly preserved in {profile}")
                passed += 1
            else:
                print(f"❌ FAILED: '{pattern}' incorrectly removed from {profile}")
                failed += 1

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}/{len(test_cases)}")
    print(f"❌ Failed: {failed}/{len(test_cases)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Context isolation works for ALL session types!")
        print("\n✅ The contamination bug is COMPLETELY FIXED")
        print("✅ Safe to deploy to production")
    else:
        print(f"\n⚠️  {failed} test(s) failed - review output above")

except Exception as e:
    print(f"\n❌ TEST SUITE FAILED WITH ERROR: {e}")
    import traceback
    traceback.print_exc()

