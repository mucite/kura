"""
Test: Universal ROM Auto-Formatting
====================================
Validates that ROM measurements are auto-formatted to Neutral-Null-Method (X-0-X)
for ALL anatomical regions, not just HWS.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from windows.physio_scribe_crossplatform import KuraEngine


def test_hws_rotation_formatting():
    """Test: HWS rotation → 30-0-60 format"""
    print("\n" + "="*70)
    print("TEST 1: HWS Rotation Auto-Formatting")
    print("="*70)
    
    engine = KuraEngine()
    transcript = "Die Rotation nach rechts ist bei etwa 30 Grad, nach links 60 Grad."
    
    soap = {"S": "", "O": "Spurling-Test: positiv", "A": "M54.2", "P": ""}
    result = engine.recover_hard_metrics(transcript, soap, profile_id="EX_HWS")
    
    obj = result["O"]
    print(f"Input: {transcript}")
    print(f"Output: {obj}")
    
    if "30-0-60" in obj and "ROM HWS (Rotation)" in obj:
        print("✅ PASS: HWS rotation formatted to 30-0-60")
        return True
    else:
        print(f"❌ FAIL: Expected '30-0-60', got: {obj}")
        return False


def test_shoulder_abduction_formatting():
    """Test: Shoulder abduction → X-0-X format"""
    print("\n" + "="*70)
    print("TEST 2: Shoulder Abduction Auto-Formatting")
    print("="*70)
    
    engine = KuraEngine()
    transcript = "Abduktion 90 Grad, Adduktion 30 Grad."
    
    soap = {"S": "Schulterschmerzen", "O": "Jobe-Test: positiv", "A": "M75.4", "P": ""}
    result = engine.recover_hard_metrics(transcript, soap, profile_id="EX_SCHULTER")
    
    obj = result["O"]
    print(f"Input: {transcript}")
    print(f"Output: {obj}")
    
    if "90-0-30" in obj and "Abd/Add" in obj:
        print("✅ PASS: Shoulder Abd/Add formatted to 90-0-30")
        return True
    else:
        print(f"❌ FAIL: Expected Abd/Add '90-0-30', got: {obj}")
        return False


def test_hip_flexion_formatting():
    """Test: Hip flexion/extension → X-0-X format"""
    print("\n" + "="*70)
    print("TEST 3: Hip Flexion/Extension Auto-Formatting")
    print("="*70)
    
    engine = KuraEngine()
    transcript = "Extension 10 Grad, Flexion 110 Grad."
    
    soap = {"S": "Hüftschmerzen", "O": "Thomas-Test: positiv", "A": "M16.1", "P": ""}
    result = engine.recover_hard_metrics(transcript, soap, profile_id="EX_HUefte")
    
    obj = result["O"]
    print(f"Input: {transcript}")
    print(f"Output: {obj}")
    
    if "10-0-110" in obj and ("ROM Hüfte (Ex/Flex)" in obj or "Ex/Flex" in obj):
        print("✅ PASS: Hip Ex/Flex formatted to 10-0-110")
        return True
    else:
        print(f"❌ FAIL: Expected Ex/Flex '10-0-110', got: {obj}")
        return False


def test_ankle_dorsiflexion_formatting():
    """Test: Ankle dorsi/plantarflexion → X-0-X format"""
    print("\n" + "="*70)
    print("TEST 4: Ankle Dorsi/Plantarflexion Auto-Formatting")
    print("="*70)
    
    engine = KuraEngine()
    transcript = "Dorsalextension 15 Grad, Plantarflexion 40 Grad."
    
    soap = {"S": "Sprunggelenkschmerzen", "O": "Schwellung Malleolus", "A": "S93.4", "P": ""}
    result = engine.recover_hard_metrics(transcript, soap, profile_id="EX_FUSS")
    
    obj = result["O"]
    print(f"Input: {transcript}")
    print(f"Output: {obj}")
    
    if "15-0-40" in obj and ("Dorsi/Plantar" in obj or "OSG" in obj):
        print("✅ PASS: Ankle Dorsi/Plantar formatted to 15-0-40")
        return True
    else:
        print(f"❌ FAIL: Expected Dorsi/Plantar '15-0-40', got: {obj}")
        return False


def test_elbow_pronation_formatting():
    """Test: Elbow pronation/supination → X-0-X format"""
    print("\n" + "="*70)
    print("TEST 5: Elbow Pronation/Supination Auto-Formatting")
    print("="*70)
    
    engine = KuraEngine()
    transcript = "Pronation 80 Grad, Supination 80 Grad."
    
    soap = {"S": "Ellbogenschmerzen", "O": "Cozen-Test: positiv", "A": "M77.1", "P": ""}
    result = engine.recover_hard_metrics(transcript, soap, profile_id="ELBOW")
    
    obj = result["O"]
    print(f"Input: {transcript}")
    print(f"Output: {obj}")
    
    if "80-0-80" in obj and ("Pro/Sup" in obj or "Unterarm" in obj):
        print("✅ PASS: Elbow Pro/Sup formatted to 80-0-80")
        return True
    else:
        print(f"❌ FAIL: Expected Pro/Sup '80-0-80', got: {obj}")
        return False


def test_wrist_radial_ulnar_formatting():
    """Test: Wrist radial/ulnar deviation → X-0-X format"""
    print("\n" + "="*70)
    print("TEST 6: Wrist Radial/Ulnar Deviation Auto-Formatting")
    print("="*70)
    
    engine = KuraEngine()
    transcript = "Radialabduktion 20 Grad, Ulnarabduktion 30 Grad."
    
    soap = {"S": "Handgelenkschmerzen", "O": "Phalen-Test: positiv", "A": "M19.04", "P": ""}
    result = engine.recover_hard_metrics(transcript, soap, profile_id="HAND")
    
    obj = result["O"]
    print(f"Input: {transcript}")
    print(f"Output: {obj}")
    
    if "20-0-30" in obj and "Rad/Uln" in obj:
        print("✅ PASS: Wrist Rad/Uln formatted to 20-0-30")
        return True
    else:
        print(f"❌ FAIL: Expected Rad/Uln '20-0-30', got: {obj}")
        return False


def test_lws_no_rom_degrees():
    """Test: LWS should use FBA/Schober, NOT degrees"""
    print("\n" + "="*70)
    print("TEST 7: LWS Uses FBA/Schober (NOT degrees)")
    print("="*70)
    
    engine = KuraEngine()
    transcript = "FBA 40 cm, Schober 10 zu 14 cm."
    
    soap = {"S": "Rückenschmerzen", "O": "", "A": "M54.5", "P": ""}
    result = engine.recover_hard_metrics(transcript, soap, profile_id="EX_LWS")
    
    obj = result["O"]
    print(f"Input: {transcript}")
    print(f"Output: {obj}")
    
    has_fba = "FBA: 40 cm" in obj or "FBA 40" in obj
    has_schober = "Schober" in obj
    no_degrees = "ROM LWS" not in obj or "0-0-" not in obj  # LWS shouldn't use degree ROM
    
    if has_fba and has_schober and no_degrees:
        print("✅ PASS: LWS uses FBA/Schober (not degrees)")
        return True
    else:
        print(f"❌ FAIL: LWS should use FBA/Schober. Got: {obj}")
        return False


def main():
    """Run all ROM auto-formatting tests"""
    print("\n" + "="*70)
    print(" UNIVERSAL ROM AUTO-FORMATTING TEST SUITE")
    print("="*70)
    print("\nValidates that ROM measurements are auto-formatted to X-0-X")
    print("for ALL joints (not just HWS)")
    print("="*70)
    
    tests = [
        ("HWS Rotation", test_hws_rotation_formatting),
        ("Shoulder Abd/Add", test_shoulder_abduction_formatting),
        ("Hip Flex/Ext", test_hip_flexion_formatting),
        ("Ankle Dorsi/Plantar", test_ankle_dorsiflexion_formatting),
        ("Elbow Pro/Sup", test_elbow_pronation_formatting),
        ("Wrist Rad/Uln", test_wrist_radial_ulnar_formatting),
        ("LWS FBA/Schober", test_lws_no_rom_degrees),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ ROM auto-formatting works for ALL joints")
        print("✅ Neutral-Null-Method (X-0-X) applied universally")
        print("✅ GKV billing-compliant format")
        print("✅ Ready for production")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("⚠️  ROM formatting needs fixes")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

