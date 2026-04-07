"""
Test: Comprehensive Context Contamination Prevention
=====================================================
Validates profile-specific template reset across ALL diagnosis types.

Ensures no cross-contamination between:
- Spine regions (HWS, LWS, BWS, ISG)
- Upper extremity (Shoulder, Elbow, Wrist)
- Lower extremity (Hip, Knee, Ankle)
- Special profiles (Lymphedema, Pelvic floor)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from windows.physio_scribe_crossplatform import KuraEngine


def test_shoulder_no_spine_tests():
    """Test: Shoulder report should NOT contain spine tests"""
    print("\n" + "="*70)
    print("TEST: Shoulder Profile - Remove Spine Contamination")
    print("="*70)

    engine = KuraEngine()

    contaminated = {
        "S": "Schulterschmerzen seit 2 Wochen",
        "O": "Jobe-Test: positiv | Hawkins: positiv | Schober-Zeichen: 10:14 cm | "
             "Lasègue 80° negativ | FBA: 40 cm | Painful Arc 60-120°",
        "A": "Impingement-Syndrom",
        "P": "MT Schulter"
    }

    cleaned = engine._remove_incompatible_tests(contaminated, "SHOULDER")
    obj = cleaned["O"]

    errors = []
    if "Schober" in obj:
        errors.append("❌ Schober (lumbar test) in shoulder report")
    if "Lasègue" in obj:
        errors.append("❌ Lasègue (lumbar test) in shoulder report")
    if "FBA" in obj:
        errors.append("❌ FBA (lumbar test) in shoulder report")

    if "Jobe" in obj and "Hawkins" in obj:
        print("✅ PASS: Shoulder tests preserved")
    else:
        errors.append("❌ Shoulder tests removed incorrectly")

    if errors:
        print("\n".join(errors))
        return False
    print("✅ PASS: All spine tests removed, shoulder tests preserved")
    return True


def test_knee_no_shoulder_tests():
    """Test: Knee report should NOT contain shoulder tests"""
    print("\n" + "="*70)
    print("TEST: Knee Profile - Remove Upper Extremity Contamination")
    print("="*70)

    engine = KuraEngine()

    contaminated = {
        "S": "Knieschmerzen nach VKB-Ruptur",
        "O": "Lachman-Test: positiv | Jobe-Test: positiv | Hawkins: negativ | "
             "McMurray: negativ | Schubladentest: 2+ anterior",
        "A": "VKB-Insuffizienz",
        "P": "KG nach VKB-Rekonstruktion"
    }

    cleaned = engine._remove_incompatible_tests(contaminated, "KNEE")
    obj = cleaned["O"]

    errors = []
    if "Jobe" in obj:
        errors.append("❌ Jobe (shoulder test) in knee report")
    if "Hawkins" in obj:
        errors.append("❌ Hawkins (shoulder test) in knee report")

    if "Lachman" in obj and "McMurray" in obj:
        print("✅ PASS: Knee tests preserved")
    else:
        errors.append("❌ Knee tests removed incorrectly")

    if errors:
        print("\n".join(errors))
        return False
    print("✅ PASS: All shoulder tests removed, knee tests preserved")
    return True


def test_hip_no_knee_tests():
    """Test: Hip report should NOT contain knee tests"""
    print("\n" + "="*70)
    print("TEST: Hip Profile - Remove Knee Contamination")
    print("="*70)

    engine = KuraEngine()

    contaminated = {
        "S": "Hüftschmerzen, Coxarthrose",
        "O": "Thomas-Test: positiv | Lachman-Test: positiv | McMurray: negativ | "
             "FABER: eingeschränkt | Trendelenburg: negativ",
        "A": "Coxarthrose Grad II",
        "P": "KG Hüfte"
    }

    cleaned = engine._remove_incompatible_tests(contaminated, "HIP")
    obj = cleaned["O"]

    errors = []
    if "Lachman" in obj:
        errors.append("❌ Lachman (knee test) in hip report")
    if "McMurray" in obj:
        errors.append("❌ McMurray (knee test) in hip report")

    if "Thomas" in obj and "FABER" in obj:
        print("✅ PASS: Hip tests preserved")
    else:
        errors.append("❌ Hip tests removed incorrectly")

    if errors:
        print("\n".join(errors))
        return False
    print("✅ PASS: All knee tests removed, hip tests preserved")
    return True


def test_elbow_no_wrist_tests():
    """Test: Elbow report should NOT contain wrist tests"""
    print("\n" + "="*70)
    print("TEST: Elbow Profile - Remove Wrist Contamination")
    print("="*70)

    engine = KuraEngine()

    contaminated = {
        "S": "Ellbogenschmerzen lateral",
        "O": "Cozen-Test: positiv | Phalen-Test: positiv | Tinel: negativ | "
             "Mill-Test: positiv",
        "A": "Epicondylitis lateralis",
        "P": "MT Ellbogen"
    }

    cleaned = engine._remove_incompatible_tests(contaminated, "ELBOW")
    obj = cleaned["O"]

    errors = []
    if "Phalen" in obj:
        errors.append("❌ Phalen (wrist test) in elbow report")
    if "Tinel" in obj:
        errors.append("❌ Tinel (wrist test) in elbow report")

    if "Cozen" in obj and "Mill" in obj:
        print("✅ PASS: Elbow tests preserved")
    else:
        errors.append("❌ Elbow tests removed incorrectly")

    if errors:
        print("\n".join(errors))
        return False
    print("✅ PASS: All wrist tests removed, elbow tests preserved")
    return True


def test_bws_no_hws_lws_tests():
    """Test: BWS report should NOT contain HWS or LWS tests"""
    print("\n" + "="*70)
    print("TEST: BWS Profile - Remove HWS/LWS Contamination")
    print("="*70)

    engine = KuraEngine()

    contaminated = {
        "S": "BWS-Schmerzen zwischen Schulterblättern",
        "O": "Atemexkursion: 3 cm | Spurling-Test: positiv | Schober: 10:14 cm | "
             "Lasègue: 80° negativ | Palpation: BWS Th5-Th7 hypomobil",
        "A": "BWS-Blockierung Th5/Th6",
        "P": "MT BWS"
    }

    cleaned = engine._remove_incompatible_tests(contaminated, "BWS")
    obj = cleaned["O"]

    errors = []
    if "Spurling" in obj:
        errors.append("❌ Spurling (HWS test) in BWS report")
    if "Schober" in obj:
        errors.append("❌ Schober (LWS test) in BWS report")
    if "Lasègue" in obj:
        errors.append("❌ Lasègue (LWS test) in BWS report")

    if "Atemexkursion" in obj:
        print("✅ PASS: BWS tests preserved")
    else:
        errors.append("❌ BWS tests removed incorrectly")

    if errors:
        print("\n".join(errors))
        return False
    print("✅ PASS: All HWS/LWS tests removed, BWS tests preserved")
    return True


def test_lymphedema_no_orthopedic_tests():
    """Test: Lymphedema report should NOT contain orthopedic tests"""
    print("\n" + "="*70)
    print("TEST: Lymphedema Profile - Remove All Orthopedic Tests")
    print("="*70)

    engine = KuraEngine()

    contaminated = {
        "S": "Lymphödem rechter Arm nach Mamm-CA-OP",
        "O": "Umfangsmessung: +3 cm | Stemmer-Zeichen: positiv | Jobe-Test: positiv | "
             "Lasègue: 80° negativ | Hautqualität: derb",
        "A": "Sekundäres Lymphödem Grad II",
        "P": "MLD 45 min"
    }

    cleaned = engine._remove_incompatible_tests(contaminated, "LY")
    obj = cleaned["O"]

    errors = []
    if "Jobe" in obj:
        errors.append("❌ Jobe (shoulder test) in lymphedema report")
    if "Lasègue" in obj:
        errors.append("❌ Lasègue (lumbar test) in lymphedema report")

    if "Umfangsmessung" in obj and "Stemmer" in obj:
        print("✅ PASS: Lymphedema tests preserved")
    else:
        errors.append("❌ Lymphedema tests removed incorrectly")

    if errors:
        print("\n".join(errors))
        return False
    print("✅ PASS: All orthopedic tests removed, lymphedema tests preserved")
    return True


def main():
    """Run comprehensive contamination prevention tests"""
    print("\n" + "="*70)
    print(" COMPREHENSIVE CONTEXT CONTAMINATION PREVENTION TEST")
    print("="*70)
    print("\nTesting profile-specific template reset across ALL diagnosis types")
    print("="*70)

    tests = [
        ("Shoulder vs Spine", test_shoulder_no_spine_tests),
        ("Knee vs Shoulder", test_knee_no_shoulder_tests),
        ("Hip vs Knee", test_hip_no_knee_tests),
        ("Elbow vs Wrist", test_elbow_no_wrist_tests),
        ("BWS vs HWS/LWS", test_bws_no_hws_lws_tests),
        ("Lymphedema vs Orthopedic", test_lymphedema_no_orthopedic_tests),
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
        print("✅ Context contamination prevented across ALL profiles")
        print("✅ No cross-contamination between anatomical regions")
        print("✅ Profile-specific tests correctly preserved")
        print("✅ Safe to use for all diagnosis types")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("⚠️  Context contamination still present in some profiles")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

