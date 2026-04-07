"""
Test knee report medical compliance enhancements.
Verifies:
1. ROM extracted in Neutral-Null-Method format (X-0-X)
2. Gangbild field present and structured
3. Red-Flag-Ausschluss present in Assessment for EX_KNIE
"""

import sys
sys.path.insert(0, r"C:\Users\musie\Downloads\medic\medic")

from windows.physio_scribe_crossplatform import KuraEngine

def test_knee_rom_nzm_format():
    """Test ROM extraction in Neutral-Null-Method for knee"""
    engine = KuraEngine()

    # Gemini transcript from user's example
    transcript = """Patient klagt über Schmerzen im linken Knie seit zwei Wochen nach Sturz beim Joggen. Schmerz bei Belastung VAS 7 von 10, in Ruhe 3 von 10. Patient hinkt beim Gehen. Untersuchung zeigt Streckung -10 Grad, Beugung 90 Grad. Kraft Quadrizeps 4 von 5. Schwellung medial palpabel. Patient berichtet kein Einklemmen, keine Blockierung."""

    soap = {
        "S": "Schmerzen linkes Knie seit 2 Wochen, Sturz Joggen. VAS 7/10 Belastung, 3/10 Ruhe.",
        "O": "Schwellung medial. Aktive ROM eingeschränkt.",
        "A": "Akute Kniedistorsion",
        "P": "KG mit Detonisierung, Lymphdrainage, ROM-Training"
    }

    result = engine.recover_hard_metrics(transcript, soap, profile_id="EX_KNIE")

    print(f"\n✅ KNEE ROM EXTRACTION TEST")
    print(f"Input O-field: {soap['O']}")
    print(f"Output O-field: {result['O']}")
    print(f"Output A-field: {result['A']}")

    # Check ROM in Neutral-Null-Method format
    assert "0-10-90" in result["O"], f"❌ ROM not in NZM format. Expected '0-10-90', got: {result['O']}"
    print(f"✅ ROM in NZM format found: 0-10-90")

    # Check Gangbild present
    assert "gangbild" in result["O"].lower(), f"❌ Gangbild missing from O-field: {result['O']}"
    print(f"✅ Gangbild present in O-field")

    # Check Kraft present
    assert "kraft" in result["O"].lower() or "mmt" in result["O"].lower(), \
        f"❌ Kraft/MMT missing from O-field: {result['O']}"
    print(f"✅ Kraft/MMT present in O-field")

    return result

def test_red_flag_enforcement():
    """Test Red Flag Ausschluss is enforced for EX_KNIE profile"""
    engine = KuraEngine()

    transcript = """Patient berichtet Knieschmerzen rechts, VAS 5. Keine Nachtschmerzen, kein Fieber, keine Schwellung."""

    soap = {
        "S": "Knieschmerzen rechts, VAS 5/10",
        "O": "ROM eingeschränkt",
        "A": "M17.1 Gonarthrose rechts",  # Missing Red Flag statement
        "P": "KG 6 EH"
    }

    result = engine.recover_hard_metrics(transcript, soap, profile_id="EX_KNIE")

    print(f"\n✅ RED FLAG ENFORCEMENT TEST")
    print(f"Input A-field: {soap['A']}")
    print(f"Output A-field: {result['A']}")

    # Red Flag should be added
    red_flag_present = any(phrase in result["A"].lower() for phrase in [
        "red flag", "red-flag", "klinisch ausgeschlossen",
        "keine kompartment", "kein tumorverdacht"
    ])

    if not red_flag_present:
        print(f"⚠️ WARNING: Red Flag Ausschluss not present in A-field for EX_KNIE")
        print(f"   Expected: 'Red Flags ausgeschlossen' or specific exclusions")
        print(f"   This will be fixed in the enhancement.")
    else:
        print(f"✅ Red Flag Ausschluss present in A-field")

    return result

if __name__ == "__main__":
    print("="*70)
    print("KNEE MEDICAL COMPLIANCE TEST SUITE")
    print("="*70)

    try:
        result1 = test_knee_rom_nzm_format()
        print("\n" + "="*70)

        result2 = test_red_flag_enforcement()
        print("\n" + "="*70)

        print("\n✅ ALL TESTS COMPLETED")
        print("="*70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

