"""
Test EX6 (Hand/Wrist) validation for grip strength and VAS pain scores.
Ensures the validator correctly detects:
1. Jamar grip strength values (even in "3/5 kg" format)
2. VAS pain scores in various formats
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from shared.billing_engine import _check_doc

def test_griffstaerke_detection():
    """Test that grip strength detection works for various formats."""

    # Should PASS - correct formats
    assert _check_doc("Griffstärke (kg)", {"O": "Jamar-Handkraft: 3 kg"})
    assert _check_doc("Griffstärke (kg)", {"O": "Jamar-Handkraft li: 3/5 kg"})  # Should now work
    assert _check_doc("Griffstärke (kg)", {"O": "Griffstärke 15 kg"})
    assert _check_doc("Griffstärke (kg)", {"O": "Grip strength: 12.5 kg"})
    assert _check_doc("Griffstärke (kg)", {"O": "Handkraft Jamar: 8 kg re, 6 kg li"})

    # Should FAIL - missing grip strength
    assert not _check_doc("Griffstärke (kg)", {"O": "ROM Handgelenk: 30-0-40"})
    assert not _check_doc("Griffstärke (kg)", {"O": "Kraftgrad 4/5"})  # This is MMT, not Jamar

    print("✅ Griffstärke detection tests passed!")

def test_vas_detection():
    """Test that VAS pain score detection works for various formats."""

    # Should PASS - correct VAS formats
    assert _check_doc("Schmerz (VAS)", {"S": "VAS 4/10"})
    assert _check_doc("Schmerz (VAS)", {"S": "Schmerz: VAS 4/10"})
    assert _check_doc("Schmerz (VAS)", {"S": "VAS: 4"})
    assert _check_doc("Schmerz (VAS)", {"S": "Patient berichtet Schmerzen VAS 7/10"})
    assert _check_doc("Schmerz (VAS)", {"O": "NRS 5"})
    assert _check_doc("Schmerz (VAS)", {"S": "Schmerz NRS 6"})

    # Should FAIL - missing VAS
    assert not _check_doc("Schmerz (VAS)", {"S": "Patient berichtet Schmerzen"})
    assert not _check_doc("Schmerz (VAS)", {"S": "Starke Schmerzen im Handgelenk"})
    assert not _check_doc("Schmerz (VAS)", {"O": "ROM eingeschränkt"})

    print("✅ VAS detection tests passed!")

def test_ex6_report_parsing():
    """Test with a real EX6 report like the user provided."""

    # The problematic report from the user
    report_soap = {
        "S": "Patient berichtet nach Gipsabnahme immer noch Schmerzen und Kraftmangel im linken Handgelenk, besonders bei Bewegungen.",
        "O": "Schonhaltung re. | ROM Handgelenk: Flexion 30°, Extension 20° | Jamar-Handkraft li: 3/5 kg | FHA: 2 cm | Endgefühl: hart-kapsulär | Keine Anzeichen für CRPS | Behandeltes Segment: Articulatio radiocarpalis (Handgelenk)",
        "A": "S52 | Red Flags klinisch ausgeschlossen",
        "P": "KG mit manuellen Techniken | Ziel: [Funktion] auf [Messwert] in [N] EH | 2x/Woche, 6 EH"
    }

    # Test grip strength detection (should NOW pass even with "3/5 kg" format)
    has_grip = _check_doc("Griffstärke (kg)", report_soap)
    print(f"Grip strength detected: {has_grip}")
    assert has_grip, "Should detect grip strength in '3/5 kg' format"

    # Test VAS detection (should FAIL - no VAS in this report)
    has_vas = _check_doc("Schmerz (VAS)", report_soap)
    print(f"VAS detected: {has_vas}")
    assert not has_vas, "Should NOT detect VAS (it's missing)"

    print("✅ Real report parsing tests passed!")

    # Now test with corrected report
    corrected_soap = {
        "S": "Patient berichtet nach Gipsabnahme immer noch Schmerzen (VAS 4/10) und Kraftmangel im linken Handgelenk, besonders bei Bewegungen.",
        "O": "Schonhaltung re. | ROM Handgelenk: Flexion 30°, Extension 20° | Jamar-Handkraft li: 3 kg | FHA: 2 cm | Endgefühl: hart-kapsulär | Keine Anzeichen für CRPS | Behandeltes Segment: Articulatio radiocarpalis (Handgelenk)",
        "A": "S52.5 | Red Flags klinisch ausgeschlossen",
        "P": "KG mit manuellen Techniken | Ziel: Schmerzreduktion auf VAS 2 und FHA 0 cm in 6 EH | 2x/Woche"
    }

    assert _check_doc("Griffstärke (kg)", corrected_soap), "Corrected: Should have grip strength"
    assert _check_doc("Schmerz (VAS)", corrected_soap), "Corrected: Should have VAS"

    print("✅ Corrected report tests passed!")

if __name__ == "__main__":
    test_griffstaerke_detection()
    test_vas_detection()
    test_ex6_report_parsing()
    print("\n🎉 ALL EX6 VALIDATION TESTS PASSED!")

