"""
Test CRPS/Neurological Red Flag Safety Logic

Verifies that the system DOES NOT create dangerous contradictions like:
- "Keine Anzeichen für CRPS" + "Verdacht auf CRPS"
- "Red Flags ausgeschlossen" when positive Tinel/Phalen tests present
"""
import pytest


def test_crps_safety_detection():
    """Test that CRPS signs prevent auto-exclusion statement."""
    from windows.physio_scribe_crossplatform import KuraEngine

    # Mock transcript with CRPS signs
    transcript = """
    Hallo Frau Weber. Der Gips am rechten Handgelenk ist seit gestern ab. 
    Die Haut glänzt so komisch und ist ganz rötlich-violett. 
    Und es brennt ständig, als läge die Hand in Brennnesseln.
    Der Daumen und der Zeigefinger kribbeln ständig, besonders nachts.
    """

    # Create mock engine
    engine = KuraEngine(license_status=True)

    # Mock SOAP dict
    soap_dict = {
        "S": "Patient berichtet brennende Schmerzen und Kribbelgefühl",
        "O": "Haut glänzend, rötlich-violette Verfärbung | VAS 8/10",
        "A": "S52 | Verdacht auf CRPS",
        "P": "Lymphdrainage"
    }

    # Apply safety check
    result = engine.recover_hard_metrics(transcript, soap_dict, profile_id="EX_HAND")

    # CRITICAL: Should NOT contain "Keine Anzeichen für CRPS"
    assert "Keine Anzeichen für CRPS" not in result["O"], \
        "SAFETY VIOLATION: 'Keine Anzeichen für CRPS' should NOT be present when CRPS signs detected!"

    # CRITICAL: Should contain warning instead of exclusion
    assert "ACHTUNG" in result["A"] or "Verdacht auf CRPS" in result["A"], \
        "SAFETY VIOLATION: Should have warning when CRPS signs present!"

    # CRITICAL: Should NOT contain "Red Flags ausgeschlossen"
    assert "Red Flags klinisch ausgeschlossen" not in result["A"], \
        "SAFETY VIOLATION: Red Flags should NOT be excluded when CRPS suspected!"

    print("✅ PASS: CRPS safety logic prevents dangerous auto-exclusion")


def test_neuro_red_flags_detection():
    """Test that positive neurological tests prevent Red Flag exclusion."""
    from windows.physio_scribe_crossplatform import KuraEngine

    transcript = """
    Ich klopfe hier mal auf die Beugeseite des Handgelenks... 
    Das Hoffmann-Tinel-Zeichen ist positiv. 
    Und wenn Sie die Handrücken für eine Minute so zusammendrücken... 
    Der Phalen-Test ist auch positiv. Es zieht in den Daumen.
    """

    engine = KuraEngine(license_status=True)

    soap_dict = {
        "S": "Parästhesien in Daumen und Zeigefinger",
        "O": "ROM n.d.",
        "A": "S52",
        "P": "Nervengleiten"
    }

    result = engine.recover_hard_metrics(transcript, soap_dict, profile_id="EX_HAND")

    # CRITICAL: Should include positive tests in O-field
    assert "Hoffmann-Tinel-Zeichen: positiv" in result["O"], \
        "Should include positive Tinel test in O-field!"
    assert "Phalen-Test: positiv" in result["O"], \
        "Should include positive Phalen test in O-field!"

    # CRITICAL: Should NOT exclude Red Flags
    assert "Red Flags klinisch ausgeschlossen" not in result["A"], \
        "SAFETY VIOLATION: Red Flags should NOT be excluded when positive neuro tests!"

    # Should have warning
    assert "ACHTUNG" in result["A"] or "Verdacht auf Nervenkompressionssyndrom" in result["A"], \
        "Should have neurological warning in A-field!"

    print("✅ PASS: Neurological red flags prevent auto-exclusion")


def test_safe_case_allows_exclusion():
    """Test that when NO red flags present, exclusion is added."""
    from windows.physio_scribe_crossplatform import KuraEngine

    transcript = """
    Handgelenk 6 Wochen nach Radiusfraktur. Gips gestern entfernt.
    Beweglichkeit: Beugen 40 Grad, Strecken 20 Grad.
    Keine Brennen, keine Verfärbung, Haut unauffällig.
    Sensibilität intakt, kein Kribbeln.
    """

    engine = KuraEngine(license_status=True)

    soap_dict = {
        "S": "Patient berichtet Steifigkeit nach Gipsabnahme",
        "O": "ROM Handgelenk: Flexion 40°, Extension 20°",
        "A": "S52.5",
        "P": "Mobilisation"
    }

    result = engine.recover_hard_metrics(transcript, soap_dict, profile_id="EX_HAND")

    # When SAFE (no CRPS signs, no positive neuro tests), should add exclusion
    assert "Red Flags klinisch ausgeschlossen" in result["A"], \
        "Should add Red Flag exclusion when NO red flags detected!"

    # Should NOT have CRPS exclusion in O (only in A as part of Red Flags)
    # But should NOT have warning
    assert "ACHTUNG" not in result["A"], \
        "Should NOT have warning when no red flags!"

    print("✅ PASS: Safe case allows Red Flag exclusion")


def test_contradiction_prevention():
    """Test that system NEVER creates contradictory statements."""
    from windows.physio_scribe_crossplatform import KuraEngine

    # Worst-case scenario: AI generates "Keine Anzeichen" AND "Verdacht auf"
    transcript = """
    Brennende Schmerzen, Haut glänzt rötlich-violett.
    Hoffmann-Tinel positiv. Phalen positiv.
    """

    engine = KuraEngine(license_status=True)

    soap_dict = {
        "S": "VAS 8/10, brennende Schmerzen",
        "O": "Keine Anzeichen für CRPS",  # ❌ WRONG - AI hallucination
        "A": "S52 | Verdacht auf CRPS | Red Flags klinisch ausgeschlossen",  # ❌ CONTRADICTION!
        "P": "Lymphdrainage"
    }

    result = engine.recover_hard_metrics(transcript, soap_dict, profile_id="EX_HAND")

    # Safety logic should REMOVE contradictions
    obj_text = result["O"]
    a_text = result["A"]

    # Check for contradictions
    has_crps_exclusion = "Keine Anzeichen für CRPS" in obj_text
    has_crps_suspicion = "Verdacht auf CRPS" in a_text
    has_red_flag_exclusion = "Red Flags klinisch ausgeschlossen" in a_text
    has_red_flag_warning = "ACHTUNG" in a_text

    # CRITICAL: These combinations are DANGEROUS and must not coexist
    assert not (has_crps_exclusion and has_crps_suspicion), \
        "LEGAL CONTRADICTION: Cannot have both 'Keine Anzeichen für CRPS' and 'Verdacht auf CRPS'!"

    assert not (has_red_flag_exclusion and has_red_flag_warning), \
        "LEGAL CONTRADICTION: Cannot have both 'Red Flags ausgeschlossen' and 'ACHTUNG' warning!"

    # If CRPS signs detected, should have warning NOT exclusion
    assert has_red_flag_warning or not has_red_flag_exclusion, \
        "When CRPS signs present, must have warning, not exclusion!"

    print("✅ PASS: Contradiction prevention works correctly")


def test_tinel_phalen_extraction():
    """Test that Tinel and Phalen tests are properly extracted to O-field."""
    from windows.physio_scribe_crossplatform import KuraEngine

    transcript = """
    Jetzt ein Test für den Nervenkanal: Ich klopfe hier mal auf die Beugeseite 
    des Handgelenks... okay, das Hoffmann-Tinel-Zeichen ist positiv. 
    Und wenn Sie die Handrücken für eine Minute so zusammendrücken... 
    nach 20 Sekunden zieht es schon in den Daumen? Ja, sofort. Es wird ganz taub.
    Das ist ein positiver Phalen-Test.
    """

    engine = KuraEngine(license_status=True)

    soap_dict = {
        "S": "Kribbeln in Daumen und Zeigefinger",
        "O": "ROM n.d.",
        "A": "S52",
        "P": "n.d."
    }

    result = engine.recover_hard_metrics(transcript, soap_dict, profile_id="EX_HAND")

    # Should extract both tests with their results
    assert "Hoffmann-Tinel-Zeichen: positiv" in result["O"], \
        "Should extract Hoffmann-Tinel test result!"
    assert "Phalen-Test: positiv" in result["O"], \
        "Should extract Phalen test result!"

    print("✅ PASS: Tinel and Phalen tests properly extracted")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CRPS/NEUROLOGICAL RED FLAG SAFETY LOGIC TEST SUITE")
    print("="*70 + "\n")

    try:
        test_crps_safety_detection()
        test_neuro_red_flags_detection()
        test_safe_case_allows_exclusion()
        test_contradiction_prevention()
        test_tinel_phalen_extraction()

        print("\n" + "="*70)
        print("✅ ALL SAFETY TESTS PASSED - System is clinically safe!")
        print("="*70 + "\n")
    except AssertionError as e:
        print(f"\n❌ SAFETY TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise

