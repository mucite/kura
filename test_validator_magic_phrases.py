"""
Test Validator Fix - Exact transcript from user feedback
Validates that all 3 magic phrases are correctly formatted:
1. ROM Knie (Ext/Flex): X-X-X (not NZM, not duplicate)
2. Kraft (MGT): X/5 (not "Stufe X von Y")
3. Red Flags klinisch ausgeschlossen (exact phrase)
"""

import sys
sys.path.insert(0, r"C:\Users\musie\Downloads\medic\medic")

from windows.physio_scribe_crossplatform import KuraEngine

def test_validator_magic_phrases():
    """Test exact validator requirements with user's transcript"""

    # Exact transcript from user
    transcript = """Therapeut: Hallo Herr Bauer, willkommen zur zweiten Sitzung. Das Pflaster ist ja schon ab, die Narbe sieht gut aus. Wie ist das Knie denn seit Dienstag?

Patientin: Ja, grüß Gott. Also, es ist halt noch ordentlich geschwollen, gell? Vor allem abends nach der Arbeit, da ist das Knie richtig dick und spannt. Ich hab dann so ein Stechen direkt unter der Kniescheibe.

Therapeut: Auf einer Skala von 1 bis 10, wenn es abends so geschwollen ist, wo liegen wir da?

Patientin: Puh, sicher bei einer 7. Wenn ich morgens aufstehe, ist es eher eine 2 oder 3, aber es ist halt steif. Ich komme noch nicht richtig in die Streckung, beim Gehen hinke ich deswegen ein bisschen.

Therapeut: Verstehe. Wir schauen uns die Beweglichkeit mal auf der Bank an. Legen Sie sich mal lang hin... okay, wenn ich das Bein aktiv strecken lasse... ja, da fehlen uns gute 10 Grad bis zur Unterlage. Und jetzt beugen, soweit es geht... (stöhnt) ...okay, bei 90 Grad ist Schluss wegen der Schwellung, richtig?

Patientin: Ja, weiter geht's nicht, da drückt's im ganzen Gelenk.

Therapeut: Okay, also Neutral-Null-Durchgang: Streckung/Beugung ist 0-10-90. Das Gelenk ist deutlich überwärmt, wir haben einen intraartikulären Erguss, der Patellatanz ist positiv. Die Kraft im Quadrizeps ist noch etwas gehemmt, ich würde sagen Stufe 3 von 5.

Patientin: Kann ich denn schon wieder Treppensteigen?

Therapeut: Ohne Geländer noch nicht, dafür ist die Stabilität im Einbeinstand noch zu schwach. Da knicken Sie noch nach innen weg. Wir machen heute eine manuelle Lymphdrainage, um den Druck rauszunehmen, und danach eine vorsichtige Mobilisation der Patella. Ziel ist, dass wir die Streckung bis nächste Woche auf 0 Grad bekommen, damit Sie wieder sauber abrollen können.

Patientin: Alles klar, packen wir's an."""

    # Simulate LLM output (potentially with wrong formats)
    llm_soap = {
        "S": "Knie geschwollen und steif. VAS 7/10 abends, 2-3/10 morgens. Hinken beim Gehen.",
        "O": "Schwellung beidseits | Lachman-Test: positiv | McMurray-Test: positiv | intraartikulärer Erguss | Patellatanz positiv",
        "A": "M17 Gonarthrose",
        "P": "Manuelle Lymphdrainage | Patellamobilisation | Ziel: Streckung 0° in 6 EH"
    }

    engine = KuraEngine()

    # Apply recovery (this should add the magic phrases)
    result = engine.recover_hard_metrics(transcript, llm_soap, profile_id="EX_KNIE")

    print("="*80)
    print("VALIDATOR MAGIC PHRASE TEST")
    print("="*80)
    print("\n📋 INPUT (LLM SOAP):")
    print(f"   O: {llm_soap['O']}")
    print(f"   A: {llm_soap['A']}")

    print("\n✨ OUTPUT (After Recovery):")
    print(f"   O: {result['O']}")
    print(f"   A: {result['A']}")

    print("\n" + "="*80)
    print("VALIDATOR CHECKS:")
    print("="*80)

    # Check 1: ROM in exact format
    print("\n✅ CHECK 1: ROM Format")
    rom_correct = "ROM Knie (Ext/Flex):" in result["O"]
    rom_clean = result["O"].count("ROM") == 1  # Only ONE ROM entry
    rom_value = "0-10-90" in result["O"]

    if rom_correct and rom_clean and rom_value:
        print(f"   ✅ PASS: ROM Knie (Ext/Flex): 0-10-90 found")
        print(f"   ✅ No duplicate ROM entries")
    else:
        print(f"   ❌ FAIL: ROM format incorrect or duplicate")
        print(f"      rom_correct={rom_correct}, rom_clean={rom_clean}, rom_value={rom_value}")
        return False

    # Check 2: Kraft in MGT format
    print("\n✅ CHECK 2: Kraft (MGT) Format")
    kraft_correct = "Kraft (MGT):" in result["O"]
    kraft_value = "3/5" in result["O"]
    kraft_no_stufe = "stufe" not in result["O"].lower()

    if kraft_correct and kraft_value and kraft_no_stufe:
        print(f"   ✅ PASS: Kraft (MGT): 3/5 found")
        print(f"   ✅ No 'Stufe' format present")
    else:
        print(f"   ❌ FAIL: Kraft format incorrect")
        print(f"      kraft_correct={kraft_correct}, kraft_value={kraft_value}")
        print(f"      kraft_no_stufe={kraft_no_stufe}")
        return False

    # Check 3: Red Flags magic phrase
    print("\n✅ CHECK 3: Red Flags Magic Phrase")
    magic_phrase = "Red Flags klinisch ausgeschlossen" in result["A"]
    has_details = "keine Kompartment-Zeichen" in result["A"]

    if magic_phrase and has_details:
        print(f"   ✅ PASS: 'Red Flags klinisch ausgeschlossen' present")
        print(f"   ✅ Profile-specific details included")
    else:
        print(f"   ❌ FAIL: Red Flag magic phrase missing")
        print(f"      magic_phrase={magic_phrase}, has_details={has_details}")
        return False

    # Check 4: Gangbild detected from "hinke"
    print("\n✅ CHECK 4: Gangbild Detection")
    gangbild_present = "Gangbild:" in result["O"]
    gangbild_correct = "Antalgisches Hinken" in result["O"] or "Extensionsdefizit" in result["O"]
    gangbild_not_normal = "unauffällig" not in result["O"] or "Antalgisches Hinken" in result["O"]

    if gangbild_present and gangbild_correct:
        print(f"   ✅ PASS: Gangbild detected from 'hinke' in transcript")
        print(f"   ✅ Correct pattern: Antalgisches Hinken (Extensionsdefizit)")
    else:
        print(f"   ❌ FAIL: Gangbild not correctly detected")
        print(f"      gangbild_present={gangbild_present}, gangbild_correct={gangbild_correct}")
        return False

    print("\n" + "="*80)
    print("📊 FINAL VALIDATOR SCORE:")
    print("="*80)
    print("✅ ROM Knie (Ext/Flex): 0-10-90 ...................... PASS")
    print("✅ Kraft (MGT): 3/5 .................................. PASS")
    print("✅ Red Flags klinisch ausgeschlossen ................. PASS")
    print("✅ Gangbild: Antalgisches Hinken .................... PASS")
    print("\n🎉 AUDIT BESTANDEN — All magic phrases present!")
    print("="*80)

    return True

if __name__ == "__main__":
    try:
        success = test_validator_magic_phrases()
        if success:
            print("\n✅ ALL VALIDATOR CHECKS PASSED")
            sys.exit(0)
        else:
            print("\n❌ VALIDATOR CHECKS FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

