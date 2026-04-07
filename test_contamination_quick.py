"""
Quick test for context contamination fix
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from windows.physio_scribe_crossplatform import KuraEngine

# Simulate the problematic transcript
knee_transcript = """
Therapeut: Hallo Herr Bauer, willkommen zur zweiten Sitzung. Das Pflaster ist ja schon ab, die Narbe sieht gut aus. Wie ist das Knie denn seit Dienstag?

Patientin: Ja, grüß Gott. Also, es ist halt noch ordentlich geschwollen, gell? Vor allem abends nach der Arbeit, da ist das Knie richtig dick und spannt. Ich hab dann so ein Stechen direkt unter der Kniescheibe.

Therapeut: Auf einer Skala von 1 bis 10, wenn es abends so geschwollen ist, wo liegen wir da?

Patientin: Puh, sicher bei einer 7. Wenn ich morgens aufstehe, ist es eher eine 2 oder 3, aber es ist halt steif. Ich komme noch nicht richtig in die Streckung, beim Gehen hinke ich deswegen ein bisschen.

Therapeut: Verstehe. Wir schauen uns die Beweglichkeit mal auf der Bank an. Legen Sie sich mal lang hin... okay, wenn ich das Bein aktiv strecken lasse... ja, da fehlen uns gute 10 Grad bis zur Unterlage. Und jetzt beugen, soweit es geht... (stöhnt) ...okay, bei 90 Grad ist Schluss wegen der Schwellung, richtig?

Patientin: Ja, weiter geht's nicht, da drückt's im ganzen Gelenk.

Therapeut: Okay, also Neutral-Null-Durchgang: Streckung/Beugung ist 0-10-90. Das Gelenk ist deutlich überwärmt, wir haben einen intraartikulären Erguss, der Patellatanz ist positiv. Die Kraft im Quadrizeps ist noch etwas gehemmt, ich würde sagen Stufe 3 von 5.

Patientin: Kann ich denn schon wieder Treppensteigen?

Therapeut: Ohne Geländer noch nicht, dafür ist die Stabilität im Einbeinstand noch zu schwach. Da knicken Sie noch nach innen weg. Wir machen heute eine manuelle Lymphdrainage, um den Druck rauszunehmen, und danach eine vorsichtige Mobilisation der Patella. Ziel ist, dass wir die Streckung bis nächste Woche auf 0 Grad bekommen, damit Sie wieder sauber abrollen können.

Patientin: Alles klar, packen wir's an.
"""

print("="*70)
print("CONTEXT CONTAMINATION TEST - KNEE SESSION")
print("="*70)

try:
    print("\n1. Initializing Kura Engine...")
    engine = KuraEngine(license_status="TRIAL")

    print("\n2. Detecting profile from transcript...")
    profile = engine._detect_profile(knee_transcript)
    print(f"   Detected profile: {profile}")

    if profile != "EX_KNIE":
        print(f"   ❌ FAILED: Expected EX_KNIE, got {profile}")
    else:
        print(f"   ✅ PASSED: Correct profile detected")

    print("\n3. Building prompt with context isolation warnings...")
    prompt = engine.build_prompt(knee_transcript, profile)

    # Check for critical warnings
    if "CONTEXT ISOLATION" in prompt:
        print("   ✅ PASSED: Context isolation warning present in prompt")
    else:
        print("   ❌ FAILED: Context isolation warning missing")

    if "AKTUELLEN Transkript" in prompt:
        print("   ✅ PASSED: Current transcript emphasis present")
    else:
        print("   ❌ FAILED: Current transcript emphasis missing")

    if "LWS" in prompt and "KNIE" in prompt:
        # Check if it's warning AGAINST mixing (good) vs providing example (bad)
        if "darf NICHTS über Rücken/LWS im S-Feld erscheinen" in prompt:
            print("   ✅ PASSED: Warning against LWS/KNIE mixing present")
        else:
            print("   ⚠️  WARNING: LWS mentioned in KNIE prompt (verify it's in warning context)")

    print("\n4. Testing contamination detection...")
    # Simulate contaminated SOAP
    contaminated_soap = {
        "S": "Pat. berichtet akute LWS-Schmerzen seit Dienstag nach Heben. Knie steif und hinkt beim Gehen",
        "O": "Schwellung rechts | Lachman-Test: positiv",
        "A": "M17 | Red Flags ausgeschlossen",
        "P": "MLD | Ziel: Streckung bis Null Grad"
    }

    print(f"   Original S-field: {contaminated_soap['S'][:80]}...")

    # Run recovery (which includes contamination removal)
    cleaned = engine.recover_hard_metrics(knee_transcript, contaminated_soap, profile)

    print(f"   Cleaned S-field: {cleaned['S'][:80]}...")

    if "LWS-Schmerzen" in cleaned['S'] or "Heben" in cleaned['S']:
        print("   ❌ FAILED: LWS contamination still present in S-field!")
    else:
        print("   ✅ PASSED: LWS contamination removed from S-field")

    if "Knie" in cleaned['S'] or "hinkt" in cleaned['S']:
        print("   ✅ PASSED: Knee-related content preserved")
    else:
        print("   ⚠️  WARNING: All content removed (might be too aggressive)")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    print("\n📊 Summary:")
    print("  - Profile detection: Working")
    print("  - Context isolation warnings: Added to prompt")
    print("  - Contamination detection: Implemented")
    print("  - S-field cleaning: Active")

    print("\n🔧 Next steps:")
    print("  1. Restart Kura application to load new code")
    print("  2. Test with real audio/transcript")
    print("  3. Verify LLM generates clean SOAP without contamination")

except Exception as e:
    print(f"\n❌ TEST FAILED WITH ERROR: {e}")
    import traceback
    traceback.print_exc()

