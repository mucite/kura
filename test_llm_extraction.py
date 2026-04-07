"""
Test script to verify LLM extraction improvements.
Run this to test if the AI properly extracts information from transcripts.
"""

test_transcript = """Therapeut: Guten Tag Frau Müller, setzen Sie sich doch. Wir haben heute die erste Verordnung über Krankengymnastik vorliegen. Diagnose vom Orthopäden ist ein akutes LWS-Syndrom. Erzählen Sie mir bitte kurz aus Ihrer Sicht: Was ist passiert und wo genau liegen die Beschwerden?

Patientin: Ja, hallo. Also eigentlich fing das letzten Donnerstag an. Ich habe in der Garage beim Aufräumen geholfen und schwere Umzugskartons gehoben. Dabei gab es plötzlich einen stechenden Schmerz im unteren Rücken, rechtsseitig. Es hat sich angefühlt wie ein kurzer Knall oder ein Reißen. Seitdem komme ich kaum noch hoch.

Therapeut: Strahlt der Schmerz irgendwohin aus? Merken Sie ein Kribbeln im Bein oder haben Sie das Gefühl, dass der Fuß kraftlos ist?

Patientin: Nein, zum Glück nicht. Es konzentriert sich wirklich auf den Bereich über dem Gesäß rechts. Aber die Schmerzen sind morgens extrem schlimm, so eine richtige Anlaufsteifigkeit. Ich brauche fast eine halbe Stunde, bis ich mich einigermaßen gerade bewegen kann. Auf einer Skala von 1 bis 10 würde ich sagen, in Bewegung ist es eine 8, im Liegen vielleicht eine 3.

Therapeut: Gut, dann schauen wir uns das mal im Stand an. Wenn ich mir Ihr Becken ansehe, fällt eine deutliche Schonhaltung nach links auf. Die paraspinale Muskulatur im Lendenbereich ist rechts massiv hyperton, also richtig hart gespannt. Ich taste jetzt einmal die Dornfortsätze ab. Haben Sie hier Schmerzen?

Patientin: Nein, direkt auf der Wirbelsäule eigentlich nicht. Aber daneben, da wo Sie jetzt drücken, das ist extrem empfindlich.

Therapeut: Okay, deutlicher Myogelose-Befund im Bereich des Musculus quadratus lumborum rechts. Versuchen Sie mal, den Oberkörper langsam nach vorne zu beugen. Sagen Sie Stopp, wenn es zu stark zieht.

Patientin: Oh, das geht gar nicht weit. Stopp! Das zieht sofort im Kreuz.

Therapeut: Alles klar, die Finger-Boden-Distanz beträgt etwa 40 Zentimeter, die Flexion der LWS ist also stark eingeschränkt. Wir gehen jetzt mal auf die Liege für die Funktionsprüfung. Das Vorlaufphänomen ist rechts positiv. Der Lasègue-Test, also das gestreckte Anheben des Beins, ist bis 80 Grad negativ, es gibt keine radikuläre Reizung. Die Kraftgrade für die Fußheber und Senker sind unauffällig, ich würde sagen 5 von 5 nach Janda.

Therapeut: Meine Einschätzung: Es handelt sich um eine akute muskuläre Dysbalance mit einer starken Schutzspannung des Quadratus lumborum nach einer mechanischen Überlastung. Ein Bandscheibenvorfall ist klinisch aktuell nicht wahrscheinlich, da keine neurologischen Ausfälle vorliegen.

Patientin: Was machen wir jetzt dagegen?

Therapeut: Wir starten heute mit einer Schmerzmittentherapie in Form von manuellen Techniken und Detonisierung der Muskulatur. Ich werde die Triggerpunkte behandeln. Für zu Hause ist Wärme ganz wichtig, um die Durchblutung zu fördern. Ich zeige Ihnen gleich noch die „Stufenlagerung" zur Entlastung. Wir sehen uns dann zweimal pro Woche, insgesamt erst mal sechs Termine.

Patientin: Alles klar, das machen wir so.

Therapeut: Gut, dann legen Sie sich bitte einmal in Bauchlage auf die Behandlungsbank."""

expected_extractions = {
    "S": [
        "Donnerstag",
        "Garage",
        "Umzugskartons",
        "stechender Schmerz",
        "rechts",
        "VAS 8/10",
        "3/10 im Liegen",
        "morgens",
        "halbe Stunde"
    ],
    "O": [
        "Schonhaltung nach links",
        "paraspinale Muskulatur rechts hyperton",
        "Myogelose",
        "Quadratus lumborum rechts",
        "FBA", "40", "cm",
        "Vorlaufphänomen rechts positiv",
        "Lasègue", "80 Grad negativ",
        "Kraftgrade 5/5"
    ],
    "A": [
        "muskuläre Dysbalance",
        "Schutzspannung",
        "Quadratus lumborum",
        "Bandscheibenvorfall", "nicht wahrscheinlich",
        "keine neurologischen Ausfälle"
    ],
    "P": [
        "manuelle Techniken",
        "Detonisierung",
        "Triggerpunkte",
        "Wärme",
        "Stufenlagerung",
        "zweimal pro Woche",
        "sechs Termine"
    ]
}

def check_extraction_quality(soap_dict):
    """
    Check if the SOAP extraction contains expected information.
    """
    score = 0
    max_score = 0
    issues = []

    for field, expected_terms in expected_extractions.items():
        field_content = soap_dict.get("soap", {}).get(field, "").lower()
        max_score += len(expected_terms)

        for term in expected_terms:
            if term.lower() in field_content:
                score += 1
            else:
                issues.append(f"Missing in {field}: '{term}'")

    # Check for "n.d." abuse
    for field in ["S", "O", "A", "P"]:
        content = soap_dict.get("soap", {}).get(field, "")
        if content.strip() in ["n.d.", "n. d.", "n.d", "nicht dokumentiert"]:
            issues.append(f"{field} field is completely empty ('n.d.') despite transcript having info")

    # Check for therapist question copying
    if "Erzählen Sie mir" in soap_dict.get("soap", {}).get("S", ""):
        issues.append("S field contains therapist question instead of patient story")

    percentage = (score / max_score * 100) if max_score > 0 else 0

    print(f"\n{'='*60}")
    print(f"EXTRACTION QUALITY: {percentage:.1f}% ({score}/{max_score} terms found)")
    print(f"{'='*60}")

    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✅ ALL EXPECTED TERMS EXTRACTED!")

    return percentage >= 70  # Pass if at least 70% extracted

if __name__ == "__main__":
    print("="*60)
    print("LLM EXTRACTION TEST")
    print("="*60)
    print("\nThis test will:")
    print("1. Load the Kura engine")
    print("2. Process the test transcript")
    print("3. Check if all key information is extracted")
    print("\nStarting test...\n")

    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "windows"))

        from windows.physio_scribe_crossplatform import KuraEngine

        print("Loading Kura engine...")
        engine = KuraEngine()

        print("\nProcessing test transcript...")
        print(f"Transcript length: {len(test_transcript)} characters\n")

        result_json = engine._generate_soap_note(test_transcript, profile_id="EX_LWS")

        # Parse JSON output with robust extraction (same as engine does)
        import json
        import re
        try:
            # Extract JSON from LLM output (may have extra text)
            json_match = re.search(r'\{.*"icd10".*"soap".*\}', result_json, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads(result_json)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ JSON parse error: {e}, using fallback")
            result = {
                "icd10": "M99.9",
                "soap": {"S": "n.d.", "O": "n.d.", "A": "n.d.", "P": "n.d."}
            }

        print("\n" + "="*60)
        print("GENERATED SOAP NOTE:")
        print("="*60)
        print(f"\nICD-10: {result.get('icd10', 'N/A')}")
        print(f"\nS: {result.get('soap', {}).get('S', 'N/A')}")
        print(f"\nO: {result.get('soap', {}).get('O', 'N/A')}")
        print(f"\nA: {result.get('soap', {}).get('A', 'N/A')}")
        print(f"\nP: {result.get('soap', {}).get('P', 'N/A')}")

        # Quality check
        passed = check_extraction_quality(result)

        if passed:
            print("\n" + "="*60)
            print("✅ TEST PASSED - Extraction quality is good!")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("❌ TEST FAILED - Extraction quality needs improvement")
            print("="*60)
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

