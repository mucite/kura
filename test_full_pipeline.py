"""
Enhanced test with full pipeline (LLM + post-processing recovery)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "windows"))

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

print("=" * 70)
print("FULL PIPELINE TEST (LLM + Post-Processing)")
print("=" * 70)
print()

try:
    from windows.physio_scribe_crossplatform import KuraEngine

    print("Loading Kura engine...")
    engine = KuraEngine()

    print("\nStep 1: Detect profile...")
    profile_id = engine._detect_profile(test_transcript)
    print(f"Detected profile: {profile_id}")

    print("\nStep 2: Generate SOAP with LLM...")
    raw_output = engine._generate_soap_note(test_transcript, profile_id)

    print("\nStep 3: Parse and post-process...")
    # Parse JSON output
    import json
    import re
    try:
        json_match = re.search(r'\{.*"icd10".*"soap".*\}', raw_output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            parsed = json.loads(raw_output)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ JSON parse error: {e}")
        parsed = {
            "icd10": "M99.9",
            "soap": {"S": "n.d.", "O": "n.d.", "A": "n.d.", "P": "n.d."}
        }

    # Apply post-processing (same as run_full_flow does)
    parsed["soap"] = engine.recover_hard_metrics(test_transcript, parsed["soap"], profile_id=profile_id)

    print("\n" + "=" * 70)
    print("AFTER FULL PIPELINE (LLM + Post-Processing):")
    print("=" * 70)


    print(f"\nICD-10: {parsed.get('icd10', 'N/A')}")
    for field in ["S", "O", "A", "P"]:
        content = parsed.get("soap", {}).get(field, "N/A")
        print(f"\n{field}:\n{content}")

    # Check key metrics
    print("\n" + "=" * 70)
    print("KEY METRICS CHECK:")
    print("=" * 70)

    metrics_found = []
    metrics_missing = []

    checks = [
        ("VAS 8/10", parsed["soap"].get("S", "")),
        ("Donnerstag", parsed["soap"].get("S", "")),
        ("Umzugskartons", parsed["soap"].get("S", "")),
        ("FBA", parsed["soap"].get("O", "")),
        ("40", parsed["soap"].get("O", "")),
        ("Lasègue", parsed["soap"].get("O", "")),
        ("80", parsed["soap"].get("O", "")),
        ("Vorlaufphänomen", parsed["soap"].get("O", "")),
        ("Kraftgrade", parsed["soap"].get("O", "")),
        ("5/5", parsed["soap"].get("O", "")),
        ("Quadratus lumborum", parsed["soap"].get("A", "")),
        ("Dysbalance", parsed["soap"].get("A", "")),
        ("Bandscheibenvorfall", parsed["soap"].get("A", "")),
        ("Triggerpunkte", parsed["soap"].get("P", "")),
        ("Wärme", parsed["soap"].get("P", "")),
        ("Stufenlagerung", parsed["soap"].get("P", "")),
        ("zweimal", parsed["soap"].get("P", "")),
        ("sechs", parsed["soap"].get("P", "")),
    ]

    for metric, field_content in checks:
        if metric.lower() in field_content.lower():
            metrics_found.append(metric)
        else:
            metrics_missing.append(metric)

    print(f"\n✅ Found ({len(metrics_found)}/18):")
    for m in metrics_found:
        print(f"   • {m}")

    if metrics_missing:
        print(f"\n❌ Missing ({len(metrics_missing)}/18):")
        for m in metrics_missing:
            print(f"   • {m}")

    extraction_rate = (len(metrics_found) / len(checks)) * 100
    print(f"\n{'=' * 70}")
    print(f"FINAL EXTRACTION RATE: {extraction_rate:.1f}%")
    print(f"{'=' * 70}")

    if extraction_rate >= 70:
        print("\n✅ EXCELLENT - Production ready!")
        sys.exit(0)
    elif extraction_rate >= 50:
        print("\n✅ GOOD - Acceptable for production with therapist review")
        sys.exit(0)
    else:
        print("\n⚠️ NEEDS IMPROVEMENT - Post-processing should recover more")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

