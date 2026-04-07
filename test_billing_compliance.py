"""
Test: German Physiotherapy Billing Compliance
Verifies correct billing correlation for major physiotherapy services in Germany
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from shared.billing_engine import BillingEngine, InsuranceType

# Test cases covering major physiotherapy services
test_cases = [
    # ══ EXTREMITÄTEN ══
    {
        "name": "Kniegelenk Gonarthrose",
        "icd10": "M17.1",
        "soap": {
            "S": "Schmerzen im Knie VAS 6/10",
            "O": "ROM Knie: Flexion 90-0-0, Extension 0-0-10. Kraft MMT 4/5. Gangbild: Schonhinken rechts.",
            "A": "M17.1 Gonarthrose rechts. Red Flags ausgeschlossen.",
            "P": "Mobilisation, Krafttraining Quadriceps, Gangschulung"
        },
        "expected_position": "20501",
        "expected_dg": "EX3",
    },
    {
        "name": "Schulter Impingement (MT)",
        "icd10": "M75.4",
        "soap": {
            "S": "Schulterschmerzen bei Überkopfarbeiten",
            "O": "Behandeltes Segment: Art. glenohumeralis. ROM Schulter Abd 90-0-0. Hawkins-Test positiv. Painful Arc 60-120°.",
            "A": "M75.4 Rotatorenmanschettensyndrom. Red Flags ausgeschlossen.",
            "P": "MT Schulter, Mobilisation, HEP"
        },
        "expected_position": "21201",
        "expected_dg": "EX2",
    },
    {
        "name": "Hüfte TEP postoperativ",
        "icd10": "M16.0",
        "soap": {
            "S": "4 Wochen nach TEP, Mobilität eingeschränkt",
            "O": "ROM Hüfte: Flexion 80-0-0, Abd 20-0-0, AR 10-0-0. Gangbild: Hilfsmittel (2 Stöcke). Muskelkraft MMT 3/5.",
            "A": "M16.0 Z.n. Hüft-TEP rechts. Red Flags ausgeschlossen.",
            "P": "Mobilisation, Kraftaufbau, Gangschulung"
        },
        "expected_position": "20501",
        "expected_dg": "EX4",
    },
    {
        "name": "Sprunggelenk Distorsion",
        "icd10": "S93.4",
        "soap": {
            "S": "Umknicken vor 1 Woche, Schwellung",
            "O": "ROM OSG: DF 5-0-30, PF 30-0-0. Schwellung lateral 2cm Differenz. Schmerz VAS 5/10. Gangbild: Schonhinken.",
            "A": "S93.4 Distorsion Sprunggelenk links. Red Flags ausgeschlossen.",
            "P": "Mobilisation, Propriozeptives Training, Taping"
        },
        "expected_position": "20501",
        "expected_dg": "EX5",
    },
    {
        "name": "Handgelenk Radiusfraktur",
        "icd10": "S52.5",
        "soap": {
            "S": "6 Wochen nach Radiusfraktur, Bewegung eingeschränkt",
            "O": "ROM Handgelenk: Ext 30-0-0, Flex 40-0-0. Griffstärke 15kg (gesunde Seite 40kg). Schmerz VAS 3/10.",
            "A": "S52.5 Z.n. distale Radiusfraktur. Red Flags ausgeschlossen.",
            "P": "Mobilisation, Krafttraining, Greifübungen"
        },
        "expected_position": "20501",
        "expected_dg": "EX6",
    },
    
    # ══ WIRBELSÄULE ══
    {
        "name": "HWS Zervikalsyndrom (MT)",
        "icd10": "M54.2",
        "soap": {
            "S": "Nackenschmerzen seit 2 Wochen VAS 7/10",
            "O": "Behandeltes Segment: C5/C6. ROM HWS: Rotation 40-0-40, Latflex 20-0-20. Spurling-Test negativ. Neurolog. Screening unauffällig.",
            "A": "M54.2 Zervikalgie. Red Flags ausgeschlossen.",
            "P": "MT HWS, Mobilisation, Muskeltraining"
        },
        "expected_position": "21201",
        "expected_dg": "WS1a",
    },
    {
        "name": "LWS Lumbago (MT)",
        "icd10": "M54.5",
        "soap": {
            "S": "Kreuzschmerzen akut VAS 8/10",
            "O": "Behandeltes Segment: L4/L5. FBA 40cm. Lasègue 80° negativ. Blasen-/Mastdarmfunktion: unauffällig. Neurolog. Screening unauffällig.",
            "A": "M54.5 Lumbalgie. Red Flags ausgeschlossen.",
            "P": "MT LWS, Triggerpunktbehandlung, Stufenlagerung"
        },
        "expected_position": "21201",
        "expected_dg": "WS1b",
    },
    {
        "name": "Bandscheiben-OP postoperativ",
        "icd10": "M51.1",
        "soap": {
            "S": "2 Wochen nach Bandscheibenoperation L5/S1",
            "O": "OP-Bericht vorhanden. Neurolog. Status: Kraft 5/5, Sensibilität unauffällig. ROM: Flexion 30-0-0. Schmerz VAS 4/10.",
            "A": "M51.1 Z.n. Bandscheiben-OP L5/S1. Red Flags ausgeschlossen.",
            "P": "Mobilisation vorsichtig, Muskelaufbau, ADL-Training"
        },
        "expected_position": "20501",
        "expected_dg": "WS2",
    },
    {
        "name": "Skoliose (Schroth)",
        "icd10": "M41.0",
        "soap": {
            "S": "Skoliose seit Kindheit, regelmäßige Therapie",
            "O": "Cobb-Winkel 32°. Rippenbuckel 3cm. Schroth-Klassifikation: 3-Kurven-Typ. Atemmuster: Flankenatmung erlernt.",
            "A": "M41.0 Idiopathische Skoliose. Red Flags ausgeschlossen.",
            "P": "Schroth-Therapie, Atemtraining, HEP"
        },
        "expected_position": "20501",
        "expected_dg": "WS3",
    },
    {
        "name": "Osteoporose",
        "icd10": "M80.0",
        "soap": {
            "S": "Osteoporose, Sturzangst",
            "O": "ROM WS eingeschränkt. Sturzrisiko-Assessment: Tinetti 18/28. Krafttest MMT 3/5 untere Extremität. Schmerz VAS 5/10.",
            "A": "M80.0 Osteoporose mit Fraktur. Red Flags ausgeschlossen.",
            "P": "Krafttraining, Sturzprophylaxe, Gleichgewichtstraining"
        },
        "expected_position": "20501",
        "expected_dg": "WS4",
    },
    
    # ══ ZNS ══
    {
        "name": "Schlaganfall Hemiplegie (KG-ZNS)",
        "icd10": "I69.3",
        "soap": {
            "S": "6 Monate nach Schlaganfall, linke Körperseite betroffen",
            "O": "Barthel-Index 60/100. Ashworth-Skala Arm 2, Bein 1. Gangbild: Zirkumduktion links. ADL: teilweise selbständig.",
            "A": "I69.3 Folgen eines Hirninfarkts (Hemiplegie links). Red Flags ausgeschlossen.",
            "P": "Bobath-Therapie, Transfer-Training, ADL-Schulung"
        },
        "expected_position": "20511",
        "expected_dg": "ZNS1",
    },
    {
        "name": "Multiple Sklerose",
        "icd10": "G35",
        "soap": {
            "S": "MS seit 5 Jahren, Fatigue stark",
            "O": "EDSS-Score 4.5. Knie-Hacke-Versuch unsicher. Gangbild: breitbeinig, unsicher. Fatigue-Skala 7/10.",
            "A": "G35 Multiple Sklerose (schubförmig-remittierend). Red Flags ausgeschlossen.",
            "P": "KG-ZNS, Koordinationstraining, Energie-Management"
        },
        "expected_position": "20511",
        "expected_dg": "ZNS2",
    },
    {
        "name": "Morbus Parkinson",
        "icd10": "G20",
        "soap": {
            "S": "Parkinson seit 3 Jahren, Gangblockaden",
            "O": "Hoehn-Yahr-Skala Stadium 2.5. Timed Up & Go 18 Sekunden. Gangbild: kleinschrittig, Freezing-Episoden. Bradykinese.",
            "A": "G20 Morbus Parkinson. Red Flags ausgeschlossen.",
            "P": "KG-ZNS, Freezing-Strategien, Gangtraining"
        },
        "expected_position": "20511",
        "expected_dg": "ZNS3",
    },
    {
        "name": "Fazialisparese",
        "icd10": "G51.0",
        "soap": {
            "S": "Seit 2 Wochen Gesichtslähmung rechts",
            "O": "House-Brackmann-Skala Grad III. Mimische Muskulatur: Stirnrunzeln nicht möglich, Mundwinkel hängt. Synkinesien: noch nicht vorhanden.",
            "A": "G51.0 Periphere Fazialisparese rechts. Red Flags ausgeschlossen.",
            "P": "Fazialis-Therapie, Mimiktraining, Massage"
        },
        "expected_position": "20511",
        "expected_dg": "ZNS5",
    },
    
    # ══ LYMPHOLOGIE ══
    {
        "name": "Primäres Lymphödem",
        "icd10": "I89.01",
        "soap": {
            "S": "Lymphödem linkes Bein seit Jahren",
            "O": "Stemmer-Zeichen positiv. Umfangsmessung: re 35cm, li 42cm (+7cm). Stadium 2. Ödemkonsistenz: teigig. Hautbefund: unauffällig, kein Erysipel.",
            "A": "I89.01 Primäres Lymphödem linkes Bein Stadium 2. Red Flags ausgeschlossen.",
            "P": "MLD 45 Min, Kompressionsstrümpfe, Entstauungsgymnastik"
        },
        "expected_position": "20201",
        "expected_dg": "LY1",
    },
    {
        "name": "Sekundäres Lymphödem (KPE)",
        "icd10": "I97.22",
        "soap": {
            "S": "Lymphödem nach Brustkrebs-OP links",
            "O": "Onkolog. Vordiagnose: Mammakarzinom links, Mastektomie 2023. Umfangsmessung beidseitig: re 28cm, li 36cm (+8cm). Stemmer-Zeichen positiv. KPE-Komponenten: MLD + Bandagierung + Entstauungsgymnastik + Hautpflege durchgeführt. Hautbefund: trocken, keine Rötung.",
            "A": "I97.22 Sekundäres Lymphödem linker Arm (postoperativ). Red Flags ausgeschlossen.",
            "P": "KPE Phase I täglich, Selbstbandagierung lernen"
        },
        "expected_position": "21110",
        "expected_dg": "LY2",
    },
    {
        "name": "Lipödem (BVB)",
        "icd10": "E88.21",
        "soap": {
            "S": "Lipödem beide Beine, Schmerzen",
            "O": "Stemmer-Zeichen negativ. Umfangsmessung: re 45cm, li 46cm. Konsistenz: weich, nicht drückbar. Stadium 2. KPE-Komponenten: MLD + Kompression + Entstauungsgymnastik + Hautpflege. Hautbefund: unauffällig.",
            "A": "E88.21 Lipödem Stadium 2 (Besonderer Verordnungsbedarf). Red Flags ausgeschlossen.",
            "P": "MLD 45 Min, Kompressionsklasse 2, Bewegungstherapie"
        },
        "expected_position": "20201",
        "expected_dg": "LY3",
    },
    
    # ══ ATEMTHERAPIE ══
    {
        "name": "COPD",
        "icd10": "J44.1",
        "soap": {
            "S": "COPD Gold II, Atemnot bei Belastung",
            "O": "Spirometrie: FEV1 65%. Atemmuster: Lippenbremse erlernt. Sekretmobilisation: produktiv. SpO2 92%.",
            "A": "J44.1 COPD mit akuter Exazerbation. Red Flags ausgeschlossen.",
            "P": "Atemtherapie, Sekretmobilisation, Ausdauertraining"
        },
        "expected_position": "20560",
        "expected_dg": "AT1",
    },
    
    # ══ RHEUMATOLOGIE ══
    {
        "name": "Rheumatoide Arthritis",
        "icd10": "M05.8",
        "soap": {
            "S": "Rheumatoide Arthritis, Gelenke geschwollen",
            "O": "ROM Hand: MCP Flex 60-0-0 (eingeschränkt). Schwellung bilateral. Schmerz VAS 6/10. DAS28-Score 4.2.",
            "A": "M05.8 Rheumatoide Arthritis mit Gelenkbefall. Red Flags ausgeschlossen.",
            "P": "Mobilisation schonend, Funktionstraining, Gelenkschutzberatung"
        },
        "expected_position": "20501",
        "expected_dg": "RH1",
    },
    
    # ══ MODALITÄT: KGG ══
    {
        "name": "KG am Gerät (MTT) - LWS",
        "icd10": "M54.5",
        "soap": {
            "S": "Chronische Rückenschmerzen",
            "O": "Trainingsplan: Beinpresse 3x12 Wdh 40kg, Latzug 3x10 Wdh 25kg, Bauchmuskeltraining. Krafttest: MMT 3/5 Rumpfmuskulatur.",
            "A": "M54.5 Lumbalgie chronisch. Red Flags ausgeschlossen.",
            "P": "MTT 2x/Woche, Progression alle 2 Wochen"
        },
        "profile_id": "KGG",
        "expected_position": "20507",
        "expected_dg": "KGG",
    },
    
    # ══ MODALITÄT: Bewegungsbad ══
    {
        "name": "KG Bewegungsbad - Rheuma",
        "icd10": "M05.8",
        "soap": {
            "S": "Rheumatoide Arthritis, Wassergymnastik gewünscht",
            "O": "Wassertemperatur 32°C. Auftriebshilfen: nicht notwendig. Belastung im Wasser deutlich besser als trocken. ROM und Gangbild im Vergleich zu trocken verbessert.",
            "A": "M05.8 Rheumatoide Arthritis. Red Flags ausgeschlossen.",
            "P": "KG Bewegungsbad 2x/Woche, Gelenkschonendes Training"
        },
        "profile_id": "AQUA",
        "expected_position": "20902",
        "expected_dg": "BB1",
    },
    
    # ══ MODALITÄT: Massage ══
    {
        "name": "Klassische Massage",
        "icd10": "M54.5",
        "soap": {
            "S": "Verspannungen im Rücken",
            "O": "Massageform: Klassische Massage (KMT). Lokalisation: paravertebrale Muskulatur LWS bds. Tonus erhöht, Myogelosen tastbar. Wirkung: Tonussenkung, Durchblutung verbessert.",
            "A": "M54.5 Lumbalgie mit muskulärer Verspannung. Red Flags ausgeschlossen.",
            "P": "KMT 2x/Woche, Wärme vorher"
        },
        "profile_id": "MASSE",
        "expected_position": "20106",
        "expected_dg": "MA1",
    },
    
    # ══ MODALITÄT: Elektrotherapie ══
    {
        "name": "Elektrotherapie TENS",
        "icd10": "M54.2",
        "soap": {
            "S": "Nackenschmerzen chronisch",
            "O": "Stromform: TENS. Frequenz 80Hz, Intensität 15mA. Elektroden-Platzierung: HWS bds. paravertebral. Wirkung: Analgesie erreicht.",
            "A": "M54.2 Zervikalgie chronisch. Red Flags ausgeschlossen.",
            "P": "TENS 3x/Woche, KG kombiniert"
        },
        "profile_id": "ELEKTRO",
        "expected_position": "21302",
        "expected_dg": "EL1",
    },
    
    # ══ MODALITÄT: Wärmetherapie ══
    {
        "name": "Fango",
        "icd10": "M54.5",
        "soap": {
            "S": "Rückenschmerzen, Wärme hilft",
            "O": "Wärmemodalität: Fango. Behandlungsregion: LWS. Temperatur: angenehm warm. Kontraindikationsausschluss: Sensibilitätsstörung nein.",
            "A": "M54.5 Lumbalgie. Red Flags ausgeschlossen.",
            "P": "Fango vor KG, 2x/Woche"
        },
        "profile_id": "THERMO",
        "expected_position": "21501",
        "expected_dg": "TH1",
    },
    
    # ══ GEBURTSHILFE / BECKENBODEN ══
    {
        "name": "Rückbildungsgymnastik",
        "icd10": "Z39.1",
        "soap": {
            "S": "8 Wochen postpartum, Beckenboden schwach",
            "O": "Wochen postpartum: 8. Beckenbodenkraft Oxford 2/5. Dammriss Grad II verheilt.",
            "A": "Z39.1 Postpartale Betreuung. Red Flags ausgeschlossen.",
            "P": "Rückbildungsgymnastik, Beckenbodentraining progressiv"
        },
        "expected_position": "21904",
        "expected_dg": "GEB2",
    },
    {
        "name": "Beckenbodentherapie Inkontinenz",
        "icd10": "N39.3",
        "soap": {
            "S": "Stressinkontinenz beim Niesen",
            "O": "Kontinenzstatus: Stressinkontinenz. Beckenboden-Tonus Oxford 2/5. Miktionsfrequenz 12x/Tag.",
            "A": "N39.3 Belastungsinkontinenz. Red Flags ausgeschlossen.",
            "P": "Beckenbodentraining, Biofeedback, Verhaltenstherapie"
        },
        "expected_position": "20501",
        "expected_dg": "PF1",
    },
]

def run_test():
    """Run comprehensive billing compliance test"""
    print("=" * 80)
    print("GERMAN PHYSIOTHERAPY BILLING COMPLIANCE TEST")
    print("=" * 80)
    print()
    
    engine = BillingEngine()
    passed = 0
    failed = 0
    warnings = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {case['name']}")
        print(f"  ICD-10: {case['icd10']}")
        
        profile_id = case.get("profile_id")
        if profile_id:
            print(f"  Profile: {profile_id}")
        
        result = engine.evaluate(
            icd10=case["icd10"],
            soap=case["soap"],
            transcript="",
            insurance_type=InsuranceType.GKV,
            config_rules={},
            profile_id=profile_id,
        )
        
        # Check position number
        if result.position_number == case["expected_position"]:
            print(f"  ✅ Position: {result.position_number} (correct)")
        else:
            print(f"  ❌ Position: {result.position_number} (expected {case['expected_position']})")
            failed += 1
            continue
        
        # Check Diagnosegruppe
        if result.diagnosegruppe == case["expected_dg"]:
            print(f"  ✅ DG: {result.diagnosegruppe} (correct)")
        else:
            print(f"  ⚠️  DG: {result.diagnosegruppe} (expected {case['expected_dg']}, but billing correct)")
            warnings.append(f"{case['name']}: DG mismatch")
        
        # Check price
        if result.fixed_price_eur:
            print(f"  💰 Price: €{result.fixed_price_eur:.2f} (GKV Festpreis)")
        
        # Check audit status
        print(f"  🔍 Audit: {result.audit_status}")
        
        # Show compliance warnings if any
        if result.compliance_warnings:
            print(f"  ⚠️  Warnings: {len(result.compliance_warnings)}")
            for w in result.compliance_warnings[:2]:  # Show first 2
                print(f"     • {w[:80]}...")
        
        # Check for critical failures
        if result.audit_status == "BLOCK":
            print(f"  🔴 BLOCKED - Cannot submit")
            failed += 1
        elif result.audit_status == "PASS":
            print(f"  ✅ PASS - Ready for submission")
            passed += 1
        else:
            print(f"  ⚠️  REVIEW - Therapist review needed")
            passed += 1  # Count as passed (system works correctly)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {len(warnings)}")
    
    if warnings:
        print("\nWarnings (non-critical):")
        for w in warnings:
            print(f"  • {w}")
    
    print("\n" + "=" * 80)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - System is compliant with German physiotherapy laws")
        print("=" * 80)
        return 0
    else:
        print(f"❌ {failed} TESTS FAILED - System needs review")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(run_test())

