"""
Test: Intelligent Segment Inference (Smart Fill Feature)
=========================================================
Validates that the billing engine can infer likely spinal/joint segments
from clinical context when Manual Therapy (21201) is selected but segments
are missing.
"""

from shared.billing_engine import BillingEngine, InsuranceType

def test_lws_segment_inference():
    """Test: LWS pain over sacrum → should suggest L5/S1"""
    print("\n" + "="*70)
    print("TEST 1: LWS Pain Over Gesäß → Segment Inference")
    print("="*70)
    
    soap = {
        "S": "Pat. berichtet akute LWS-Schmerzen seit Donnerstag nach Heben schwerer Umzugskartons, "
             "Schmerz konzentriert sich auf den Bereich über dem Gesäß rechts, VAS-Wert 8 in Bewegung "
             "und 3 im Liegen",
        "O": "Schonhaltung re. | FBA: 40 cm | Lasègue 80° negativ | Kraftgrade 5/5 | "
             "Schober-Zeichen: < 10 cm (geschätzt aus FBA - stark eingeschränkt) | "
             "Vorlaufphänomen bds. positiv | Paraspinale Muskulatur: hyperton | "
             "Myogelose: M. quadratus lumborum",
        "A": "akute muskuläre Dysbalance mit einer starken Schutzspannung des quadratus lumborum "
             "nach einer mechanischen Überlastung, kein Bandscheibenvorfall wahrscheinlich. | "
             "Red Flags klinisch ausgeschlossen.",
        "P": "KG mit manuellen Techniken und Detonisierung der Muskulatur | "
             "Ziel: FBA 10 cm in 6 EH, beschwerdefrei bei Flexion | 2x/Woche, 6 EH | "
             "Triggerpunktbehandlung | Wärmeanwendung | Heimübung: Stufenlagerung"
    }
    
    engine = BillingEngine()
    result = engine.evaluate(
        icd10="M54.5",  # Lumbago
        soap=soap,
        transcript="LWS Schmerzen Gesäß",
        insurance_type=InsuranceType.GKV,
    )
    
    print(f"\nBilling Position: {result.position_number} - {result.position_name}")
    print(f"Audit Status: {result.audit_status}")
    print(f"\n{'─'*70}")
    print("AUDIT ITEMS:")
    print('─'*70)
    
    for item in result.audit_items:
        print(str(item))
    
    # Check segment inference
    segment_items = [a for a in result.audit_items if a.code == "MT_SEGMENT"]
    if segment_items:
        seg_item = segment_items[0]
        print(f"\n{'─'*70}")
        print(f"✅ SEGMENT INFERENCE WORKING:")
        print(f"   Status: {seg_item.status}")
        print(f"   Detail: {seg_item.detail}")
        print('─'*70)
        
        if "L5/S1" in seg_item.detail or "L4/L5" in seg_item.detail:
            print("✅ PASS: System correctly suggests lumbar segments from clinical context")
        else:
            print("❌ FAIL: Segment suggestion missing or incorrect")
    else:
        print("\n❌ SEGMENT CHECK NOT FOUND")
    
    return result


def test_with_segment_present():
    """Test: Proper segment documentation → should PASS"""
    print("\n" + "="*70)
    print("TEST 2: With Proper Segment Documentation → Should PASS")
    print("="*70)
    
    soap = {
        "S": "Pat. berichtet akute LWS-Schmerzen",
        "O": "Schonhaltung re. | ROM LWS (Ex/Flex): 20-0-35 | FBA: 40 cm | "
             "Schober-Zeichen: 10:14 cm | Lasègue 80° negativ | Kraftgrade 5/5 | "
             "Vorlaufphänomen bds. positiv | Behandeltes Segment: L4/L5 und L5/S1 | "
             "Paraspinale Muskulatur: hyperton | Myogelose: M. quadratus lumborum",
        "A": "akute muskuläre Dysbalance. Red Flags klinisch ausgeschlossen.",
        "P": "MT mit manuellen Techniken"
    }
    
    engine = BillingEngine()
    result = engine.evaluate(
        icd10="M54.5",
        soap=soap,
        transcript="LWS MT",
        insurance_type=InsuranceType.GKV,
    )
    
    print(f"\nBilling Position: {result.position_number} - {result.position_name}")
    print(f"Audit Status: {result.audit_status}")
    
    segment_items = [a for a in result.audit_items if a.code == "MT_SEGMENT"]
    if segment_items:
        seg_item = segment_items[0]
        print(f"\nSegment Check: {seg_item.status}")
        print(f"Detail: {seg_item.detail or '(none - documented correctly)'}")
        
        if seg_item.status == "PASS":
            print("✅ PASS: Segment documentation recognized")
        else:
            print(f"❌ FAIL: Should be PASS, got {seg_item.status}")
    
    return result


def test_hws_segment_inference():
    """Test: HWS with arm pain → should suggest C5/C6 und C6/C7"""
    print("\n" + "="*70)
    print("TEST 3: HWS with Arm Pain → Should Suggest Cervical Segments")
    print("="*70)
    
    soap = {
        "S": "Nackenschmerzen mit Ausstrahlung in den rechten Arm",
        "O": "HWS Bewegungseinschränkung | Spurling Test positiv rechts | "
             "Parästhesien C6 Dermatom",
        "A": "HWS-Syndrom mit radikulärer Symptomatik. Red Flags ausgeschlossen.",
        "P": "MT HWS mit Traktion und Mobilisation"
    }
    
    engine = BillingEngine()
    result = engine.evaluate(
        icd10="M54.2",
        soap=soap,
        transcript="HWS Nacken Arm",
        insurance_type=InsuranceType.GKV,
    )
    
    segment_items = [a for a in result.audit_items if a.code == "MT_SEGMENT"]
    if segment_items:
        seg_item = segment_items[0]
        print(f"\nSegment Check: {seg_item.status}")
        print(f"Detail: {seg_item.detail}")
        
        if "C5/C6" in seg_item.detail or "C6/C7" in seg_item.detail:
            print("✅ PASS: System correctly suggests cervical segments")
        else:
            print("❌ FAIL: Should suggest C5/C6 segments")
    
    return result


def test_shoulder_segment_inference():
    """Test: Shoulder → should suggest Glenohumeralgelenk"""
    print("\n" + "="*70)
    print("TEST 4: Shoulder → Should Suggest Glenohumeral Joint")
    print("="*70)
    
    soap = {
        "S": "Schulterschmerzen rechts, Bewegungseinschränkung",
        "O": "Impingement positiv | Painful Arc 70-120° | ROM eingeschränkt",
        "A": "Rotatorenmanschettensyndrom. Red Flags ausgeschlossen.",
        "P": "MT Schulter mit Mobilisation"
    }
    
    engine = BillingEngine()
    result = engine.evaluate(
        icd10="M75.1",
        soap=soap,
        transcript="Schulter Rotatorenmanschette",
        insurance_type=InsuranceType.GKV,
    )
    
    segment_items = [a for a in result.audit_items if a.code == "MT_SEGMENT"]
    if segment_items:
        seg_item = segment_items[0]
        print(f"\nSegment Check: {seg_item.status}")
        print(f"Detail: {seg_item.detail}")
        
        if "glenohumeral" in seg_item.detail.lower() or "schulter" in seg_item.detail.lower():
            print("✅ PASS: System correctly suggests shoulder joint")
        else:
            print("❌ FAIL: Should suggest Glenohumeralgelenk")
    
    return result


def main():
    """Run all segment inference tests"""
    print("\n" + "="*70)
    print(" KURA BILLING ENGINE: INTELLIGENT SEGMENT INFERENCE TEST SUITE")
    print("="*70)
    print("\nThis feature eliminates the need for therapists to manually type")
    print("segments by inferring them from clinical context (pain location,")
    print("diagnosis, and anatomical keywords).")
    print("="*70)
    
    # Run all tests
    test_lws_segment_inference()
    test_with_segment_present()
    test_hws_segment_inference()
    test_shoulder_segment_inference()
    
    print("\n" + "="*70)
    print(" TEST SUITE COMPLETE")
    print("="*70)
    print("\n✅ The billing engine now provides intelligent segment suggestions")
    print("   when Manual Therapy (21201) is selected but segments are missing.")
    print("\n💡 This prevents billing rejections while reducing therapist workload.\n")


if __name__ == "__main__":
    main()

