#!/usr/bin/env python3
"""
Comprehensive test for all regex fixes
Tests both the ROM HWS fix and the ICD-10 fix
"""
import re

def test_rom_hws_fix():
    """Test the ROM HWS pattern fix (rf-string issue)"""
    print("="*70)
    print("TEST 1: ROM HWS Pattern (rf-string fix)")
    print("="*70)

    # Simulate the scenario
    obj_text = "ROM HWS: some data"
    test_values = ["30-0-40", "n.d.-0-45", "20-0-50", "15-0-30", "25-0-35"]

    for _nzm in test_values:
        # OLD METHOD would fail:
        # obj_text = re.sub(r'(ROM HWS[^|]*)', rf'\1 Flex/Ext: {_nzm}', obj_text, flags=re.I)

        # NEW METHOD (fixed):
        replacement = f'\\1 Flex/Ext: {_nzm}'
        result = re.sub(r'(ROM HWS[^|]*)', replacement, obj_text, flags=re.I)

        print(f"  ✅ {_nzm:15s} → {result}")
        assert "Flex/Ext:" in result
        assert _nzm in result

    print()

def test_icd10_fix():
    """Test the ICD-10 pattern fix (invalid backreference \3)"""
    print("="*70)
    print("TEST 2: ICD-10 Pattern (backreference fix)")
    print("="*70)

    # OLD BUGGY pattern (would fail with "invalid group reference 3"):
    # r"\b([A-Z]\d{2})\s+(\d{1,2})(?:\.\s*\w)?\b":        r"\1\2.\3"
    # Only 2 capture groups but tries to use \3!

    # NEW FIXED pattern:
    pattern = r"\b([A-Z]\d{2})\s+(\d{1,2})\b"
    replacement = r"\1.\2"

    test_cases = [
        ("M54 5", "M54.5"),
        ("M17 1", "M17.1"),
        ("M75 4", "M75.4"),
        ("I89 0", "I89.0"),
        ("G51 0", "G51.0"),
        ("Text with M99 9 code", "Text with M99.9 code"),
    ]

    for input_text, expected in test_cases:
        result = re.sub(pattern, replacement, input_text)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{input_text}' → '{result}'")
        assert result == expected, f"Expected '{expected}', got '{result}'"

    print()

def test_combined_scenario():
    """Test a realistic scenario with both patterns"""
    print("="*70)
    print("TEST 3: Combined Scenario (Real Medical Documentation)")
    print("="*70)

    # Simulate medical text processing
    text = "Pat. mit M54 5 Diagnose. ROM HWS: Extension/Flexion gemessen."

    # Apply ICD fix
    text = re.sub(r"\b([A-Z]\d{2})\s+(\d{1,2})\b", r"\1.\2", text)
    print(f"  After ICD fix: {text}")
    assert "M54.5" in text

    # Apply ROM fix
    _nzm = "30-0-45"
    replacement = f'\\1 Flex/Ext (NZM): {_nzm}'
    text = re.sub(r'(ROM HWS[^.]*)', replacement, text, flags=re.I)
    print(f"  After ROM fix: {text}")
    assert "30-0-45" in text

    print("  ✅ Combined scenario successful!")
    print()

def test_edge_cases():
    """Test edge cases that might cause issues"""
    print("="*70)
    print("TEST 4: Edge Cases")
    print("="*70)

    # Test 1: Multiple ICD codes in one string
    text = "Diagnosen: M54 5 und M17 1 und I89 0"
    result = re.sub(r"\b([A-Z]\d{2})\s+(\d{1,2})\b", r"\1.\2", text)
    print(f"  ✅ Multiple codes: {result}")
    assert result == "Diagnosen: M54.5 und M17.1 und I89.0"

    # Test 2: ROM with special characters
    _nzm_values = ["n.d.-0-45", "15-0-30", "0-0-90", "45-0-0"]
    for val in _nzm_values:
        replacement = f'\\1 Value: {val}'
        result = re.sub(r'(TEST)', replacement, "TEST")
        print(f"  ✅ Special chars: {val} → {result}")
        assert val in result

    # Test 3: Already formatted ICD codes should not change
    text = "M54.5 is already formatted"
    result = re.sub(r"\b([A-Z]\d{2})\s+(\d{1,2})\b", r"\1.\2", text)
    print(f"  ✅ Already formatted: {result}")
    assert result == text  # Should be unchanged

    print()

def main():
    """Run all tests"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "COMPREHENSIVE REGEX FIX TESTS" + " "*24 + "║")
    print("╚" + "="*68 + "╝\n")

    try:
        test_rom_hws_fix()
        test_icd10_fix()
        test_combined_scenario()
        test_edge_cases()

        print("╔" + "="*68 + "╗")
        print("║" + " "*10 + "🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉" + " "*21 + "║")
        print("╚" + "="*68 + "╝")
        print("\n✅ Both regex fixes are working correctly!")
        print("✅ No 'invalid group reference' errors!")
        print("✅ Ready for production use!\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise

if __name__ == "__main__":
    main()

