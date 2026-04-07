#!/usr/bin/env python3
"""
Test script to verify the ICD-10 regex fix
Tests the fix for "invalid group reference 3 at position 6" error
"""
import re

def test_icd_regex_fix():
    """Test the fixed ICD-10 regex pattern"""

    print("Testing ICD-10 regex fix...")

    # The OLD BUGGY pattern (would fail):
    # r"\b([A-Z]\d{2})\s+(\d{1,2})(?:\.\s*\w)?\b":        r"\1\2.\3"
    # This has only 2 capture groups but tries to use \3!

    # The NEW FIXED pattern:
    # r"\b([A-Z]\d{2})\s+(\d{1,2})\b":                     r"\1.\2"

    # According to the comment in the code, we need to handle:
    # 1. "M 54.5" → "M54.5" (space before dot)
    # 2. "M 54. 5" → "M54.5" (space before and after dot)
    # But actually, these are DIFFERENT patterns!

    # First, let's test the pattern that's actually in the code now
    test_cases = [
        ("M54 5", "M54.5"),   # "M54 5" → "M54.5" (space between code parts, no dot)
        ("M17 1", "M17.1"),   # "M17 1" → "M17.1"
        ("M75 4", "M75.4"),   # "M75 4" → "M75.4"
        ("I89 0", "I89.0"),   # "I89 0" → "I89.0"
        ("G51 0", "G51.0"),   # "G51 0" → "G51.0"
    ]

    pattern = r"\b([A-Z]\d{2})\s+(\d{1,2})\b"
    replacement = r"\1.\2"

    print("\n✅ Testing NEW pattern (should work):")
    for input_text, expected in test_cases:
        try:
            result = re.sub(pattern, replacement, input_text)
            print(f"   '{input_text}' → '{result}' (expected: '{expected}')")
            assert result == expected, f"Expected {expected}, got {result}"
        except Exception as e:
            print(f"   ❌ FAILED: '{input_text}' → ERROR: {e}")
            raise

    print("\n✅ All ICD-10 regex tests passed!")

    # Also test that the pattern doesn't break normal ICD codes
    normal_cases = [
        ("M54.5", "M54.5"),  # Already formatted correctly
        ("M17.1 Gonarthrose", "M17.1 Gonarthrose"),  # In context
    ]

    print("\n✅ Testing that normal ICD codes are not affected:")
    for input_text, expected in normal_cases:
        result = re.sub(pattern, replacement, input_text)
        print(f"   '{input_text}' → '{result}'")
        assert result == expected, f"Expected {expected}, got {result}"

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED! The regex fix is working correctly.")
    print("="*60)

if __name__ == "__main__":
    test_icd_regex_fix()

