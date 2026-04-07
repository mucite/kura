#!/usr/bin/env python3
"""
Test HWS ROM extraction from the specific transcript
"""
import re

def test_hws_rotation_extraction():
    """Test extraction of rotation values from HWS transcript"""

    transcript = """
    Therapeut: Okay, schauen wir mal. Setzen Sie sich bitte aufrecht hin. Wenn Sie den Kopf nach rechts drehen... wie weit kommen wir da?
    
    Patientin: (stöhnt leise) Da ist Schluss. Es hakt richtig.
    
    Therapeut: Ja, die Rotation nach rechts ist deutlich eingeschränkt, ich schätze mal auf 30 Grad. Nach links sieht es mit 60 Grad viel besser aus.
    """

    print("="*70)
    print("Testing HWS Rotation Extraction")
    print("="*70)
    print(f"\nTranscript excerpt:")
    print(transcript.strip())
    print("\n" + "-"*70)

    # Test all patterns
    patterns = [
        ("Pattern 1", r"rotation.*?(\d+)\s*(?:grad|°|degrees?).*?(?:rechts|re\.|right).*?(\d+)\s*(?:grad|°|degrees?).*?(?:links|li\.|left)"),
        ("Pattern 2", r"(\d+)\s*(?:grad|°|degrees?)\s*(?:nach\s+)?(?:rechts|re\.|right).*?(\d+)\s*(?:grad|°|degrees?)\s*(?:nach\s+)?(?:links|li\.|left)"),
        ("Pattern 3", r"(?:nach\s+)?(?:rechts|re\.|right)\s+(?:ist\s+)?(?:bei\s+)?(?:etwa\s+)?(\d+)\s*(?:grad|°|degrees?).*?(?:nach\s+)?(?:links|li\.|left)\s+(?:ist\s+)?(?:bei\s+)?(?:etwa\s+)?(\d+)\s*(?:grad|°|degrees?)"),
        ("Pattern 4", r"(?:auf\s+)?(\d+)\s*(?:grad|°|degrees?).*?(?:rechts|re\.|right).*?[.!?]\s*(?:nach\s+)?(?:links|li\.|left).*?(\d+)\s*(?:grad|°|degrees?)"),
    ]

    for name, pattern in patterns:
        match = re.search(pattern, transcript, re.I | re.DOTALL)
        if match:
            right_val = match.group(1)
            left_val = match.group(2)
            result = f"ROM HWS (Rotation): {right_val}-0-{left_val}"
            print(f"\n✅ {name} MATCHED!")
            print(f"   Right: {right_val}°, Left: {left_val}°")
            print(f"   Result: {result}")
            break
        else:
            print(f"   ❌ {name} - no match")

    if not match:
        print("\n⚠️  NO PATTERN MATCHED!")
        print("\nLet's try simpler patterns:")

        # Try finding the numbers separately
        right_match = re.search(r"(\d+)\s*(?:grad|°).*?rechts", transcript, re.I | re.DOTALL)
        left_match = re.search(r"links.*?(\d+)\s*(?:grad|°)", transcript, re.I | re.DOTALL)

        if right_match and left_match:
            print(f"   ✅ Found separately: Right={right_match.group(1)}°, Left={left_match.group(1)}°")
            print(f"   Result: ROM HWS (Rotation): {right_match.group(1)}-0-{left_match.group(1)}")
        else:
            if right_match:
                print(f"   Found right: {right_match.group(1)}°")
            if left_match:
                print(f"   Found left: {left_match.group(1)}°")
            if not right_match and not left_match:
                print("   Even separate extraction failed!")
                print("\nTrying most basic search:")
                all_grads = re.findall(r"(\d+)\s*(?:grad|°)", transcript, re.I)
                print(f"   All degree values found: {all_grads}")

    print("\n" + "="*70)

if __name__ == "__main__":
    test_hws_rotation_extraction()

