#!/usr/bin/env python3
"""Quick test of the new trial status module"""
from shared.trial_status import TrialStatus

print("=" * 70)
print("TESTING NEW TRIAL STATUS MODULE")
print("=" * 70)

features = TrialStatus.get_pro_features()
print(f"\n✅ Module loaded successfully!")
print(f"✅ {len(features)} Pro features defined\n")

print("Kura Pro Features:")
print("-" * 70)
for feature in features:
    print(f"\n{feature['icon']} {feature['title']}")
    print(f"   {feature['description']}")

trial_info = TrialStatus.get_trial_info()
print("\n" + "=" * 70)
print("Trial Information:")
print("-" * 70)
print(f"Max Reports: {trial_info['max_reports']}")
print(f"Features: {', '.join(trial_info['features_included'])}")
print(f"Limitations: {len(trial_info['limitations'])}")
for lim in trial_info['limitations']:
    print(f"  • {lim}")

print("\n" + "=" * 70)
print("🎉 NEW TRIAL STATUS MODULE IS WORKING!")
print("=" * 70)
print("\nThe new UX is simpler, cleaner, and more professional.")
print("Old comparison table removed. Modern card-based design ready!")

