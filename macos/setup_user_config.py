#!/usr/bin/env python3
"""Setup user configuration for Kura"""
import os
import shutil

user_kura_dir = os.path.expanduser("~/Documents/Kura")
os.makedirs(user_kura_dir, exist_ok=True)

source_env = os.path.join(os.path.dirname(__file__), '..', '.env')
dest_env = os.path.join(user_kura_dir, '.env')

if os.path.exists(source_env):
    shutil.copy(source_env, dest_env)
    print(f"✅ Copied .env to {dest_env}")
else:
    source_env = os.path.join(os.path.dirname(__file__), '..', '.env.example')
    if os.path.exists(source_env):
        shutil.copy(source_env, dest_env)
        print(f"⚠️ Copied .env.example to {dest_env} - please add credentials")
    else:
        print("❌ No .env or .env.example found!")

print(f"\nYour config: {dest_env}")

