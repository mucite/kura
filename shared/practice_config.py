"""
Practice Configuration Manager
Handles multi-practice deployment with § 125 Abs. 1 SGB V compliance
"""
import json
import logging
import os
import platform
import sys
from datetime import datetime

from ._compat import fix_windows_encoding

fix_windows_encoding()

logger = logging.getLogger("kura.practice_config")

class PracticeConfig:
    """Manages practice-specific configurations"""
    
    def __init__(self, practice_name: str = None, practice_file: str = None):
        """
        Initialize practice config
        
        Args:
            practice_name: Name of the practice
            practice_file: Path to practice config JSON
        """
        # Platform-specific default path
        if practice_file is None:
            if platform.system() == "Windows":
                practice_file = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Kura", "kura_practice.json")
            else:
                practice_file = os.path.expanduser("~/.kura_practice.json")

        # Ensure directory exists
        practice_dir = os.path.dirname(practice_file)
        try:
            os.makedirs(practice_dir, exist_ok=True)
        except Exception as dir_err:
            print(f"Warning: Could not create practice config directory: {dir_err}")

        self.practice_file = practice_file
        self.practice_name = practice_name
        self.config = self._load_or_create_config()
        
    def _load_or_create_config(self):
        """Load existing practice config or create default"""
        if os.path.exists(self.practice_file):
            try:
                with open(self.practice_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._default_config()
        else:
            return self._default_config()
    
    def _default_config(self):
        """Default multi-practice compliant configuration"""
        return {
            "version": "2026.1.0-multi-practice",
            "compliance_standard": "§ 125 Abs. 1 SGB V",
            "practice": {
                "name": self.practice_name or "Standard Praxis",
                "license_number": "",  # Betriebsstättennummer
                "location": "",
                "therapist_ids": []
            },
            "billing": {
                "MT": "21201",  # Manuelle Therapie
                "KG": "20501",  # Krankengymnastik
                "KPE": "21110"  # KPE Phase I
            },
            "icd10_rules": {
                "HWS": {
                    "keywords": ["Nacken", "Hals", "Kopfschmerz", "Spannungskopfschmerz"],
                    "primary_code": "M54.2",
                    "alternative_code": "G44.2",
                    "billing_default": "21201"
                },
                "LWS": {
                    "keywords": ["Rücken", "Lumbal", "Lendenwirbel"],
                    "primary_code": "M54.5",
                    "alternative_code": "M51.1",  # Only with surgery
                    "billing_default": "20501"
                },
                "Schulter": {
                    "keywords": ["Schulter", "Arm", "Impingement"],
                    "primary_code": "M75.5",
                    "alternative_code": "M75.0",
                    "billing_default": "21201"
                }
            },
            "audit_rules": {
                "mandatory_fields": ["S", "O", "A", "P"],
                "min_objective_length": 20,
                "red_flags": [
                    "neurologisch",
                    "Taubheit",
                    "Kraftverlust",
                    "Parese",
                    "Reflexverlust",
                    "Querschnitt",
                    "Schlaganfall"
                ],
                "rom_keywords": ["ROM", "Beweglichkeit", "Flexion", "Extension", "Rotation", "Lateralflexion"],
                "compliance_warnings": []
            },
            "multi_user": {
                "enabled": False,
                "users": []
            },
            "data_protection": {
                "dsgvo_compliant": True,
                "local_processing_only": True,
                "data_retention_days": 365,
                "encryption_enabled": True
            },
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
    
    def save(self):
        """Save practice config to file"""
        self.config["last_modified"] = datetime.now().isoformat()
        try:
            with open(self.practice_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ Practice config saved: {self.practice_file}")
        except Exception as write_err:
            print(f"Practice config save error: {write_err}")

    def add_user(self, user_id: str, username: str, role: str = "therapist"):
        """Add therapist user to practice"""
        if "multi_user" not in self.config:
            self.config["multi_user"] = {"enabled": True, "users": []}
        
        self.config["multi_user"]["enabled"] = True
        self.config["multi_user"]["users"].append({
            "user_id": user_id,
            "username": username,
            "role": role,
            "created_at": datetime.now().isoformat()
        })
        self.save()
        logger.info(f"User {username} added to practice {self.practice_name}")

    def get_icd10_for_keywords(self, keywords: list):
        """Get recommended ICD-10 code based on keywords"""
        keywords_lower = [k.lower() for k in keywords]
        
        for category, rules in self.config["icd10_rules"].items():
            category_keywords = [k.lower() for k in rules["keywords"]]
            if any(kw in " ".join(keywords_lower) for kw in category_keywords):
                return {
                    "category": category,
                    "primary": rules["primary_code"],
                    "alternative": rules["alternative_code"],
                    "billing": rules["billing_default"]
                }
        
        # Default fallback
        return {
            "category": "Unknown",
            "primary": "M54.2",
            "alternative": "M54.5",
            "billing": "20501"
        }
    
    def validate_compliance(self, soap_dict: dict):
        """Validate SOAP note for § 125 SGB V compliance"""
        issues = []
        
        # Check mandatory fields
        for field in self.config["audit_rules"]["mandatory_fields"]:
            if field not in soap_dict or not soap_dict[field]:
                issues.append(f"⚠️ Mandatory field '{field}' is missing")
        
        # Check objective minimum length
        if "O" in soap_dict:
            if len(str(soap_dict["O"])) < self.config["audit_rules"]["min_objective_length"]:
                issues.append("⚠️ Objektiver Befund zu kurz für Audit")
        
        # Check for red flags
        all_text = " ".join(str(v) for v in soap_dict.values()).lower()
        for flag in self.config["audit_rules"]["red_flags"]:
            if flag.lower() in all_text:
                issues.append(f"🚩 Red flag detected: {flag}")
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "practice": self.practice_name,
            "compliance_standard": "§ 125 Abs. 1 SGB V"
        }


