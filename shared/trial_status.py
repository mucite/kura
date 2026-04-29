"""
Simple Trial Status Module
Clean, modern UX for showing trial information and upgrade path
"""


class TrialStatus:
    """Simple trial status information"""

    @staticmethod
    def get_pro_features() -> list[dict]:
        """Return list of Kura Pro features with icons"""
        return [
            {
                "icon": "📋",
                "title": "79 Abrechnungsziffern",
                "description": "Alle Heilmittel vollständig abrechenbar"
            },
            {
                "icon": "🎯",
                "title": "ICD-spezifische Regeln",
                "description": "Präzise Validierung für jeden Diagnose-Code"
            },
            {
                "icon": "✅",
                "title": "Pflichtfeld-Prüfung",
                "description": "MT-Segmente, Barthel-Index, Red Flags"
            },
            {
                "icon": "🔄",
                "title": "Updates bis 31.12.2027 inkl.",
                "description": "Ab 2028 optional 79 €/Jahr für Regel-Updates"
            },
            {
                "icon": "∞",
                "title": "Unbegrenzte Nutzung",
                "description": "Keine Beschränkung der Berichte"
            },
            {
                "icon": "💎",
                "title": "Premium Support",
                "description": "Prioritäts-Hilfe bei Fragen"
            }
        ]

    @staticmethod
    def get_trial_info() -> dict:
        """Return trial information"""
        return {
            "max_reports": 5,
            "features_included": ["KG", "MT", "KG-ZNS", "MLD"],
            "limitations": [
                "Nur 4 Basis-Abrechnungsziffern",
                "Keine automatischen Updates",
                "Begrenzt auf 5 Berichte"
            ]
        }

