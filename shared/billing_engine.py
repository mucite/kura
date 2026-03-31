"""
Kura Dual Billing Engine
========================
GKV — deterministic rules engine, Heilmittelkatalog § 125 SGB V (01.01.2026)
PKV — AI-assisted recommendation engine, GebüTh market reference
BG  — Berufsgenossenschaft, follows GKV codes, DGUV pricing

Design principle: the two systems are completely separated.
Never mix GKV fixed prices with PKV flexibility, and never guarantee
PKV reimbursement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Insurance type ─────────────────────────────────────────────────────────────

class InsuranceType(Enum):
    GKV = "GKV"   # Gesetzliche Krankenversicherung — strict §125 SGB V
    PKV = "PKV"   # Private Krankenversicherung — GebüTh market reference
    BG  = "BG"    # Berufsgenossenschaft — work accident, DGUV, GKV codes


# ── Audit tick-box item ────────────────────────────────────────────────────────

@dataclass
class AuditItem:
    code: str          # machine key, e.g. "ROM", "VAS", "RED_FLAG_PARESE"
    label: str         # human label
    status: str        # "PASS" | "WARN" | "FAIL" | "BLOCK"
    detail: str = ""   # specific missing info or instruction

    @property
    def icon(self) -> str:
        return {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FEHLT]", "BLOCK": "[STOP]"}.get(self.status, "[?]")

    def __str__(self) -> str:
        base = f"{self.icon} {self.label}"
        return f"{base}: {self.detail}" if self.detail else base


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class BillingResult:
    insurance_type: InsuranceType
    position_number: str
    position_name: str
    diagnosegruppe: str
    diagnosegruppe_desc: str
    legal_basis: str
    session_duration_min: int
    risk_level: str                          # "OK" | "WARN" | "BLOCK"
    # Audit
    audit_items: list = field(default_factory=list)    # list[AuditItem]
    audit_status: str = "PASS"              # "PASS" | "REVIEW" | "BLOCK"
    compliance_warnings: list = field(default_factory=list)
    required_documentation: list = field(default_factory=list)
    # GKV-spezifisch
    max_units_regelfall: int = 0
    requires_langfrist_approval: bool = False
    fixed_price_eur: Optional[float] = None        # Festpreis §125 SGB V — unveränderlich
    # PKV-spezifisch
    price_range_eur: Optional[tuple] = None        # GebüTh-Orientierungswerte (Ober-/Untergrenze)
    pkv_praxispreis_eur: Optional[float] = None    # Praxiseigener PKV-Preis (aus config_override.json)
    reimbursement_likelihood: str = ""             # "HOCH" | "MITTEL" | "GERING"
    optimization_hints: list = field(default_factory=list)
    # BG-spezifisch (Berufsgenossenschaft / DGUV)
    bg_surcharge_pct: float = 0.0
    bg_extra_docs: list = field(default_factory=list)

    def format_audit_report(self) -> str:
        """Gibt einen prüfbereiten Textblock für Zwischenablage/PDF zurück."""
        lines = []
        for item in self.audit_items:
            lines.append(str(item))
        status_icon = {"PASS": "✅", "REVIEW": "⚠️", "BLOCK": "🔴"}.get(self.audit_status, "❓")
        lines.append(f"\nSTATUS: {status_icon} {self.audit_status}")
        return "\n".join(lines)

    def format_billing_line(self) -> str:
        """Einzeilige Abrechnungszeile für den Berichtskopf."""
        ins = self.insurance_type.value
        if self.insurance_type == InsuranceType.GKV:
            # Festpreis §125 SGB V — gesetzlich fixiert, keine Abweichung möglich
            price = f"€{self.fixed_price_eur:.2f} (Festpreis §125 SGB V)" if self.fixed_price_eur else ""
        elif self.insurance_type == InsuranceType.BG:
            # DGUV-Vergütung = GKV-Satz + Aufschlag
            if self.fixed_price_eur:
                aufschlag = f"+{self.bg_surcharge_pct:.0f}% DGUV"
                price = f"€{self.fixed_price_eur:.2f} ({aufschlag})"
            else:
                price = ""
        else:
            # PKV: entweder praxiseigener Preis oder GebüTh-Orientierungswert
            if self.pkv_praxispreis_eur:
                price = f"€{self.pkv_praxispreis_eur:.2f} (Praxispreis PKV)"
            elif self.price_range_eur:
                price = (f"€{self.price_range_eur[0]:.0f}–{self.price_range_eur[1]:.0f}"
                         f" (GebüTh-Orientierungswert)")
            else:
                price = ""
        regelfall = f" | max. {self.max_units_regelfall} EH" if self.max_units_regelfall else ""
        langfrist = " | ⚠️ Langfristgenehmigung erforderlich" if self.requires_langfrist_approval else ""
        return f"[{ins}] {self.position_number} – {self.position_name} | {price}{regelfall}{langfrist}"


# ── Heilmittelkatalog — VOLLSTÄNDIG ───────────────────────────────────────────
# Anlage 2 Rahmenempfehlungen §125 SGB V, gültig 01.01.2026
# Quelle: GKV-Spitzenverband Lesefassung 01.01.2026
#
# Struktur je Diagnosegruppe:
#   desc        — Beschreibung
#   heilmittel  — Therapieform
#   position    — Positionsnummer (Standard)
#   name        — Positionsbezeichnung
#   duration    — Regelbehandlungszeit (Minuten)
#   regelfall   — Max. Einheiten Regelfall
#   langfristig — Langfristgenehmigung notwendig
#   icd         — ICD-10-Präfixe (für Lookup)
#   docs        — Pflichtdokumentation für §106b
#   optional_mt — KG kann zu MT aufgewertet werden wenn indiziert

_HMK: dict[str, dict] = {

    # ══ EXTREMITÄTEN ══════════════════════════════════════════════════════════

    "EX1a": {
        "desc": "Periphere Lähmungen/Paresen (Extremitäten)",
        "heilmittel": "KG-ZNS", "position": "20511",
        "name": "Krankengymnastik ZNS (Bobath/PNF/Vojta)",
        "duration": 45, "regelfall": 10, "langfristig": True,
        "icd": ["G54", "G55", "G56", "G57", "G58", "G60", "G61", "G62"],
        "docs": ["Neurolog. Befund", "Tonus (Ashworth-Skala)", "ADL-Status", "Funktionsziel"],
    },
    "EX1b": {
        "desc": "Gelenkstörungen degenerativ/entzündlich/posttraumatisch (Extremitäten)",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "optional_mt": True,
        "icd": ["M00", "M01", "M05", "M06", "M07", "M08",
                "M20", "M21", "M22", "M24", "M25"],
        "docs": ["ROM (Neutral-Null)", "Schmerz (VAS)", "Funktionsziel"],
    },
    "EX2": {
        "desc": "Schultergelenk – Kapsel-/Sehnenläsionen, Impingementsyndrom",
        "heilmittel": "MT", "position": "20701",
        "name": "Manuelle Therapie",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M75"],
        "docs": ["ROM Schulter (Abd/Flex/AR/IR)", "Endgefühl", "Painful Arc", "Krafttest (Jobe/Hawkins)"],
    },
    "EX3": {
        "desc": "Kniegelenk – Gonarthrose, postoperativ, Meniskusläsion",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": False,
        # optional_mt intentionally absent: HMK does not list Kniegelenk as MT indication.
        "icd": ["M17", "M22", "M23", "S82", "S83"],
        "docs": ["ROM Knie (Flex/Ext)", "Umfang", "Kraft (MMT)", "Gangbild", "Knienachgiebigkeit"],
    },
    "EX4": {
        "desc": "Hüftgelenk – Coxarthrose, postoperativ (TEP)",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M16", "S72"],
        "docs": ["ROM Hüfte (Flex/Abd/AR)", "Gangbild", "Muskelkraft (MMT)"],
    },
    "EX5": {
        "desc": "Sprunggelenk/Fuß – Arthrose, posttraumatisch, Achillessehne",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": False,
        # optional_mt removed: Sprunggelenk/Fuß not listed as MT indication in HMK 2026
        "icd": ["M19.0", "M79.6", "S93", "M77.3"],
        "docs": ["ROM Sprunggelenk (DF/PF)", "Schmerz (VAS)", "Gangbild"],
    },
    "EX6": {
        "desc": "Hand/Handgelenk/Ellbogen – Arthrose, Fraktur-Reha, Karpaltunnel, Epikondylitis",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "optional_mt": True,
        "icd": ["M19.0", "M65", "G54.2", "M77.0", "M77.1", "M79.2",
                "S52", "S62", "S60", "S61", "S63", "S64", "S68",  # wrist/hand fractures & trauma
                "T92"],  # sequelae of upper limb injuries
        "docs": ["ROM Hand/Ellbogen (Neutral-Null)", "Griffstärke (kg)", "Schmerz (VAS)"],
    },

    # ══ WIRBELSÄULE ═══════════════════════════════════════════════════════════

    "WS1a": {
        "desc": "HWS/BWS – segmentale Funktionsstörung, Zervikalsyndrom",
        "heilmittel": "MT", "position": "20701",
        "name": "Manuelle Therapie",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M50", "M53", "M54.0", "M54.1", "M54.2", "M99.0", "M99.1"],
        "docs": ["Segmentbefund (C/Th)", "Endgefühl", "Neurolog. Screening", "ROM HWS"],
    },
    "WS1b": {
        "desc": "LWS/ISG – segmentale Funktionsstörung, Lumbago, Ischialgie",
        "heilmittel": "MT", "position": "20701",
        "name": "Manuelle Therapie",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M54.3", "M54.4", "M54.5", "M99.3", "M99.4", "M99.5"],
        "docs": ["Segmentbefund (L/S)", "FBA (Finger-Boden-Abstand)", "Schober-Zeichen", "Lasègue"],
    },
    "WS2": {
        "desc": "Wirbelsäule postoperativ / Bandscheibenoperation",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 10, "langfristig": False,
        "icd": ["M51", "M96", "Z96.6"],
        "docs": ["OP-Bericht vorhanden", "Neurolog. Status", "ROM (Neutral-Null)", "Schmerz (VAS)"],
    },
    "WS3": {
        "desc": "Skoliose – konservativ/postoperativ (Schroth/3D-Therapie)",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung (Schroth/3D)",
        "duration": 20, "regelfall": 10, "langfristig": True,
        "icd": ["M41"],
        "docs": ["Cobb-Winkel", "Rippenbuckel", "Schroth-Klassifikation", "Atemmuster"],
    },
    "WS4": {
        "desc": "Osteoporose – Frakturprophylaxe, Schmerztherapie",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": True,
        "icd": ["M80", "M81"],
        "docs": ["ROM (Neutral-Null)", "Sturzrisiko-Assessment", "Krafttest (MMT)", "Schmerz (VAS)"],
    },

    # ══ ZNS ════════════════════════════════════════════════════════════════════

    "ZNS1": {
        "desc": "Hemiplegie/Hemiparese – Schlaganfall, Hirnverletzung",
        "heilmittel": "KG-ZNS", "position": "20511",
        "name": "Krankengymnastik ZNS nach Bobath",
        "duration": 45, "regelfall": 10, "langfristig": True,
        "icd": ["G81", "I69", "G82", "G83", "S14", "S24", "S34"],
        "docs": ["Barthel-Index", "Ashworth-Skala", "Ganganalyse", "ADL-Status"],
    },
    "ZNS2": {
        "desc": "Multiple Sklerose / Ataxie / spinale Erkrankungen",
        "heilmittel": "KG-ZNS", "position": "20511",
        "name": "Krankengymnastik ZNS",
        "duration": 45, "regelfall": 10, "langfristig": True,
        "icd": ["G35", "G11", "G12", "G95"],
        "docs": ["EDSS-Score", "Koordinationstest (Knie-Hacke)", "Gangbild", "Fatigue-Skala"],
    },
    "ZNS3": {
        "desc": "Morbus Parkinson / extrapyramidale Erkrankungen",
        "heilmittel": "KG-ZNS", "position": "20511",
        "name": "Krankengymnastik ZNS",
        "duration": 45, "regelfall": 10, "langfristig": True,
        "icd": ["G20", "G21"],
        "docs": ["Hoehn-Yahr-Skala", "Timed Up & Go", "Gangbild", "Freezing-Protokoll"],
    },
    "ZNS4": {
        "desc": "Zerebralparese / frühkindliche Hirnschädigung",
        "heilmittel": "KG-ZNS", "position": "20511",
        "name": "Krankengymnastik ZNS (Vojta/Bobath)",
        "duration": 45, "regelfall": 10, "langfristig": True,
        "icd": ["G80"],
        "docs": ["Tonus (Ashworth-Skala)", "Motorische Meilensteine", "ADL-Status", "Barthel-Index"],
    },
    "ZNS5": {
        "desc": "Periphere Fazialisparese / Hirnnervenparesen",
        "heilmittel": "KG-ZNS", "position": "20511",
        "name": "Krankengymnastik ZNS",
        "duration": 45, "regelfall": 6, "langfristig": False,
        "icd": ["G51", "G52", "G53"],
        "docs": ["House-Brackmann-Skala", "Mimische Muskulatur (Befund)", "Synkinesien"],
    },

    # ══ LYMPHOLOGIE ════════════════════════════════════════════════════════════

    "LY1": {
        "desc": "Primäres Lymphödem (kongenital/idiopathisch)",
        "heilmittel": "MLD", "position": "21101",
        "name": "Manuelle Lymphdrainage 45 Min",
        "duration": 45, "regelfall": 6, "langfristig": True,
        "icd": ["Q82.0", "I89.0"],
        "docs": ["Stemmer-Zeichen", "Umfangsmessung (cm)", "Stadium (1-3)", "Ödemkonsistenz"],
    },
    "LY2": {
        "desc": "Sekundäres Lymphödem (postoperativ, Post-Cancer, Bestrahlung)",
        "heilmittel": "KPE", "position": "21110",
        "name": "Komplexe Physikalische Entstauungstherapie Phase I",
        "duration": 60, "regelfall": 6, "langfristig": True,
        "icd": ["I89.0", "I97.2", "I97.89", "C77", "C78", "C79"],
        "docs": ["Ödem-Stadium", "Umfangsmessung beidseitig", "Stemmer-Zeichen", "Onkolog. Vordiagnose"],
    },
    "LY3": {
        "desc": "Lipödem (kombiniert mit Lymphödem)",
        "heilmittel": "MLD", "position": "21101",
        "name": "Manuelle Lymphdrainage 45 Min",
        "duration": 45, "regelfall": 6, "langfristig": True,
        "icd": ["E88.2"],
        "docs": ["Stemmer-Zeichen", "Umfangsmessung (cm)", "Konsistenz", "Stadium (1-3)"],
    },
    "LY4": {
        "desc": "Chronisch venöse Insuffizienz mit sekundärem Lymphödem",
        "heilmittel": "MLD", "position": "21100",
        "name": "Manuelle Lymphdrainage 30 Min",
        "duration": 30, "regelfall": 6, "langfristig": False,
        "icd": ["I83", "I87"],
        "docs": ["Umfangsmessung (cm)", "Ödemkonsistenz", "Stemmer-Zeichen"],
    },

    # ══ ATEMWEGE ═══════════════════════════════════════════════════════════════

    "AT1": {
        "desc": "Chronische Atemwegserkrankungen (COPD, Asthma, Mukoviszidose)",
        "heilmittel": "AT-KG", "position": "20560",
        "name": "Krankengymnastik atemtherapeutisch",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["J44", "J45", "J96", "J98", "E84"],
        "docs": ["Spirometrie", "Atemmuster", "Sekretmobilisation", "SpO2"],
    },
    "AT2": {
        "desc": "Postoperative Atemtherapie / Thoraxchirurgie",
        "heilmittel": "AT-KG", "position": "20560",
        "name": "Krankengymnastik atemtherapeutisch",
        "duration": 20, "regelfall": 10, "langfristig": False,
        "icd": ["J95", "Z96.3"],
        "docs": ["Spirometrie", "Sekretmobilisation", "SpO2", "Atemzugvolumen"],
    },

    # ══ RHEUMATOLOGIE ══════════════════════════════════════════════════════════

    "RH1": {
        "desc": "Rheumatoide Arthritis / entzündliche Gelenkerkrankungen",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": True,
        "icd": ["M05", "M06", "M07", "M08", "M45", "M46"],
        "docs": ["ROM (Neutral-Null)", "Schwellung/Rötung", "Schmerz (VAS)", "Funktionsscore"],
    },

    # ══ GRUPPENBEHANDLUNG ══════════════════════════════════════════════════════

    "GR1": {
        "desc": "Krankengymnastik Gruppenbehandlung (2–5 Patienten)",
        "heilmittel": "KG-Gruppe", "position": "20503",
        "name": "Krankengymnastik Gruppenbehandlung klein",
        "duration": 25, "regelfall": 6, "langfristig": False,
        "icd": [],   # group therapy — prescribed alongside individual KG
        "docs": ["Gruppenindikation", "Teilnehmerzahl (2-5)", "Therapieziel"],
    },
}

# ── ICD-10 → Diagnosegruppe index ──────────────────────────────────────────────

_ICD_TO_DG: dict[str, str] = {}
for _dg, _entry in _HMK.items():
    for _icd in _entry["icd"]:
        _ICD_TO_DG[_icd.upper()] = _dg


def _match_dg(icd10: str) -> Optional[str]:
    """Prefix-match ICD-10 code to Diagnosegruppe (most specific wins)."""
    icd = icd10.strip().upper()
    for length in (6, 5, 4, 3, 2):
        if icd[:length] in _ICD_TO_DG:
            return _ICD_TO_DG[icd[:length]]
    return None


# ── GKV fixed prices — Bundesbasiszulassung ab 01.01.2026 ────────────────────
# Source: Vergütungsvereinbarung §125 Abs. 1 SGB V, GKV-Spitzenverband / ZVK / IFK
# ⚠️  VERIFY before each billing year: prices are renegotiated annually.
# These figures reflect the 2026 agreed rates. Praxis-specific Zulassungsverträge
# may deviate — always check your individual Kassenzulassung.

_GKV_PRICES: dict[str, float] = {
    "20500": 30.83,   # KG Erstbefundung 30 min (ab 01.01.2026)
    "20501": 24.57,   # KG Einzelbehandlung 20 min (ab 01.01.2026)
    "20502": 24.57,   # KG Hausbesuch 20 min
    "20503": 13.76,   # KG Gruppe 2–5 Pat. 25 min
    "20504": 10.29,   # KG Gruppe 6–8 Pat. 45 min
    "20510": 36.87,   # KG-ZNS Erstbefundung 30 min
    "20511": 42.69,   # KG-ZNS Einzelbehandlung 45 min
    "20512": 42.69,   # KG-ZNS Hausbesuch 45 min
    "20560": 24.57,   # KG atemtherapeutisch 20 min
    "20700": 36.87,   # MT Erstbefundung 30 min
    "20701": 29.23,   # MT Folgebehandlung 20 min
    "21100": 24.57,   # MLD Teilbehandlung 30 min
    "21101": 36.87,   # MLD Standardbehandlung 45 min
    "21102": 46.26,   # MLD Ganzbehandlung 60 min
    "21110": 58.42,   # KPE Phase I 60 min
    "21111": 46.26,   # KPE Phase II 30 min
}

# ── PKV market price ranges (GebüTh reference 2026) ──────────────────────────
# ⚠️ Orientierungswerte — kein Rechtsanspruch, Erstattung vertragsabhängig

_PKV_RANGES: dict[str, tuple] = {
    "20501": (25.0,  70.0),
    "20502": (30.0,  80.0),
    "20503": (15.0,  35.0),
    "20510": (40.0,  90.0),
    "20511": (48.0, 100.0),
    "20560": (25.0,  65.0),
    "20700": (42.0,  95.0),
    "20701": (32.0,  80.0),
    "21100": (25.0,  55.0),
    "21101": (38.0,  80.0),
    "21102": (52.0, 115.0),
    "21110": (65.0, 140.0),
    "21111": (52.0, 115.0),
}

# ── BG surcharges (DGUV typical, varies by Träger) ────────────────────────────

_BG_SURCHARGE_PCT: dict[str, float] = {
    "20501": 18.0, "20511": 20.0, "20701": 22.0,
    "21101": 18.0, "21110": 20.0, "20560": 18.0,
}

_BG_EXTRA_DOCS = [
    "D-Arzt-Bericht vorhanden",
    "Unfallhergang dokumentiert",
    "BG-Fallnummer / Aktenzeichen",
    "Erstbehandlungsdatum",
]

# ── Red flags — block billing until physician clearance ───────────────────────

_RED_FLAGS: dict[str, str] = {
    "Parese":           "Lähmungszeichen",
    "Kraftverlust":     "akuter Kraftverlust",
    "Taubheit":         "Sensibilitätsverlust",
    "Reflexverlust":    "Reflexausfall",
    "Stuhlinkontinenz": "Blasen-/Mastdarmkontrolle",
    "Harninkontinenz":  "Blasen-/Mastdarmkontrolle",
    "Cauda":            "Cauda-equina-Syndrom V.a.",
    "Querschnitt":      "Querschnittslähmung V.a.",
    "Fraktur":          "ungeklärte Fraktur",
    "Tumorverdacht":    "Tumorverdacht",
    "Meningismus":      "Meningismus",
}

# ── Domain-specific required doc checkers ─────────────────────────────────────

_DOC_CHECKERS: dict = {
    "ROM (Neutral-Null)":              lambda t: bool(re.search(r"\d+ - \d+ - \d+|\d+-\d+-\d+", t)),
    "ROM Schulter (Abd/Flex/AR/IR)":   lambda t: bool(re.search(r"\d+ - \d+ - \d+", t)) and "schulter" in t,
    "ROM Knie (Flex/Ext)":             lambda t: bool(re.search(r"\d+ - \d+ - \d+", t)) and "knie" in t,
    "ROM Hüfte (Flex/Abd/AR)":         lambda t: bool(re.search(r"\d+ - \d+ - \d+", t)) and "hüfte" in t,
    "ROM HWS":                         lambda t: bool(re.search(r"\d+ - \d+ - \d+", t)) and any(k in t for k in ["hws", "hals", "zervikal", "c0", "c1", "c2"]),
    "ROM Sprunggelenk (DF/PF)":        lambda t: bool(re.search(r"\d+ - \d+ - \d+", t)) and any(k in t for k in ["sprung", "osg", "usg"]),
    "Schmerz (VAS)":                   lambda t: bool(re.search(r"vas\s*\d|schmerz.*\d+/10|\d+/10", t)),
    "Schober-Zeichen":                 lambda t: "schober" in t,
    "Lasègue":                         lambda t: "lasègue" in t or "lasegue" in t,
    "Stemmer-Zeichen":                 lambda t: "stemmer" in t,
    "Umfangsmessung (cm)":             lambda t: bool(re.search(r"\d+\s*cm", t)),
    "Stadium (1-3)":                   lambda t: bool(re.search(r"stadium\s*[1-3]", t)),
    "Ödemkonsistenz":                  lambda t: any(k in t for k in ["konsistenz", "weich", "teigig", "hart", "fibros"]),
    "Ashworth-Skala":                  lambda t: "ashworth" in t,
    "Timed Up & Go":                   lambda t: "timed up" in t or " tug" in t,
    "Barthel-Index":                   lambda t: "barthel" in t,
    "ADL-Status":                      lambda t: "adl" in t or "selbständig" in t or "barthel" in t,
    "Ganganalyse":                     lambda t: any(k in t for k in ["gangbild", "ganganalyse", "zirkumduktion", "steppergang", "trendelenburg"]),
    "Gangbild":                        lambda t: any(k in t for k in ["gangbild", "gang", "zirkumduktion"]),
    "Koordinationstest (Knie-Hacke)":  lambda t: "knie-hacke" in t or "kniehacke" in t or "koordination" in t,
    "FBA (Finger-Boden-Abstand)":      lambda t: "fba" in t or "finger-boden" in t,
    "Segmentbefund (C/Th)":            lambda t: any(k in t for k in ["segmentbefund", "segment", "c1", "c2", "c3", "c4", "c5", "th"]),
    "Segmentbefund (L/S)":             lambda t: any(k in t for k in ["segmentbefund", "l1", "l2", "l3", "l4", "l5", "s1", "lws"]),
    "Behandeltes Segment":             lambda t: bool(re.search(
        r'\b(?:segment(?:befund)?|[Cc]\d/[Cc]\d|[Cc]\d/[Tt]h?\d|[Ll]\d/[Ll]\d|[Ll]\d/[Ss]\d|'
        r'L4/L5|L5/S1|C5/C6|C6/C7|BWS|ISG|'
        # Extremity joint names also satisfy the §125 segment documentation requirement
        r'radiokarpal(?:gelenk)?|handgelenk|handwurzel|mcp|pip|dip|metakarpal|'
        r'glenohumer(?:al)?|schultergelenk|akromioklavikular|'
        r'kniegelenk|tibiofemorales?\s+gelenk|patellofemoral|'
        r'sprunggelenk|osg|usg|talokalkanear|'
        r'h[üu]ftgelenk|koxofemoral|'
        r'ellbogengelenk|humeroradiares?\s+gelenk|humeroulnar)\b', t, re.I)),
    "Blasen-/Mastdarmfunktion":        lambda t: bool(re.search(
        r'blasen|mastdarm|harninkontinenz|stuhlinkontinenz|miktion|defäkation|'
        r'blasen.?mastdarm|kontinenz', t, re.I)),
    "Endgefühl":                       lambda t: "endgefühl" in t or "end feel" in t,
    "Neurolog. Screening":             lambda t: any(k in t for k in ["reflex", "sensibilität", "kraft", "mmt", "neurolog"]),
    "Neurolog. Befund":                lambda t: any(k in t for k in ["neurolog", "reflex", "sensibilität", "ashworth", "barthel"]),
    "Krafttest (Jobe/Hawkins)":        lambda t: any(k in t for k in ["jobe", "hawkins"]),
    "Painful Arc":                     lambda t: "painful arc" in t or "schmerzbogen" in t,
    "Cobb-Winkel":                     lambda t: "cobb" in t,
    "Rippenbuckel":                    lambda t: "rippenbuckel" in t or "rippe" in t,
    "Schroth-Klassifikation":          lambda t: "schroth" in t,
    "Atemmuster":                      lambda t: "atemmuster" in t or "atemtyp" in t or "atemfrequenz" in t,
    "Spirometrie":                     lambda t: "spirometrie" in t or "fev" in t or "fvc" in t,
    "Sekretmobilisation":              lambda t: "sekret" in t or "husten" in t,
    "SpO2":                            lambda t: "spo2" in t or "sauerstoff" in t or "o2" in t,
    "Onkolog. Vordiagnose":            lambda t: any(k in t for k in ["karzinom", "tumor", "chemo", "bestrahlung", "mastektomie", "onkol"]),
    "Umfangsmessung beidseitig":       lambda t: bool(re.search(r"\d+\s*cm.{0,20}\d+\s*cm|re\..*cm.*li\.|li\..*cm.*re\.", t)),
    "Muskelkraft (MMT)":               lambda t: "mmt" in t or "muskelkraft" in t or re.search(r"kraft.*[0-5]/5", t) is not None,
    "Kraft (MMT)":                     lambda t: "mmt" in t or re.search(r"kraft.*[0-5]/5", t) is not None,
    "Griffstärke (kg)":                lambda t: "griffstärke" in t or "grip" in t,
    "Knienachgiebigkeit":              lambda t: "knienachgiebigkeit" in t,
    "Patella-Mobilität":               lambda t: "patella" in t,
    "Sturzrisiko-Assessment":          lambda t: any(k in t for k in ["sturzrisiko", "timed up", "berg-balance", "tinetti"]),
    "Funktionsziel":                   lambda t: any(k in t for k in ["ziel", "funktionsziel", "outcome"]),
    "OP-Bericht vorhanden":            lambda t: any(k in t for k in ["op-bericht", "operationsbericht", "postoperativ", "post-op"]),
    "Hoehn-Yahr-Skala":                lambda t: "hoehn" in t or "hoehn-yahr" in t,
    "Freezing-Protokoll":              lambda t: "freezing" in t,
    "EDSS-Score":                      lambda t: "edss" in t,
    "Fatigue-Skala":                   lambda t: "fatigue" in t or "erschöpfung" in t,
    "House-Brackmann-Skala":           lambda t: "house" in t or "brackmann" in t or "house-brackmann" in t,
    "Mimische Muskulatur (Befund)":    lambda t: "mimisch" in t or "fazialis" in t,
    "Synkinesien":                     lambda t: "synkinese" in t,
    "Motorische Meilensteine":         lambda t: "meilenstein" in t or "motorik" in t,
    "Gruppenindikation":               lambda t: "gruppe" in t,
    "Teilnehmerzahl (2-5)":            lambda t: bool(re.search(r"[2-5]\s*(?:patienten|teilnehmer)", t)),
    "Therapieziel":                    lambda t: any(k in t for k in ["ziel", "therapieziel", "outcome"]),
    "Schwellung/Rötung":               lambda t: any(k in t for k in ["schwellung", "rötung", "inflammat"]),
    "Funktionsscore":                  lambda t: any(k in t for k in ["funktionsscore", "das28", "haq", "sf-36"]),
    "D-Arzt-Bericht vorhanden":        lambda t: "d-arzt" in t or "durchgangsarzt" in t,
    "Unfallhergang dokumentiert":      lambda t: "unfall" in t or "hergang" in t,
    "BG-Fallnummer / Aktenzeichen":    lambda t: "bg-fall" in t or "aktenzeichen" in t or "fallnummer" in t,
    "Erstbehandlungsdatum":            lambda t: "erstbehandlung" in t,
}


def _check_doc(doc_name: str, soap: dict) -> bool:
    text = " ".join(str(v) for v in soap.values()).lower()
    checker = _DOC_CHECKERS.get(doc_name)
    if checker:
        return checker(text)
    return any(w.lower() in text for w in doc_name.split() if len(w) > 3)


# ── GKV engine ────────────────────────────────────────────────────────────────

class _GKVEngine:
    """
    Deterministic. No AI freedom.
    Rules-first: Heilmittelkatalog → tick-box audit → warn → block.
    """

    def evaluate(
        self,
        icd10: str,
        soap: dict,
        transcript: str,
        config_rules: dict,
    ) -> BillingResult:
        audit: list[AuditItem] = []
        risk = "OK"
        audit_status = "PASS"

        # ── 1. Map ICD → Diagnosegruppe ────────────────────────────────────────
        dg = _match_dg(icd10)
        if not dg:
            audit.append(AuditItem(
                "DG_LOOKUP", "Diagnosegruppe §125",
                "WARN",
                f"ICD {icd10} keiner Anlage-2-Gruppe zuordenbar — manuelle Prüfung"
            ))
            risk = "WARN"
            dg = self._fallback_dg(soap, transcript)
        else:
            audit.append(AuditItem("DG_LOOKUP", "Diagnosegruppe §125", "PASS",
                                   f"{dg}: {_HMK[dg]['desc']}"))

        entry = _HMK[dg]
        position = entry["position"]

        # ── 2. Config-level override ───────────────────────────────────────────
        config_pos = self._config_position(icd10, config_rules)
        if config_pos and config_pos != position:
            position = config_pos
            audit.append(AuditItem("CONFIG_OVERRIDE", "Praxis-Konfiguration",
                                   "PASS", f"Position {position} aus Gist-Konfig übernommen"))

        # ── 3. MT indication detected — WARN, never auto-upgrade ─────────────
        # §125 SGB V: only the doctor's prescription authorises MT (20701).
        # The therapist cannot self-authorise the upgrade; auto-upgrading from
        # KG to MT based on transcript content is Abrechnungsbetrug.
        # We flag it so the therapist can check their prescription.
        if entry.get("optional_mt") and self._mt_indicated(soap, transcript):
            audit.append(AuditItem(
                "MT_UPGRADE", "MT-Techniken dokumentiert",
                "WARN",
                "Prüfen Sie Ihr Rezept: Ist 'Manuelle Therapie' explizit verordnet? "
                "Nur dann ist 20701 abrechenbar. Rezept KG -> bleibt 20501."
            ))

        # ── 4. Mandatory SOAP fields ───────────────────────────────────────────
        obj = soap.get("O", "")
        subj = soap.get("S", "")
        plan = soap.get("P", "")
        assess = soap.get("A", "")

        audit.append(AuditItem("SOAP_S", "S-Feld (Subjektiv)",
                               "PASS" if len(subj) > 15 else "FAIL",
                               "" if len(subj) > 15 else "Subjektiver Befund zu kurz oder fehlt"))
        audit.append(AuditItem("SOAP_A", "A-Feld (Assessment/Diagnose)",
                               "PASS" if len(assess) > 10 else "FAIL",
                               "" if len(assess) > 10 else "Diagnose fehlt"))
        audit.append(AuditItem("SOAP_P", "P-Feld (Therapieplan)",
                               "PASS" if len(plan) > 15 else "FAIL",
                               "" if len(plan) > 15 else "Therapieplan fehlt oder zu unspezifisch"))

        # ── 5. Befunddichte §106b ──────────────────────────────────────────────
        min_len = 60 if position in ("20701", "20511", "20510") else 20
        if len(obj) >= min_len:
            audit.append(AuditItem("OBJ_DENSITY", f"O-Feld Mindestdichte (>={min_len} Zeichen)",
                                   "PASS", f"{len(obj)} Zeichen"))
        else:
            audit.append(AuditItem("OBJ_DENSITY", f"O-Feld Mindestdichte (>={min_len} Zeichen)",
                                   "FAIL", f"Nur {len(obj)} Zeichen — {min_len - len(obj)} fehlen"))
            risk = "WARN"

        # ── 6. Neutral-Null ROM ────────────────────────────────────────────────
        if position in ("20701", "20501") and "°" in obj:
            has_nn = bool(re.search(r"\d+ - \d+ - \d+|\d+-\d+-\d+", obj))
            audit.append(AuditItem("ROM_FORMAT", "ROM Neutral-Null-Methode",
                                   "PASS" if has_nn else "WARN",
                                   "" if has_nn else "Grad (°) ohne Neutral-Null-Format — bitte [Ext]-[0]-[Flex] verwenden"))

        # ── 7. VAS pain score ──────────────────────────────────────────────────
        has_vas = bool(re.search(r"vas\s*\d|schmerz.*\d+/10|\d+/10", (subj + obj).lower()))
        audit.append(AuditItem("VAS", "Schmerzquantifizierung (VAS/NRS)",
                               "PASS" if has_vas else "WARN",
                               "" if has_vas else "VAS-Score nicht dokumentiert — empfohlen für §106b"))

        # ── 8. Required documentation per Diagnosegruppe ──────────────────────
        for doc in entry["docs"]:
            present = _check_doc(doc, soap)
            audit.append(AuditItem(
                f"DOC_{doc.upper().replace(' ', '_')[:20]}",
                doc,
                "PASS" if present else "FAIL",
                "" if present else f"Pflichtfeld für {dg} fehlt"
            ))
            if not present:
                risk = "WARN"

        # ── 9. Required tests from remote config ───────────────────────────────
        rule = config_rules.get(f"ICD10_{icd10.replace('.', '_')}", {})
        for test in rule.get("required_tests", []):
            present = _check_doc(test, soap)
            audit.append(AuditItem(
                f"CFG_{test.upper()[:20]}",
                f"{test} (Praxisregel)",
                "PASS" if present else "FAIL",
                "" if present else f"Pflichttest laut Konfiguration fehlt"
            ))
            if not present:
                risk = "WARN"

        # ── 10a. MT segment documentation — mandatory for 20701 ───────────────
        if position == "20701":
            seg_checker = _DOC_CHECKERS["Behandeltes Segment"]
            has_seg = seg_checker(obj.lower() + " " + assess.lower())
            audit.append(AuditItem(
                "MT_SEGMENT", "Behandeltes Segment (MT §125 SGB V Pflicht)",
                "PASS" if has_seg else "FAIL",
                "" if has_seg else
                "Fehlendes Segment (z.B. L4/L5, C5/C6) — 20701 ohne Segmentangabe nicht abrechenbar."
            ))
            if not has_seg:
                risk = "WARN"

        # ── 10b. Blasen-/Mastdarmfunktion in O-Feld — Cauda-equina screening ─
        # LWS and MT cases: documenting that bladder/bowel was checked is a
        # clinical safety standard (Cauda equina exclusion). It must appear in O.
        is_ws_mt = position == "20701" and any(
            k in (obj + subj + assess).lower()
            for k in ["lws", "lumbal", "l4", "l5", "s1", "ischias", "bandscheib"]
        )
        if is_ws_mt:
            bm_checker = _DOC_CHECKERS["Blasen-/Mastdarmfunktion"]
            has_bm = bm_checker(obj.lower())   # must be in O, not just anywhere
            audit.append(AuditItem(
                "CAUDA_SCREEN", "Blasen-/Mastdarmfunktion dokumentiert (O-Feld)",
                "PASS" if has_bm else "WARN",
                "" if has_bm else
                "Cauda-equina-Screening fehlt im Objektiven Befund. "
                "Bitte ergänzen: 'Blasen-/Mastdarmfunktion: unauffällig' (oder Befund)."
            ))

        # ── 11. Red flag check — BLOCKS billing ────────────────────────────────
        red = self._red_flag_audit(soap)
        audit.extend(red)
        if any(a.status == "BLOCK" for a in red):
            risk = "BLOCK"

        # ── 11. Red-Flag exclusion in Assessment ───────────────────────────────
        has_rf_exclusion = "red flag" in assess.lower() or "ausgeschlossen" in assess.lower()
        audit.append(AuditItem("RF_EXCLUSION", "Red-Flag-Ausschluss im Assessment",
                               "PASS" if has_rf_exclusion else "WARN",
                               "" if has_rf_exclusion else "Fehlender Red-Flag-Ausschluss im A-Feld"))

        # ── Determine overall audit status ─────────────────────────────────────
        if any(a.status == "BLOCK" for a in audit):
            audit_status = "BLOCK"
        elif any(a.status in ("FAIL", "WARN") for a in audit):
            audit_status = "REVIEW"
        else:
            audit_status = "PASS"

        # ── BG extras ─────────────────────────────────────────────────────────
        bg_docs = []
        is_bg = False  # will be overridden by BillingEngine dispatcher

        return BillingResult(
            insurance_type=InsuranceType.GKV,
            position_number=position,
            position_name=_HMK.get(dg, entry)["name"] if position == entry["position"] else "Manuelle Therapie",
            diagnosegruppe=dg,
            diagnosegruppe_desc=entry["desc"],
            legal_basis="§ 125 SGB V | Anlage 2 Rahmenempfehlungen (01.01.2026) | § 106b Prüfung",
            session_duration_min=entry["duration"],
            risk_level=risk,
            audit_items=audit,
            audit_status=audit_status,
            compliance_warnings=[str(a) for a in audit if a.status != "PASS"],
            required_documentation=entry["docs"],
            max_units_regelfall=entry["regelfall"],
            requires_langfrist_approval=entry.get("langfristig", False),
            fixed_price_eur=_GKV_PRICES.get(position),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _config_position(self, icd10: str, rules: dict) -> Optional[str]:
        key = f"ICD10_{icd10.replace('.', '_')}"
        rule = rules.get(key) or rules.get("general", {})
        return rule.get("priority_code")

    def _fallback_dg(self, soap: dict, transcript: str) -> str:
        text = (transcript + " " + soap.get("P", "")).lower()
        if any(k in text for k in ["bobath", "pnf", "zns", "schlaganfall", "hemiplegie",
                                    "hemiparese", "parkinson", "multiple sklerose", "insult"]):
            return "ZNS1"
        if any(k in text for k in ["lymph", "ödem", "mld", "kpe", "entstauung", "stemmer"]):
            return "LY1"
        if any(k in text for k in ["skoliose", "schroth", "rippenbuckel"]):
            return "WS3"
        if any(k in text for k in ["schulter", "impingement", "rotatorenmanschette"]):
            return "EX2"
        if any(k in text for k in ["knie", "gonarthrose", "meniskus"]):
            return "EX3"
        if any(k in text for k in ["hws", "nacken", "zervikal"]):
            return "WS1a"
        if any(k in text for k in ["copd", "asthma", "atemweg"]):
            return "AT1"
        if any(k in text for k in ["handgelenk", "handwurzel", "radiusfraktur", "radius",
                                    "finger", "fingergelenk", "metakarpal", "phalanx",
                                    "ellbogen", "karpaltunnel", "ulna"]):
            return "EX6"
        if any(k in text for k in ["manuelle therapie", " mt ", "traktion", "gleitmobilisation"]):
            return "WS1b"
        return "WS1b"

    def _mt_indicated(self, soap: dict, transcript: str) -> bool:
        text = (transcript + " " + soap.get("P", "")).lower()
        return any(k in text for k in ["manuelle therapie", " mt ", "traktion",
                                        "gleitmobilisation", "manipulation"])

    def _red_flag_audit(self, soap: dict) -> list:
        items = []
        text = " ".join(str(v) for v in soap.values()).lower()
        for flag, label in _RED_FLAGS.items():
            idx = text.find(flag.lower())
            if idx == -1:
                continue
            after = text[idx: idx + 80]
            negated = any(n in after for n in
                          ["negativ", "ausgeschlossen", "kein", "ohne", "unauffällig", "verneint"])
            if negated:
                items.append(AuditItem(f"RF_{flag.upper()}", f"Red Flag: {label}",
                                       "PASS", f"{flag} dokumentiert und ausgeschlossen"))
            else:
                items.append(AuditItem(f"RF_{flag.upper()}", f"Red Flag: {label}",
                                       "BLOCK",
                                       f"{flag} ohne Ausschluss — ärztliche Abklärung vor Therapiefortsetzung!"))
        return items


# ── PKV engine ────────────────────────────────────────────────────────────────

class _PKVEngine:
    """
    AI-assisted. GebüTh as market reference only — not legally binding.
    ⚠️ Never guarantee PKV reimbursement.
    """

    def evaluate(
        self,
        icd10: str,
        soap: dict,
        transcript: str,
        pkv_preise: Optional[dict] = None,
    ) -> BillingResult:
        """
        pkv_preise: dict[Positionsnummer → float] aus config_override.json.
        Beispiel: {"20701": 72.00, "20501": 55.00}
        Wenn gesetzt, gilt der Praxispreis statt des GebüTh-Orientierungswerts.
        GKV-Festpreise werden dadurch nicht berührt.
        """
        audit: list[AuditItem] = []
        pkv_preise = pkv_preise or {}

        audit.append(AuditItem("PKV_HINWEIS", "PKV-Abrechnungshinweis", "WARN",
                               "Keine gesetzlichen Festpreise — Preis frei vereinbar (GebüTh als Orientierung)"))
        audit.append(AuditItem("PKV_ERSTATTUNG", "Erstattungshinweis", "WARN",
                               "Erstattung abhängig vom individuellen Versicherungsvertrag des Patienten"))

        dg = _match_dg(icd10) or "EX1b"
        entry = _HMK[dg]
        position = entry["position"]

        if entry.get("optional_mt") and self._mt_indicated(soap, transcript):
            position = "20701"
            audit.append(AuditItem("MT_UPGRADE", "MT-Erstbefundung empfohlen", "PASS",
                                   "20700 (30 Min) bei Neupatient separat abrechenbar (~€7 Aufschlag)"))

        # Praxispreis hat Vorrang — GebüTh nur als Orientierungswert wenn kein Praxispreis gesetzt
        praxispreis = pkv_preise.get(position)
        price_range = _PKV_RANGES.get(position, (25.0, 65.0))

        if praxispreis:
            audit.append(AuditItem("PKV_PRAXISPREIS", "Praxiseigener PKV-Preis",
                                   "PASS", f"€{praxispreis:.2f} (aus Praxiskonfiguration)"))
        else:
            audit.append(AuditItem("PKV_GEBUETH", "GebüTh-Orientierungswert",
                                   "WARN",
                                   f"€{price_range[0]:.0f}–{price_range[1]:.0f} — "
                                   "Praxispreis in config_override.json unter 'pkv_preise' hinterlegen"))

        likelihood = self._score_likelihood(icd10, soap)
        hints = self._hints(soap, transcript, position)

        # Quality audit items (same as GKV, but advisory only)
        obj = soap.get("O", "")
        subj = soap.get("S", "")

        audit.append(AuditItem("OBJ_QUALITY", "O-Feld Qualität",
                               "PASS" if len(obj) > 80 else "WARN",
                               f"{len(obj)} Zeichen — PKV prüft Befunddichte bei Retaxation"))
        has_nn = bool(re.search(r"\d+ - \d+ - \d+|\d+-\d+-\d+", obj))
        audit.append(AuditItem("ROM_FORMAT", "ROM Neutral-Null",
                               "PASS" if has_nn else "WARN",
                               "" if has_nn else "Neutral-Null erhöht PKV-Erstattungswahrscheinlichkeit"))
        has_vas = bool(re.search(r"vas\s*\d|\d+/10", (subj + obj).lower()))
        audit.append(AuditItem("VAS", "VAS-Score",
                               "PASS" if has_vas else "WARN",
                               "" if has_vas else "VAS-Score erhöht PKV-Akzeptanz"))

        for doc in entry["docs"]:
            present = _check_doc(doc, soap)
            audit.append(AuditItem(f"DOC_{doc[:20]}", doc,
                                   "PASS" if present else "WARN",
                                   "" if present else "Empfohlen für PKV-Prüfung"))

        if likelihood == "GERING":
            audit.append(AuditItem("PKV_LIKELIHOOD", "Erstattungswahrscheinlichkeit",
                                   "WARN",
                                   "GERING — Kostenvoranschlag und Begründungsschreiben empfohlen"))
        else:
            audit.append(AuditItem("PKV_LIKELIHOOD", "Erstattungswahrscheinlichkeit",
                                   "PASS", likelihood))

        audit_status = "REVIEW" if any(a.status == "WARN" for a in audit) else "PASS"

        return BillingResult(
            insurance_type=InsuranceType.PKV,
            position_number=position,
            position_name=entry["name"] + " (PKV)",
            diagnosegruppe=dg,
            diagnosegruppe_desc=entry["desc"],
            legal_basis="PKV — freie Preisgestaltung | GebüTh als Orientierungswert | kein §125 SGB V",
            session_duration_min=entry["duration"],
            risk_level="OK" if likelihood != "GERING" else "WARN",
            audit_items=audit,
            audit_status=audit_status,
            compliance_warnings=[str(a) for a in audit if a.status == "WARN"],
            required_documentation=entry["docs"],
            max_units_regelfall=0,
            pkv_praxispreis_eur=praxispreis,
            price_range_eur=price_range,
            reimbursement_likelihood=likelihood,
            optimization_hints=hints,
        )

    def _mt_indicated(self, soap: dict, transcript: str) -> bool:
        text = (transcript + " " + soap.get("P", "")).lower()
        return any(k in text for k in ["manuelle therapie", " mt ", "traktion", "gleitmobilisation"])

    def _score_likelihood(self, icd10: str, soap: dict) -> str:
        score = 0
        obj = soap.get("O", "")
        if re.search(r"\d+ - \d+ - \d+|\d+-\d+-\d+", obj): score += 2
        if re.search(r"vas\s*\d|\d+/10", (soap.get("S", "") + obj).lower()): score += 1
        if len(obj) > 100: score += 2
        if icd10 not in ("M99.9", "N/A", ""): score += 2
        if re.search(r"\d+\s*(cm|°|grad)", obj.lower()): score += 1
        if len(soap.get("P", "")) > 60: score += 1
        return "HOCH" if score >= 7 else "MITTEL" if score >= 4 else "GERING"

    def _hints(self, soap: dict, transcript: str, position: str) -> list:
        hints = []
        obj = soap.get("O", "")
        p = soap.get("P", "")
        if not re.search(r"\d+ - \d+ - \d+", obj):
            hints.append("💡 Neutral-Null-Werte im O-Feld erhöhen PKV-Erstattung erheblich.")
        if re.search(r"vas\s*\d|\d+/10", (soap.get("S", "") + obj).lower()) is None:
            hints.append("💡 VAS-Score im S-Feld — PKV prüft Schmerzquantifizierung.")
        if len(p) < 60:
            hints.append("💡 P-Feld präzisieren — PKV erwartet konkretes Outcome-Ziel.")
        if position == "20701":
            hints.append("💡 MT-Erstbefundung (20700, 30 Min) bei Neupatient separat abrechenbar.")
        if len(obj) < 80:
            hints.append("💡 Ausführlicheres O-Feld reduziert Retaxationsrisiko.")
        return hints


# ── BG engine (wraps GKV, adds DGUV surcharge + extra docs) ──────────────────

class _BGEngine:
    def __init__(self):
        self._gkv = _GKVEngine()

    def evaluate(self, icd10: str, soap: dict, transcript: str,
                 config_rules: dict) -> BillingResult:
        result = self._gkv.evaluate(icd10, soap, transcript, config_rules)
        result.insurance_type = InsuranceType.BG

        # Add BG-specific doc checks
        for doc in _BG_EXTRA_DOCS:
            present = _check_doc(doc, soap)
            result.audit_items.append(AuditItem(
                f"BG_{doc[:20]}", f"BG-Pflicht: {doc}",
                "PASS" if present else "FAIL",
                "" if present else "Pflichtdokument für BG-Abrechnung fehlt"
            ))
            if not present and result.audit_status != "BLOCK":
                result.audit_status = "REVIEW"

        result.bg_surcharge_pct = _BG_SURCHARGE_PCT.get(result.position_number, 18.0)
        if result.fixed_price_eur:
            result.fixed_price_eur = round(
                result.fixed_price_eur * (1 + result.bg_surcharge_pct / 100), 2
            )
        result.bg_extra_docs = _BG_EXTRA_DOCS
        result.legal_basis = "DGUV / Vertrag Ärzte-UV-Träger | §125 SGB V Codes | §106b Prüfung"
        result.compliance_warnings = [str(a) for a in result.audit_items if a.status != "PASS"]
        return result


# ── Public entry point ────────────────────────────────────────────────────────

class BillingEngine:
    """
    Single entry point. Dispatches to GKV, PKV, or BG engine.

    Usage:
        result = BillingEngine().evaluate(
            icd10="M54.5",
            soap={"S": "...", "O": "...", "A": "...", "P": "..."},
            transcript="...",
            insurance_type=InsuranceType.GKV,
            config_rules={},   # from ConfigManager.billing_rules
        )
        print(result.format_audit_report())
        print(result.format_billing_line())
    """

    def __init__(self):
        self._gkv = _GKVEngine()
        self._pkv = _PKVEngine()
        self._bg  = _BGEngine()

    def evaluate(
        self,
        icd10: str,
        soap: dict,
        transcript: str,
        insurance_type: InsuranceType = InsuranceType.GKV,
        config_rules: Optional[dict] = None,
        pkv_preise: Optional[dict] = None,
    ) -> BillingResult:
        """
        config_rules : aus ConfigManager.billing_rules  (GKV/BG)
        pkv_preise   : aus ConfigManager.pkv_preise     (PKV — praxiseigene Preise)
                       z.B. {"20701": 72.00, "20501": 55.00}
                       GKV-Festpreise werden dadurch nicht verändert.
        """
        rules = config_rules or {}
        if insurance_type == InsuranceType.BG:
            return self._bg.evaluate(icd10, soap, transcript, rules)
        if insurance_type == InsuranceType.PKV:
            return self._pkv.evaluate(icd10, soap, transcript, pkv_preise=pkv_preise)
        return self._gkv.evaluate(icd10, soap, transcript, rules)
