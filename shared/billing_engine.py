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
        "heilmittel": "MT", "position": "21201",
        "name": "Manuelle Therapie",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M75"],
        # 3 billing-critical items only (§106b): segment + provocation test + ROM
        "docs": ["Behandeltes Segment (Art. glenohumeralis)", "Provokationstest (Hawkins/Jobe)",
                 "ROM Schulter (Abd/Flex/AR/IR)"],
    },
    "EX3": {
        "desc": "Kniegelenk – Gonarthrose, postoperativ, Meniskusläsion",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M17", "M22", "M23", "S82", "S83"],
        # 3 billing-critical items (§106b): ROM + strength + functional test
        "docs": ["ROM Knie (Flex/Ext)", "Kraft (MMT)", "Gangbild"],
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
        "heilmittel": "MT", "position": "21201",
        "name": "Manuelle Therapie",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M50", "M53", "M54.0", "M54.1", "M54.2", "M99.0", "M99.1"],
        # 3 billing-critical: segment (MT-Pflicht) + provocation + ROM
        "docs": ["Behandeltes Segment (C/Th)", "Provokationstest (Spurling/Slump)", "ROM HWS"],
    },
    "WS1b": {
        "desc": "LWS/ISG – segmentale Funktionsstörung, Lumbago, Ischialgie",
        "heilmittel": "MT", "position": "21201",
        "name": "Manuelle Therapie",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M54.3", "M54.4", "M54.5", "M99.3", "M99.4", "M99.5"],
        # 3 billing-critical: segment (MT-Pflicht) + FBA + Lasègue (neurolog. exclusion)
        "docs": ["Behandeltes Segment (L/S)", "FBA (Finger-Boden-Abstand)", "Lasègue"],
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
        "heilmittel": "MLD", "position": "20201",
        "name": "Manuelle Lymphdrainage 45 Min",
        "duration": 45, "regelfall": 6, "langfristig": True,
        # ICD with stage suffixes (Oct 2024 rule: terminal code required on prescription)
        "icd": ["Q82.0", "Q82.00", "Q82.01", "Q82.02", "I89.0", "I89.00", "I89.01", "I89.02"],
        "docs": ["Stemmer-Zeichen", "Umfangsmessung (cm)", "Stadium (1-3)", "Ödemkonsistenz",
                 "Hautbefund (Rötung/Hyperkeratose)"],
    },
    "LY2": {
        "desc": "Sekundäres Lymphödem (postoperativ, Post-Cancer, Bestrahlung)",
        "heilmittel": "KPE", "position": "21110",
        "name": "Komplexe Physikalische Entstauungstherapie Phase I",
        "duration": 60, "regelfall": 6, "langfristig": True,
        "icd": ["I97.2", "I97.21", "I97.22", "I97.89", "C77", "C78", "C79"],
        "docs": ["Ödem-Stadium", "Umfangsmessung beidseitig", "Stemmer-Zeichen",
                 "Onkolog. Vordiagnose", "KPE-Komponenten", "Hautbefund (Rötung/Hyperkeratose)"],
    },
    "LY3": {
        # Lipödem is BVB (Besonderer Verordnungsbedarf), NOT LHB — extrabudgetary through 31.12.2025
        # Must always be prescribed alongside KPE, not MLD alone
        "desc": "Lipödem (Besonderer Verordnungsbedarf — BVB bis 31.12.2025)",
        "heilmittel": "MLD+KPE", "position": "20201",
        "name": "Manuelle Lymphdrainage 45 Min",
        "duration": 45, "regelfall": 6, "langfristig": False, "bvb": True,
        "icd": ["E88.2", "E88.20", "E88.21", "E88.22"],
        "docs": ["Stemmer-Zeichen", "Umfangsmessung (cm)", "Konsistenz", "Stadium (1-3)",
                 "KPE-Komponenten", "Hautbefund (Rötung/Hyperkeratose)"],
    },
    "LY4": {
        "desc": "Chronisch venöse Insuffizienz mit sekundärem Lymphödem",
        "heilmittel": "MLD", "position": "20205",
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
        "heilmittel": "KG-Gruppe", "position": "20601",
        # Gist: KG_Gruppe = 20601 (€13.26) — 20503 is Übungsgruppe (lower fee)
        "name": "Krankengymnastik Gruppenbehandlung",
        "duration": 45, "regelfall": 6, "langfristig": False,
        "icd": [],   # group therapy — prescribed alongside individual KG
        "docs": ["Gruppenindikation", "Teilnehmerzahl (2-5 Pat.)", "Therapieziel (gemeinsames Gruppenziel)"],
    },

    # ══ MASSAGETHERAPIE ════════════════════════════════════════════════════════

    "MA1": {
        "desc": "Klassische Massage (KMT) / Bindegewebsmassage (BGM) / Segmentmassage",
        "heilmittel": "Massage", "position": "20106",
        # Gist: Massage_KMT = 20106 (€21.63), Massage_BGM = 20107 (€25.98), Massage_UW = 20102 (€33.75)
        "name": "Klassische Massagetherapie",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M54", "M79.1", "M62.4"],
        "docs": ["Massageform (KMT/BGM/Segment)", "Lokalisation + Befund (Tonus/Myogelosen)", "Wirkung (Tonussenkung/Durchblutung)"],
    },
    "MA2": {
        "desc": "Unterwasserdruckstrahlmassage (UWM)",
        "heilmittel": "Massage-UW", "position": "20102",
        "name": "Unterwasserdruckstrahlmassage",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M54", "M79.3", "G82"],
        "docs": ["Wassertemperatur (°C)", "Druck (bar)", "Behandlungsregion", "Wirkung"],
    },

    # ══ ELEKTROTHERAPIE ════════════════════════════════════════════════════════

    "EL1": {
        "desc": "Elektrotherapie – TENS / diadynamische Ströme / Interferenzstrom",
        "heilmittel": "Elektro", "position": "21302",
        # Gist: Elektro = 21302 (€8.43)
        "name": "Elektrotherapie",
        "duration": 15, "regelfall": 6, "langfristig": False,
        "icd": ["M54", "M79.1", "M25.5"],
        "docs": ["Stromform (TENS/IFC/Galvano)", "Frequenz (Hz) + Intensität (mA)", "Elektroden-Platzierung", "Wirkung (Analgesie/Muskelstimulation)"],
    },
    "EL2": {
        "desc": "Elektrotherapie bei Lähmungen – EMS / neuromuskuläre Stimulation",
        "heilmittel": "Elektro-Lähmung", "position": "21303",
        # Gist: Elektro_Lahmung = 21303 (€18.70)
        "name": "Elektrotherapie bei Lähmungen",
        "duration": 20, "regelfall": 10, "langfristig": True,
        "icd": ["G57", "G58", "G82", "S14", "S24"],
        "docs": ["Lähmungsgrad (MRC 0-5)", "Stimulationsparameter (Hz/mA)", "Elektroden-Platzierung", "Muskelkontraktion: vorhanden / fehlend"],
    },

    # ══ THERMOTHERAPIE ═════════════════════════════════════════════════════════

    "TH1": {
        "desc": "Wärmetherapie – Fango / Heiße Rolle / Wärmestrahler",
        "heilmittel": "Thermo", "position": "21501",
        # Gist: Fango = 21501 (€16.16), Heisse_Rolle = 21530 (€13.47)
        "name": "Warmpackung (Fango/Heiße Rolle)",
        "duration": 20, "regelfall": 6, "langfristig": False,
        "icd": ["M54", "M79.1"],
        "docs": ["Wärmemodalität (Fango/Heiße Rolle/Strahler)", "Behandlungsregion", "Temperatur (°C oder subjektiv: angenehm warm)", "Kontraindikationsausschluss (Sensibilitätsstörung: nein)"],
    },
    "TH2": {
        "desc": "Kältetherapie – Kryotherapie / Eispackung",
        "heilmittel": "Kryotherapie", "position": "21534",
        # Gist: Kaelte = 21534 (€11.95)
        "name": "Kälteanwendung",
        "duration": 15, "regelfall": 6, "langfristig": False,
        "icd": ["M25.5", "S00", "S60", "S80", "S90"],
        "docs": ["Kältemodalität (Eispack/Kältespray/Kryokammer)", "Behandlungsregion", "Schwellung (Umfang cm)", "Kontraindikationsausschluss (Kälteurtikaria/Durchblutungsstörung: nein)"],
    },

    # ══ KG IM BEWEGUNGSBAD / AQUATHERAPIE ══════════════════════════════════════

    "BB1": {
        "desc": "Krankengymnastik im Bewegungsbad – Einzelbehandlung (Aquatherapie)",
        "heilmittel": "KG-BB", "position": "20902",
        # Gist: KG_BB_Einzel = 20902 (€33.87)
        "name": "Krankengymnastik im Bewegungsbad (Einzeln)",
        "duration": 30, "regelfall": 6, "langfristig": False,
        "icd": [],  # BB1 selected ONLY via AQUA profile_id, never via ICD lookup
        "docs": ["Wassertemperatur (°C)", "Auftriebshilfen (vorhanden / nicht notwendig)", "Belastungsstatus im Wasser", "ROM und Gangbild im Vergleich zu trocken"],
    },

    # ══ GERÄTEGESTÜTZTE KG / MTT ══════════════════════════════════════════════

    "KGG": {
        "desc": "Krankengymnastik am Gerät / Medizinische Trainingstherapie (MTT)",
        "heilmittel": "KGG", "position": "20507",
        # Gist-confirmed: KG_Gerät = 20507 (€55.81) — distinct from 20501 KG Einzelbehandlung (€29.63)
        "name": "Krankengymnastik am Gerät",
        "duration": 45, "regelfall": 6, "langfristig": False,
        "icd": [],   # No specific ICD — DG assigned via fallback from M54/M17/M75/M16/S-codes
        "docs": ["Trainingsplan (Gerät + Last + Wdh)", "Krafttest (MRC 0-5 oder Dynamometer)", "Therapieziel"],
    },

    # ══ GEBURTSHILFE / RÜCKBILDUNG ════════════════════════════════════════════
    # Gist: ICD10_O80 → Geburtsvorbereitung (21901) / Rückbildungsgymnastik (21904)

    "GEB1": {
        "desc": "Geburtsvorbereitung – pränatale Atemtherapie, Beckenbodentraining",
        "heilmittel": "KG", "position": "21901",
        "name": "Geburtshilfliche Leistungen – Vorbereitung",
        "duration": 45, "regelfall": 6, "langfristig": False,
        "icd": ["O34.1", "O26", "O30", "Z34"],
        "docs": ["Schwangerschaftswoche (SSW)", "Beckenbodenbefund", "Atemtechnik-Stand"],
    },
    "GEB2": {
        "desc": "Rückbildungsgymnastik – postnatale Beckenbodenreha",
        "heilmittel": "KG", "position": "21904",
        "name": "Geburtshilfliche Leistungen – Rückbildung",
        "duration": 45, "regelfall": 6, "langfristig": False,
        "icd": ["O70", "O71", "N81.0", "N81.8", "Z39"],
        "docs": ["Wochen postpartum", "Beckenbodenkraft (Oxford 0-5)", "Dammriss-/Narbengrad (falls vorhanden)"],
    },

    # ══ BECKENBODEN / KONTINENZ ═══════════════════════════════════════════════

    "PF1": {
        "desc": "Beckenbodentherapie – Harninkontinenz, Beckenbodendysfunktion, postpartum",
        "heilmittel": "KG", "position": "20501",
        "name": "Krankengymnastik Einzelbehandlung",
        "duration": 20, "regelfall": 10, "langfristig": False,
        "icd": ["N39.3", "N39.4", "N81", "N81.1", "N81.2", "N81.8", "O34.1",
                "N40", "R32"],  # incl. Prostatahyperplasie + funktionelle Harninkontinenz
        "docs": ["Kontinenzstatus (Harnverlust-Typ: Stress/Drang/Misch)", "Beckenboden-Tonus (0-5)", "Miktionsfrequenz"],
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


# ── GKV fixed prices — Loaded dynamically from data files ────────────────────
# Source: data/gkv_prices_YYYY.json (updated annually)
# Vergütungsvereinbarung §125 Abs. 1 SGB V, GKV-Spitzenverband / ZVK / IFK
# ⚠️  Prices are negotiated annually and loaded from JSON for easy updates

def _load_gkv_prices() -> dict[str, float]:
    """Load GKV prices from data file. Falls back to 2026 if current year not available."""
    try:
        import os
        import sys
        from pathlib import Path

        # Determine data directory
        if getattr(sys, 'frozen', False):
            base = Path(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)))
        else:
            base = Path(__file__).parent.parent

        data_dir = base / "data"

        # Try current year first
        from datetime import datetime
        current_year = datetime.now().year
        pricing_file = data_dir / f"gkv_prices_{current_year}.json"

        if not pricing_file.exists():
            pricing_file = data_dir / "gkv_prices_2026.json"

        if pricing_file.exists():
            import json
            try:
                with open(pricing_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load pricing file {pricing_file.name}: {e}")
                data = {}

            # Flatten all categories into single price dict
            prices = {}
            for category in data.values():
                if isinstance(category, dict):
                    for position, details in category.items():
                        if isinstance(details, dict) and "price_eur" in details:
                            prices[position] = details["price_eur"]

            return prices

    except Exception as e:
        print(f"Warning: Could not load pricing data from file: {e}")

    # Fallback to hardcoded 2026 prices
    return {
        "20300": 34.34, "20500": 30.83, "20501": 29.63, "20507": 55.81,
        "20502": 29.63, "20503": 13.76, "20504": 10.29, "20510": 36.87,
        "20511": 42.69, "20512": 42.69, "20560": 29.63, "21200": 42.71,
        "21201": 35.59, "20201": 53.94, "20202": 71.94, "20205": 35.97,
        "21100": 53.94, "21110": 58.42, "21111": 46.26, "21901": 11.40,
        "21904": 11.40, "20102": 33.75, "20106": 21.63, "20107": 25.98,
        "20108": 21.63, "20601": 13.26, "20401": 8.43, "21302": 8.43,
        "21303": 18.70, "21310": 14.48, "21312": 27.61, "21501": 16.16,
        "21517": 7.43, "21530": 13.47, "21531": 14.66, "21534": 11.95,
        "20902": 33.87, "21004": 24.16, "21005": 15.97,
    }

# Load prices at module import
_GKV_PRICES: dict[str, float] = _load_gkv_prices()

# ── PKV market price ranges (GebüTh reference 2026) ──────────────────────────
# ⚠️ Orientierungswerte — kein Rechtsanspruch, Erstattung vertragsabhängig

_PKV_RANGES: dict[str, tuple] = {
    "20501": (30.0,  80.0),
    "20507": (55.0, 130.0),   # KGG/MTT 45 min
    "20502": (30.0,  80.0),
    "20503": (15.0,  35.0),
    "20510": (40.0,  90.0),
    "20511": (48.0, 100.0),
    "20560": (30.0,  75.0),
    "21200": (48.0, 100.0),
    "21201": (38.0,  90.0),
    "20201": (55.0, 120.0),
    "20202": (72.0, 150.0),
    "20205": (36.0,  80.0),
    "21100": (55.0, 120.0),
    "21110": (65.0, 140.0),
    "21111": (52.0, 115.0),
    "21901": (20.0,  55.0),   # Geburtsvorbereitung
    "21904": (20.0,  55.0),   # Rückbildungsgymnastik
    "20106": (22.0,  55.0),   # KMT
    "20107": (26.0,  65.0),   # BGM
    "20102": (34.0,  80.0),   # UW-Massage
    "20601": (14.0,  35.0),   # KG-Gruppe
    "21302": (10.0,  28.0),   # Elektrotherapie
    "21303": (20.0,  55.0),   # Elektro bei Lähmungen
    "21501": (18.0,  45.0),   # Fango
    "21534": (13.0,  38.0),   # Kältetherapie
    "20902": (35.0,  90.0),   # KG Bewegungsbad
}

# ── BG surcharges (DGUV typical, varies by Träger) ────────────────────────────

_BG_SURCHARGE_PCT: dict[str, float] = {
    "20501": 18.0, "20511": 20.0, "21201": 22.0,
    "20201": 18.0, "21110": 20.0, "20560": 18.0,
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
    "Tumorverdacht":    "Tumorverdacht",
    "Meningismus":      "Meningismus",
}

# ── Domain-specific required doc checkers ─────────────────────────────────────

_DOC_CHECKERS: dict = {
    "ROM (Neutral-Null)":              lambda t: bool(re.search(r"\d+\s*-\s*0\s*-\s*\d+", t)),
    "ROM Schulter (Abd/Flex/AR/IR)":   lambda t: "schulter" in t and bool(re.search(
        r"\d+\s*-\s*0\s*-\s*\d+|"                  # valid NZM: X-0-X (80-120-60 rejected)
        r"(?:flex|ext|abd|add|aro|iro):\s*\d+",     # labeled measurement with actual number
        t, re.I)),
    "ROM Knie (Flex/Ext)":             lambda t: bool(re.search(r"\d+ - \d+ - \d+", t)) and "knie" in t,
    "ROM Hüfte (Flex/Abd/AR)":         lambda t: "hüfte" in t and bool(re.search(
        r"(?:flexion|abd|aro|iro|ext|rom hüfte|nzm).*\d+|"   # any named value with a number
        r"\d+\s*[-/]\s*0\s*[-/]\s*\d+|"                       # NZM format: 90-0-0 or 90/0/0
        r"\d+ - \d+ - \d+",                                    # NZM with spaces
        t, re.I)),
    "ROM HWS":                         lambda t: bool(re.search(r"\d+\s*-\s*\d+\s*-\s*\d+", t)) and any(k in t for k in ["hws", "hals", "zervikal", "c0", "c1", "c2"]),
    "ROM Sprunggelenk (DF/PF)":        lambda t: (
        # Accept either: explicit NZM digits with ankle keyword, OR keyword-based presence
        # (acute injury may prevent measurement — n.d. with field present is acceptable)
        any(k in t for k in ["rom osg", "dorsalextension", "plantarflexion", "df/pf"]) or
        (bool(re.search(r"\d+ - \d+ - \d+|\d+-\d+-\d+", t)) and
         any(k in t for k in ["sprung", "osg", "usg"]))
    ),
    "Schmerz (VAS)":                   lambda t: bool(re.search(
        r"vas[:\s]*\d+(?:/10)?|"  # VAS: 4 or VAS 4/10
        r"schmerz[:\s]*vas[:\s]*\d+|"  # Schmerz: VAS 4
        r"schmerz.*\d+/10|"  # Schmerz VAS 4/10
        r"nrs[:\s]*\d+|"  # NRS 4
        r"schmerz.*nrs|"  # Schmerz NRS
        r"vas\s*\d", t)),
    "Schober-Zeichen":                 lambda t: "schober" in t,
    "Lasègue":                         lambda t: "lasègue" in t or "lasegue" in t,
    "Stemmer-Zeichen":                 lambda t: "stemmer" in t,
    "Umfangsmessung (cm)":             lambda t: bool(re.search(r"\d+\s*cm|umfangsmessung|umfangsdiff", t, re.I)),
    "Stadium (1-3)":                   lambda t: bool(re.search(r"stadium\s*[1-3]", t)),
    "Ödemkonsistenz":                  lambda t: any(k in t for k in [
        "konsistenz", "weich", "teigig", "hart", "fibros",
        "prall", "gespannt", "fest", "verhärtet", "induriert", "derb",
        "irreversibel", "reversibel", "ödemkonsistenz", "pitting", "verhärtung",
    ]),
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
    "HWS-ROM":                         lambda t: bool(re.search(r"\d+\s*-\s*\d+\s*-\s*\d+", t)) and any(k in t for k in ["hws", "hals", "zervikal", "rotation", "flexion", "latflex"]),
    "Neurolog. Screening":             lambda t: any(k in t for k in ["reflex", "sensibilität", "kraft", "mmt", "neurolog", "spurling", "dermatom", "sensibil", "parästhes", "taubheit", "unauffällig"]),
    "Neurolog. Befund":                lambda t: any(k in t for k in ["neurolog", "reflex", "sensibilität", "ashworth", "barthel"]),
    "Krafttest (Jobe/Hawkins)":        lambda t: any(k in t for k in ["jobe", "hawkins", "nicht testbar", "nicht wertbar", "schmerzinhibition"]),
    "Provokationstest (Hawkins/Jobe)": lambda t: any(k in t for k in ["hawkins", "jobe", "nicht testbar", "nicht wertbar", "schmerzinhibition"]),
    # Spurling/Slump: present AND not explicitly "nicht durchgeführt"
    "Provokationstest (Spurling/Slump)": lambda t: bool(re.search(r"spurling|slump", t, re.I)) and not bool(
        re.search(r"(?:spurling|slump)[^.]{0,40}nicht\s+durchge?f[üu]hrt", t, re.I)
    ),
    "Painful Arc":                     lambda t: "painful arc" in t or "schmerzbogen" in t or "blockade" in t or "bewegungslimitierung" in t,
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
    "Griffstärke (kg)":                lambda t: (
        "griffstärke" in t or "grip" in t or "jamar" in t or
        bool(re.search(r"(?:jamar|handkraft|grip).*?\d+(?:[,./]\d+)?\s*kg", t)) or
        bool(re.search(r"\d+(?:[,./]\d+)?\s*kg.*(?:jamar|handkraft|grip)", t))
    ),
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
    # KPE 4 mandatory components (§125 SGB V): MLD + Kompression + Entstauungsgymnastik + Hautpflege
    "KPE-Komponenten":                 lambda t: (
        any(k in t for k in ["mld", "lymphdrainage"]) and
        any(k in t for k in ["kompression", "bandagier", "kompressionsklasse", "strumpf"]) and
        any(k in t for k in ["entstauungsgymnastik", "übung", "bewegungsübung", "aktivierung"])
    ),
    # Lymphedema skin assessment (mandatory in LY audit)
    # Accepts positive findings (rötung, hyperkeratose) AND documented absence
    # (keine rötung, haut unauffällig, reizlos) — both constitute a complete skin assessment.
    "Hautbefund (Rötung/Hyperkeratose)": lambda t: (
        any(k in t for k in [
            "hautbefund", "rötung", "erythem", "hyperkeratose", "papillomatose",
            "haut trocken", "haut unauffällig", "kein erysipel", "erysipel",
            "reizlos", "keine rötung", "keine hitze", "unauffällig", "trocken",
            "haut intakt", "haut ist intakt", "intakte haut",
            "keine läsion", "keine infektion", "keine anzeichen",
            "gut gepflegt", "gepflegte haut", "geschlossen",
        ])
        or bool(re.search(r"keine\s+\w+\s*r[oö]t|haut\s+\w+\s+intakt", t))
    ),
    # Volume difference ≥10% clinical threshold for lymphedema significance
    "Umfangsdifferenz ≥10%":           lambda t: bool(re.search(
        r"(?:10|1[1-9]|[2-9]\d)\s*%|differenz.*\d+\s*cm|\d+\s*cm.*differenz|"
        r"re\..*\d+.*li\..*\d+|li\..*\d+.*re\..*\d+|"
        r"(?:mehr|größer|kleiner|unterschied).*\d+\s*cm", t
    )),
}


def _check_doc(doc_name: str, soap: dict, transcript: str = "") -> bool:
    # Search SOAP first, then fall back to transcript — the therapist may have
    # stated a finding verbally (transcript) even if the LLM failed to extract it.
    soap_text = " ".join(str(v) for v in soap.values()).lower()
    text = soap_text + " " + transcript.lower()
    checker = _DOC_CHECKERS.get(doc_name)
    if checker:
        return checker(text)
    words = re.findall(r'\w+', doc_name)
    return any(w.lower() in text for w in words if len(w) > 3)


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
        profile_id: Optional[str] = None,
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
        # PASS suppressed — DG_LOOKUP only emits on failure

        # ── 1b. Profile override for modality-specific sessions ───────────────
        # Modality profiles (KGG, ELEKTRO, THERMO, etc.) are identified by transcript
        # content, not ICD codes. Without this override M54.5 would route to WS1b/MT
        # even when the actual session is KG am Gerät (20507).
        if profile_id and profile_id in _PROFILE_TO_DG:
            # Guard: do NOT downgrade a specific LY subtype (LY2, LY3) that _match_dg()
            # already resolved correctly (e.g. I97.21 → LY2) back to LY1 via the generic
            # "LY" → "LY1" mapping.  Only apply the override when _match_dg() was
            # inconclusive (dg == "LY1" from default) or a non-LY profile is being set.
            _would_override_to = _PROFILE_TO_DG[profile_id]
            _skip = profile_id == "LY" and dg in ("LY2", "LY3")
            if not _skip:
                dg = _would_override_to
            # Remove DG_LOOKUP warning that may have been emitted above — the profile
            # override resolves the ICD ambiguity; no manual check needed.
            audit = [a for a in audit if a.code != "DG_LOOKUP"]
            risk = "OK"

        # ── 1c. Transcript-based LY1 → LY2 upgrade ───────────────────────────
        # ICD I89.x covers both primary and secondary lymphedema (same code space).
        # When the transcript explicitly names a secondary etiology, upgrade the
        # Diagnosegruppe label to LY2 (Sekundäres Lymphödem) for correct display.
        # Important: the therapist holds an MLD prescription, not a KPE prescription.
        # We therefore keep the MLD billing position (20201 base, upgraded by duration
        # logic below) and only use LY2's description/docs — NOT its KPE position 21110.
        _ly2_via_transcript = False
        if dg == "LY1" and transcript:
            _t = transcript.lower()
            _secondary_triggers = [
                "mamma-ablation", "mastektomie", "mastectomy",
                "prostatektomie", "prostata-op", "prostata op", "nach der prostata",
                "neck-dissection", "neck dissection", "nach neck",
                "axillaer", "sentinel", "axillary",
                "nach bestrahlung", "nach strahlentherapie", "nach chemotherapie",
                "zervix-karzinom", "zervixkarzinom", "cervix-ca", "cervix ca",
                "sekundäres lymphödem", "sekundaeres lymphoedem",
                "nach tumorektomie", "nach tumor-op",
                "nach rektum-op", "nach darm-op", "nach gynäkolog",
                "beckenlymphödem", "beckenlymphoedem",
                "nach op lymph", "lymphknotenentfernung",
            ]
            if any(trig in _t for trig in _secondary_triggers):
                dg = "LY2"
                _ly2_via_transcript = True

        entry = _HMK[dg]
        position = entry["position"]

        # ── LY2 + MLD prescription guard ─────────────────────────────────────
        # LY2 can be billed as KPE (21110) or MLD (20201/20203/20205) depending
        # on the prescription.
        #
        # Case A — transcript-based upgrade (secondary etiology detected):
        #   The therapist is performing MLD (that's why they're dictating a
        #   lymphology session). Always keep MLD base position.
        #
        # Case B — ICD-based LY2 (I97.x / C77 etc.) but transcript shows MLD:
        #   Therapist explicitly mentions MLD treatment → MLD position.
        #   Otherwise leave at 21110 (KPE prescription assumed).
        _mld_indicators = ["mld-", "mld ", "mld:", "mld,", "mld.", "der mld",
                           "lymphdrainage", "manuelle lymphdrainage",
                           "lymphabflusswege"]  # "an den Lymphabflusswegen" = MLD treatment
        if dg == "LY2" and position == "21110":
            if _ly2_via_transcript:
                # Case A: always MLD when secondary etiology came from transcript
                position = _HMK["LY1"]["position"]  # "20201" — MLD base
            elif transcript:
                # Case B: MLD only when explicitly mentioned
                _t_low = transcript.lower()
                if any(ind in _t_low for ind in _mld_indicators):
                    position = _HMK["LY1"]["position"]  # "20201" — MLD base
                    _ly2_via_transcript = True  # suppress KPE_4COMP (MLD session)

        # ── 2. Config-level override ───────────────────────────────────────────
        config_pos = self._config_position(icd10, config_rules)
        if config_pos and config_pos != position:
            position = config_pos
            # Resync entry/dg so docs and position_name reflect the overridden position
            for _override_dg, _override_entry in _HMK.items():
                if _override_entry.get("position") == config_pos:
                    dg = _override_dg
                    entry = _override_entry
                    break

        # ── 3. MT indication detected — WARN, never auto-upgrade ─────────────
        # §125 SGB V: only the doctor's prescription authorises MT (21201).
        # The therapist cannot self-authorise the upgrade; auto-upgrading from
        # KG to MT based on transcript content is Abrechnungsbetrug.
        # We flag it so the therapist can check their prescription.
        if entry.get("optional_mt") and self._mt_indicated(soap, transcript):
            audit.append(AuditItem(
                "MT_UPGRADE", "MT-Techniken dokumentiert",
                "WARN",
                "Prüfen Sie Ihr Rezept: Ist 'Manuelle Therapie' explizit verordnet? "
                "Nur dann ist 21201 abrechenbar. Rezept KG -> bleibt 20501."
            ))

        # ── 4. Mandatory SOAP fields ───────────────────────────────────────────
        obj = soap.get("O", "")
        subj = soap.get("S", "")
        plan = soap.get("P", "")
        assess = soap.get("A", "")

        # SOAP field checks
        if len(subj) <= 10:
            audit.append(AuditItem("SOAP_S", "S-Feld (Subjektiv/Anamnese)", "FAIL",
                                   "Patientenanamnese fehlt — §106b erfordert dokumentierte "
                                   "Beschwerdeschilderung des Patienten (Hauptbeschwerde, Schmerz, Ziel)"))
            risk = "WARN"

        if len(assess) <= 10:
            audit.append(AuditItem("SOAP_A", "A-Feld (Assessment/Diagnose)", "FAIL",
                                   "Diagnose fehlt"))

        # ── 5. Befunddichte §106b — only meaningful for high-scrutiny positions ──
        if position in ("21201", "20511", "20510") and len(obj) < 60:
            audit.append(AuditItem("OBJ_DENSITY", "O-Feld Befunddichte",
                                   "FAIL", f"Nur {len(obj)} Zeichen — mindestens 60 für MT/ZNS erforderlich"))
            risk = "WARN"

        # ── 6. Neutral-Null ROM format — FAIL only for MT (21201), not a WARN elsewhere ─
        # Valid NZM requires 0 in the middle: X-0-X.  80-120-60 is NOT valid NZM.
        if position == "21201" and bool(re.search(r"°|\bgrad\b", obj, re.I)):
            has_nn = bool(re.search(r"\d+\s*-\s*0\s*-\s*\d+", obj))
            if not has_nn:
                audit.append(AuditItem("ROM_FORMAT", "ROM Neutral-Null-Methode",
                                       "FAIL",
                                       "Grad (°) dokumentiert aber kein [Ext]-[0]-[Flex] Format — "
                                       "Pflicht fuer MT-Abrechnung (21201). "
                                       "Beispiel: Abd/Add: 90-0-30 (nicht 80-120-60)"))
                risk = "WARN"

        # ── 6a. Anatomy mismatch — LWS-specific tests in extremity/shoulder profile ──
        _LWS_TESTS = {"fba": "FBA (Finger-Boden-Abstand)", "finger-boden": "FBA",
                      "lasègue": "Lasègue-Test", "lasegue": "Lasègue-Test"}
        _EXTREMITY_DGS = {"EX1", "EX2", "EX3", "EX4", "EX5", "EX6"}
        if dg in _EXTREMITY_DGS:
            obj_lower = obj.lower()
            wrong_tests = [label for kw, label in _LWS_TESTS.items() if kw in obj_lower]
            # deduplicate labels
            wrong_tests = list(dict.fromkeys(wrong_tests))
            if wrong_tests:
                audit.append(AuditItem(
                    "ANATOMY_MISMATCH", "Anatomie-Konflikt: LWS-Tests im Extremitäten-Profil",
                    "WARN",
                    f"{', '.join(wrong_tests)} sind LWS/ISG-Diagnostik (Diagnosegruppe WS1b) "
                    f"und haben im {dg}-Schulter/Extremitäten-Bericht keinen klinischen Wert. "
                    "Ein Prüfer wird den Bericht als inkonsistent zurückweisen."
                ))

        # ── 6b. n.d. ROM fields — required measurements must not be 'no data' ────
        # For MT positions: if the therapist wrote ROM fields but filled them with n.d.,
        # the insurance cannot verify the starting point for the therapy goal.
        if position == "21201":
            nd_fields = [
                name for name, pattern in [
                    ("Flexion",   r"flex(?:ion)?[^|.]{0,25}n\.d\."),
                    ("Extension", r"ext(?:ension)?[^|.]{0,25}n\.d\."),
                    ("Abduktion", r"abd(?:uktion)?[^|.]{0,25}n\.d\."),
                    ("ARO/IRO",   r"(?:aro|iro)[^|.]{0,25}n\.d\."),
                    ("Endgefühl", r"endgef[üu]hl[^|.]{0,25}n\.d\."),
                    ("Painful Arc", r"painful arc[^|.]{0,25}n\.d\."),
                ]
                if re.search(pattern, obj, re.I)
            ]
            if nd_fields:
                audit.append(AuditItem(
                    "ROM_ND", "ROM-Pflichtfelder mit 'n.d.' (keine Daten)",
                    "FAIL",
                    f"Keine Messwerte für: {', '.join(nd_fields)}. "
                    "Die Krankenkasse muss den Ausgangsbefund kennen, "
                    "um das Therapieziel (z.B. Abd 120°) zu bewilligen. "
                    "Fehlende Messung = kein nachvollziehbarer Behandlungsfortschritt."
                ))
                risk = "WARN"

        # ── 7. Required documentation per Diagnosegruppe — emit only on FAIL ────
        # When LY2 was reached via transcript secondary-etiology detection but the
        # prescription is MLD (not KPE), use LY1's doc requirements so that
        # KPE-specific fields (KPE-Komponenten, Onkolog. Vordiagnose) are not
        # flagged as missing in what is legitimately an MLD session.
        _docs_to_check = _HMK["LY1"]["docs"] if _ly2_via_transcript else entry["docs"]
        for doc in _docs_to_check:
            if not _check_doc(doc, soap, transcript):
                audit.append(AuditItem(
                    f"DOC_{doc.upper().replace(' ', '_')[:20]}",
                    doc, "FAIL", f"Pflichtfeld für {dg} fehlt"
                ))
                risk = "WARN"

        # ── 9. Required tests from remote config — emit only on FAIL ──────────
        # Skip any test already covered by entry["docs"] to avoid duplicate FEHLT items.
        rule = config_rules.get(f"ICD10_{icd10.replace('.', '_')}", {})
        _dg_docs_lower = {d.lower() for d in entry.get("docs", [])}
        for test in rule.get("required_tests", []):
            if test.lower() in _dg_docs_lower:
                continue   # already checked in section 7 above
            if not _check_doc(test, soap, transcript):
                audit.append(AuditItem(
                    f"CFG_{test.upper()[:20]}",
                    f"{test} (Praxisregel)", "FAIL", "Pflichttest laut Konfiguration fehlt"
                ))
                risk = "WARN"

        # ── 10a. MT segment documentation — mandatory for 21201 ───────────────
        if position == "21201":
            seg_checker = _DOC_CHECKERS["Behandeltes Segment"]
            has_seg = seg_checker(obj.lower() + " " + assess.lower())

            # Smart Fill: Infer segment from clinical context if missing
            suggested_segment = ""
            if not has_seg:
                suggested_segment = self._infer_segment_from_context(soap, icd10, profile_id)

            audit.append(AuditItem(
                "MT_SEGMENT", "Behandeltes Segment (MT §125 SGB V Pflicht)",
                "PASS" if has_seg else "FAIL",
                "" if has_seg else
                (f"Fehlendes Segment (z.B. L4/L5, C5/C6) — 21201 ohne Segmentangabe nicht abrechenbar. "
                 f"Vorschlag: {suggested_segment}" if suggested_segment else
                 "Fehlendes Segment (z.B. L4/L5, C5/C6) — 21201 ohne Segmentangabe nicht abrechenbar.")
            ))
            if not has_seg:
                risk = "WARN"

        # ── 10b. Blasen-/Mastdarmfunktion in O-Feld — Cauda-equina screening ─
        # LWS and MT cases: documenting that bladder/bowel was checked is a
        # clinical safety standard (Cauda equina exclusion). It must appear in O.
        is_ws_mt = position == "21201" and any(
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

        # ── 10c. KPE 4-component documentation (LY2/LY3) ─────────────────────
        # Skip for transcript-based LY2 upgrades: the prescription is MLD, not KPE.
        # KPE requirements only apply when LY2 was resolved from the ICD (I97.x / C77).
        if dg in ("LY2", "LY3") and not _ly2_via_transcript:
            kpe_checker = _DOC_CHECKERS["KPE-Komponenten"]
            has_kpe = kpe_checker((obj + plan).lower())
            audit.append(AuditItem(
                "KPE_4COMP", "KPE 4 Komponenten (§125 SGB V Pflicht)",
                "PASS" if has_kpe else "FAIL",
                "" if has_kpe else
                "KPE: MLD + Kompressionsbandagierung + Entstauungsgymnastik + Hautpflege "
                "müssen pro Sitzung dokumentiert sein — fehlendes Element = Down-Coding zu MLD."
            ))
            if not has_kpe:
                risk = "WARN"

        # ── 10d. Lipödem BVB-Hinweis ──────────────────────────────────────────
        if dg == "LY3":
            audit.append(AuditItem(
                "LY3_BVB", "Lipödem — Besonderer Verordnungsbedarf",
                "WARN",
                "Lipödem (E88.2x) ist BVB, nicht LHB. Rezept muss als "
                "'Besonderer Verordnungsbedarf' (extrabudgetär) ausgestellt sein — bis 31.12.2025."
            ))

        # ── 10e. Red flag check — BLOCKS billing ──────────────────────────────
        red = self._red_flag_audit(soap)
        audit.extend(red)
        if any(a.status == "BLOCK" for a in red):
            risk = "BLOCK"

        # ── 10e-LY. LY-specific transcript red flags (Erysipel / DVT / kardiale Dekompensation) ──
        # These are absolute contraindications for MLD — must BLOCK billing immediately.
        if transcript:
            _t_low = transcript.lower()
            _ly_rf_patterns = [
                (r"erysipel",         "Erysipel-Verdacht",           "Erysipel-Verdacht: MLD absolut kontraindiziert — sofort Arzt!"),
                (r"tiefe.*venen|venenthrombose|tvt\b|thrombose",
                                      "Thrombose-Verdacht",           "Tiefe Venenthrombose V.a.: MLD kontraindiziert — Notfall!"),
                (r"kardiale.*dekompensation|herzinsuffizienz.*akut|akute.*herzinsuffizienz|atemnot.*herz",
                                      "Kardiale Dekompensation",      "Akute kardiale Dekompensation: MLD kontraindiziert — Notarzt!"),
            ]
            for _pat, _lbl, _msg in _ly_rf_patterns:
                if re.search(_pat, _t_low):
                    # Check for negation in transcript context
                    _match = re.search(_pat, _t_low)
                    if _match:
                        # Only check PRE-match context for negation (up to 60 chars before).
                        # Post-match window is kept short (25 chars) to avoid false negatives:
                        # e.g. "tiefe Venenthrombose. Es wurde KEINE MLD durchgeführt" — the
                        # "keine" refers to MLD, not to the thrombosis. A wide post-window
                        # would incorrectly suppress the BLOCK.
                        _pre_ctx  = _t_low[max(0, _match.start()-60): _match.start()]
                        _post_ctx = _t_low[_match.end(): _match.end()+25]
                        _ctx = _pre_ctx + _post_ctx
                        _negated = any(n in _ctx for n in [
                            "kein ", "keine ", "nicht ", "ausgeschlossen", "verdacht ausgeräumt",
                            "negativ", "ohne anzeichen"
                        ])
                        if not _negated:
                            audit.append(AuditItem(
                                f"RF_LY_{_lbl.upper().replace(' ', '_')}",
                                f"Red Flag (LY): {_lbl}",
                                "BLOCK", _msg
                            ))
                            risk = "BLOCK"

        # ── 10f. Red-Flag exclusion — only FAIL when genuinely missing ─────────
        # inject_audit_stamps() always adds the phrase, so this fires only when
        # the A-field is empty or the scribe pipeline didn't run.
        has_rf_exclusion = bool(re.search(r"red.flag|ausgeschlossen|kein(?:e)?\s+red", assess, re.I))
        if not has_rf_exclusion:
            audit.append(AuditItem("RF_EXCLUSION", "Red-Flag-Ausschluss im Assessment",
                                   "FAIL",
                                   "Red-Flag-Ausschluss fehlt im A-Feld — §106b Pflicht"))
            risk = "WARN"
        # RX_START_WINDOW is a process reminder, not a per-session doc item — omitted from
        # the audit list to avoid constant noise; reflected in compliance_warnings text only.

        # ── MLD duration upgrade: 45-min → 60-min when transcript confirms 60 minutes ──
        # 20201 = MLD 45 Min (€53.94)  →  20203 = MLD Ganzbehandlung 60 Min (€71.94)
        # Heilmittelkatalog §125 SGB V: pos. 20203 is the correct 60-min MLD position.
        # Only auto-upgrade when the therapist explicitly documented the session length.
        if position == "20201" and re.search(r"\b60\s*(?:min(?:uten?)?)?\b", transcript, re.I):
            position = "20203"
            entry = dict(entry)
            entry["position"] = "20203"
            entry["name"] = "MLD Ganzbehandlung 60 Min"
            entry["duration"] = 60
        # ── MLD duration downgrade: 45-min → 30-min when transcript confirms only 30 minutes ──
        # 20201 = MLD 45 Min  →  20205 = MLD Teilbehandlung 30 Min (€35.97)
        # Applies when therapist explicitly states "30 Minuten MLD" or "MLD-30".
        elif position == "20201" and re.search(r"\b30\s*(?:min(?:uten?)?)?\b", transcript, re.I):
            position = "20205"
            entry = dict(entry)
            entry["position"] = "20205"
            entry["name"] = "MLD Teilbehandlung 30 Min"
            entry["duration"] = 30

        # ── Early-termination flag: vorzeitiger Abbruch requires manual review ──
        # When the therapist explicitly documents an early session termination
        # ("vorzeitig beenden", "Abbruch", etc.) AND the position is a partial
        # session (20205), add a WARN so the claim is flagged for human review.
        _t_low = transcript.lower()
        if position == "20205" and any(k in _t_low for k in [
            "vorzeitig", "abbruch", "abgebrochen", "musste beenden", "frühzeitig",
            "vorzeitigen", "behandlung abbrechen", "sitzung unterbrech",
        ]):
            audit.append(AuditItem(
                "ABBRUCH_TEILBEHANDLUNG",
                "Abbruchgrund 30-Min-Teilbehandlung",
                "WARN",
                "Vorzeitiger Behandlungsabbruch dokumentiert — manuelle Prüfung erforderlich",
            ))
            risk = "WARN"

        # ── Determine overall audit status ─────────────────────────────────────
        if any(a.status == "BLOCK" for a in audit):
            audit_status = "BLOCK"
        elif any(a.status in ("FAIL", "WARN") for a in audit):
            audit_status = "REVIEW"
        else:
            audit_status = "PASS"

        return BillingResult(
            insurance_type=InsuranceType.GKV,
            position_number=position,
            position_name=entry["name"],
            diagnosegruppe=dg,
            diagnosegruppe_desc=entry["desc"],
            legal_basis="§ 125 SGB V | Anlage 2 Rahmenempfehlungen (01.01.2026) | § 106b Prüfung",
            session_duration_min=entry["duration"],
            risk_level=risk,
            audit_items=audit,
            audit_status=audit_status,
            compliance_warnings=(
                [str(a) for a in audit if a.status not in ("PASS",)] +
                ["Rezeptfrist: Behandlungsbeginn innerhalb 28 Tagen ab Rezeptdatum (14 Tage bei 'dringend')"]
            ),
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
        # NOTE: "lymphdrainage" and "mld" are treatment TECHNIQUES also used for acute
        # orthopaedic injuries (ankle sprains, haematoma). Requiring actual disease terms
        # prevents mis-routing a foot/ankle session to LY1/MLD billing.
        if any(k in text for k in ["lymphoedem", "lymphödeme", "kpe", "entstauung",
                                    "stemmer-zeichen", "lipoedem", "lipoedema",
                                    "stemmer positiv", "stemmer ist positiv",
                                    "stemmer ist negativ", "stemmer negativ",
                                    "prostata-op", "prostata op",
                                    "beckenlymphoedem", "sekundaeres lymphoedem"]):
            return "LY1"
        # Foot / ankle — check before generic WS fallback
        if any(k in text for k in ["sprunggelenk", "außenknöchel", "aussenknöchel",
                                    "malleolus", "osg", "usg", "achillessehne",
                                    "talofibulare", "calcaneus", "fersenschmerz",
                                    "plantarfasziitis", "hallux", "peroneus"]):
            return "EX5"
        if any(k in text for k in ["beckenboden", "inkontinenz", "harninkontinenz",
                                    "stressinkontinenz", "dranginkontinenz", "kontinenz",
                                    "beckenorgane", "prostatektomie", "postpartum"]):
            return "PF1"
        if any(k in text for k in ["kgg", "gerät", "trainingstherapie", "medizinische trainings",
                                    "beinpresse", "latzug", "ergometer", "krafttraining am",
                                    "mtt ", "gerätegestützt"]):
            return "KGG"
        if any(k in text for k in ["bewegungsbad", "aquatherapie", "wassergymnastik",
                                    "unterwassergymnastik", "hydrotherapie", "pool "]):
            return "BB1"
        if any(k in text for k in ["bindegewebsmassage", "bgm ", "klassische massage", "kmt ",
                                    "segmentmassage", "myogelose", "triggerpunkt-massage"]):
            return "MA1"
        if any(k in text for k in ["unterwasserdruckstrahl", "uw-massage", "uwm "]):
            return "MA2"
        if any(k in text for k in ["fango", "heiße rolle", "heisse rolle", "warmpackung",
                                    "wärmetherapie", "wärmestrahler", "rotlicht"]):
            return "TH1"
        if any(k in text for k in ["kältetherapie", "eispack", "kryotherapie", "eis-", "kälte "]):
            return "TH2"
        if any(k in text for k in ["tens ", "interferenzstrom", "ifc ", "galvano", "diadynamisch",
                                    "elektrotherapie", "reizstrom"]):
            return "EL1"
        if any(k in text for k in ["ems ", "elektrostimulation lähmung", "neuromuskuläre stimulation"]):
            return "EL2"
        if any(k in text for k in ["gruppentherapie", "gruppenbehandlung", "kurstherapie"]):
            return "GR1"
        if any(k in text for k in ["skoliose", "schroth", "rippenbuckel"]):
            return "WS3"
        if any(k in text for k in ["hws", "nacken", "zervikal", "trapezius",
                                   "schädelbasis", "scaleni", "subokzipital",
                                   "spannungskopfschmerz", "kopfschmerz", "c5", "c6", "c7"]):
            return "WS1a"
        if any(k in text for k in ["iliosakral", "isg", " lws", "lumbal", "lumbago",
                                   "ischialgie", "ischiasschmerz", "kreuzschmerz",
                                   "vorlauf-test", "vorlauftest", "blockierung im"]):
            return "WS1b"
        if any(k in text for k in ["schulter", "impingement", "rotatorenmanschette"]):
            return "EX2"
        if any(k in text for k in ["knie", "gonarthrose", "meniskus"]):
            return "EX3"
        if any(k in text for k in ["copd", "asthma", "atemweg"]):
            return "AT1"
        if any(k in text for k in ["handgelenk", "handwurzel", "radiusfraktur", "radius",
                                    "finger", "fingergelenk", "metakarpal", "phalanx",
                                    "ellbogen", "karpaltunnel", "ulna"]):
            return "EX6"
        if any(k in text for k in ["manuelle therapie", " mt ", "traktion", "gleitmobilisation"]):
            return "WS1b"
        # Default: Generic KG session for musculoskeletal conditions
        return "EX1b"

    def _mt_indicated(self, soap: dict, transcript: str) -> bool:
        text = (transcript + " " + soap.get("P", "")).lower()
        return any(k in text for k in ["manuelle therapie", " mt ", "traktion",
                                        "gleitmobilisation", "manipulation"])

    def _red_flag_audit(self, soap: dict) -> list:
        items = []
        obj = soap.get("O", "").lower()
        text_all = " ".join(str(v) for v in soap.values()).lower()

        # Special handling for "Taubheit" - check for "pelzig" which is benign numbness/tingling
        # "pelzig" is a common symptom descriptor that shouldn't trigger RED FLAG block
        has_pelzig = "pelzig" in text_all

        for flag, label in _RED_FLAGS.items():
            idx = text_all.find(flag.lower())
            if idx == -1:
                continue

            # Extended context check
            before = text_all[max(0, idx-40): idx]
            after = text_all[idx: idx + 120]
            context = before + after

            # Check for negation or documentation of screening
            negated = any(n in context for n in [
                "negativ", "ausgeschlossen", "kein", "keine", "nicht", "ohne", "unauffällig",
                "verneint", "normal", "regelrecht", "o.b.n.", "obn", "geprüft",
                "screening negativ", "test negativ"
            ])

            # Special case: "Taubheit" with "pelzig" - treat as documented symptom, not emergency
            if flag.lower() == "taubheit" and has_pelzig:
                if any(k in obj for k in ["pelzig", "missempfindung", "parästhesie", "kribbel"]):
                    continue  # PASS suppressed — documented screening, no alert needed

            # Special case: "Taubheitsgefühle" — compound word is a symptom descriptor,
            # not a confirmed pathological finding. If "gefühl" follows the match within
            # the same word, treat same as negated unless also in O-field as a finding.
            if flag.lower() == "taubheit":
                match_end = idx + len("taubheit")
                if text_all[match_end:match_end + 8].startswith("sgefühl") or \
                   text_all[match_end:match_end + 8].startswith("sgefuhl"):
                    if flag.lower() not in obj:
                        continue  # Compound "Taubheitsgefühle" in S/A without O-finding — not a block

            if negated:
                pass  # PASS suppressed — red flag excluded, nothing to show
            else:
                # Warn instead of Block if symptom is documented in O-field (clinical assessment present)
                if flag.lower() in obj:
                    items.append(AuditItem(f"RF_{flag.upper()}", f"Red Flag: {label}",
                                           "WARN",
                                           f"{flag} im O-Feld dokumentiert — bitte ärztlichen Ausschluss im A-Feld ergänzen"))
                else:
                    items.append(AuditItem(f"RF_{flag.upper()}", f"Red Flag: {label}",
                                           "BLOCK",
                                           f"{flag} ohne Ausschluss — ärztliche Abklärung vor Therapiefortsetzung!"))
        return items

    def _infer_segment_from_context(self, soap: dict, icd10: str, profile_id: str = None) -> str:
        """
        Smart Fill: Infer likely spinal/joint segment from clinical context.
        Eliminates need for therapist to manually type segment every time.

        Returns suggested segment string (e.g., "L4/L5 und L5/S1") or empty string if can't infer.
        """
        text = (soap.get("S", "") + " " + soap.get("O", "") + " " + soap.get("A", "")).lower()

        # LWS (Lumbar Spine) - most common
        if any(k in text for k in ["lws", "lumbal", "lumbago", "ischias", "iliosakral", "kreuzbein"]):
            # Pain location determines likely segments
            if "gesäß" in text or "gluteal" in text or "kreuzbein" in text:
                return "L5/S1 (geschätzt aus Schmerzlokalisation Gesäß/Kreuzbein)"
            elif "leiste" in text or "hüfte" in text:
                return "L3/L4 und L4/L5 (geschätzt aus Schmerzausstrahlung Leiste/Hüfte)"
            elif "bein" in text or "oberschenkel" in text:
                return "L4/L5 und L5/S1 (geschätzt aus Ausstrahlung ins Bein)"
            else:
                # Default LWS segments
                return "L4/L5 und L5/S1 (geschätzt aus LWS-Diagnose)"

        # HWS (Cervical Spine)
        elif any(k in text for k in ["hws", "zervikal", "nacken", "halswirbel"]):
            if "kopfschmerz" in text or "schwindel" in text:
                return "C1/C2 und C2/C3 (geschätzt aus Kopfschmerz/Schwindel)"
            elif "arm" in text or "schulter" in text:
                return "C5/C6 und C6/C7 (geschätzt aus Armausstrahlung)"
            else:
                return "C5/C6 und C6/C7 (geschätzt aus HWS-Diagnose)"

        # BWS (Thoracic Spine)
        elif any(k in text for k in ["bws", "thorakal", "brustwirbel", "rippe"]):
            return "Th6/Th7 und Th7/Th8 (geschätzt aus BWS-Diagnose)"

        # ISG (Sacroiliac Joint)
        elif any(k in text for k in ["isg", "iliosakral", "sakroiliak", "si-gelenk"]):
            return "ISG bds. (geschätzt aus ISG-Problematik)"

        # Shoulder
        elif any(k in text for k in ["schulter", "glenohumeral", "akromioklavikular", "rotatorenmanschette"]):
            return "Glenohumeralgelenk re/li (geschätzt aus Schulterdiagnose)"

        # Knee
        elif any(k in text for k in ["knie", "tibiofemoral", "patellofemoral", "menisk"]):
            return "Kniegelenk re/li (geschätzt aus Kniediagnose)"

        # Hip
        elif any(k in text for k in ["hüfte", "hüft", "koxofemoral", "tep"]):
            return "Hüftgelenk re/li (geschätzt aus Hüftdiagnose)"

        # Ankle
        elif any(k in text for k in ["sprunggelenk", "osg", "malleolus", "achilles"]):
            return "Oberes Sprunggelenk re/li (geschätzt aus OSG-Diagnose)"

        # Hand/Wrist
        elif any(k in text for k in ["handgelenk", "radioulnar", "radiokarpal"]):
            return "Radiokarpalgelenk re/li (geschätzt aus Handgelenk-Diagnose)"

        # Elbow
        elif any(k in text for k in ["ellbogen", "ellenbogen", "humeroradial", "humeroulnar"]):
            return "Ellbogengelenk re/li (geschätzt aus Ellbogendiagnose)"

        # ICD-based inference
        if icd10:
            if icd10.startswith("M54.5") or icd10.startswith("M54.4"):  # Lumbago/LWS
                return "L4/L5 und L5/S1 (geschätzt aus ICD M54.5/M54.4)"
            elif icd10.startswith("M54.2"):  # HWS
                return "C5/C6 und C6/C7 (geschätzt aus ICD M54.2)"
            elif icd10.startswith("M75"):  # Shoulder
                return "Glenohumeralgelenk (geschätzt aus ICD M75)"
            elif icd10.startswith("M17"):  # Knee
                return "Kniegelenk (geschätzt aus ICD M17)"

        # Can't infer - return empty (will trigger manual entry prompt)
        return ""


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
        profile_id: Optional[str] = None,
    ) -> BillingResult:
        """
        pkv_preise: dict[Positionsnummer → float] aus config_override.json.
        Beispiel: {"21201": 72.00, "20501": 55.00}
        Wenn gesetzt, gilt der Praxispreis statt des GebüTh-Orientierungswerts.
        GKV-Festpreise werden dadurch nicht berührt.
        """
        audit: list[AuditItem] = []
        pkv_preise = pkv_preise or {}

        dg = _match_dg(icd10) or "EX1b"
        if profile_id and profile_id in _PROFILE_TO_DG:
            dg = _PROFILE_TO_DG[profile_id]
        entry = _HMK[dg]
        position = entry["position"]

        if entry.get("optional_mt") and self._mt_indicated(soap, transcript):
            position = "21201"

        # Praxispreis has priority; GebüTh is advisory fallback
        praxispreis = pkv_preise.get(position)
        price_range = _PKV_RANGES.get(position, (25.0, 65.0))
        price_str = (f"€{praxispreis:.2f} (Praxispreis)"
                     if praxispreis
                     else f"€{price_range[0]:.0f}–{price_range[1]:.0f} (GebüTh-Orientierung)")

        # ── 1. PKV info — emit only when praxispreis is missing (actionable) ──
        if not praxispreis:
            audit.append(AuditItem(
                "PKV_INFO", "PKV-Abrechnung",
                "WARN",
                f"{price_str} | Praxispreis in config_override.json hinterlegen fuer exakten Betrag."
            ))

        likelihood = self._score_likelihood(icd10, soap)
        hints = self._hints(soap, transcript, position)

        # ── 2. Missing mandatory documentation (WARN = PKV may reject) ────────
        for doc in entry["docs"]:
            present = _check_doc(doc, soap, transcript)
            if not present:
                audit.append(AuditItem(
                    f"DOC_{doc[:20]}", doc,
                    "WARN", "Fehlt — PKV-Retaxation wahrscheinlicher ohne diesen Befund"
                ))

        # ── 3. Erstattungswahrscheinlichkeit — only surface if LOW ─────────────
        if likelihood == "GERING":
            audit.append(AuditItem("PKV_LIKELIHOOD", "Erstattungswahrscheinlichkeit",
                                   "WARN",
                                   "GERING — Kostenvoranschlag und Begründungsschreiben empfohlen"))

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
        if re.search(r"\d+ - \d+ - \d+|\d+-\d+-\d+", obj):
            score += 2
        if re.search(r"vas\s*\d|schmerz.*\d+/10|nrs\s*\d", (soap.get("S", "") + obj).lower()):
            score += 1
        if len(obj) > 100:
            score += 2
        if icd10 not in ("M99.9", "N/A", ""):
            score += 2
        if re.search(r"\d+\s*(cm|°|grad)", obj.lower()):
            score += 1
        if len(soap.get("P", "")) > 60:
            score += 1
        return "HOCH" if score >= 7 else "MITTEL" if score >= 4 else "GERING"

    def _hints(self, soap: dict, transcript: str, position: str) -> list:
        hints = []
        obj = soap.get("O", "")
        p = soap.get("P", "")
        if not re.search(r"\d+ - \d+ - \d+", obj):
            hints.append("💡 Neutral-Null-Werte im O-Feld erhöhen PKV-Erstattung erheblich.")
        if re.search(r"vas\s*\d|schmerz.*\d+/10|nrs\s*\d", (soap.get("S", "") + obj).lower()) is None:
            hints.append("💡 VAS-Score im S-Feld — PKV prüft Schmerzquantifizierung.")
        if len(p) < 60:
            hints.append("💡 P-Feld präzisieren — PKV erwartet konkretes Outcome-Ziel.")
        if position == "21201":
            hints.append("💡 MT-Erstbefundung (21200, 30 Min) bei Neupatient separat abrechenbar.")
        if len(obj) < 80:
            hints.append("💡 Ausführlicheres O-Feld reduziert Retaxationsrisiko.")
        return hints


# ── BG engine (wraps GKV, adds DGUV surcharge + extra docs) ──────────────────

class _BGEngine:
    def __init__(self):
        self._gkv = _GKVEngine()

    def evaluate(self, icd10: str, soap: dict, transcript: str,
                 config_rules: dict, profile_id: Optional[str] = None) -> BillingResult:
        result = self._gkv.evaluate(icd10, soap, transcript, config_rules, profile_id=profile_id)
        result.insurance_type = InsuranceType.BG

        # Add BG-specific doc checks
        for doc in _BG_EXTRA_DOCS:
            present = _check_doc(doc, soap, transcript)
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

# ── Profile → Diagnosegruppe override for modality-specific sessions ─────────
# These profiles are identified by transcript content (not ICD codes).
# Without this map the GKV engine would fall through to the ICD-based DG
# (e.g. M54.5 → WS1b/21201) and produce the wrong billing position.
_PROFILE_TO_DG: dict[str, str] = {
    # ── Anatomy profiles — bypass ICD lookup so a bad/generic ICD never
    #    mis-routes to the wrong billing position ──────────────────────
    "EX_FUSS":    "EX5",   # 20501 Fuß / Sprunggelenk
    "EX_KNIE":    "EX3",   # 20501 Knie
    "EX_HUefte":  "EX4",   # 20501 Hüfte
    "EX_SCHULTER":"EX2",   # 21201 Schulter (MT)
    "EX_HAND":    "EX6",   # 21201 Hand/Handgelenk
    "EX_HWS":     "WS1a",  # 21201 HWS
    "EX_LWS":     "WS1b",  # 21201 LWS/ISG
    "MT":         "WS1b",  # 21201 Manuelle Therapie WS
    "LY":         "LY1",   # 20201 Lymphologie
    "ZNS_ADULT":  "ZNS1",  # 20511 Neurologie adult
    "ZNS_FAZ":    "ZNS1",  # 20511 Fazialisparese
    "AT":         "AT1",   # 20560 Atemtherapie
    "RHEUM":      "EX1a",  # 20501 Rheuma (entzündlich)
    "GEB":        "GEB2",  # 21904 Geburtshilfe / Rückbildung
    # ── Modality profiles — identified by transcript content, not ICD ─
    "KGG":    "KGG",   # 20507 KG am Gerät / MTT
    "ELEKTRO": "EL1",  # 21302 Elektrotherapie
    "THERMO": "TH1",   # 21501 Wärmetherapie / Fango
    "MASSE":  "MA1",   # 20106 Klassische Massage / KMT
    "UWM":    "MA2",   # 20102 Unterwasserdruckstrahlmassage
    "AQUA":   "BB1",   # 20902 KG im Bewegungsbad
    "GRUPPE": "GR1",   # 20601 KG Gruppenbehandlung
    "BECKEN": "PF1",   # 20501 Beckenbodentherapie
}


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
            profile_id="KGG",  # optional — overrides ICD-based billing for modality profiles
        )
        # Use the logger for output
        logger.info(result.format_audit_report())
        logger.info(result.format_billing_line())
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
        profile_id: Optional[str] = None,
    ) -> BillingResult:
        """
        config_rules : aus ConfigManager.billing_rules  (GKV/BG)
        pkv_preise   : aus ConfigManager.pkv_preise     (PKV — praxiseigene Preise)
                       z.B. {"21201": 72.00, "20501": 55.00}
                       GKV-Festpreise werden dadurch nicht verändert.
        profile_id   : aus PhysioScribe._detect_profile() — overrides ICD-based DG
                       for modality-specific profiles (KGG, ELEKTRO, THERMO, etc.)
        """
        rules = config_rules or {}
        if insurance_type == InsuranceType.BG:
            return self._bg.evaluate(icd10, soap, transcript, rules, profile_id=profile_id)
        if insurance_type == InsuranceType.PKV:
            return self._pkv.evaluate(icd10, soap, transcript, pkv_preise=pkv_preise,
                                      profile_id=profile_id)
        return self._gkv.evaluate(icd10, soap, transcript, rules, profile_id=profile_id)
