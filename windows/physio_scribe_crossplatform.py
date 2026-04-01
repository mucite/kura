"""
Kura Engine — Windows
Inference: llama-cpp-python (GGUF) + faster-whisper (CPU)
Clinical logic: identical to macOS (LearningManager, post-processing pipeline, compliance checks)
"""
import gc
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config_manager import ConfigManager
from shared.learning_manager import LearningManager


class KuraEngine:
    def __init__(self):
        self.learning_mgr = LearningManager()
        self.config = ConfigManager()
        self._setup_paths()
        self._check_concurrent_instances()
        self._check_system_resources()
        self._init_models()
        self.billing_rules = self.config.billing_rules
        self.audit_rules = self.config.audit_rules
        self.llm_config = self.config.data.get("llm_config", {})
        self.whisper_config = self.config.data.get("whisper_config", {})

    # ── Path setup ────────────────────────────────────────────────────────────

    def _setup_paths(self):
        if getattr(sys, "frozen", False):
            base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        else:
            base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.model_dir = os.path.join(base, "models")
        self.stt_model_path = os.path.join(self.model_dir, "whisper-large-v3-turbo")

    # ── System checks ─────────────────────────────────────────────────────────

    def _check_concurrent_instances(self):
        try:
            import psutil
            curr = os.getpid()
            for p in psutil.process_iter(["pid", "cmdline"]):
                if p.pid != curr and any("Kura" in str(a) for a in p.info.get("cmdline", [])):
                    print("⚠️ Concurrent Kura instance detected.")
        except Exception:
            pass

    def _check_system_resources(self):
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < 2.0:
                print(f"⚠️ Low RAM: {available_gb:.1f} GB available. Kura needs at least 2 GB free.")
        except Exception:
            pass

    # ── Model loading ─────────────────────────────────────────────────────────

    def _init_models(self):
        print("🩺 Loading local medical-grade models...")

        # LLM (GGUF via llama-cpp-python)
        try:
            from llama_cpp import Llama

            candidates = [
                os.path.join(self.model_dir, "Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B-Instruct.Q4_K_M.gguf"),
                os.path.join(self.model_dir, "Llama-3.2-3B-Instruct.Q4_K_M.gguf"),
                os.path.join(self.model_dir, "llama-3.2-3b-medical.Q4_K_M.gguf"),
                os.path.join(self.model_dir, "Mistral-7B-Instruct-v0.1.Q4_K_M.gguf"),
            ]

            llm_path = next((p for p in candidates if os.path.exists(p)), None)
            if not llm_path:
                raise FileNotFoundError(
                    "No GGUF model found in models/ directory.\n"
                    "Download Llama-3.2-3B-Instruct-Q4_K_M.gguf from HuggingFace and place it in models/."
                )

            print(f"✅ Loading LLM: {os.path.basename(llm_path)}")
            self.llm = Llama(
                model_path=llm_path,
                n_ctx=2048,
                n_threads=min(os.cpu_count() or 4, 8),
                n_gpu_layers=0,  # CPU-only; set >0 if CUDA available
                verbose=False,
            )
            print("✅ LLM loaded")
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

        # STT (faster-whisper)
        try:
            from faster_whisper import WhisperModel

            # Try local model directory first, fall back to HuggingFace name
            if os.path.isdir(self.stt_model_path):
                stt_src = self.stt_model_path
            else:
                stt_src = "large-v3-turbo"
                print(f"⚠️ Local Whisper model not found, will download: {stt_src}")

            self.whisper = WhisperModel(stt_src, device="cpu", compute_type="int8")
            print("✅ Whisper STT loaded")
        except ImportError:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

        gc.collect()
        print("✅ Kura Engine ready (100% local, DSGVO-konform)")

    # ── Transcript cleaning ────────────────────────────────────────────────────

    def clean_transcript(self, transcript: str) -> str:
        """Fix common Whisper hallucinations in German physiotherapy terminology."""
        corrections = {
            r"Bobert|Bobat|Bobart": "Bobath",
            r"Stämmer|Stemmerzeichen": "Stemmer-Zeichen",
            r"Kompressionsstrübe|Strübe": "Kompressionsstrümpfe",
            r"Lymphödes|Lymphödemen": "Lymphödem",
            r"Anastomosen|Anastomo": "Anastomosen",
            r"Lenertschroth|Schrot-Therapie": "Lehnert-Schroth",
            r"ischio-choraler|ischo-cural": "ischiocrurale",
            r"autochtoner": "autochthoner",
            r"Lasek|Lasegge|Laseque": "Lasègue",
            r"Psoasdehnung": "Psoas-Dehnung",
            r"Finger-Bodenabstand|FBA": "Finger-Boden-Abstand (FBA)",
            r"Sanddemobilisation|Sandmobilisation": "sanfte Mobilisation",
            r"VKB Plastik|VKB-Plastik": "Vorderes Kreuzband (VKB) Plastik",
            r"Kranial": "kranial",
            r"M, M, T": "MMT (Manual Muscle Test)",
            r"Kopfgelenk|C1|C2": "Kopfgelenksregion (C1/C2)",
            r"Hinterkopfschmerz": "okzipitaler Kopfschmerz",
            r"Kinnretraktion": "Kinn-Retraktion (Deep Neck Flexor Training)",
            r"Detonisierung": "Detonisierung (Tonussenkung)",
            r"CNMD|CMND|CNMT|C\.N\.M\.D\.": "CMD",
            r"Knieknadi\w*|Knienadig\w*|Knie.?Nadi\w*|Knienachgibigkeit": "Knienachgiebigkeit",
            r"Rotatorenmanschete\b|Rotatoren.?Manschette": "Rotatorenmanschette",
            r"Impingmentsyndrom|Impingement.?Syndrom": "Impingementsyndrom",
            r"Plantarfasciitis|Plantar.?Fasziitis": "Plantarfasziitis",
            r"Karpaltunnel.?Syndrom": "Karpaltunnelsyndrom",
            r"Epikondylitis|Epicondylitis": "Epikondylitis",
            r"Propriozepzion|Propioception|Propiozeption": "Propriozeption",
            r"Tendinapathie|Tendinopatie": "Tendinopathie",
            r"Patellofemorales?\s?Schmerz.?Syndrom": "Patellofemoralschmerzsyndrom",
        }
        for wrong, correct in corrections.items():
            transcript = re.sub(wrong, correct, transcript, flags=re.IGNORECASE)

        # Context-aware Schenkelhalsfraktur vs Schulterhalsfraktur disambiguation
        if re.search(r'schulterhalsfraktur', transcript, re.I):
            hip_context = any(k in transcript.lower() for k in [
                'hüft', 'huefte', 'femur', 'schenkel', 'oberschenkel',
                'tep', 'hüft-tep', 'hüftprothese', 'prothese',
                'schenkelhals', 'pertrochantär', 'pertrochantar',
            ])
            if hip_context:
                transcript = re.sub(
                    r'schulterhalsfraktur', 'Schenkelhalsfraktur',
                    transcript, flags=re.I
                )

        transcript = re.sub(
            r'schen(?:k|g)el\s*hals\s*fr?ak?tur',
            'Schenkelhalsfraktur', transcript, flags=re.I
        )

        # Dynamic corrections from Gist whisper_config.medical_corrections
        for wrong, correct in self.whisper_config.get("medical_corrections", {}).items():
            try:
                transcript = re.sub(wrong, correct, transcript, flags=re.IGNORECASE)
            except re.error:
                pass
        return transcript

    # ── Diagnosis Profiles ────────────────────────────────────────────────────
    # Priority: higher number wins. Age constraints narrow the match.
    # Triggers are checked case-insensitively against the full transcript.
    _PROFILES = {
        "ZNS_PAD": {
            "label":    "Neuro-Paediatrie (Bobath-Kind / CP)",
            "billing":  "20511",
            "priority": 100,
            "age_max":  17,
            "triggers": [
                "spastische tetraparese", "spastische diparese", "spastische hemiparese",
                "zerebralparese", "infantile cerebralparese", " cp ", "cp-kind",
                "dyskinetisch", "athetose", "vojta-kind", "bobath-kind",
                "entwicklungsverzoegerung", "fruehgeburt", "perinatale schaedigung",
                "fazio-oral", "schluckstoerung kind",
            ],
            "icd_prefix": ["G80", "P91", "Q"],
            "checklist": [
                "Kopfkontrolle: vorhanden / eingeschraenkt / nicht moeglich",
                "Langsitz: stabil / mit Abstuetzung / nicht moeglich",
                "Tonusregulation: Ashworth-Grad (0-4) je Extremitaet",
                "Primitive Reflexe: ATNR / Moro / Saugreflexe (integriert / persistent)",
                "GMFCS-Level (I-V)",
                "Greiffunktion: palmar / lateral / Pinzettengriff / nicht moeglich",
                "Kommunikation / Kognition: altersgerecht / verzoegert",
                "Hilfsmittel: Orthesen / Stehstaender / Rollstuhl (vorhanden / geplant)",
            ],
        },
        "ZNS_FAZ": {
            "label":    "Fazialisparese",
            "billing":  "20511",
            "priority": 90,
            "triggers": [
                "fazialisparese", "fazialis", "bell", "nervus facialis",
                "gesichtslaehm", "gesichtsnerv",
            ],
            "icd_prefix": ["G51"],
            "checklist": [
                "House-Brackmann-Grad (I-VI)",
                "Synkinesien: vorhanden / nicht vorhanden",
                "Augenlidschluss: vollstaendig / unvollstaendig (Lagophthalmus)",
                "Mundwinkel-Symmetrie: in Ruhe / bei Bewegung",
                "EMG-Befund (falls vorliegend, sonst n.d.)",
            ],
        },
        "ZNS_ADULT": {
            "label":    "Neurologie adult",
            "billing":  "20511",
            "priority": 80,
            "age_min":  18,
            "triggers": [
                "schlaganfall", "apoplex", "hemiplegie", "hemiparese",
                "parkinson", "morbus parkinson", "multiple sklerose", "ms-erkrankung",
                "querschnitt", "sht", "schaedel-hirn-trauma", "hirnverletzung",
                "ataxie", "spastik",
            ],
            "icd_prefix": ["G20", "G35", "I69", "G81", "G82", "S06"],
            "checklist": [
                "Barthel-Index (0-100)",
                "Ashworth-Skala (0-4) fuer betroffene Extremitaet",
                "Berg Balance Scale (0-56) ODER TUG-Test (Sekunden)",
                "10m Gehtest (Sekunden + Hilfsmittel)",
                "ADL-Status: selbstaendig / mit Hilfe / abhaengig",
                "Bei Parkinson: Hoehn-Yahr (1-5)",
                "Bei MS: EDSS + Fatigue (vorhanden / nicht vorhanden)",
            ],
        },
        "LY": {
            "label":    "Lymphologie / Entstauung",
            "billing":  "20201",
            "priority": 70,
            "triggers": [
                "lymphoedem", "lymph", "mld", "kpe", "entstauung", "stemmer",
                "lipoedem", "mastektomie", "axillaer", "sentinel", "erysipel",
                "sekundaeres oedema",
            ],
            "icd_prefix": ["I89", "Q82", "C77", "I97"],
            "checklist": [
                "Stadium: 1 (reversibel) / 2 (irreversibel) / 3 (Elephantiasis)",
                "Stemmer-Zeichen: positiv / negativ + Lokalisation",
                "Umfangsmessung beidseitig in cm (Koerperstelle angeben)",
                "Konsistenz: weich / teigig / hart / fibrosiert",
                "Hautbefund: Roetung / Hyperkeratose / Papillomatose",
            ],
        },
        "AT": {
            "label":    "Atemtherapie",
            "billing":  "20560",
            "priority": 60,
            "triggers": [
                "copd", "asthma", "atemweg", "mukoviszidose", "atemtherapie",
                "sekretmobilisation", "ateminsuffizienz", "lungenfibrose",
                "pneumonie", "pleuraerguss",
            ],
            "icd_prefix": ["J44", "J45", "J96", "E84", "J18"],
            "checklist": [
                "Spirometrie: FEV1 / FVC (Liter + % Soll), sonst n.d.",
                "Atemfrequenz (in Ruhe)",
                "SpO2 (%)",
                "Hustenstoss: effektiv / ineffektiv",
                "Sekret: Menge / Farbe / Konsistenz",
                "Atemhilfsmuskulatur: aktiv / nicht aktiv",
            ],
        },
        "ONKO": {
            "label":    "Onkologie-Reha",
            "billing":  "20501",
            "priority": 65,
            "triggers": [
                "onkologie", "krebserkrankung", "tumorpatientin", "chemotherapie",
                "bestrahlung", "reha nach krebs", "fatigue syndrom", "kachexie",
                "mammakarzinom", "kolonkarzinom",
            ],
            "icd_prefix": ["Z08", "Z09", "C", "Z85"],
            "checklist": [
                "Fatigue-Skala (0-10 oder BFI, FACIT-Fatigue)",
                "Karnofsky-Index oder ECOG-Score",
                "Belastbarkeit: MET oder Gehstrecke (m)",
                "Kraft (MRC 0-5) je nach betroffenem Bereich",
                "Nebenwirkungen: Neuropathie / Narbe / Lymphoedem (vorhanden / nicht vorhanden)",
            ],
        },
        "RHEUM": {
            "label":    "Rheumatologie / Entzuendliche Erkrankung",
            "billing":  "20501",
            "priority": 55,
            "triggers": [
                "rheuma", "rheumatoide arthritis", "ra ", "psoriasis-arthritis",
                "ankylosierende spondylitis", "morbus bechterew", "systemischer lupus",
                "sle ", "gicht", "entzuendlich",
            ],
            "icd_prefix": ["M05", "M06", "M45", "M07", "L40.5"],
            "checklist": [
                "DAS28 (falls vorliegend) oder klinische Aktivitaetsbeurteilung",
                "Morgendliche Steifigkeit: Dauer in Minuten",
                "Gelenkschwellung: befallene Gelenke benennen",
                "Kraftminderung: Jamar-Handkraft (kg) re / li",
                "CRP / BSG (falls aus Akte bekannt)",
                "BASFI (bei Spondylitis, falls vorhanden)",
            ],
        },
        "MT": {
            "label":    "Manuelle Therapie",
            "billing":  "21201",
            "priority": 50,
            "triggers": [
                "manuelle therapie", " mt ", "traktion", "gleitmobilisation",
                "manipulation", "gelenkmobilisation", "hvla", "facettensyndrom",
                "iliosakralgelenk", "isg",
            ],
            "icd_prefix": ["M54", "M51", "M47", "M45"],
            "checklist": [
                "Behandeltes Segment: z.B. L4/L5 oder C5/C6 (MT-Pflichtangabe fuer 21201)",
                "ROM Neutral-Null je Bewegungsebene: [Ext]-[0]-[Flex]",
                "Endgefuehl: fest-elastisch / fest / leer / hart (je Richtung)",
                "Palpation: Druckdolenz + exakte Lokalisation",
                "Schmerz: VAS x/10",
                "Provokationstest: Lasegue / Spurling / Slump (positiv / negativ)",
                "LWS: Schober-Zeichen (cm zu cm), FBA (cm)",
                "Blasen-/Mastdarmfunktion: unauffaellig / gestaert (Cauda-equina-Screening)",
            ],
        },
        "EX_SCHULTER": {
            "label":    "Extremitaeten Schulter (EX2)",
            "billing":  "21201",
            "priority": 45,
            "triggers": [
                "schulter", "rotatorenmanschette", "impingement", "supraspinatus",
                "bizepssehne", "acromion", "omarthrose", "bankart", "slap",
                "frozen shoulder",
            ],
            "icd_prefix": ["M75"],
            "checklist": [
                "Hawkins-Test: positiv / negativ",
                "Jobe-Test (Empty Can): positiv / negativ",
                "Painful Arc: Grad-Bereich angeben",
                "ROM: Flexion / Abduktion / ARO / IRO (Neutral-Null-Methode)",
                "Kraftgrad MRC (0-5): Abduktion / ARO",
            ],
        },
        "EX_KNIE": {
            "label":    "Extremitaeten Knie (EX3)",
            "billing":  "21201",
            "priority": 44,
            "triggers": [
                "knie", "gonarthrose", "vkb", "kreuzband", "hkb",
                "meniskus", "patella", "knieschmerz", "knie-tep", "tkep",
            ],
            "icd_prefix": ["M17", "M23", "S83"],
            "checklist": [
                "Umfang Knie beidseits in cm (Oedemmass)",
                "Lachman-Test: positiv / negativ",
                "McMurray-Test: positiv / negativ (Innen- / Aussenmeniskus)",
                "ROM: Extension / Flexion (Grad)",
                "VAS-Score (0-10)",
            ],
        },
        "EX_HWS": {
            "label":    "HWS / Zervikalsyndrom",
            "billing":  "21201",
            "priority": 43,
            "triggers": [
                "hws", "halswirbel", "zervikalsyndrom", "cervical", "nacken",
                "kopfschmerz", "okzipital", "torticollis", "schleudertrauma",
            ],
            "icd_prefix": ["M54.2", "M50", "G44"],
            "checklist": [
                "Behandeltes Segment: C__/C__ oder C__/Th__ (MT-Pflichtangabe fuer 21201)",
                "ROM HWS: Flexion / Extension / Latflex re+li / Rotation re+li (Grad)",
                "Endgefuehl: fest-elastisch / fest / leer / muskulaer (je Richtung)",
                "Spurling-Test: positiv / negativ (mit Seitenangabe)",
                "Neurologisches Screening: Reflexe / Sensibilitaet / Kraft C5-C8",
                "VAS (0-10)",
                "Schmerzmuster: lokal / ausstrahlend (Dermatom angeben)",
            ],
        },
        "EX_LWS": {
            "label":    "LWS / Lumbalgie",
            "billing":  "21201",
            "priority": 42,
            "triggers": [
                "lws", "lendenwirbel", "lumbalgie", "lumboischialgie", "ischiasschmerz",
                "bandscheibenvorfall", "lumbago", "rücken", "wirbelsaeule",
            ],
            "icd_prefix": ["M54.4", "M54.5", "M51"],
            "checklist": [
                "Behandeltes Segment: L__/L__ oder L__/S__ (MT-Pflichtangabe fuer 21201)",
                "Lasegue-Test: Grad + Seite (z.B. re. positiv bei 45 Grad)",
                "Schober-Zeichen: X cm zu Y cm",
                "FBA (Finger-Boden-Abstand): X cm",
                "ROM LWS: Flexion / Extension / Latflex (Neutral-Null-Methode)",
                "Neurologisches Screening: Reflexe ASR/PSR / Sensibilitaet / Kraft L3-S1",
                "Blasen-/Mastdarmfunktion: unauffaellig / gestaert (Cauda-equina-Screening)",
                "VAS (0-10)",
            ],
        },
        "EX_HUefte": {
            "label":    "Extremitaeten Huefte (EX4)",
            "billing":  "21201",
            "priority": 41,
            "triggers": [
                "huefte", "coxarthrose", "hüftprothese", "htep", "trochanter",
                "piriformis", "femur", "coxa", "hüftgelenk",
            ],
            "icd_prefix": ["M16", "Z96.6", "M70.6"],
            "checklist": [
                "ROM Huefte: Flexion / Extension / ABD / ADD / IRO / ARO (Grad)",
                "Trendelenburg-Zeichen: positiv / negativ",
                "Thomas-Handgriff: positiv / negativ",
                "Kraft Huefte (MRC 0-5): Abduktion / Extension",
                "VAS (0-10)",
            ],
        },
        "EX_FUSS": {
            "label":    "Extremitaeten Fuss / Sprunggelenk (EX5)",
            "billing":  "21201",
            "priority": 40,
            "triggers": [
                "fuss", "sprunggelenk", "osg", "usg", "achillessee", "plantarfasziitis",
                "hallux", "fersenschmerz", "peroneus", "bandruptur",
            ],
            "icd_prefix": ["M79.3", "M72.2", "S93"],
            "checklist": [
                "ROM OSG: Dorsalextension / Plantarflexion (Grad)",
                "ROM USG: Pronation / Supination (Grad)",
                "Stabilitaetstest: Schubladentest / Talarneigung (positiv / negativ)",
                "Einbeinstand: stabil / instabil (Sekunden)",
                "Oedemmass Sprunggelenk: Umfang in cm",
            ],
        },
        "GER": {
            "label":    "Geriatrie / Sturzpraevention",
            "billing":  "20501",
            "priority": 35,
            "age_min":  65,
            "triggers": [
                "geriatrie", "sturz", "sturzrisiko", "demenz", "osteoporose",
                "gebrechlichkeit", "frailty", "sarkopenie", "gangstörung alter",
            ],
            "icd_prefix": ["M81", "Z74", "F00", "F01", "R26"],
            "checklist": [
                "TUG-Test (Timed Up and Go): Sekunden",
                "Chair Stand Test (5x): Sekunden",
                "Berg Balance Scale (0-56) ODER Tinetti-Test",
                "Ganggeschwindigkeit (m/s)",
                "Sturzanamnese: Anzahl Stuerze letztes Jahr",
                "Hilfsmittel: Rollator / Gehstock / keine",
            ],
        },
        "POST_OP": {
            "label":    "Postoperative Reha",
            "billing":  "20501",
            "priority": 30,
            "triggers": [
                "post-op", "postoperativ", "nach der op", "nach op", "postoperativer",
                "op-wunde", "narbe", "nahtdehiszenz", "prothesenversorgung",
            ],
            "icd_prefix": ["Z96", "Z47", "T84"],
            "checklist": [
                "OP-Datum und OP-Art (aus Arztbrief)",
                "Wundzustand: reizlos / Roetung / Sekretion",
                "Belastungsstatus: Vollbelastung / Teilbelastung X kg / Entlastung",
                "ROM aktuell vs. Ziel-ROM (Grad)",
                "Schwellung: Umfang cm (Vergleich Gegenseite)",
                "Schmerzfreiheit bei Belastung: ja / nein (VAS)",
            ],
        },
        "KG": {
            "label":    "Krankengymnastik allgemein",
            "billing":  "20501",
            "priority": 0,
            "triggers": [],   # fallback — always matches last
            "icd_prefix": [],
            "checklist": [
                "Schmerzlokalisation und -qualitaet (VAS 0-10)",
                "ROM mit Gradangabe fuer betroffene Gelenke",
                "Muskelkraft MMT (0-5) fuer betroffene Gruppen",
                "Palpation: Tonusbefund, Druckdolenz",
                "Funktionelles Therapieziel (SMART formuliert)",
            ],
        },
    }

    def _detect_profile(self, transcript: str) -> str:
        """
        Diagnosis-First profile detection.
        1. Detect patient age from transcript (e.g. '4 Jahre alt').
        2. Score every profile by trigger matches + age constraints.
        3. Return the highest-priority matching profile ID.
        """
        t = transcript.lower()

        # Age extraction — "4 Jahre alt", "4-jaehrig", "4 J."
        age = None
        m = re.search(r'(\d{1,2})\s*(?:jahre?\s*alt|j\b|-jaehrig)', t)
        if m:
            age = int(m.group(1))

        best_id = "KG"
        best_priority = -1

        for pid, prof in self._PROFILES.items():
            if pid == "KG":
                continue  # evaluated as fallback

            # Age constraints
            if age is not None:
                if prof.get("age_max") is not None and age > prof["age_max"]:
                    continue
                if prof.get("age_min") is not None and age < prof["age_min"]:
                    continue
            else:
                # No age in transcript — skip paediatric profiles
                if prof.get("age_max", 999) <= 17:
                    continue

            priority = prof.get("priority", 0)
            if priority <= best_priority:
                continue

            if any(trigger in t for trigger in prof.get("triggers", [])):
                best_id = pid
                best_priority = priority

        return best_id

    def _profile_checklist(self, profile_id: str) -> str:
        prof = self._PROFILES.get(profile_id, self._PROFILES["KG"])
        label    = prof["label"]
        billing  = prof["billing"]
        items    = "\n".join(f"- {item}" for item in prof["checklist"])
        icd_hint = ", ".join(prof.get("icd_prefix", [])) or "nach Befund"
        return (
            f"PROFIL: {label}  |  Abrechnung: {billing}\n"
            f"ICD-10-Hinweis: {icd_hint}\n"
            f"PFLICHTFELDER O-Feld:\n{items}"
        )

    def build_prompt(self, transcript: str, profile_id: str = "KG") -> str:
        learning_notes = self.learning_mgr.get_relevant_prefs(transcript)
        style_injection = f"\nBEVORZUGTE CODES DES THERAPEUTEN:\n{learning_notes}\n" if learning_notes else ""
        checklist = self._profile_checklist(profile_id)
        prof = self._PROFILES.get(profile_id, self._PROFILES["KG"])

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Du bist ein klinischer Dokumentationsexperte fuer deutsche Physiotherapie (Paragraph 106b SGB V).
DIAGNOSE-PROFIL: {prof["label"]}  |  Abrechnung: {prof["billing"]}
{style_injection}
EXTRAKTIONSREGELN (ABSOLUT VERBINDLICH):
1. Extrahiere AUSSCHLIESSLICH Informationen aus dem Transkript.
2. Fehlende Werte: schreibe "n.d." (nicht dokumentiert) — NIEMALS erfinden.
3. Zahlen EXAKT: "VAS 7" nicht "starke Schmerzen", "+4cm" nicht "Schwellung".
4. Neutral-Null-Methode: [Ext]-[0]-[Flex], Beispiel Knie: "0-0-90".
5. Red Flags IMMER im A-Feld: "Red Flags klinisch ausgeschlossen." (oder benennen).
6. DIAGNOSEN GEHOEREN IN A, NICHT IN S: ICD-10-Codes, Erkrankungsbezeichnungen (z.B. "Gonarthrose", "Bandscheibenvorfall", "Lymphödem"), Diagnose-Aussagen und Vordiagnosen NIEMALS in S schreiben. S enthaelt NUR subjektive Patientenaussagen.

PROFIL-PFLICHTFELDER (diese Felder MUESSEN im O-Feld erscheinen):
{checklist}

SOAP-STRUKTUR:
S: Hauptbeschwerde + Schmerzlokalisation + VAS x/10 + Dauer + Ausloeser
O: ALLE klinischen Messwerte und Tests des Profils — KEINE Zusammenfassungen
A: ICD-10-Diagnose | Differentialdiagnose | Red-Flag-Ausschluss
P: Heilmittel ({prof["billing"]}) + Technik + Frequenz + messbares Funktionsziel

{{
  "icd10": "[spezifischer ICD-10-Code]",
  "soap": {{"S": "...", "O": "...", "A": "...", "P": "..."}},
  "billing_suggestion": "{prof["billing"]}"
}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Transkript analysieren: {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""

    # ── Inference ──────────────────────────────────────────────────────────────

    def _transcribe(self, audio_path: str) -> str:
        wcfg = self.whisper_config
        segments, _ = self.whisper.transcribe(
            audio_path,
            language=wcfg.get("language", "de"),
            initial_prompt=wcfg.get("initial_prompt", "Physiotherapie Befund. Neutral-Null-Methode. VAS Schmerzskala."),
            temperature=wcfg.get("temperature", 0.0),
            condition_on_previous_text=wcfg.get("condition_on_previous_text", True),
            beam_size=wcfg.get("beam_size", 5),
            best_of=wcfg.get("best_of", 5),
        )
        return " ".join(seg.text for seg in segments)

    def _generate_soap_note(self, transcript: str, profile_id: str = "KG") -> str:
        prompt = self.build_prompt(transcript, profile_id)
        cfg = self.llm_config
        output = self.llm(
            prompt,
            max_tokens=cfg.get("max_tokens", 1800),
            stop=cfg.get("stop_tokens", ["<|eot_id|>", "<|end_header_id|>", "```"]),
            temperature=cfg.get("temperature", 0.15),
            top_p=cfg.get("top_p", 0.9),
            repeat_penalty=cfg.get("repetition_penalty", 1.1),
        )
        raw = output["choices"][0]["text"]
        return "{" + raw if not raw.strip().startswith("{") else raw

    # ── Post-processing pipeline ───────────────────────────────────────────────

    def recover_hard_metrics(self, transcript: str, soap_dict: dict) -> dict:
        """Safety net: if the therapist SAID it, it MUST appear in O."""
        obj_text = soap_dict.get("O", "")

        schober = re.search(r"Schober.*?(\d+)\s*(?:zu|bis|-)\s*(\d+)", transcript, re.I)
        if schober and "Schober" not in obj_text:
            obj_text += f" | Schober-Zeichen: {schober.group(1)} - {schober.group(2)}"

        vas = re.search(r"VAS\s*(\d+)", transcript, re.I)
        if vas and "VAS" not in soap_dict.get("S", ""):
            soap_dict["S"] = soap_dict.get("S", "") + f" (VAS {vas.group(1)}/10)"

        vas_match = re.search(r"(?:Schmerz|VAS).*?(\d+)\s*(?:von|/)\s*10", transcript, re.I)
        if vas_match and "VAS" not in soap_dict.get("S", ""):
            soap_dict["S"] = f"VAS {vas_match.group(1)}/10. " + soap_dict.get("S", "")

        if "lasegue" in transcript.lower() or "lasek" in transcript.lower():
            if "lasègue" not in obj_text.lower():
                deg = re.search(r"(?:lasegue|lasek).*?(\d+)\s*(?:grad|°)", transcript, re.I)
                deg_val = deg.group(1) if deg else "positiv"
                obj_text += f" | Lasègue-Test: {deg_val}° positiv."

        ashworth = re.search(r"Ashworth.*?(\d+)", transcript, re.I)
        if ashworth and "Ashworth" not in soap_dict.get("O", ""):
            obj_text += f" | Ashworth-Skala: {ashworth.group(1)}"

        tug = re.search(r"Timed Up and Go.*?(\d+)\s*Sekunden", transcript, re.I)
        if tug and "Timed Up and Go" not in soap_dict.get("O", ""):
            obj_text += f" | Timed Up & Go: {tug.group(1)}s"

        rom_match = re.search(r"(?:Abduktion|Rotation).*?(\d+)\s*(?:zu|bis)\s*0\s*(?:zu|bis)\s*(\d+)", transcript, re.I)
        if rom_match and "-" not in soap_dict.get("O", ""):
            obj_text += f" | ROM: {rom_match.group(1)} - 0 - {rom_match.group(2)}"

        for test in ["Jobe", "Hawkins", "Neer"]:
            if test.lower() in transcript.lower() and test not in soap_dict.get("O", ""):
                obj_text += f" | {test}-Test: positiv."

        cm_metrics = re.findall(r"([+-]\d+\s*cm)", transcript, re.I)
        if cm_metrics and "cm" not in soap_dict.get("O", ""):
            obj_text += f" | Umfangsdifferenz: {', '.join(cm_metrics)}"

        stadium = re.search(r"Stadium\s*[1-3]", transcript, re.I)
        if stadium and "Stadium" not in soap_dict.get("O", ""):
            obj_text = f"{stadium.group(0)}, " + obj_text

        soap_dict["O"] = obj_text
        return soap_dict

    def apply_medical_corrections(self, soap_dict: dict) -> dict:
        """Standardize medical terminology and fix Whisper hallucinations in SOAP text."""
        simple_fixes = {
            "Axiola": "Axilla",
            "Sanddemobilisation": "sanfte Mobilisation",
            "chonic": "chronisch",
            "Hinterkopfschmerz": "okzipitaler Kopfschmerz",
            "Kinnretraktion": "Kinn-Retraktion",
            "VKB Plastik": "VKB-Plastik",
            "Lasek": "Lasègue-Test",
            "Tuberculum Mayus": "Tuberculum majus",
            "Gleno-Humeral-Gelenk": "Glenohumeralgelenk",
            "Jobe Test": "Jobe-Test",
            "Hawkins Test": "Hawkins-Test",
            "Supraspinatus-Szene": "Supraspinatussehne",
            "Jove-Test": "Jobe-Test",
            "Mayus": "majus",
            "Bobad": "Bobath",
            "Bobert": "Bobath",
            "P N F": "PNF",
            "Bobart": "Bobath",
            "Mama-Karzinom": "Mamma-Karzinom",
            "Lymphödes": "Lymphödem",
        }
        regex_fixes = {
            r"Laseck|Lasegge|Laseque": "Lasègue-Test",
            r"Schoberzeichen|Schober Zeichen": "Schober-Zeichen",
            r"(\d+)\s*zu\s*(\d+)": r"\1 - \2",
            r"CNMD|CMND|CNMT|C\.N\.M\.D\.": "CMD",
            r"Knieknadi\w*|Knienadig\w*|Knie.?Nadi\w*|Knienachgibigkeit": "Knienachgiebigkeit",
            r"Rotatorenmanschete\b|Rotatoren.?Manschette": "Rotatorenmanschette",
            r"Impingmentsyndrom|Impingement.?Syndrom": "Impingementsyndrom",
            r"Plantarfasciitis|Plantar.?Fasziitis": "Plantarfasziitis",
            r"Karpaltunnel.?Syndrom": "Karpaltunnelsyndrom",
            r"Propriozepzion|Propioception|Propiozeption": "Propriozeption",
            r"Tendinapathie|Tendinopatie": "Tendinopathie",
            r"Patellofemorales?\s?Schmerz.?Syndrom": "Patellofemoralschmerzsyndrom",
        }

        for key in ["S", "O", "A", "P"]:
            text = soap_dict.get(key, "")
            if not text:
                continue
            for wrong, right in simple_fixes.items():
                text = re.compile(re.escape(wrong), re.IGNORECASE).sub(right, text)
            for pattern_str, right in regex_fixes.items():
                text = re.sub(pattern_str, right, text, flags=re.IGNORECASE)
            soap_dict[key] = text

        return soap_dict

    def inject_audit_stamps(self, soap: dict) -> dict:
        if "red flag" not in soap.get("A", "").lower():
            soap["A"] = soap.get("A", "") + " | Red Flags klinisch ausgeschlossen."
        return soap

    def _inject_ly_staging(self, transcript: str, soap_dict: dict) -> dict:
        """For LY domain: infer lymphedema stadium and inject into A-field if missing."""
        t = transcript.lower()
        a_field = soap_dict.get("A", "")

        if re.search(r"stadium\s*[1-3]", a_field, re.I):
            return soap_dict

        swelling = any(w in t for w in [
            "geschwollen", "schwellung", "ödem", "ödematös", "anschwellen",
            "prall", "gespannt", "dick", "aufgetrieben"
        ])
        if not swelling:
            return soap_dict

        is_hard    = any(w in t for w in ["hart", "fibrosiert", "nicht eindrückbar", "derb", "induriert"])
        is_soft    = any(w in t for w in ["weich", "eindrückbar", "delle", "pitting", "morgens besser", "reversibel"])
        is_massive = any(w in t for w in ["elephantiasis", "massiv", "extrem", "riesig"])
        is_postop  = any(w in t for w in ["post-op", "postoperativ", "postop", "op ", "nach der op",
                                           "mastektomie", "axilläre", "sentinel"])

        if is_massive:
            stadium = "Stadium 3 (Elephantiasis)"
        elif is_hard:
            stadium = "Stadium 2 (irreversibel, fibrosiert)"
        elif is_soft:
            stadium = "Stadium 1 (reversibel, pitting)"
        elif is_postop:
            stadium = "Stadium 1–2 (postoperativ, noch zu klassifizieren)"
        else:
            stadium = "Stadium 1–2 (Klassifikation ausstehend — Stemmer-Zeichen prüfen)"

        staging_note = f"Lymphödem {stadium}."
        if "Stadium" not in a_field:
            soap_dict["A"] = f"{staging_note} {a_field}".strip()

        return soap_dict

    # Negation words that mean a condition is ruled OUT — do not migrate these
    _NEGATION_RE = re.compile(
        r'\b(?:kein(?:e[rns]?)?|nicht|ohne|ausgeschlossen|'
        r'kein\s+hinweis|kein\s+verdacht|negativer?\s+befund|'
        r'unauffällig|regelrecht|o\.?b\.?n\.?)\b',
        re.I
    )

    def _migrate_diagnoses_from_s_to_a(self, soap: dict) -> dict:
        """
        Safety net: if the AI placed a diagnosis in S, extract it and prepend to A.
        Skips negated conditions ("kein Bandscheibenvorfall") — those stay in S/A
        as differential exclusions, not as confirmed diagnoses.
        """
        s = soap.get("S", "")
        a = soap.get("A", "")

        sentences = re.split(r'(?<=[.!?])\s+|(?<=\|)\s*', s)

        icd_re = re.compile(r'\b[A-Z]\d{2}\.?\d*\b')
        diag_prefix_re = re.compile(
            r'(?:diagnose|vordiagnose|arztdiagnose)\s*:', re.I
        )
        diag_phrase_re = re.compile(
            r'\b(?:bekannte[rns]?|leidet\s+(?:an|unter)|diagnostizierte[rns]?)\b', re.I
        )
        condition_re = re.compile(
            r'\b(?:gonarthrose|koxarthrose|omarthrose|arthrose|arthritis|'
            r'bandscheibenvorfall|bandscheibenprotrusion|diskushernie|'
            r'spinalkanalstenose|spondylolisthese|spondylose|'
            r'lymphödem|lipödem|'
            r'hemiparese|hemiplegie|ataxie|parkinson|multiple\s+sklerose|'
            r'frozen\s+shoulder|impingementsyndrom|rotatorenmanschetten(?:ruptur|riss)?|'
            r'meniskusriss|meniskusläsion|kreuzbandruptur|vkb[\-\s](?:ruptur|riss|plastik)|'
            r'osteoporose|fibromyalgie|copd)\b',
            re.I
        )

        kept_sentences = []
        extracted = []

        for sent in sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue

            # Skip if the condition is negated within this sentence
            if condition_re.search(sent_stripped) and self._NEGATION_RE.search(sent_stripped):
                kept_sentences.append(sent_stripped)
                continue

            is_diagnostic = (
                icd_re.search(sent_stripped)
                or diag_prefix_re.search(sent_stripped)
                or (diag_phrase_re.search(sent_stripped) and condition_re.search(sent_stripped))
            )

            if is_diagnostic:
                extracted.append(sent_stripped)
            else:
                kept_sentences.append(sent_stripped)

        if not extracted:
            return soap

        soap["S"] = " ".join(kept_sentences).strip() or "n.d."

        for frag in extracted:
            if frag[:30] not in a:
                a = frag + " | " + a if a else frag
        soap["A"] = a.strip(" |").strip()

        return soap

    # Terms that are out-of-scope for each profile domain.
    # If these appear as diagnoses in A (not as exclusions), they are hallucinations.
    _PROFILE_FORBIDDEN_A: dict = {
        "EX_SCHULTER": [
            r"gonarthrose", r"koxarthrose", r"meniskus(?:riss|läsion)",
            r"kreuzband(?:ruptur)?", r"vkb[\-\s](?:ruptur|riss|plastik)",
            r"bandscheibenvorfall", r"bandscheibenprotrusion", r"diskushernie",
            r"spinalkanalstenose", r"spondylolisthese",
            r"schenkelhalsfraktur", r"schenkelhals",
            r"hüftfraktur", r"femurhalsfraktur",
            r"lymphödem", r"lipödem", r"entstauung",
            r"copd", r"asthma", r"atemweg",
        ],
        "EX_KNIE": [
            r"impingementsyndrom", r"omarthrose", r"rotatorenmanschette",
            r"bankart", r"slap.läsion", r"frozen\s+shoulder",
            r"lymphödem", r"lipödem",
            r"copd", r"asthma",
        ],
        "EX_LWS": [
            r"gonarthrose", r"omarthrose", r"impingementsyndrom",
            r"lymphödem", r"copd",
        ],
        "EX_HWS": [
            r"gonarthrose", r"omarthrose", r"impingementsyndrom",
            r"lymphödem", r"copd",
        ],
        "EX_HUFTE": [
            r"impingementsyndrom", r"omarthrose", r"rotatorenmanschette",
            r"lymphödem", r"copd",
        ],
        "EX_FUSS": [
            r"impingementsyndrom", r"omarthrose", r"rotatorenmanschette",
            r"gonarthrose", r"lymphödem", r"copd",
        ],
        "LY": [
            r"gonarthrose", r"omarthrose", r"impingementsyndrom",
            r"bandscheibenvorfall", r"copd",
        ],
        "AT": [
            r"gonarthrose", r"omarthrose", r"impingementsyndrom",
            r"bandscheibenvorfall", r"lymphödem", r"meniskus",
        ],
        "MT": [
            r"lymphödem", r"copd", r"asthma",
        ],
    }

    def _clean_hallucinated_regions(self, soap: dict, icd: str, profile_id: str = "KG") -> dict:
        """
        Remove out-of-scope diagnosis terms from the A field.
        Only removes terms that are NOT negated (negated = already a ruled-out differential).
        """
        forbidden = self._PROFILE_FORBIDDEN_A.get(profile_id, [])
        if not forbidden:
            return soap

        a = soap.get("A", "")
        a_sentences = re.split(r'(?<=[.!?|])\s*', a)
        cleaned = []

        for sent in a_sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue
            is_hallucination = False
            for pattern in forbidden:
                if re.search(pattern, sent_stripped, re.I):
                    # Keep it if it's a negated exclusion ("kein Hinweis auf Gonarthrose")
                    if self._NEGATION_RE.search(sent_stripped):
                        break
                    is_hallucination = True
                    print(f"[SanityCheck] Removed off-profile term '{pattern}' from A: {sent_stripped[:60]}")
                    break
            if not is_hallucination:
                cleaned.append(sent_stripped)

        soap["A"] = " ".join(cleaned).strip()
        return soap

    def _inject_bladder_bowel_into_objective(self, transcript: str, soap: dict) -> dict:
        """
        Cauda-equina safety net for LWS/MT cases.
        If the therapist mentioned bladder/bowel function but the AI buried it
        in S or A instead of O, extract and place it explicitly in O.
        """
        t = transcript.lower()

        # Only relevant for spinal/MT cases
        is_spinal = any(k in t for k in [
            "lws", "lumbal", "l4", "l5", "s1", "ischias",
            "bandscheib", "manuelle therapie", " mt ", "hws",
        ])
        if not is_spinal:
            return soap

        # Already in O — nothing to do
        o = soap.get("O", "")
        if re.search(r'blasen|mastdarm|harninkontinenz|stuhlinkontinenz|miktion', o, re.I):
            return soap

        # Detect the finding from transcript
        negated = bool(re.search(
            r'keine?\s+blasen.?mastdarm|kein(?:e)?\s+(?:harn|stuhl)inkontinenz|'
            r'blasen.?mastdarm(?:funktion|kontrolle|störung)?\s*(?:unauffällig|negativ|o\.?b\.?n\.?|nicht\s+gestört)|'
            r'miktion\s+(?:regelrecht|unauffällig)|'
            r'kontinenz\s+erhalten',
            t, re.I
        ))
        positive = bool(re.search(
            r'blasen.?mastdarm(?:störung|inkontinenz|verlust|ausfall)|'
            r'harninkontinenz|stuhlinkontinenz|'
            r'miktion\s+(?:gestört|eingeschränkt|schmerzhaft)',
            t, re.I
        ))

        if negated:
            finding = "Blasen-/Mastdarmfunktion: unauffällig (Cauda-equina-Screening: negativ)"
        elif positive:
            finding = "Blasen-/Mastdarmfunktion: GESTAERT — sofortige aerztliche Abklaerung erforderlich!"
        else:
            return soap  # not mentioned — don't fabricate

        soap["O"] = (o.rstrip(" |") + " | " + finding).lstrip(" |")
        return soap

    def rom_sanity_check(self, transcript: str, parsed: dict) -> dict:
        obj = parsed["soap"].get("O", "")
        t_nums = set(re.findall(r"\b\d+\b", transcript))
        for l, r in re.findall(r"(\d+)-0-(\d+)", obj):
            if l not in t_nums or r not in t_nums:
                parsed.setdefault("compliance_check", [])
                parsed["compliance_check"].append(f"⚠️ ROM Halluzination? {l}-0-{r}!")
        return parsed

    # ── Billing ────────────────────────────────────────────────────────────────

    def suggest_billing(self, icd10: str, soap: dict, transcript: str):
        codes = self.config.billing_codes
        t_low = transcript.lower()
        plan_text = soap.get("P", "").lower()
        obj_text = soap.get("O", "").lower()
        full_text = f"{obj_text} {plan_text} {t_low}"

        is_neuro = any(k in full_text for k in ["bobath", "pnf", "neuro", "zns", "hemiparese", "ataxie", "spastik", "insult", "schlaganfall"])
        is_lymph = any(k in full_text for k in ["mld", "lymph", "ödem", "kpe", "entstauung", "stemmer"])
        is_ortho_mt = any(k in full_text for k in ["manuelle therapie", " mt ", "traktion", "gleitmobilisation", "manipulation", "mobilisation"])

        res_icd = icd10
        if is_neuro:
            if not icd10.startswith(("G", "I69")):
                res_icd = "I69.3"
        elif is_lymph:
            if not icd10.startswith("I89"):
                res_icd = "I89.0"
        else:
            hip_fracture_ctx = any(k in t_low for k in [
                "schenkelhalsfraktur", "schenkelhals", "hüftfraktur", "femurhalsfraktur",
                "pertrochantär", "pertrochantar", "subtrochantär",
            ])
            hip_ctx = any(k in t_low for k in [
                "hüft", "huefte", "koxarthrose", "hüft-tep", "hüftprothese",
                "femur", "oberschenkelhals",
            ])

            if hip_fracture_ctx:
                is_osteoporotic = any(k in t_low for k in ["osteoporose", "osteoporotisch", "knochendichte"])
                res_icd = "M80.05" if is_osteoporotic else "S72.0"
            elif icd10.startswith("M81") and any(k in t_low for k in ["fraktur", "bruch", "gebrochen"]):
                res_icd = "M80.05" if hip_ctx else "M80.08"
            elif "schulter" in t_low and not icd10.startswith("M75"):
                res_icd = "M75.4"
            elif "knie" in t_low and not icd10.startswith("M17"):
                res_icd = "M17.1"
            elif hip_ctx and not icd10.startswith(("M16", "M80", "S72")):
                res_icd = "M16.1"
            elif any(k in t_low for k in ["hexenschuss", "lumbago", "ischiasschmerz", "lws", "rücken"]):
                res_icd = "M54.5"
                if any(k in t_low for k in ["ausstrahlung", "lasegue", "radikulär", "bein", "wade"]):
                    res_icd = "M51.1"

        if "krankengymnastik" in plan_text or " kg" in plan_text:
            if is_neuro:
                return res_icd, codes.get("KG_ZNS", "20511")
            return res_icd, codes.get("KG", "20501")

        if is_neuro:
            return res_icd, codes.get("KG_ZNS", "20511")
        if is_ortho_mt:
            return res_icd, codes.get("MT", "21201")
        if is_lymph:
            return res_icd, codes.get("MLD", "20201")

        return res_icd, codes.get("KG", "20501")

    # ── Compliance ────────────────────────────────────────────────────────────

    def compliance_check(self, soap: dict, billing_code: str) -> list:
        warns = []
        obj = soap.get("O", "").lower()

        for f in self.audit_rules.get("red_flags", []):
            idx = obj.find(f.lower())
            if idx != -1:
                after = obj[idx: idx + 70]
                if not any(n in after for n in ["negativ", "unauffällig", "keine", "kein", "ausgeschlossen", "normal"]):
                    warns.append(f"🔴 NOTFALL: {f.upper()}!")

        if billing_code in ["21201", "20511"] and len(obj) < 60:
            warns.append(f"📋 DOKU: Befunddichte zu gering für {billing_code}.")

        if "°" in obj and not re.search(r"\d+-\d+-\d+", obj):
            warns.append("⚠️ HINWEIS: Bitte Neutral-Null-Methode nutzen.")

        return warns if warns else ["✅ Dokumentation GKV-konform."]

    # ── JSON parsing ───────────────────────────────────────────────────────────

    def parse_robust_json(self, text: str) -> dict:
        text = text.strip()
        if not text.startswith("{"):
            text = "{" + text
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                clean = re.sub(r",\s*([\]}])", r"\1", match.group())
                data = json.loads(clean)
                s = data.get("soap", {})
                return {
                    "icd10": data.get("icd10", "M99.9"),
                    "soap": {k: s.get(k, "N/A") for k in "SOAP"},
                    "billing_suggestion": data.get("billing_suggestion", "20501"),
                }
        except Exception as e:
            print(f"JSON parse error: {e}")

        return {
            "icd10": "M99.9",
            "soap": {k: "Fehler" for k in "SOAP"},
            "billing_suggestion": "20501",
        }

    # ── Main flow ─────────────────────────────────────────────────────────────

    def run_full_flow(self, audio_path: str, status_callback=None, insurance_type=None):
        from shared.billing_engine import BillingEngine, InsuranceType
        if insurance_type is None:
            insurance_type = InsuranceType.GKV

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if status_callback:
            status_callback("✍️ Transkription...")
        raw_t = self._transcribe(audio_path)
        transcript = self.clean_transcript(raw_t)

        profile_id = self._detect_profile(transcript)
        prof_label = self._PROFILES[profile_id]["label"]
        if status_callback:
            status_callback(f"🧠 KI-Analyse [{prof_label}]...")
        raw_output = self._generate_soap_note(transcript, profile_id)

        if status_callback:
            status_callback("🔍 Validierung...")
        parsed = self.parse_robust_json(raw_output)

        # ICD correction (profile detection + keyword-based upgrade)
        icd, _ = self.suggest_billing(parsed["icd10"], parsed["soap"], transcript)
        parsed["icd10"] = icd

        parsed["soap"] = self.apply_medical_corrections(parsed["soap"])
        parsed["soap"] = self.recover_hard_metrics(transcript, parsed["soap"])
        if profile_id == "LY":
            parsed["soap"] = self._inject_ly_staging(transcript, parsed["soap"])
        parsed["soap"] = self._migrate_diagnoses_from_s_to_a(parsed["soap"])
        parsed["soap"] = self._clean_hallucinated_regions(parsed["soap"], icd, profile_id)
        parsed["soap"] = self._inject_bladder_bowel_into_objective(transcript, parsed["soap"])
        parsed["soap"] = self.inject_audit_stamps(parsed["soap"])
        parsed = self.rom_sanity_check(transcript, parsed)

        # Dual billing engine: GKV deterministic / PKV AI-assisted / BG DGUV
        billing_result = BillingEngine().evaluate(
            icd10=icd,
            soap=parsed["soap"],
            transcript=transcript,
            insurance_type=insurance_type,
            config_rules=self.billing_rules,
            pkv_preise=self.config.pkv_preise,
        )

        return {
            "icd10": icd,
            "soap": parsed["soap"],
            "billing_suggestion": billing_result.position_number,
            "billing_result": billing_result,
            "compliance_check": billing_result.compliance_warnings,
            "transcript": transcript,
            "profile_id": profile_id,
            "profile_label": prof_label,
        }

    def cleanup(self):
        self.llm = None
        self.whisper = None
        gc.collect()
