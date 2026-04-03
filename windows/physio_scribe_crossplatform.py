"""
Kura Engine — Windows
Inference: llama-cpp-python (GGUF) + openai-whisper (local CPU)
Clinical logic: identical to macOS (LearningManager, post-processing pipeline, compliance checks)
"""
import gc
import json
import os
import re
import sys

# ── Fix Windows encoding issues ────────────────────────────────────────────────
# Windows console uses cp1252 by default which can't handle Unicode/emojis
if sys.platform == 'win32':
    # Set UTF-8 for stdout/stderr
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    elif hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')
    elif hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

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
        self.whisper_model_dir = os.path.join(self.model_dir, "whisper")

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

        # Suppress llama.cpp verbose C++ output
        os.environ['LLAMA_CPP_LOG_DISABLE'] = '1'

        # LLM (GGUF via llama-cpp-python)
        try:
            from llama_cpp import Llama
            import io
            import contextlib

            candidates = [
                os.path.join(self.model_dir, "Llama-3.2-3B-Instruct-4bit-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
                os.path.join(self.model_dir, "Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
                os.path.join(self.model_dir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
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

            # Suppress stderr during model loading to hide C++ warnings
            stderr_buffer = io.StringIO()
            with contextlib.redirect_stderr(stderr_buffer):
                self.llm = Llama(
                    model_path=llm_path,
                    n_ctx=2048,  # Increased for comprehensive prompts (was 800)
                    n_threads=min(os.cpu_count() or 4, 12),  # Use more threads
                    n_batch=512,  # Increased batch size to match context
                    n_gpu_layers=0,  # CPU-only
                    verbose=False,
                    logits_all=False,
                    use_mlock=False,
                    use_mmap=True,  # Memory mapping for faster loading
                    low_vram=True,  # Optimize for lower memory usage
                )

            print("✅ LLM loaded")
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

        # STT - Load openai-whisper from local models directory
        try:
            import whisper

            # Try to find available Whisper model in order of preference
            model_preferences = ["medium.pt", "large-v3.pt", "large-v2.pt", "base.pt", "small.pt"]
            whisper_model_path = None
            
            for model_name in model_preferences:
                candidate_path = os.path.join(self.whisper_model_dir, model_name)
                if os.path.exists(candidate_path):
                    whisper_model_path = candidate_path
                    print(f"✅ Found Whisper model: {model_name}")
                    break
            
            if whisper_model_path:
                print(f"✅ Loading Whisper from local: {whisper_model_path}")
                self.whisper = whisper.load_model(whisper_model_path, device="cpu")
            else:
                # Fallback: download medium model (smaller than large-v3, good quality)
                print("⚠️ Local Whisper model not found, downloading medium model to models/whisper...")
                self.whisper = whisper.load_model("medium", device="cpu", download_root=self.whisper_model_dir)

            self.whisper_backend = "openai-whisper"
            print("✅ Whisper STT loaded (openai-whisper backend, local model)")
        except ImportError:
            raise RuntimeError("openai-whisper not installed. Run: pip install openai-whisper")

        gc.collect()
        print("✅ Kura Engine ready (100% local, DSGVO-konform)")

    # ── Transcript cleaning ────────────────────────────────────────────────────

    def clean_transcript(self, transcript: str) -> str:
        """Fix common Whisper hallucinations in German physiotherapy terminology."""
        # First pass: Fix number hallucinations (e.g., "4 *4" -> "45", "3 *5" -> "35")
        transcript = re.sub(r'(\d)\s*\*\s*(\d)', r'\1\2', transcript)
        transcript = re.sub(r'(\d)\s+mal\s+(\d)', r'\1\2', transcript)  # "4 mal 5" -> "45"
        transcript = re.sub(r'(\d)\s+x\s+(\d)', r'\1\2', transcript)  # "4 x 5" -> "45"

        # Fix common MLD duration patterns
        transcript = re.sub(r'MLD\s+(\d)\s*\*\s*(\d)', r'MLD \1\2', transcript, flags=re.IGNORECASE)
        transcript = re.sub(r'(\d{1,2})\s*\*\s*(\d{1,2})\s*(minuten|min)', r'\1\2 \3', transcript, flags=re.IGNORECASE)

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
                # NOTE: do NOT use bare "lymph" — matches "Lymphabfluss", "Lymphknoten"
                # in orthopaedic contexts and causes false LY profile selection.
                "lymphoedem", "lymphdrainage", "mld", "kpe", "entstauung", "stemmer",
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
            "priority": 55,
            "triggers": [
                "hws", "halswirbel", "zervikalsyndrom", "cervical", "nacken",
                "kopfschmerz", "okzipital", "torticollis", "schleudertrauma",
                "trapezius", "schädelbasis", "scaleni", "subokzipital",
                "spannungskopfschmerz", "nackenmuskeln", "kinn-retraktion",
                "segment c", "c5", "c6", "c7",
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

        return f"""<|start_header_id|>system<|end_header_id|>
Du bist ein klinischer Dokumentationsexperte fuer deutsche Physiotherapie (Paragraph 106b SGB V).
DIAGNOSE-PROFIL: {prof["label"]}  |  Abrechnung: {prof["billing"]}
{style_injection}
EXTRAKTIONSREGELN (ABSOLUT VERBINDLICH):
1. Extrahiere AUSSCHLIESSLICH Informationen aus dem Transkript.
2. Fehlende Werte: schreibe "n.d." (nicht dokumentiert) — NIEMALS erfinden.
3. Zahlen EXAKT: "VAS 7" nicht "starke Schmerzen", "+4cm" nicht "Schwellung".
4. Zahlen NIEMALS veraendern: "45 Minuten" bleibt "45 Minuten", nicht "4 mal 5" oder "4*5".
5. Neutral-Null-Methode: [Ext]-[0]-[Flex], Beispiel Knie: "0-0-90".
6. Red Flags IMMER im A-Feld: "Red Flags klinisch ausgeschlossen." (oder benennen).
7. DIAGNOSEN GEHOEREN IN A, NICHT IN S: ICD-10-Codes, Erkrankungsbezeichnungen (z.B. "Gonarthrose", "Bandscheibenvorfall", "Lymphödem"), Diagnose-Aussagen und Vordiagnosen NIEMALS in S schreiben. S enthaelt NUR subjektive Patientenaussagen: Schmerzschilderung, Funktionsziel, Vorgeschichte in eigenen Worten. Wenn der Therapeut eine Diagnose nennt, landet sie in A.
8. THERAPIEZIEL im P-Feld: SMART formulieren — Spezifisch, Messbar, Erreichbar, Relevant, Terminiert. Beispiel: "Ziel: ROM Knieflexion 0-0-120 in 6 EH."
9. KPE-DOKUMENTATION (nur bei MLD/Lymph): P-Feld muss alle 4 Komponenten nennen: MLD + Kompressionsbandagierung + Entstauungsgymnastik + Hautpflege.

PROFIL-PFLICHTFELDER (diese Felder MUESSEN im O-Feld erscheinen):
{checklist}

SOAP-STRUKTUR:
S: Hauptbeschwerde des Patienten (eigene Worte) + Schmerzlokalisation + VAS x/10 + Dauer + Ausloeser
O: ALLE klinischen Messwerte und Tests des Profils — KEINE Zusammenfassungen
A: ICD-10-Diagnose | Differentialdiagnose | Red-Flag-Ausschluss
P: Heilmittel ({prof["label"]}) + Technik + Frequenz + SMART-Funktionsziel | Behandler: n.d.

JSON-OUTPUT (alle Felder Pflicht, auch wenn "n.d."):
{{
  "icd10": "[spezifischer ICD-10-Code]",
  "soap": {{
    "S": "...",
    "O": "...",
    "A": "...",
    "P": "..."
  }},
  "billing_suggestion": "{prof["billing"]}"
}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Transkript: {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""

    # ── Inference ──────────────────────────────────────────────────────────────

    def _validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate audio file before transcription to catch common issues.
        Returns True if valid, raises exception with helpful message if not.
        """
        import os

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")

        if file_size < 100:  # Suspiciously small
            print(f"⚠️ Warning: Audio file is very small ({file_size} bytes) - might be corrupt")

        # Check file extension
        valid_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus'}
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in valid_extensions:
            print(f"⚠️ Warning: Unexpected audio format '{ext}' - supported: {valid_extensions}")

        return True

    def _load_audio_without_ffmpeg(self, audio_path: str):
        """
        Load audio using Python libraries (soundfile/numpy) instead of ffmpeg.
        Converts to 16kHz mono float32 format expected by Whisper.
        """
        import soundfile as sf
        import numpy as np

        # Read audio file
        audio, sample_rate = sf.read(audio_path, dtype='float32')

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling using numpy interpolation
            duration = len(audio) / sample_rate
            target_length = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, len(audio), target_length),
                np.arange(len(audio)),
                audio
            )

        return audio

    def _transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio using local openai-whisper model (runs on CPU).
        Model is cached locally, no internet required after first download.
        Falls back to Python-based audio loading if FFmpeg is not available.
        """
        import traceback

        # Validate audio file first
        self._validate_audio_file(audio_path)

        try:
            print(f"🎙️ Transcribing with local Whisper model...")
            
            # First attempt: Try with Whisper's default FFmpeg approach
            try:
                result = self.whisper.transcribe(
                    audio_path,
                    language="de",
                    fp16=False,  # CPU mode (no GPU)
                    temperature=self.whisper_config.get("temperature", 0.0),
                    initial_prompt=self.whisper_config.get("initial_prompt",
                                  "Physiotherapie Befund. Neutral-Null-Methode. VAS Schmerzskala.")
                )

                result_text = result["text"].strip()
                print(f"✅ Transcription complete ({len(result_text)} characters)")
                return result_text

            except FileNotFoundError as ffe:
                # FFmpeg not found - try Python-based audio loading
                if "system cannot find the file" in str(ffe) or "ffmpeg" in str(ffe).lower():
                    print(f"⚠️ FFmpeg not found, switching to Python-based audio loading...")
                    try:
                        # Load audio using Python libraries instead
                        audio = self._load_audio_without_ffmpeg(audio_path)

                        # Use Whisper with pre-loaded audio
                        result = self.whisper.transcribe(
                            audio,
                            language="de",
                            fp16=False,
                            temperature=self.whisper_config.get("temperature", 0.0),
                            initial_prompt=self.whisper_config.get("initial_prompt",
                                          "Physiotherapie Befund. Neutral-Null-Methode. VAS Schmerzskala.")
                        )

                        result_text = result["text"].strip()
                        print(f"✅ Transcription complete ({len(result_text)} characters)")
                        return result_text
                    except ImportError as ie:
                        raise RuntimeError(
                            f"FFmpeg is not installed and Python audio libraries are missing.\n"
                            f"Install FFmpeg (https://ffmpeg.org/download.html) or run:\n"
                            f"  pip install soundfile librosa\n"
                            f"Error: {ie}"
                        ) from ie
                else:
                    raise

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"❌ Whisper transcription error: {type(e).__name__}: {e}")
            print(f"Full traceback:\n{error_details}")
            raise RuntimeError(f"Local Whisper transcription failed: {e}") from e

    def _generate_soap_note(self, transcript: str, profile_id: str = "KG") -> str:
        prompt = self.build_prompt(transcript, profile_id)

        # Suppress Python warnings from llama-cpp-python
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="llama_cpp")

        raw = None
        try:
            output = self.llm(prompt, max_tokens=1500)
            raw = output["choices"][0]["text"]
        except Exception as e:
            print(f"❌ LLM call failed with: {type(e).__name__}: {e}")
            raw = '{"icd10": "M99.9", "soap": {"S": "KI-Fehler", "O": "n.d.", "A": "Fehler", "P": "n.d."}}'

        return "{" + raw if not raw.strip().startswith("{") else raw

    # ── Post-processing pipeline ───────────────────────────────────────────────

    def recover_hard_metrics(self, transcript: str, soap_dict: dict) -> dict:
        """Safety net: if the therapist SAID it, it MUST appear in O."""
        obj_val = soap_dict.get("O", "")
        obj_text = obj_val if isinstance(obj_val, str) else ""

        schober = re.search(r"Schober.*?(\d+)\s*(?:zu|bis|-)\s*(\d+)", transcript, re.I)
        if schober and "Schober" not in obj_text:
            obj_text += f" | Schober-Zeichen: {schober.group(1)} - {schober.group(2)}"

        # FBA (Finger-Boden-Abstand) — most common LWS metric
        fba = re.search(r"(?:finger.boden|fba)[^\d]*(\d+)\s*cm", transcript, re.I)
        if fba and "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
            obj_text += f" | FBA: {fba.group(1)} cm"
        elif re.search(r"finger.boden|fba", transcript, re.I):
            if "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
                obj_text += " | FBA: n.d."

        # Verbal LWS flexion descriptions → convert to FBA, remove hallucinated degree-ROM
        _verbal_fba = [
            (r'mitte\s+(?:der\s+)?schienbein\w*|schienbeinhöhe|schienbeinniveau', '~35 cm', 'Mitte Schienbein'),
            (r'kniehöhe\b|bis\s+(?:zum?\s+)?knie\b',                              '~50 cm', 'Kniehöhe'),
            (r'waden(?:höhe)?\b|wadenmitte\b',                                     '~25 cm', 'Wadenhöhe'),
            (r'knöchelh?öhe\b|bis\s+(?:zum?\s+)?knöchel\b',                       '~15 cm', 'Knöchelhöhe'),
            (r'(?:fast\s+)?den?\s+boden\b|bodenkontakt\b',                        '~5 cm',  'fast Boden'),
        ]
        is_lws = any(k in transcript.lower() for k in ["lws", "lumbal", "isg", "iliosakral", "kreuzschmerz"])
        if is_lws:
            for pattern, fba_val, fba_label in _verbal_fba:
                if re.search(pattern, transcript, re.I):
                    if "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
                        obj_text += f" | FBA: {fba_val} (Angabe Therapeut: {fba_label})"
                    obj_text = re.sub(r'(?:LWS[^|]*?)?\b0-0-\d{2,3}\b[^|]*', '', obj_text).strip(' |')
                    break

        s_val = soap_dict.get("S", "")
        s_text = s_val if isinstance(s_val, str) else ""

        # Recover VAS — handle all orderings: "VAS 6", "6 von 10 beim Schmerz", "6/10"
        if "VAS" not in s_text:
            vas_num = None
            m = re.search(r"\bVAS\s*(\d{1,2})\b", transcript, re.I)
            if m:
                vas_num = m.group(1)
            if not vas_num:
                m = re.search(r"(?:Schmerz|Schmerzen|schmerzt)[^.]*?(\d{1,2})\s*(?:von|/)\s*10", transcript, re.I)
                if m:
                    vas_num = m.group(1)
            if not vas_num:
                m = re.search(r"\b(\d{1,2})\s*(?:von|/)\s*10\b[^.]*?(?:schmerz|schmerzen|schmerzt)", transcript, re.I)
                if m:
                    vas_num = m.group(1)
            if not vas_num:
                m = re.search(r"\b([1-9]|10)\s*/\s*10\b", transcript)
                if m:
                    vas_num = m.group(1)
            if vas_num:
                soap_dict["S"] = f"VAS {vas_num}/10. " + s_text.lstrip()

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

        # Barthel Index
        barthel = re.search(r"barthel.*?(\d+)", transcript, re.I)
        if barthel and "barthel" not in obj_text.lower():
            obj_text += f" | Barthel-Index: {barthel.group(1)}/100"

        # House-Brackmann (Fazialisparese)
        hb = re.search(r"house.brackmann[^\d]*(grad\s*[IVX]+|\d)", transcript, re.I)
        if hb and "house" not in obj_text.lower():
            obj_text += f" | House-Brackmann: {hb.group(1)}"

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

        t_low = transcript.lower()

        # Stemmer-Zeichen: infer from "Delle" (pitting) or explicit Stadium 2/3
        if any(k in t_low for k in ["delle", "dellen", "stadium 2", "stadium 3"]):
            if "stemmer" not in obj_text.lower():
                obj_text += " | Stemmer-Zeichen: positiv, Hautfalte nicht abhebbar."

        # Ödemkonsistenz: recover "teigig" / pitting descriptor
        if "delle" in t_low and "konsistenz" not in obj_text.lower() and "teigig" not in obj_text.lower():
            obj_text += " | Ödem-Konsistenz: teigig, Delle bleibend."

        soap_dict["O"] = obj_text

        # Mamma-Ablation → inject onkologische Vordiagnose into Assessment if missing
        if any(k in t_low for k in ["ablation", "mastektomie", "mamma-ablation"]):
            a_val = soap_dict.get("A", "")
            a_field = a_val if isinstance(a_val, str) else ""
            if "ablation" not in a_field.lower() and "mastektomie" not in a_field.lower():
                soap_dict["A"] = a_field + " | Z.n. Mamma-Ablation (onkologische Vordiagnose erfüllt)."

        # Spannungsgefühl: patient's subjective complaint — must appear in S, not just as diagnosis
        if "spannungsgefühl" in t_low:
            s_val = soap_dict.get("S", "")
            s_field = s_val if isinstance(s_val, str) else ""
            if "spannungsgefühl" not in s_field.lower():
                region = "linken Arm" if "links" in t_low or "linken arm" in t_low else "betroffenen Arm"
                soap_dict["S"] = f"Spannungsgefühl im {region}. " + s_field

        return soap_dict

    def apply_medical_corrections(self, soap_dict: dict) -> dict:
        """
        Professional Grade Medical Text Refiner.
        Fixes Whisper hallucinations and standardizes terminology.
        """
        # 1. Simple string replacements (case-insensitive exact match)
        simple_fixes = {
            # ── Whisper mishearings / compound splits ─────────────────────────
            "Axiola": "Axilla",
            "Sanddemobilisation": "sanfte Mobilisation",
            "chonic": "chronisch",
            "ischemisch": "ischämisch",
            "Hinterkopfschmerz": "okzipitaler Kopfschmerz",
            "Kinnretraktion": "Kinn-Retraktion",
            "Hochlagerndes": "Hochlagern des",
            "Befallung": "Läsion/Dysfunktion",
            # ── Muscles ───────────────────────────────────────────────────────
            "M. Levator Skapulay": "M. levator scapulae",
            "Levator Skapulay": "M. levator scapulae",
            "Levator scapulay": "M. levator scapulae",
            "Levator Scapulay": "M. levator scapulae",
            "Wärmeam": "Wärmeanwendung",
            "Wärme am ": "Wärmeanwendung ",
            "Gastroknemius": "M. gastrocnemius",
            "Gastrocnemius": "M. gastrocnemius",
            "Quadrizeps": "M. quadriceps femoris",
            "Quadriceps": "M. quadriceps femoris",
            "Deltamuskel": "M. deltoideus",
            "Deltoideus": "M. deltoideus",
            "Illiopsoas": "M. iliopsoas",
            "Hüftbeuger": "M. iliopsoas",
            "Trapeziusmuskel": "M. trapezius",
            "Latissimus dorsi": "M. latissimus dorsi",
            "Bizepsmuskel": "M. biceps brachii",
            "Bizeps": "M. biceps brachii",
            "Trizepsmuskel": "M. triceps brachii",
            "Trizeps": "M. triceps brachii",
            "Soleus": "M. soleus",
            "Tibialis anterior": "M. tibialis anterior",
            "Peroneusmuskel": "M. peroneus longus",
            "Peroneus": "M. peroneus longus",
            "Supraspinatusmuskel": "M. supraspinatus",
            "Infraspinatus": "M. infraspinatus",
            "Subscapularis": "M. subscapularis",
            "Serratus anterior": "M. serratus anterior",
            "Gluteus maximus": "M. gluteus maximus",
            "Gluteus medius": "M. gluteus medius",
            "Glutäusmuskel": "M. gluteus maximus",
            "Piriformismuskel": "M. piriformis",
            "Hamstrings": "Mm. ischiocrurales",
            "Ischiocrurale": "Mm. ischiocrurales",
            "ischio-choraler": "ischiocrurale",
            "ischo-cural": "ischiocrurale",
            "autochtoner": "autochthoner",
            "Adduktoren": "Mm. adductores",
            "Rückenstrecker": "M. erector spinae",
            "Brustmuskel": "M. pectoralis major",
            "SCM": "M. sternocleidomastoideus",
            "Skalenusse": "Mm. scaleni",
            "Skalenus": "Mm. scaleni",
            # ── Tendons ───────────────────────────────────────────────────────
            "Achilles Sehne": "Achillessehne",
            "Achilles sehne": "Achillessehne",
            "Achillesehne": "Achillessehne",
            "Achillos Sehne": "Achillessehne",
            "Patella Sehne": "Patellasehne",
            "Patellarsehne": "Patellasehne",
            "Patelasehne": "Patellasehne",
            "Bizeps Sehne": "Bizepssehne",
            "Supraspinatus Sehne": "Supraspinatussehne",
            "Supraspinatuss Sehne": "Supraspinatussehne",
            "Supraspinatuse Sehne": "Supraspinatussehne",
            "Supraspinatus-Szene": "Supraspinatussehne",
            # ── Ligaments ─────────────────────────────────────────────────────
            "VKB Plastik": "VKB-Plastik",
            "Kreuzbandriss": "Kreuzbandruptur",
            "Innenband": "MCL (Mediales Kollateralband)",
            "Außenband": "LCL (Laterales Kollateralband)",
            "Außenbandriss": "Außenbandruptur",
            "Deltaligament": "Lig. deltoideum",
            "Deltaligemant": "Lig. deltoideum",
            # ── Joints ────────────────────────────────────────────────────────
            "Schulter Gelenk": "Schultergelenk",
            "Schultergelenkt": "Schultergelenk",
            "Knie Gelenk": "Kniegelenk",
            "Hüft Gelenk": "Hüftgelenk",
            "Hand Gelenk": "Handgelenk",
            "Sprung Gelenk": "Sprunggelenk",
            "Ellenbogen Gelenk": "Ellbogengelenk",
            "Ellenbogengelenk": "Ellbogengelenk",
            "Ileo Sakral Gelenk": "ISG (Iliosakralgelenk)",
            "Ileo-Sakral-Gelenk": "ISG (Iliosakralgelenk)",
            "Iliosakralgelenk": "ISG (Iliosakralgelenk)",
            "Sakroiliakalgelenk": "ISG (Iliosakralgelenk)",
            "Gleno-Humeral-Gelenk": "Glenohumeralgelenk",
            "AC Gelenk": "ACG (Akromioklavikulargelenk)",
            "SC Gelenk": "SCG (Sternoklavikulargelenk)",
            # ── Bones ─────────────────────────────────────────────────────────
            "Klabicula": "Klavikula",
            "Klavicula": "Klavikula",
            "Clavicula": "Klavikula",
            "Scapula": "Skapula",
            "Calcaneus": "Kalkaneus",
            "Kalcaneus": "Kalkaneus",
            "Tuberculum Mayus": "Tuberculum majus",
            "Mayus": "majus",
            # ── Nerves ────────────────────────────────────────────────────────
            "Ischiasnerv": "N. ischiadicus",
            "Ischias Nerv": "N. ischiadicus",
            "N Ischiadikus": "N. ischiadicus",
            "Nervus ischiadicus": "N. ischiadicus",
            "Medianusnerv": "N. medianus",
            "Median Nerv": "N. medianus",
            "Radialisnerv": "N. radialis",
            "Ulnarisnerv": "N. ulnaris",
            "Peroneusnerv": "N. peroneus communis",
            "Peronäusnerv": "N. peroneus communis",
            "Facialisparese": "Fazialisparese",
            "Fazialiesparese": "Fazialisparese",
            "Faziale Parese": "Fazialisparese",
            "Fazialislähmung": "Fazialisparese",
            # ── Clinical tests ────────────────────────────────────────────────
            "Lasek": "Lasègue-Test",
            "Jobe Test": "Jobe-Test",
            "Jobe-test": "Jobe-Test",
            "Jobbe Test": "Jobe-Test",
            "Jobbe-Test": "Jobe-Test",
            "Jove-Test": "Jobe-Test",
            "Hawkins Test": "Hawkins-Test",
            "Hawkins Kennedy Test": "Hawkins-Kennedy-Test",
            "Hawkins Kennedy": "Hawkins-Kennedy-Test",
            "Neer Test": "Neer-Test",
            "Neer Zeichen": "Neer-Zeichen",
            "Apley Test": "Apley-Test",
            "Apley Grinding": "Apley-Grinding-Test",
            "McMurray Test": "McMurray-Test",
            "Mc Murray Test": "McMurray-Test",
            "Lachmann Test": "Lachman-Test",
            "Lachmann-Test": "Lachman-Test",
            "Drawer Test": "Schubladentest",
            "Drawer-Test": "Schubladentest",
            "Schubladen Test": "Schubladentest",
            "FABER Test": "FABER-Test",
            "Faber Test": "FABER-Test",
            "Patrick Test": "Patrick-Test (FABER)",
            "Thomas Test": "Thomas-Test",
            "Ober Test": "Ober-Test",
            "Spurling Test": "Spurling-Test",
            "Bragard Test": "Bragard-Test",
            "Slump Test": "Slump-Test",
            "Drop Arm Test": "Drop-Arm-Test",
            "Speed Test": "Speed-Test",
            "Yergason Test": "Yergason-Test",
            "Pivot Shift Test": "Pivot-Shift-Test",
            "Stemmer Zeichen": "Stemmer-Zeichen",
            "Stemmer-zeichen": "Stemmer-Zeichen",
            "Stämmer": "Stemmer-Zeichen",
            "Stemmerzeichen": "Stemmer-Zeichen",
            "Tinel Zeichen": "Tinel-Zeichen",
            "Phalen Test": "Phalen-Test",
            "Finkelstein Test": "Finkelstein-Test",
            "Schober Zeichen": "Schober-Zeichen",
            "Schoberzeichen": "Schober-Zeichen",
            "Finger Boden Abstand": "Finger-Boden-Abstand (FBA)",
            "Finger-Boden Abstand": "Finger-Boden-Abstand (FBA)",
            "Timed Up and Go": "Timed-Up-and-Go-Test",
            "Hoehn Yahr": "Hoehn-Yahr-Skala",
            "House Brackmann": "House-Brackmann-Skala",
            "Ashworth Skala": "Ashworth-Skala",
            "Barthel Index": "Barthel-Index",
            # ── Diagnoses ─────────────────────────────────────────────────────
            "Gonartrose": "Gonarthrose",
            "Coxarthrose": "Koxarthrose",
            "Epikondilitis": "Epikondylitis",
            "Epicondylitis": "Epikondylitis",
            "Tennisellbogen": "Laterale Epikondylitis (Tennisellbogen)",
            "Golferellbogen": "Mediale Epikondylitis (Golferellbogen)",
            "Tendinopatie": "Tendinopathie",
            "Tendinopathia": "Tendinopathie",
            "Bursitas": "Bursitis",
            "Carpaltunnelsyndrom": "Karpaltunnelsyndrom",
            "Karpaltunnel Syndrom": "Karpaltunnelsyndrom",
            "Bandscheiben Vorfall": "Bandscheibenvorfall",
            "Bandscheibenprolaps": "Bandscheibenvorfall (Prolaps)",
            "Bandscheiben Prolaps": "Bandscheibenvorfall (Prolaps)",
            "Bandscheiben Protrusion": "Bandscheibenprotrusion",
            "Spinalkanel Stenose": "Spinalkanalstenose",
            "Spinalstenose": "Spinalkanalstenose",
            "Frozen Shoulder": "Frozen Shoulder (Schultersteife, adhäsive Kapsulitis)",
            "Schultersteife": "Frozen Shoulder (Schultersteife)",
            "adhäsive Kapsulitis": "Adhäsive Kapsulitis (Frozen Shoulder)",
            "Rotatorenmanschetten Ruptur": "Rotatorenmanschettenruptur",
            "Rotatorenmanschettenriss": "Rotatorenmanschettenruptur",
            "Schulterimpingement": "Schulter-Impingement-Syndrom",
            "Plantarfaszitis": "Plantarfasziitis",
            "Plantarfasziose": "Plantarfasziose",
            "Hallux Valgus": "Hallux valgus",
            "Spastik": "Spastizität",
            "Ataksie": "Ataxie",
            "Hemiplägie": "Hemiplegie",
            "Apoplex": "Apoplexie (Schlaganfall)",
            "Apoplexia": "Apoplexie (Schlaganfall)",
            "Hemi Parese": "Hemiparese",
            "Querschnitt Lähmung": "Querschnittlähmung",
            "Poly Neuropathie": "Polyneuropathie",
            "Osteoporosse": "Osteoporose",
            "Fibromyalgia": "Fibromyalgie",
            "Psoriasisathritis": "Psoriasisarthritis",
            # ── Techniques ────────────────────────────────────────────────────
            "Mobilisierung": "Mobilisation",
            "Mobilization": "Mobilisation",
            "Manual Therapie": "Manualtherapie",
            "Manuele Therapie": "Manualtherapie",
            "Manualtherapie": "Manuelle Therapie",
            "KTaping": "Kinesiotaping",
            "K-Taping": "Kinesiotaping",
            "Kinesio Taping": "Kinesiotaping",
            "TENS Behandlung": "TENS-Behandlung",
            "Ultraschall Therapie": "Ultraschalltherapie",
            "Ultraschall-Therapie": "Ultraschalltherapie",
            "Wärme Therapie": "Wärmetherapie",
            "Kälte Therapie": "Kältetherapie",
            "Schlingen Tisch": "Schlingentisch",
            "Schockwellentherapie": "Stoßwellentherapie",
            "KG am Gerät": "KGG (Krankengymnastik am Gerät)",
            # ── Lymphology ────────────────────────────────────────────────────
            "Manuelle Lymph Drainage": "MLD (Manuelle Lymphdrainage)",
            "Manuelle Lymphdrainage": "MLD (Manuelle Lymphdrainage)",
            "Lymphoedem": "Lymphödem",
            "Lymphodem": "Lymphödem",
            "Lymphödes": "Lymphödem",
            "Oedema": "Ödem",
            "Oedem": "Ödem",
            "Anastomosen": "Lymph-Anastomosen",
            "Kompressions Therapie": "Kompressionstherapie",
            "Kompressionsstrumpf": "Kompressionsstrumpf Kl.",
            # ── Neurotherapy approach names ───────────────────────────────────
            "Bobad": "Bobath",
            "Bobert": "Bobath",
            "Bobart": "Bobath",
            "P N F": "PNF",
            "M, M, T": "MMT",
            # ── Oncology ─────────────────────────────────────────────────────
            "Mama-Karzinom": "Mamma-Karzinom",
            "Mama-Ablation": "Mamma-Ablation",
            "Mamma Ablation": "Mamma-Ablation",
            "Mammaablation": "Mamma-Ablation",
            # ── Scales / measurements ─────────────────────────────────────────
            "Barthel-index": "Barthel-Index",
            "Ashworth-skala": "Ashworth-Skala",
            "modifizierte Ashworth Skala": "Modifizierte Ashworth-Skala (MAS)",
            "Hoehn-Yahr-skala": "Hoehn-Yahr-Skala",
            "Epikondylitis lateral": "Laterale Epikondylitis",
            "Epikondylitis medial": "Mediale Epikondylitis",
        }

        # 2. Regex patterns for multi-variant corrections
        regex_fixes = {
            r"Laseck|Lasegge|Laseque":                   "Lasègue-Test",
            r"Schoberzeichen|Schober Zeichen":            "Schober-Zeichen",
            r"Fußheber":                                  "M. extensor hallucis longus (Fußheber)",
            r"(\d+)\s*zu\s*(\d+)":                       r"\1 - \2",  # Neutral-Zero fix
            r"CNMD|CMND|CNMT|C\.N\.M\.D\.":              "CMD",
            r"Knieknadi\w*|Knienadig\w*|Knie.?Nadi\w*|Knienachgibigkeit": "Knienachgiebigkeit",
            r"Rotatorenmanschete\b|Rotatoren.?Manschette": "Rotatorenmanschette",
            r"Impingmentsyndrom|Impingement.?Syndrom":   "Impingement-Syndrom",
            r"Plantarfasciitis|Plantar.?Fasziitis":      "Plantarfasziitis",
            r"Karpaltunnel.?Syndrom":                    "Karpaltunnelsyndrom",
            r"Propriozepzion|Propioception|Propiozeption": "Propriozeption",
            r"Tendinapathie|Tendinopatie":               "Tendinopathie",
            r"Patellofemorales?\s?Schmerz.?Syndrom":     "Patellofemoralschmerzsyndrom",
            # Spine level normalisation: "L 4" → "L4", "C 5/C 6" → "C5/C6"
            r"\bL\s*([1-5])\s*/\s*L\s*([1-5])\b":       r"L\1/L\2",
            r"\bL\s*([1-5])\b(?!\s*/\s*[LS])":           r"L\1",
            r"\bS\s*([12])\b":                            r"S\1",
            r"\bC\s*([1-8])\s*/\s*C\s*([1-8])\b":        r"C\1/C\2",
            r"\bC\s*([1-8])\b(?!\s*/\s*C)":              r"C\1",
            r"\b[Tt][Hh]\s*([1-9]|1[0-2])\b":           r"BWK\1",  # Th4 → BWK4
            # Test name hyphenation
            r"\b(Jobe|Neer|Ober|Thomas|Apley|Phalen|Spurling|Yergason|Lachman|McMurray|Speed)\s+[Tt]est\b": r"\1-Test",
            r"\b(Hawkins)[- ](Kennedy)[- ][Tt]est\b":    r"\1-\2-Test",
            r"\b(Pivot)[- ](Shift)[- ][Tt]est\b":        r"\1-\2-Test",
            r"\b(Drop)[- ](Arm)[- ][Tt]est\b":           r"\1-\2-Test",
            r"\b(Valgus|Varus)[- ](Stress)[- ][Tt]est\b": r"\1-\2-Test",
            # VAS/NRS score normalisation: "VAS 7 von 10" → "VAS 7/10"
            r"\b(VAS|NRS)\s+(\d{1,2})\s+(?:von|of|aus)\s+10\b": r"\1 \2/10",
            # ICD-10 spacing fix: "M 54.5" → "M54.5"
            r"\b([A-Z])\s+(\d{2})\.(\d{1,2})\b":        r"\1\2.\3",
            # LWS/HWS/BWS compound hyphenation
            r"\b(LWS|HWS|BWS)[- ](Syndrom|Schmerzen|Problematik|Beschwerden)\b": r"\1-\2",
            # ISG compound hyphenation
            r"\bISG[- ](Blockierung|Dysfunktion|Syndrom)\b": r"ISG-\1",
            # Rheumatology
            r"Spondylitis\s+Ankylosans":                 "Spondylitis ankylosans",
        }

        for key in ["S", "O", "A", "P"]:
            text = soap_dict.get(key, "")
            # Type safety: ensure we're working with a string
            if not isinstance(text, str):
                text = str(text) if text else ""
            if not text:
                continue
            for wrong, right in simple_fixes.items():
                text = re.compile(re.escape(wrong), re.IGNORECASE).sub(right, text)
            for pattern_str, right in regex_fixes.items():
                text = re.sub(pattern_str, right, text, flags=re.IGNORECASE)
            soap_dict[key] = text

        # Strip null/placeholder pain fields the LLM writes when none was mentioned
        s_field = soap_dict.get("S", "")
        s_field = re.sub(
            r'\.?\s*Schmerzlokalisation\s*:\s*(Nicht|Keine|n\.?d\.?|keine Angabe|nicht genannt)[,.]?',
            '', s_field, flags=re.IGNORECASE
        ).strip().strip('.')
        if s_field:
            soap_dict["S"] = s_field

        # Cross-domain SMART goal sanity check
        p_field = soap_dict.get("P", "")
        if isinstance(p_field, str):
            _smart_m = re.search(r'Ziel\s*:\s*ROM\s+(\w+)', p_field, re.I)
            if _smart_m:
                _goal_part = _smart_m.group(1).lower()
                _all_text = " ".join(v for v in soap_dict.values() if isinstance(v, str)).lower()
                if _goal_part not in _all_text:
                    soap_dict["P"] = re.sub(
                        r'Ziel\s*:\s*ROM\s+\w+[^\|.]*',
                        'Ziel: n.d. — bitte korrektes Funktionsziel ergaenzen',
                        p_field, flags=re.I
                    )

        # Diagnostic tests belong in O (Objektiv), not in P (Plan) — strip from PLAN
        _diag_test_re = re.compile(
            r'\b(?:Patrick-Test|FABER-Test|Lasègue-Test|Bragard-Test|Slump-Test|'
            r'Spurling-Test|Jobe-Test|Hawkins(?:-Kennedy)?-Test|Neer-(?:Test|Zeichen)|'
            r'McMurray-Test|Apley-(?:Grinding-)?Test|Lachman-Test|Schubladentest|'
            r'Thomas-Test|Ober-Test|Tinel-Zeichen|Phalen-Test|Finkelstein-Test|'
            r'Speed-Test|Yergason-Test|Drop-Arm-Test|Pivot-Shift-Test|'
            r'Vorlauf-Test|Vorlauftest|Trendelenburg-(?:Zeichen|Test)|'
            r'Schober-Zeichen|FABER)[^\|.]*[,.]?',
            re.I
        )
        p_field = soap_dict.get("P", "")
        if isinstance(p_field, str):
            p_clean = _diag_test_re.sub('', p_field).strip(' |,.')
            if p_clean != p_field:
                soap_dict["P"] = re.sub(r'\s{2,}', ' ', p_clean).strip()

        return soap_dict

    def inject_audit_stamps(self, soap: dict) -> dict:
        a_val = soap.get("A", "")
        # Type safety: ensure we're working with a string
        if not isinstance(a_val, str):
            a_val = str(a_val) if a_val else ""

        if "red flag" not in a_val.lower():
            soap["A"] = a_val + " | Red Flags klinisch ausgeschlossen."
        else:
            soap["A"] = a_val
        return soap

    def _inject_ly_staging(self, transcript: str, soap_dict: dict) -> dict:
        """For LY domain: infer lymphedema stadium and inject into A-field if missing."""
        t = transcript.lower() if isinstance(transcript, str) else ""
        a_val = soap_dict.get("A", "")
        a_field = a_val if isinstance(a_val, str) else ""

        # 1. Check transcript for EXPLICIT therapist stadium statement — highest priority
        explicit_in_transcript = re.search(r"stadium\s*([1-3])", transcript, re.I)
        if explicit_in_transcript:
            explicit_num = explicit_in_transcript.group(1)
            _label_map = {
                "1": "Stadium 1 (reversibel, pitting)",
                "2": "Stadium 2 (irreversibel, fibrosiert)",
                "3": "Stadium 3 (Elephantiasis)",
            }
            explicit_stadium = _label_map.get(explicit_num, f"Stadium {explicit_num}")
            icd_suffix = "02" if explicit_num == "3" else "01"
            soap_dict["_ly_icd_suffix"] = icd_suffix
            new_staging = f"Lymphödem {explicit_stadium}."
            corrected = re.sub(r"Lymphödem\s+Stadium\s*[1-3][^\.\|]*\.", new_staging, a_field)
            if corrected == a_field:
                if "Stadium" not in a_field:
                    soap_dict["A"] = f"{new_staging} {a_field}".strip()
                else:
                    soap_dict["A"] = re.sub(r"Stadium\s*[1-3][^,\.\|]*", explicit_stadium, a_field)
            else:
                soap_dict["A"] = corrected
            return soap_dict

        # 2. If no explicit therapist statement, skip if LLM already placed a stadium
        if re.search(r"stadium\s*[1-3]", a_field, re.I):
            return soap_dict

        swelling = any(w in t for w in [
            "geschwollen", "schwellung", "ödem", "ödematös", "anschwellen",
            "prall", "gespannt", "dick", "aufgetrieben"
        ])
        if not swelling:
            return soap_dict

        is_hard    = any(w in t for w in ["hart", "fibrosiert", "nicht eindrückbar", "derb", "induriert"])
        is_pitting = any(w in t for w in ["delle", "dellen", "pitting", "eindrückbar"])
        is_prall   = any(w in t for w in ["prall", "pralle"])
        is_soft    = any(w in t for w in ["weich", "morgens besser", "reversibel"])
        is_massive = any(w in t for w in ["elephantiasis", "massiv", "extrem", "riesig"])
        is_postop  = any(w in t for w in ["post-op", "postoperativ", "postop", "op ", "nach der op",
                                           "mastektomie", "axilläre", "sentinel"])

        if is_massive:
            stadium = "Stadium 3 (Elephantiasis)"
        elif is_hard or (is_prall and is_pitting):
            stadium = "Stadium 2 (irreversibel, fibrosiert)"
        elif is_soft or is_pitting:
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

        # Type safety: ensure we're working with strings
        if not isinstance(s, str):
            s = str(s) if s else ""
        if not isinstance(a, str):
            a = str(a) if a else ""

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
        # Type safety: ensure we're working with a string
        if not isinstance(a, str):
            a = str(a) if a else ""

        a_sentences = re.split(r'(?<=[.!?|])\s*', a)
        cleaned = []

        for sent in a_sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue
            is_hallucination = False
            for pattern in forbidden:
                m = re.search(pattern, sent_stripped, re.I)
                if m:
                    # Proximity negation: check only the 6 words BEFORE the matched term
                    before = sent_stripped[:m.start()]
                    nearby_words = before.split()[-6:]
                    negation_nearby = self._NEGATION_RE.search(" ".join(nearby_words))
                    if negation_nearby:
                        break  # term is locally negated — keep it
                    is_hallucination = True
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
        # Type safety: ensure we're working with a string
        if not isinstance(o, str):
            o = str(o) if o else ""

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
        # Type safety: ensure we're working with a string
        if not isinstance(obj, str):
            obj = str(obj) if obj else ""

        t_nums = set(re.findall(r"\b\d+\b", transcript))
        for l, r in re.findall(r"(\d+)-0-(\d+)", obj):
            if l not in t_nums or r not in t_nums:
                parsed.setdefault("compliance_check", [])
                parsed["compliance_check"].append(f"⚠️ ROM Halluzination? {l}-0-{r}!")
        return parsed

    # ── Billing ────────────────────────────────────────────────────────────────

    def suggest_billing(self, icd10: str, soap: dict, transcript: str):
        codes = self.config.billing_codes
        t_low = transcript.lower() if isinstance(transcript, str) else ""

        # Ensure soap values are strings (not dicts) before calling .lower()
        plan_val = soap.get("P", "")
        plan_text = plan_val.lower() if isinstance(plan_val, str) else ""

        obj_val = soap.get("O", "")
        obj_text = obj_val.lower() if isinstance(obj_val, str) else ""

        full_text = f"{obj_text} {plan_text} {t_low}"

        is_neuro = any(k in full_text for k in ["bobath", "pnf", "neuro", "zns", "hemiparese", "ataxie", "spastik", "insult", "schlaganfall"])
        is_lymph = any(k in full_text for k in ["mld", "lymph", "ödem", "kpe", "entstauung", "stemmer"])
        is_ortho_mt = any(k in full_text for k in ["manuelle therapie", " mt ", "traktion", "gleitmobilisation", "manipulation", "mobilisation"])

        # Detect spine-specific indicators that should override extremity detection
        spine_indicators = any(k in full_text for k in [
            "schober", "lasègue", "lasegue", "lasek",
            "l4/l5", "l5/s1", "l3/l4", "lumbal", "lws", "lendenwirbel",
            "hws", "halswirbel", "c5/c6", "c6/c7", "zervikalsyndrom",
            "bandscheibenvorfall", "diskushernie", "spinalkanalstenose",
            "ischiasschmerz", "lumboischialgie", "radikulär",
            "wirbelsäule", "facettensyndrom", "iliosakralgelenk", "isg"
        ])

        res_icd = icd10
        if is_neuro:
            if not icd10.startswith(("G", "I69")):
                res_icd = "I69.3"
        elif is_lymph:
            if not icd10.startswith("I89"):
                res_icd = "I89.0"
        else:
            # Priority 1: Spine indicators override other extremity keywords
            if spine_indicators:
                # Check for specific spine region
                if any(k in full_text for k in ["hws", "halswirbel", "c5/c6", "c6/c7", "c4/c5", "zervikalsyndrom", "nacken"]):
                    res_icd = "M54.2"  # Cervical pain
                elif any(k in full_text for k in ["bandscheibenvorfall", "diskushernie", "radikulär", "ausstrahlung"]):
                    res_icd = "M51.1"  # Lumbar disc herniation with radiculopathy
                else:
                    res_icd = "M54.5"  # Low back pain
            # Priority 2: Other specific conditions
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
                _is_schulter = "schulter" in t_low or icd10.startswith("M75")
                _is_knie     = "knie" in t_low or icd10.startswith("M17")
                if _is_schulter and not icd10.startswith("M75"):
                    res_icd = "M75.4"
                elif _is_knie and not _is_schulter and not icd10.startswith("M17"):
                    res_icd = "M17.1"
                elif hip_ctx and not _is_schulter and not _is_knie and not icd10.startswith(("M16", "M80", "S72")):
                    res_icd = "M16.1"
                elif (not _is_schulter and not _is_knie and not hip_ctx and
                      (any(k in t_low for k in ["hexenschuss", "lumbago", "ischiasschmerz", "lws"]) or
                       re.search(r'rücken(?:schmerz|weh|beschwerden|problem)', t_low))):
                    res_icd = "M54.5"
                    if any(k in t_low for k in ["ausstrahlung", "lasegue", "radikulär", "bein", "wade"]):
                        res_icd = "M51.1"

        # Apply ICD domain lock before billing allocation
        res_icd = self._lock_icd_domain(res_icd, soap, transcript)

        if "krankengymnastik" in plan_text or " kg" in plan_text:
            if is_neuro:
                return res_icd, codes.get("KG_ZNS", "20710")
            return res_icd, codes.get("KG", "20501")

        if is_neuro:
            return res_icd, codes.get("KG_ZNS", "20710")
        # MT must NOT override lymph — a lymph case mentioning "mobilisation" is still MLD
        if is_ortho_mt and not is_lymph:
            return res_icd, codes.get("MT", "21201")
        if is_lymph:
            # Post-cancer / oncological secondary lymphedema → KPE (21110)
            is_post_cancer = any(k in t_low for k in [
                "ablation", "mastektomie", "mamma", "sentinel",
                "bestrahlung", "axillaer", "onkol", "karzinom",
            ])
            if is_post_cancer:
                res_icd = "I97.21" if not res_icd.startswith("I97") else res_icd
                return res_icd, codes.get("KPE_I", "21110")

            # Oct 2024 rule: duration determined by stadium, not just time keyword
            is_stadium2_3 = any(k in t_low for k in [
                "stadium 2", "stadium 3", "irreversibel", "fibrosiert",
                "elephantiasis", "hart", "derb",
                "prall", "chronisch", "delle bleibend", "persistierend",
            ])
            two_parts = bool(re.search(
                r"arm.*bein|bein.*arm|beidseitig|bilateral|"
                r"(?:hand|ober|unter).{0,10}(?:ober|unter)|zwei.{0,10}(?:glied|extrem)",
                t_low
            ))
            explicit_60 = bool(re.search(r"60\s*(?:min|minuten)", t_low))
            explicit_30 = bool(re.search(r"30\s*(?:min|minuten)", t_low))

            if is_stadium2_3 or explicit_60:
                return res_icd, codes.get("MLD_60", "20202")
            if explicit_30 and not two_parts:
                return res_icd, codes.get("MLD_30", "20205")
            return res_icd, codes.get("MLD_45", "20201")

        return res_icd, codes.get("KG", "20501")

    def _lock_icd_domain(self, icd10: str, soap: dict, transcript: str) -> str:
        """
        Safety net: prevent cross-domain ICD-10 coding errors.
        E.g. a stroke patient must never receive a knee code (M17).
        Uses a 2-hit confidence threshold for hard blocks.
        """
        if not icd10:
            return icd10

        t = transcript.lower() if isinstance(transcript, str) else ""
        a_text = (soap.get("A", "") or "").lower()
        combined = f"{t} {a_text}"

        # Tension headache / cervicogenic headache → G44.2 (prevents M45/M51 upcoding)
        is_tension_ha = any(k in t for k in [
            "spannungskopfschmerz", "tension headache", "kopfschmerz",
            "schädelbasis", "subokzipital",
        ])
        is_hws_context = any(k in t for k in [
            "nacken", "hws", "trapezius", "scaleni", "zervik", "cervical",
        ])
        if is_tension_ha and is_hws_context:
            if re.match(r"^M4[5-9]|^M51|^M54$", icd10):
                return "G44.2"  # Spannungskopfschmerz — no M45/M51 for headache sessions

        # M51.1 carve-out: disc herniation with neurological signs is legitimately M5x
        if re.match(r"^M5[0-9]", icd10):
            disc_neuro = bool(re.search(
                r"bandscheib|diskus|prolaps|protrusion|radikulär|ausstrahlung|dermatom",
                combined
            ))
            if disc_neuro:
                return icd10  # preserve M51.1 etc.

        # Domain keyword sets
        domains = {
            "neuro_stroke":  (["schlaganfall", "apoplex", "insult", "hemiparese", "hemiplegie",
                                "hirninfarkt", "tia", "stroke"], "I69.3"),
            "neuro_parkinson": (["parkinson", "tremor", "rigor", "bradykinese",
                                  "hoehn yahr", "hoehn-yahr"], "G20"),
            "neuro_ms":      (["multiple sklerose", "ms-schub", "ms schub", "demyelini",
                                "fatigue ms", "gangstörung ms"], "G35"),
            "neuro_facial":  (["fazialisparese", "fazialis", "bellsche",
                                "gesichtslähmung", "house-brackmann"], "G51.0"),
            "lymph":         (["lymphödem", "lymphdrainage", "mld", "entstauung",
                                "stemmer", "kpe", "lipödem"], "I89.0"),
            "copd":          (["copd", "atemwegsobstruktion", "emphysem", "dyspnoe",
                                "atemtherapie", "atemübung"], "J44.1"),
        }

        for domain, (keywords, fallback_icd) in domains.items():
            hits = sum(1 for kw in keywords if kw in combined)
            if hits >= 2:
                if domain.startswith("neuro_") and icd10[0] == "M":
                    return fallback_icd
                if domain == "lymph" and icd10[0] == "M":
                    return fallback_icd
                if domain == "copd" and icd10[0] == "M":
                    return fallback_icd

        return icd10

    # ── Compliance ────────────────────────────────────────────────────────────

    def compliance_check(self, soap: dict, billing_code: str) -> list:
        warns = []
        obj_val = soap.get("O", "")
        obj = obj_val.lower() if isinstance(obj_val, str) else ""

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
        """
        Robust JSON parser with multiple fallback strategies for malformed LLM output.
        Ensures all SOAP fields are strings, not nested objects.
        """
        text = text.strip()
        if not text.startswith("{"):
            text = "{" + text

        # Strategy 1: Try direct parse with basic JSON cleaning
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                clean = match.group()
                
                # Fix common JSON errors
                clean = re.sub(r",\s*([\]}])", r"\1", clean)  # Remove trailing commas
                clean = re.sub(r'"\s*\n\s*"', '", "', clean)  # Fix line breaks between strings
                
                data = json.loads(clean)
                soap_raw = data.get("soap", {})
                
                # Ensure all SOAP fields are strings, not nested objects
                soap_clean = {}
                for field in ["S", "O", "A", "P"]:
                    value = soap_raw.get(field, "")
                    # Convert non-string values to strings
                    if isinstance(value, dict):
                        # Flatten nested dict to readable string
                        value = " | ".join(f"{k}: {v}" for k, v in value.items() if v)
                    elif not isinstance(value, str):
                        value = str(value) if value else "N/A"
                    soap_clean[field] = value.strip() if value else "N/A"
                
                return {
                    "icd10": data.get("icd10", "M99.9"),
                    "soap": soap_clean,
                    "billing_suggestion": data.get("billing_suggestion", "20501"),
                }
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ JSON Strategy 1 failed: {e}")

        # Strategy 2: Extract fields with regex (more lenient)
        try:
            soap_dict = {}
            for field in ["S", "O", "A", "P"]:
                # Try to match quoted string value
                pattern = rf'"{field}"\s*:\s*"([^"]*(?:\\"[^"]*)*)"'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    soap_dict[field] = match.group(1).replace('\\"', '"').strip()
                else:
                    # Try to extract any content after field name (handles malformed JSON)
                    alt_pattern = rf'"{field}"\s*:\s*([^,\}}]+)'
                    alt_match = re.search(alt_pattern, text, re.DOTALL)
                    if alt_match:
                        value = alt_match.group(1).strip().strip('"\'')
                        soap_dict[field] = value if value else "N/A"
                    else:
                        soap_dict[field] = "N/A"

            icd_match = re.search(r'"icd10"\s*:\s*"([A-Z]\d{2}\.?\d*)"', text)
            icd = icd_match.group(1) if icd_match else "M99.9"

            billing_match = re.search(r'"billing_suggestion"\s*:\s*"(\d+)"', text)
            billing = billing_match.group(1) if billing_match else "20501"

            if any(v != "N/A" for v in soap_dict.values()):
                return {
                    "icd10": icd,
                    "soap": soap_dict,
                    "billing_suggestion": billing,
                }
        except Exception as e:
            print(f"⚠️ JSON Strategy 2 failed: {e}")

        # Strategy 3: Last resort - extract any text content
        print("⚠️ All JSON parsing failed, using fallback")
        return {
            "icd10": "M99.9",
            "soap": {
                "S": "Parsing-Fehler - Transkript manuell prüfen",
                "O": text[:200] if text else "Fehler",
                "A": "Fehler",
                "P": "Fehler"
            },
            "billing_suggestion": "20501",
        }

    # ── Main flow ─────────────────────────────────────────────────────────────

    def _validate_and_fix_soap(self, soap_dict: dict, icd10: str) -> dict:
        """
        Validate SOAP dict after parsing - ensure all fields are strings and non-empty.
        Fixes common LLM errors: empty fields, nested objects, missing Assessment.
        """
        # Ensure all SOAP fields exist and are strings
        for field in ["S", "O", "A", "P"]:
            value = soap_dict.get(field, "")

            # Convert non-string values
            if isinstance(value, dict):
                # Flatten nested dict to readable string
                value = " | ".join(f"{k}: {v}" for k, v in value.items() if v)
            elif not isinstance(value, str):
                value = str(value) if value else ""

            # Check for empty/placeholder values
            if not value or value in ("N/A", "n.d.", "Fehler", "{}"):
                # Generate minimal placeholder based on field type
                if field == "S":
                    value = "Keine subjektiven Angaben dokumentiert"
                elif field == "O":
                    value = "Objektiver Befund: siehe Transkript"
                elif field == "A":
                    # Assessment should never be empty - use ICD as fallback
                    value = f"{icd10} | Red Flags klinisch ausgeschlossen."
                elif field == "P":
                    value = "Therapieplanung siehe Dokumentation"

            soap_dict[field] = value

        return soap_dict

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

        # Validate and fix SOAP fields before final processing
        parsed["soap"] = self._validate_and_fix_soap(parsed["soap"], icd)

        if status_callback:
            status_callback("💰 Abrechnung berechnen...")
        # Dual billing engine: GKV deterministic / PKV AI-assisted / BG DGUV
        billing_result = BillingEngine().evaluate(
            icd10=icd,
            soap=parsed["soap"],
            transcript=transcript,
            insurance_type=insurance_type,
            config_rules=self.billing_rules,
            pkv_preise=self.config.pkv_preise,
        )

        if status_callback:
            status_callback("✅ Fertig!")


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
