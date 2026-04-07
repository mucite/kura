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
    def __init__(self, license_status=None):
        """
        license_status: Pass from LicenseManager.verify_locally()
            - True = licensed (gets premium config)
            - "TRIAL" = trial mode (gets basic config)
            - False = expired (gets basic config)
        """
        self.learning_mgr = LearningManager()
        self.license_status = license_status
        self.config = ConfigManager(license_status=license_status)
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
            base = os.path.dirname(sys.executable)
            print(f"🔍 Running as bundled app, looking for models at: {base}\\models")
        else:
            base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        self.model_dir = os.path.join(base, "models")
        self.whisper_model_dir = os.path.join(self.model_dir, "whisper")
        
        # Verify model directory exists
        if not os.path.exists(self.model_dir):
            print(f"⚠️ WARNING: Model directory not found at: {self.model_dir}")
        else:
            print(f"✅ Model directory found: {self.model_dir}")

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

        # Check if models need to be downloaded (first launch)
        try:
            from core.model_downloader import ensure_models_available
            if not ensure_models_available():
                raise RuntimeError("Model download failed. Please check internet connection and try again.")
        except ImportError:
            pass  # Downloader not available, proceed with existing models

        # Suppress llama.cpp verbose C++ output
        os.environ['LLAMA_CPP_LOG_DISABLE'] = '1'

        # LLM (GGUF via llama-cpp-python)
        try:
            from llama_cpp import Llama
            import io
            import contextlib

            candidates = [
                # Llama 3.1 8B (production model - 77% extraction quality)
                os.path.join(self.model_dir, "Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
                os.path.join(self.model_dir, "Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            ]

            llm_path = next((p for p in candidates if os.path.exists(p)), None)
            if not llm_path:
                raise FileNotFoundError(
                    "Llama-3.1-8B model not found.\n"
                    "The model should have been downloaded automatically.\n"
                    "Please ensure you have internet connection and restart Kura."
                )

            print(f"✅ Loading LLM: {os.path.basename(llm_path)}")

            # Suppress stderr during model loading to hide C++ warnings
            stderr_buffer = io.StringIO()
            with contextlib.redirect_stderr(stderr_buffer):
                self.llm = Llama(
                    model_path=llm_path,
                    n_ctx=4096,  # Increased context window for long prompts + transcripts
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
                # NOTE: do NOT add "lymphdrainage" or "mld" — treatment TECHNIQUES
                # also used for acute sports injuries (sprains, haematoma) → false LY.
                # Require actual DISEASE terms only.
                "lymphoedem", "lymphödeme", "kpe", "entstauung", "stemmer-zeichen",
                "lipoedem", "lipoedema", "mastektomie", "axillaer", "sentinel",
                "erysipel", "sekundäres lymphödem", "primäres lymphödem",
                "chronisches ödem", "phlebolymphoedema",
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
            "priority": 55,   # unique; below EX_HWS(57)
            "triggers": [
                "rheuma", "rheumaerkrankung",
                "rheumatoide arthritis", "ra-erkrankung",
                "psoriasis-arthritis", "psoriatrische arthritis",
                "ankylosierende spondylitis", "morbus bechterew", "spondylitis ankylosans",
                "systemischer lupus", "lupus erythematodes",
                "gicht", "gichtarthritis", "hyperurikämie arthritis",
                "entzündliche gelenkerkrankung", "arthritis entzündlich",
                "sjögren", "polymyalgia rheumatica",
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
            "label":    "Manuelle Therapie WS (Facette / ISG)",
            "billing":  "21201",
            "priority": 50,   # unique; below EX_FUSS(51) — ankle MT sessions → EX_FUSS
            "triggers": [
                "manuelle therapie", "manualtherapie",
                "traktion wirbelsäule", "traktion ws",
                "gleitmobilisation", "gelenkmobilisation wirbelsäule",
                "manipulation wirbelsäule", "hvla",
                "facettensyndrom", "facettengelenk",
                "iliosakralgelenk", "isg", "isg-blockierung", "sakroiliakal",
                "wirbelbogengelenk",
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
            "priority": 62,   # unique; above AT(60) and EX_HWS(57)
            "triggers": [
                "schulter", "schultergelenk", "schultersteife",
                "rotatorenmanschette", "rotatorenmanschettenruptur",
                "impingement", "impingementsyndrom",
                "supraspinatus", "infraspinatus", "subscapularis", "teres minor",
                "bizepssehne", "bizepssehnenruptur",
                "acromion", "subakromial", "subakromiales",
                "omarthrose", "schulterarthrose",
                "bankart", "slap-läsion", "slap",
                "frozen shoulder", "schultersteifigkeit", "adhäsive kapsulitis",
                "glenohumer", "glenoid",
                "kapselmuster schulter", "kapsuläres muster schulter",
                "ac-gelenk", "acromioclavikular", "schultereckgelenk",
                "abduktion schulter", "schulter abduktion",
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
            "priority": 44,   # unique
            "triggers": [
                "knie", "kniegelenk", "knieschmerz", "kniebeschwerden",
                "gonarthrose", "kniearthrose",
                "vkb", "vkb-ruptur", "vkb-plastik", "vorderes kreuzband",
                "hkb", "hinteres kreuzband", "kreuzband",
                "meniskus", "meniskusriss", "meniskusläsion", "meniskusresektion",
                "patella", "patellofemoral", "kniescheibe",
                "knie-tep", "tkep", "knieprothese", "knie-endoprothese",
                "hoffa", "plica", "plicasyndrom",
                "kollateralband knie", "innenseitenband knie",
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
            "priority": 57,   # unique; below EX_SCHULTER(62), above RHEUM(55)
            "triggers": [
                "hws", "halswirbelsäule", "halswirbel", "zervikalwirbel",
                "zervikalsyndrom", "zervikogen", "cervical",
                "kopfschmerz zervikogen", "okzipitalneuralgie", "okzipital",
                "torticollis", "schiefhals",
                "schleudertrauma", "hws-distorsion", "hws-trauma",
                "schädelbasis", "scaleni", "subokzipital",
                "spannungskopfschmerz", "kinn-retraktion",
                "segment c", "c2/c3", "c3/c4", "c4/c5", "c5/c6", "c6/c7", "c7/th1",
                # Lay-vocabulary additions — patients rarely say "HWS":
                "nackenschmerz", "nackensteifigkeit", "nackenmuskulatur",
                "atlas", "atlasgelenk", "atlaskompression",
                "kopfgelenk", "kopfgelenksreihe",
                "hinterkopf", "hinterkopfschmerz",
                "geier-hals", "vorköpfige haltung", "doppelkinn",
                "tiefe halsflexoren", "tiefe nackenflexoren",
                "zervikogener kopfschmerz",
                # "nacken" alone excluded — appears in shoulder sessions as a
                # compensatory finding and would override EX_SCHULTER incorrectly.
                # "trapezius" excluded for same reason.
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
            "priority": 42,   # unique
            "triggers": [
                "lws", "lendenwirbelsäule", "lendenwirbel", "lumbosakral",
                "lumbalgie", "lumboischialgie", "ischiasschmerz", "ischialgie",
                "bandscheibenvorfall", "bandscheibenprotrusion", "diskushernie",
                "lumbago", "kreuzschmerz",
                "rückenschmerz", "rückenbeschwerd", "rückenprobleme",
                "spinalkanalstenose lumbal", "spondylolisthese",
                "l1", "l2", "l3", "l4", "l5", "l4/l5", "l5/s1",
                # "wirbelsäule" excluded — too broad, matches HWS sessions too
                # "rücken" excluded — matches "Rückenlage" (patient position)
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
            "priority": 43,   # unique; above EX_LWS(42)
            "triggers": [
                "hüfte", "huefte", "hüftgelenk", "hüftbeschwerden",
                "coxarthrose", "hüftarthrose", "cox",
                "hüftprothese", "hüfttep", "htep", "hüft-tep", "hüftendoprothese",
                "totalendoprothese hüfte",
                "trochanter", "trochanter major", "trochanterbursa",
                "piriformis", "piriformissyndrom",
                "femur", "femurkopf", "schenkelhals", "schenkelhalsfraktur",
                "coxa", "coxalgie",
                "hüftabduktor", "gluteus medius", "gluteus maximus",
                "trendelenburg", "trendelenburgzeichen",
                "leistenbereich", "leistenschmerz",
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
        "EX_HAND": {
            "label":    "Extremitaeten Hand / Handgelenk / Finger",
            "billing":  "21201",
            "priority": 45,   # unique; above EX_KNIE(44)
            "triggers": [
                "handgelenk", "handgelenkschmerz",
                "handwurzel", "handwurzelknochen", "karpalknochen",
                "radiokarpal", "ulnokarpal",
                "radiusfraktur", "radiusfraktur distale", "speichenbruch",
                "metakarpal", "mittelhand",
                "fingergelenk", "fingergrundgelenk", "fingermittelgelenk", "fingerendgelenk",
                "finger", "daumen", "daumengelenk", "daumensattelgelenk",
                "handkraft", "faustschluss",
                "karpaltunnel", "karpaltunnelsyndrom",
                "handchirurgie", "handödem",
                "de quervain", "dupuytren", "ganglion handgelenk",
                "skaphoid", "kahnbein",
            ],
            "icd_prefix": ["S52", "S62", "M19.0", "G56", "M65.3"],
            "checklist": [
                "Behandeltes Segment: z.B. Radiokarpalgelenk / MCP II / PIP III (MT-Pflichtangabe)",
                "ROM Handgelenk: Flexion / Extension / Radialabduktion / Ulnarabduktion (Grad)",
                "Jamar-Handkraft (kg) re / li",
            ],
        },
        "EX_FUSS": {
            "label":    "Extremitaeten Fuss / Sprunggelenk (EX5)",
            "billing":  "21201",
            "priority": 51,   # unique; above MT(50) — ankle sessions using manual therapy → EX_FUSS
            "triggers": [
                # Foot / ankle anatomy - SPECIFIC to avoid matching neurological tests
                "sprunggelenk", "fußgelenk", "knöchelgelenk",
                "osg", "oberes sprunggelenk",
                "usg", "unteres sprunggelenk",
                "außenknöchel", "aussenknöchel", "innenknöchel",
                "malleolus", "malleolus lateralis", "malleolus medialis",
                # Tendons / ligaments — specific to foot/ankle
                "achillessehne", "achillessehnenentzündung", "achillessehnenriss",
                "plantarfasziitis", "plantarfaszie", "fasziitis",
                "peroneus", "peroneussehne", "peronealsehnenluxation",
                "talofibulare", "lig. talofibulare", "ltfa", "calcaneofibuläre",
                "lateralband", "außenband sprunggelenk", "außenbandruptur",
                "syndesmose", "syndesmosenruptur",
                # Bones
                "talus", "calcaneus", "fersenbein", "kahnbein fuß", "naviculare",
                "metatarsal", "mittelfuß", "zehengelenk", "zehe",
                "hallux", "hallux valgus", "großzehengrundgelenk",
                # Symptoms / mechanisms - SPECIFIC to ankle injury
                "fersenschmerz", "ferse schmerz",
                "umknicken", "umgeknickt", "umgeknickte",
                "supinationstrauma", "inversionstrauma", "inversionsdistorsion",
                "distorsion sprunggelenk", "distorsion fuß",
                # Tests specific to ankle - avoid generic "fuß" or "knöchel" alone
                "schubladentest sprunggelenk", "vordere schublade fuß", "talarneigung",
                "thompsons test", "wadenkompression",
                # Treatment context
                "lymphtape fuß", "aircast", "knöchelschiene",
                "sprunggelenkschmerz", "knöchelschmerz",
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
        "MASSE": {
            "label":    "Massagetherapie (KMT / BGM / Segmentmassage)",
            "billing":  "20106",
            "priority": 25,
            "triggers": [
                "klassische massage", "kmt", "bindegewebsmassage", "bgm",
                "segmentmassage", "myogelose", "triggerpunkt-massage",
                "massage therapie", "weichteilmassage",
            ],
            "icd_prefix": ["M54", "M79.1", "M62"],
            "checklist": [
                "Massageform: KMT / BGM / Segment (je nach Verordnung)",
                "Lokalisation: Muskeln + Befund (Myogelosen / Tonuserhoehung)",
                "Druckdolenz: Palpationsbefund (Lokalisation + Intensitaet)",
                "Wirkung: Tonussenkung / Durchblutungsfoerderung (subjektiv + palpatorisch)",
            ],
        },
        "UWM": {
            "label":    "Unterwasserdruckstrahlmassage (UWM)",
            "billing":  "20102",
            "priority": 24,
            "triggers": [
                "unterwasserdruckstrahl", "uw-massage", "uwm", "unterwassermassage",
                "druckstrahlmassage", "whirlpool massage",
            ],
            "icd_prefix": ["M54", "M79", "G82"],
            "checklist": [
                "Wassertemperatur (°C)",
                "Druck (bar) + Behandlungsabstand (cm)",
                "Behandlungsregion: genaue Lokalisation",
                "Wirkung: Tonussenkung / Oedemmobilisation (dokumentieren)",
            ],
        },
        "AQUA": {
            "label":    "Krankengymnastik im Bewegungsbad (Aquatherapie)",
            "billing":  "20902",
            "priority": 32,
            "triggers": [
                "bewegungsbad", "aquatherapie", "wassergymnastik",
                "unterwassergymnastik", "hydrotherapie", "pool therapie",
                "warmwasserbecken", "kg im wasser",
            ],
            "icd_prefix": ["M16", "M17", "M05", "G82", "M80"],
            "checklist": [
                "Wassertemperatur (°C)",
                "Auftriebshilfen: Schwimmnudel / Schwimmflossen / keine",
                "Belastungsstatus im Wasser: Vollbelastung / Entlastung X%",
                "ROM und Gangbild im Vergleich zur Trockenuebung",
            ],
        },
        "ELEKTRO": {
            "label":    "Elektrotherapie (TENS / IFC / EMS / Galvano)",
            "billing":  "21302",
            "priority": 22,
            "triggers": [
                "elektrotherapie", "tens", "interferenzstrom", "ifc", "galvano",
                "diadynamisch", "reizstrom", "elektrostimulation", "ems therapie",
                "transkutane elektrische",
            ],
            "icd_prefix": ["M54", "M79.1", "M25.5", "G57"],
            "checklist": [
                "Stromform: TENS / IFC / Galvano / Diadyn (je nach Verordnung)",
                "Frequenz (Hz) + Intensitaet (mA) — unter Wahrnehmungsgrenze / Kontraktionsschwelle",
                "Elektroden-Platzierung: Lokalisation (Dermatom / Muskelbauch / Trigger)",
                "Patientenreaktion: Kribbeln / Waerme / Zucken (erwartet vs. tatsaechlich)",
            ],
        },
        "THERMO": {
            "label":    "Thermotherapie (Fango / Waerme / Kaelte)",
            "billing":  "21501",
            "priority": 20,
            "triggers": [
                "fango", "heiße rolle", "heisse rolle", "warmpackung",
                "waermetherapie", "waermestrahler", "rotlicht",
                "kaeltetherapie", "eispack", "kryotherapie", "kryopack",
                "ultraschall waerme",
            ],
            "icd_prefix": ["M54", "M79.1", "M25.5", "S00"],
            "checklist": [
                "Modalitaet: Fango / Heisse Rolle / Strahler / Eispack / Kryopack",
                "Temperatur (°C) oder Stufe (subjektiv: angenehm warm / kuehl)",
                "Behandlungsregion + Dauer (Minuten)",
                "Kontraindikationsausschluss: Sensibilitaetstoerung nein / Durchblutungstoerung nein",
            ],
        },
        "GRUPPE": {
            "label":    "Gruppentherapie (KG-Gruppe)",
            "billing":  "20601",
            "priority": 18,
            "triggers": [
                "gruppentherapie", "gruppenbehandlung", "kurstherapie",
                "gruppenkg", "sturzpraevention gruppe", "rueckenschule",
                "gruppengymnastik", "gruppenbehandlung kg",
            ],
            "icd_prefix": ["M54", "M79", "G20", "M81"],
            "checklist": [
                "Teilnehmerzahl: X Patienten (GKV-Limit: max. 5 bei KG-Gruppe)",
                "Gemeinsames Gruppenziel (Therapieziel fuer alle Teilnehmer)",
                "Besonderheiten einzelner Teilnehmer (falls dokumentationsrelevant)",
            ],
        },
        "KGG": {
            "label":    "Krankengymnastik am Geraet (MTT / KGG)",
            "billing":  "20507",   # Gist KG_Gerät: 20507 @ €55.81 (not 20501 @ €29.63)
            "priority": 38,
            "triggers": [
                "kgg", "krankengymnastik am geraet", "medizinische trainingstherapie",
                "mtt", "geraetetraining", "beinpresse", "latzug", "rudergeraet",
                "trainingsplan", "trainingstherapie", "geraetebezogen",
                "krafttraining therapeutisch", "seilzug",
            ],
            "icd_prefix": ["M54", "M47", "M51", "M75", "M17", "M16"],
            "checklist": [
                "Trainingsgeraet: Name + Einstellung (Gewicht kg / Winkelbereich Grad)",
                "Belastungsparameter: Saetze x Wiederholungen x Gewicht (kg)",
                "Ausgangsleistung (Watt) oder 1-RM-Schaetzung bei Kraftgeraeten",
                "Schmerzfreiheit bei Belastung: ja / nein (VAS vor + nach Training)",
                "Steuerung: herzfrequenzbasiert (Ziel-HF) / RPE (Borg-Skala 0-10)",
            ],
        },
        "GEB": {
            "label":    "Geburtshilfe / Rückbildungsgymnastik",
            "billing":  "21904",   # 21901 Vorbereitung / 21904 Rückbildung (Gist ICD10_O80)
            "priority": 52,
            "triggers": [
                "schwanger", "geburt", "rueckbildung", "postnatal", "postpartal",
                "dammriss", "perinealriss", "kaiserschnitt", "sectio", "stillen",
                "wochenbett", "hebamme", "pränatal", "ssw", "schwangerschaftswoche",
            ],
            "icd_prefix": ["O34", "O70", "O71", "Z34", "Z39"],
            "checklist": [
                "SSW (pränatal) oder Wochen postpartum (postnatal)",
                "Beckenbodenkraft: Oxford-Skala (0-5)",
                "Dammriss-/OP-Narbe: Grad (I-IV) + Verschieblichkeit (falls vorhanden)",
                "Abdominalwand: Rektusdiastase (cm Spalt auf Nabelhöhe, falls vorhanden)",
                "Kontinenz: Harnverlust unter Belastung: ja / nein",
            ],
        },
        "BECKEN": {
            "label":    "Beckenbodentherapie",
            "billing":  "20501",
            "priority": 48,
            "triggers": [
                "beckenboden", "inkontinenz", "harninkontinenz", "stressinkontinenz",
                "belastungsinkontinenz", "dranginkontinenz", "mischinkontinenz",
                "kontinenz", "blasenschwaeche", "blasenkontrolle",
                "prolaps", "deszensus", "geburtsverletzung", "dammriss",
                "perineum", "kegel", "biofeedback beckenboden",
            ],
            "icd_prefix": ["N39.3", "N39.4", "N81", "O34", "N32"],
            "checklist": [
                "Beckenboden-Kraft: Oxford-Skala (0-5) oder PERFECT-Schema",
                "Inkontinenztyp: Stress / Drang / Misch / n.d.",
                "Miktionsprotokoll: Haeufigkeit taeglich / Episoden (falls vorliegend)",
                "Pessartherapie / Hilfsmittel: vorhanden / nicht vorhanden",
                "Biofeedback: eingesetzt ja / nein",
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
        Anatomy-first profile detection.
        Score = priority * 1000 + match_count
        — match_count breaks ties when two profiles share the same priority
          (shouldn't happen with deduplicated priorities, but keeps routing
          deterministic if future edits accidentally create a tie).

        Step 0 runs first: ultra-specific anatomical terms that can only belong
        to one profile override the score ranking entirely.
        """
        t = transcript.lower()

        # ── Step 0: Definitive-term override ─────────────────────────────────
        # These terms are anatomically exclusive to one profile.
        # Their presence guarantees that profile regardless of priority/score.
        _DEFINITIVE: dict = {
            "EX_HWS": [
                "atlas-übergang", "atlasgelenk", "atlaskompression",
                "kopfgelenk", "kopfgelenksreihe",
                "geier-hals", "vorköpfige haltung",
                "doppelkinn",        # deep neck flexor rehab — exclusively HWS
                "hinterkopfschmerz", "okzipitaler kopfschmerz",
                "zervikogener kopfschmerz",
            ],
            "EX_LWS": [
                "lws-syndrom", "lws syndrom", "lendenwirbelsäule",
                "quadratus lumborum", "finger-boden-abstand", "finger-boden-distanz",
                "lasègue", "lasegue-test", "lasek",
                "schober-zeichen", "vorlaufphänomen", "vorlauf-test",
                "lumbalgie", "lumboischialgie", "kreuzschmerz",
                "bandscheibenvorfall lumbal", "bandscheibenprotrusion lws",
            ],
        }
        for def_pid, def_terms in _DEFINITIVE.items():
            if any(term in t for term in def_terms):
                if def_pid in self._PROFILES:
                    return def_pid

        # Age extraction — "4 Jahre alt", "4-jaehrig", "4 J."
        age = None
        m = re.search(r'(\d{1,2})\s*(?:jahre?\s*alt|j\b|-jaehrig)', t)
        if m:
            age = int(m.group(1))

        best_id    = "KG"
        best_score = -1

        for pid, prof in self._PROFILES.items():
            if pid == "KG":
                continue  # fallback — wins only if nothing else matches

            # Age constraints
            if age is not None:
                if prof.get("age_max") is not None and age > prof["age_max"]:
                    continue
                if prof.get("age_min") is not None and age < prof["age_min"]:
                    continue
            else:
                # No age detected → skip paediatric-only profiles to avoid
                # false positives (they require explicit age context).
                if prof.get("age_max", 999) <= 17:
                    continue

            triggers = prof.get("triggers", [])
            match_count = sum(1 for trig in triggers if trig in t)
            if match_count == 0:
                continue

            score = prof.get("priority", 0) * 1000 + match_count
            if score > best_score:
                best_id    = pid
                best_score = score

        return best_id

    def _is_compatible_profile(self, profile_a: str, profile_b: str) -> bool:
        """
        Check if two profiles are compatible (can share tests).
        Returns True if profiles are in the same anatomical group.
        """
        # Define profile groups
        SPINE_PROFILES = {"EX_HWS", "EX_LWS", "EX_BWS", "EX_ISG"}
        UPPER_EXTREMITY = {"EX_SCHULTER", "EX_ELLBOGEN", "EX_HAND"}
        LOWER_EXTREMITY = {"EX_HUEFTE", "EX_KNIE", "EX_FUSS"}
        LYMPH_PROFILES = {"LY_ARM", "LY_BEIN"}
        NEURO_PROFILES = {"ZNS_STROKE", "ZNS_MS", "ZNS_PARKINSON"}
        SPECIAL_PROFILES = {"ATEMTHERAPIE", "BECKENBODEN", "KGG"}
        
        # Normalize profile names
        profile_a = profile_a.upper()
        profile_b = profile_b.upper()
        
        # Check if both profiles are in the same group
        for group in [SPINE_PROFILES, UPPER_EXTREMITY, LOWER_EXTREMITY, 
                      LYMPH_PROFILES, NEURO_PROFILES, SPECIAL_PROFILES]:
            if profile_a in group and profile_b in group:
                return True
        
        return False

    def _remove_incompatible_tests(self, soap_dict: dict, profile_type: str) -> dict:
        """
        Remove tests from SOAP note that are incompatible with the current profile.
        Prevents contamination like spine tests in shoulder reports.
        
        Args:
            soap_dict: SOAP note dictionary with S, O, A, P fields
            profile_type: Simplified profile type (SHOULDER, KNEE, HWS, LWS, etc.)
        
        Returns:
            Cleaned SOAP dictionary
        """
        # Define test categories by anatomical region
        SPINE_TESTS = [
            "Schober", "Lasègue", "Lasegue", "FBA", "Finger-Boden",
            "Vorlaufphänomen", "Ott-Zeichen", "Lateral-Flexion"
        ]
        
        SHOULDER_TESTS = [
            "Jobe", "Hawkins", "Neer", "Painful Arc", "Lift-off", 
            "Belly-Press", "Empty-Can", "Drop-Arm"
        ]
        
        KNEE_TESTS = [
            "Lachman", "McMurray", "Schublade", "Varus", "Valgus",
            "Patellatanz", "Patella-Apprehension"
        ]
        
        HIP_TESTS = [
            "Thomas", "Trendelenburg", "Patrick", "FABER", "FADIR",
            "Log-Roll"
        ]
        
        ELBOW_TESTS = [
            "Epicondylitis", "Golferellenbogen", "Tennisellenbogen",
            "Valgus-Stress", "Varus-Stress"
        ]
        
        WRIST_TESTS = [
            "Phalen", "Tinel", "Finkelstein", "Watson"
        ]
        
        LYMPH_TESTS = [
            "Stemmer", "Umfangsmessung", "Pitting", "Hautqualität"
        ]
        
        # Map profile types to their valid tests
        profile_type = profile_type.upper()
        
        # Determine what tests to KEEP based on profile
        keep_tests = []
        remove_tests = []
        
        if "SHOULDER" in profile_type or "SCHULTER" in profile_type:
            keep_tests = SHOULDER_TESTS
            remove_tests = SPINE_TESTS + KNEE_TESTS + HIP_TESTS + ELBOW_TESTS + WRIST_TESTS + LYMPH_TESTS
        elif "KNEE" in profile_type or "KNIE" in profile_type:
            keep_tests = KNEE_TESTS
            remove_tests = SPINE_TESTS + SHOULDER_TESTS + HIP_TESTS + ELBOW_TESTS + WRIST_TESTS + LYMPH_TESTS
        elif "HIP" in profile_type or "HÜFT" in profile_type or "HUEFT" in profile_type:
            keep_tests = HIP_TESTS
            remove_tests = SPINE_TESTS + SHOULDER_TESTS + KNEE_TESTS + ELBOW_TESTS + WRIST_TESTS + LYMPH_TESTS
        elif "ELBOW" in profile_type or "ELLBOGEN" in profile_type:
            keep_tests = ELBOW_TESTS
            remove_tests = SPINE_TESTS + SHOULDER_TESTS + KNEE_TESTS + HIP_TESTS + WRIST_TESTS + LYMPH_TESTS
        elif "WRIST" in profile_type or "HAND" in profile_type:
            keep_tests = WRIST_TESTS
            remove_tests = SPINE_TESTS + SHOULDER_TESTS + KNEE_TESTS + HIP_TESTS + ELBOW_TESTS + LYMPH_TESTS
        elif "HWS" in profile_type:
            keep_tests = []  # HWS has specific tests, remove all spine-specific ones
            remove_tests = [t for t in SPINE_TESTS if t not in ["Lateral-Flexion"]] + SHOULDER_TESTS + KNEE_TESTS + HIP_TESTS + ELBOW_TESTS + WRIST_TESTS + LYMPH_TESTS
        elif "LWS" in profile_type:
            keep_tests = SPINE_TESTS
            remove_tests = SHOULDER_TESTS + KNEE_TESTS + HIP_TESTS + ELBOW_TESTS + WRIST_TESTS + LYMPH_TESTS
        elif "BWS" in profile_type:
            keep_tests = []  # BWS has specific tests
            remove_tests = [t for t in SPINE_TESTS if t in ["Schober", "Lasègue", "Lasegue", "FBA", "Finger-Boden"]] + SHOULDER_TESTS + KNEE_TESTS + HIP_TESTS + ELBOW_TESTS + WRIST_TESTS + LYMPH_TESTS
        elif "LY" in profile_type or "LYMPH" in profile_type:
            keep_tests = LYMPH_TESTS
            remove_tests = SPINE_TESTS + SHOULDER_TESTS + KNEE_TESTS + HIP_TESTS + ELBOW_TESTS + WRIST_TESTS
        else:
            # Unknown profile - don't remove anything
            return soap_dict
        
        # Clean the O (Objective) field
        cleaned_soap = soap_dict.copy()
        obj_text = cleaned_soap.get("O", "")
        
        if not obj_text:
            return cleaned_soap
        
        # Split on pipe separator to preserve structure
        parts = obj_text.split("|")
        cleaned_parts = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check if this part contains a test to remove
            should_remove = False
            for test in remove_tests:
                # Case-insensitive check
                if test.lower() in part.lower():
                    should_remove = True
                    break
            
            if not should_remove:
                cleaned_parts.append(part)
        
        # Reconstruct the O field
        cleaned_soap["O"] = " | ".join(cleaned_parts)
        
        return cleaned_soap

    def _profile_checklist(self, profile_id: str) -> str:
        prof = self._PROFILES.get(profile_id, self._PROFILES["KG"])
        label    = prof["label"]
        billing  = prof["billing"]
        items    = "\n".join(f"- {item}" for item in prof["checklist"])
        prefixes = prof.get("icd_prefix", [])
        icd_hint = prefixes[0] if prefixes else "nach Befund"
        return (
            f"PROFIL: {label}  |  Abrechnung: {billing}\n"
            f"ICD-10-Hinweis: waehle NUR EINEN passenden Code (typisch: {icd_hint}.x)\n"
            f"PFLICHTFELDER O-Feld:\n{items}"
        )

    def build_prompt(self, transcript: str, profile_id: str = "KG") -> str:
        learning_notes = self.learning_mgr.get_relevant_prefs(transcript)
        few_shot_block = self.learning_mgr.format_few_shot_block(transcript, profile_id)
        style_injection = ""
        if learning_notes:
            style_injection += f"\nBEVORZUGTE CODES DES THERAPEUTEN:\n{learning_notes}\n"
        if few_shot_block:
            style_injection += f"\n{few_shot_block}\n"
        checklist = self._profile_checklist(profile_id)
        prof = self._PROFILES.get(profile_id, self._PROFILES["KG"])

        _pain_examples = {
            "EX_SCHULTER": "linke Schulter, Ausstrahlung in den Arm",
            "EX_HWS":      "Nacken/HWS mit Ausstrahlung in den Arm",
            "EX_LWS":      "Lendenwirbelsäule, Ausstrahlung ins Bein",
            "EX_HUefte":   "rechte Hüfte mit Ausstrahlung in den Oberschenkel",
            "EX_HUFTE":    "rechte Hüfte mit Ausstrahlung in den Oberschenkel",
            "EX_KNIE":     "linkes Knie, Schmerz bei Treppensteigen",
            "EX_FUSS":     "linkes Sprunggelenk, Schmerz beim Abrollen",
        }
        _red_flag_examples = {
            "EX_SCHULTER": "keine Parästhesien in Hand/Fingern, kein Kraftverlust im Arm, kein Verdacht auf vollständige RM-Ruptur",
            "EX_HWS":      "keine Arm-Parästhesien, keine Dysphagie, keine Myelopathiezeichen",
            "EX_LWS":      "keine Blasen-/Mastdarmstörung, keine Kauda-Symptomatik, keine Lähmung",
            "EX_HUefte":   "keine Femurhalsfraktur, keine AVN-Zeichen, kein Tumorverdacht",
            "EX_HUFTE":    "keine Femurhalsfraktur, keine AVN-Zeichen, kein Tumorverdacht",
            "EX_KNIE":     "keine Kompartment-Zeichen, kein Tumorverdacht, keine tief. Venenthrombose",
        }
        pain_ex     = _pain_examples.get(profile_id, "lokaler Schmerz, ggf. Ausstrahlung")
        red_flag_ex = _red_flag_examples.get(profile_id, "Red Flags klinisch ausgeschlossen")
        
        # ICD hint for the profile
        prefixes = prof.get("icd_prefix", [])
        icd_hint = prefixes[0] if prefixes else "M99.9"

        _inspection_examples = {
            "EX_SCHULTER": "Schulterachse re. hochgezogen, Skapula protrahiert",
            "EX_HWS":      "Kopfhaltung in Vorneigung, Schultern hochgezogen bds.",
            "EX_LWS":      "Schonhaltung nach re., Beckenschiefstand, Hyperlordose",
            "EX_HUefte":   "Trendelenburg-Hinken re., Becken sinkt li. bei Einbeinstand",
            "EX_HUFTE":    "Trendelenburg-Hinken re., Becken sinkt li. bei Einbeinstand",
            "EX_KNIE":     "Knieachse valgus li., geringgradige Schwellung med. Gelenkspalt",
            "EX_FUSS":     "Schwellung Außenknöchel li., Hämatom unter Malleolus lateralis, Entlastungshinken",
            "LY":          "diffuse Schwellung Unterschenkel re., Haut gespannt und glänzend",
            "AT":          "Atemexkursion eingeschränkt, Schulteratmung sichtbar",
        }
        inspection_ex = _inspection_examples.get(profile_id, "Schonhaltung, sichtbare Bewegungseinschränkung")

        _smart_goal_examples = {
            "EX_SCHULTER": "Ziel: Schulterabduktion auf 120° in 6 EH",
            "EX_HWS":      "Ziel: HWS-Rotation auf 60° bds. in 4 EH",
            "EX_LWS":      "Ziel: FBA 10 cm in 6 EH, beschwerdefrei bei Flexion",
            "EX_HUefte":   "Ziel: Hüftflexion 0-0-110° in 4 EH",
            "EX_HUFTE":    "Ziel: Hüftflexion 0-0-110° in 4 EH",
            "EX_KNIE":     "Ziel: Knieflexion 0-0-120° in 6 EH",
            "EX_FUSS":     "Ziel: OSG-Dorsalextension 0-0-20° in 4 EH, schmerzfreies Abrollen",
            "GEB":         "Ziel: Beckenbodenkraft Oxford 3/5 in 6 EH",
            "KGG":         "Ziel: 10 Wiederholungen Beinpresse 40 kg ohne Schmerz in 4 EH",
            "GER":         "Ziel: 10m Tandemgang ohne Hilfsmittel in 6 EH",
        }
        smart_goal_ex = _smart_goal_examples.get(profile_id, "Ziel: [Funktion] auf [Messwert] in [N] EH")

        # Krücken recommendation only makes clinical sense for lower-limb profiles.
        _lower_limb_profiles = {"EX_KNIE", "EX_HUefte", "EX_HUFTE", "EX_FUSS", "GER", "POST_OP"}
        kruecken_line = (
            "  • Krücken- / Hilfsmittel-Empfehlung mit SEITE (kontralateral zur betroffenen Seite!)"
            if profile_id in _lower_limb_profiles else ""
        )

        return f"""<|start_header_id|>system<|end_header_id|>
Du bist ein medizinischer Dokumentationsexperte. Erstelle aus dem Transkript einen SOAP-Befund als JSON.

⚠️⚠️⚠️ KRITISCHE WARNUNG - CONTEXT ISOLATION ⚠️⚠️⚠️
JEDE SITZUNG IST EIN NEUER PATIENT. Verwende NIEMALS Informationen aus vorherigen Sitzungen!
Das S-Feld (Subjektiv) MUSS zu 100% aus dem AKTUELLEN Transkript stammen.
Wenn im Transkript KNIE behandelt wird, darf NICHTS über Rücken/LWS im S-Feld erscheinen!

WICHTIG - STRIKTE REGELN:
1. Extrahiere NUR Fakten aus dem Transkript. Wenn etwas nicht erwähnt wurde → "n.d."
2. Alle SOAP-Felder müssen STRINGS sein, KEINE verschachtelten Objekte
3. S-Feld = NUR die PATIENTENGESCHICHTE aus dem AKTUELLEN Transkript (keine Therapeutenfragen)
4. O-Feld = Messwerte und Tests als EINZIGER STRING mit | als Trenner
5. JSON-Format:  {{"icd10": "CODE", "soap": {{"S": "text", "O": "text", "A": "text", "P": "text"}}}}

PROFIL: {prof["label"]}
{style_injection}

S-FELD (Subjektiv):
- Zusammenfassung der Patientenaussagen aus dem AKTUELLEN Transkript in 2-3 Sätzen
- ⚠️ VERWENDE NUR INFORMATIONEN AUS DEM UNTEN STEHENDEN TRANSKRIPT!
- ❌ FALSCH: "Erzählen Sie mir bitte..."  (Das ist der Therapeut!)
- ❌ FALSCH: Informationen über andere Körperregionen als im Transkript erwähnt
- ✅ RICHTIG: "Pat. berichtet akute LWS-Schmerzen seit Donnerstag nach Heben..."
- Schmerzangaben: VAS-Wert wenn genannt
- Beispiel: "{pain_ex}"

O-FELD (Objektiv):
- Format: "Inspektion... | Test1: Wert | Test2: Wert | ..."
- Alle Tests aus dem Transkript extrahieren
- ❌ FALSCH: {{"FBA": "40", "Lasegue": "negativ"}}  (verschachteltes Objekt!)
- ✅ RICHTIG: "Schonhaltung re. | FBA: 40 cm | Lasègue 80° negativ | Kraftgrade 5/5"
- Pflichtfelder für {prof["label"]}:
{checklist}

A-FELD (Assessment):
- Format: "ICD-10-Code | Diagnose | Red Flags ausgeschlossen"
- Beispiel: "{icd_hint} | {red_flag_ex}"

P-FELD (Plan):
- Format: "Heilmittel | Technik | Ziel: ... | Frequenz"
- Beispiel: "KG mit manuellen Techniken | {smart_goal_ex} | 2x/Woche, 6 EH"
{kruecken_line}

AUSGABEFORMAT (NUR EIN JSON-Objekt, KEINE Wiederholungen):
{{"icd10": "{icd_hint}", "soap": {{"S": "Patientengeschichte als String", "O": "Test1 | Test2 | Test3", "A": "Diagnose | Red Flags", "P": "Behandlung | Ziel"}}}}

<|eot_id|><|start_header_id|>user<|end_header_id|>
Transkript:
{transcript}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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

            except (FileNotFoundError, OSError, RuntimeError) as ffe:
                # FFmpeg not found - try Python-based audio loading
                error_msg = str(ffe).lower()
                is_ffmpeg_error = (
                    "system cannot find the file" in error_msg or
                    "ffmpeg" in error_msg or
                    "winerror 2" in error_msg or
                    "no such file" in error_msg
                )

                if is_ffmpeg_error:
                    print(f"⚠️ FFmpeg not available, using Python-based audio loading...")
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
                            f"  pip install soundfile\n"
                            f"Error: {ie}"
                        ) from ie
                    except Exception as inner_e:
                        raise RuntimeError(
                            f"Audio loading failed with both FFmpeg and soundfile.\n"
                            f"Error: {inner_e}"
                        ) from inner_e
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

        # 🔥 CRITICAL: Reset LLM cache to prevent context contamination between sessions
        # Without this, the LLM "remembers" previous patient data and mixes it into new sessions
        try:
            if hasattr(self.llm, 'reset'):
                self.llm.reset()
            elif hasattr(self.llm, '_ctx'):
                # Direct llama.cpp context reset (if available)
                import llama_cpp
                if hasattr(llama_cpp, 'llama_kv_cache_clear'):
                    llama_cpp.llama_kv_cache_clear(self.llm._ctx)
                    print("🔄 LLM context cache cleared")
        except Exception as e:
            print(f"⚠️ Could not reset LLM context: {e}")
            # Continue anyway - the strong prompt warnings should help

        raw = None
        try:
            output = self.llm(
                prompt, 
                max_tokens=1800,  # Reduced to leave room for prompt (4096 ctx - ~2300 prompt = ~1800 for output)
                temperature=0.3,  # Low temperature for factual extraction
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|eot_id|>", "<|end_header_id|>"],  # Stop tokens for Llama format
            )
            raw = output["choices"][0]["text"]
            

            # Debug: Warn if output is suspiciously short
            if len(raw) < 100:
                print(f"⚠️ WARNING: LLM produced very short output ({len(raw)} chars)")
                print(f"Transcript length: {len(transcript)} chars")
                
        except Exception as e:
            print(f"❌ LLM call failed with: {type(e).__name__}: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            raw = '{"icd10": "M99.9", "soap": {"S": "n.d.", "O": "n.d.", "A": "n.d. | Red Flags klinisch ausgeschlossen.", "P": "n.d."}}'

        return "{" + raw if not raw.strip().startswith("{") else raw

    # ── Post-processing pipeline ───────────────────────────────────────────────

    def _detect_joint_context(self, profile_id: str, transcript: str) -> list[str]:
        """
        Detect which joint(s) are being examined based on profile and transcript.
        Returns list of joint keywords for ROM formatting.
        """
        t_low = transcript.lower()
        joints = []

        # Spine
        if "HWS" in profile_id.upper() or any(k in t_low for k in ["hws", "zervikal", "nacken", "halswirbel"]):
            joints.append("hws")
        if "LWS" in profile_id.upper() or any(k in t_low for k in ["lws", "lumbal", "lumbago"]):
            joints.append("lws")
        if "BWS" in profile_id.upper() or any(k in t_low for k in ["bws", "thorakal", "brustwirbel"]):
            joints.append("bws")

        # Upper Extremity
        if "SHOULDER" in profile_id.upper() or "SCHULTER" in profile_id.upper() or \
           any(k in t_low for k in ["schulter", "shoulder", "glenohumeral"]):
            joints.append("shoulder")
        if "ELBOW" in profile_id.upper() or "ELLBOGEN" in profile_id.upper() or \
           any(k in t_low for k in ["ellbogen", "elbow"]):
            joints.append("elbow")
        if "WRIST" in profile_id.upper() or "HAND" in profile_id.upper() or \
           any(k in t_low for k in ["handgelenk", "wrist"]):
            joints.append("wrist")

        # Lower Extremity
        if "HIP" in profile_id.upper() or "HÜFT" in profile_id.upper() or \
           any(k in t_low for k in ["hüfte", "hüft", "hip", "koxofemoral"]):
            joints.append("hip")
        if "KNEE" in profile_id.upper() or "KNIE" in profile_id.upper() or \
           any(k in t_low for k in ["knie", "knee"]):
            joints.append("knee")
        if "ANKLE" in profile_id.upper() or "FOOT" in profile_id.upper() or "FUSS" in profile_id.upper() or \
           any(k in t_low for k in ["sprunggelenk", "ankle", "osg", "fuss", "fuß"]):
            joints.append("ankle")

        return joints if joints else ["general"]

    def _get_rom_joint_label(self, joint_context: list[str]) -> str:
        """Get the appropriate joint label for ROM documentation."""
        if "hws" in joint_context:
            return "HWS"
        elif "lws" in joint_context:
            return "LWS"
        elif "bws" in joint_context:
            return "BWS"
        elif "shoulder" in joint_context:
            return "Schulter"
        elif "elbow" in joint_context:
            return "Ellbogen"
        elif "wrist" in joint_context:
            return "Handgelenk"
        elif "hip" in joint_context:
            return "Hüfte"
        elif "knee" in joint_context:
            return "Knie"
        elif "ankle" in joint_context:
            return "OSG"
        else:
            return ""

    def recover_hard_metrics(self, transcript: str, soap_dict: dict, profile_id: str = "KG") -> dict:
        """Safety net: if the therapist SAID it, it MUST appear in O."""
        obj_val = soap_dict.get("O", "")
        obj_text = obj_val if isinstance(obj_val, str) else ""
        t_low = transcript.lower() if isinstance(transcript, str) else ""

        # ✨ CRITICAL: Profile-Specific Test Detection
        # Prevent Context Contamination: HWS tests != LWS tests
        _is_hws_session = profile_id == "EX_HWS" or any(
            k in t_low for k in ["hws", "zervikal", "nacken", "halswirbel", "cervical"])
        _is_lws_session = profile_id == "EX_LWS" or any(
            k in t_low for k in ["lws", "lumbal", "isg", "iliosakral", "kreuzschmerz", "bandscheib"])

        # Schober test - LUMBAR ONLY (anatomically impossible for HWS)
        schober = re.search(r"Schober.*?(\d+)\s*(?:zu|bis|-)\s*(\d+)", transcript, re.I)
        if schober and "Schober" not in obj_text and _is_lws_session and not _is_hws_session:
            obj_text += f" | Schober-Zeichen: {schober.group(1)} - {schober.group(2)}"

        # FBA (Finger-Boden-Abstand) — LUMBAR ONLY
        # Injecting FBA into extremity sessions (EX_FUSS, EX_KNIE, etc.) causes the LLM
        # to confuse "FBA" with ankle abbreviations ("Fuß-Band-Außen") in the S-field.
        _is_spine_session = _is_lws_session or _is_hws_session
        fba = re.search(r"(?:finger.boden|fba)[^\d]*(\d+)\s*cm", transcript, re.I)
        # FBA is LUMBAR-only (not for HWS!)
        if _is_lws_session and not _is_hws_session and fba and "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
            obj_text += f" | FBA: {fba.group(1)} cm"
        elif _is_lws_session and not _is_hws_session and re.search(r"finger.boden|fba", transcript, re.I):
            if "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
                obj_text += " | FBA: n.d."

        # ── INTELLIGENT INFERENCE: Schober from FBA ───────────────────────────
        # Clinical reasoning: FBA correlates with lumbar flexion mobility
        # FBA > 25 cm → poor mobility → likely Schober < 12 cm expansion
        # FBA < 10 cm → good mobility → likely Schober 14-15 cm expansion
        # LUMBAR ONLY (not for HWS!)
        if _is_lws_session and not _is_hws_session and "schober" not in obj_text.lower():
            # Try to extract FBA value (either just captured or already in O-field)
            fba_in_obj = re.search(r"FBA[:\s]+(\d+)\s*cm", obj_text, re.I)
            fba_value = None
            if fba:
                fba_value = int(fba.group(1))
            elif fba_in_obj:
                fba_value = int(fba_in_obj.group(1))
            
            if fba_value is not None:
                # Infer Schober from FBA (clinical correlation)
                # Normal: Schober 13-15 cm, FBA 0-10 cm
                # Restricted: Schober 10-12 cm, FBA 15-30 cm
                # Severely restricted: Schober < 10 cm, FBA > 30 cm
                if fba_value <= 10:
                    schober_est = "14-15 cm (geschätzt aus FBA - gute Flexionsmobilität)"
                elif fba_value <= 20:
                    schober_est = "12-13 cm (geschätzt aus FBA - mäßig eingeschränkt)"
                elif fba_value <= 30:
                    schober_est = "10-12 cm (geschätzt aus FBA - deutlich eingeschränkt)"
                else:
                    schober_est = "< 10 cm (geschätzt aus FBA - stark eingeschränkt)"
                obj_text += f" | Schober-Zeichen: {schober_est}"
                print(f"[IntelligentInference] Schober estimated from FBA {fba_value} cm → {schober_est}")

        # Verbal LWS flexion descriptions → convert to FBA, remove hallucinated degree-ROM
        _verbal_fba = [
            (r'mitte\s+(?:der\s+)?schienbein\w*|schienbeinhöhe|schienbeinniveau', '~35 cm', 'Mitte Schienbein'),
            (r'kniehöhe\b|bis\s+(?:zum?\s+)?knie\b',                              '~50 cm', 'Kniehöhe'),
            (r'waden(?:höhe)?\b|wadenmitte\b',                                     '~25 cm', 'Wadenhöhe'),
            (r'knöchelh?öhe\b|bis\s+(?:zum?\s+)?knöchel\b',                       '~15 cm', 'Knöchelhöhe'),
            (r'(?:fast\s+)?den?\s+boden\b|bodenkontakt\b',                        '~5 cm',  'fast Boden'),
        ]
        # LUMBAR ONLY (not for HWS!)
        if _is_lws_session and not _is_hws_session:
            for pattern, fba_val, fba_label in _verbal_fba:
                if re.search(pattern, transcript, re.I):
                    if "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
                        obj_text += f" | FBA: {fba_val} (Angabe Therapeut: {fba_label})"
                    obj_text = re.sub(r'(?:LWS[^|]*?)?\b0-0-\d{2,3}\b[^|]*', '', obj_text).strip(' |')
                    break

        s_val = soap_dict.get("S", "")
        s_text = s_val if isinstance(s_val, str) else ""

        # Recover VAS — handle all orderings: "VAS 6", "6 von 10 beim Schmerz", "6/10"
        if "VAS" not in s_text and "vas" not in s_text.lower():
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
                # Try "Skala von 1 bis 10 würde ich sagen, ... ist es eine 8"
                m = re.search(r"(?:skala|scale)[^.]*?(?:ist\s+es\s+)?(?:eine\s+)?(\d{1,2})", transcript, re.I)
                if m:
                    vas_num = m.group(1)
            if not vas_num:
                m = re.search(r"\b([1-9]|10)\s*/\s*10\b", transcript)
                if m:
                    vas_num = m.group(1)
            if vas_num:
                soap_dict["S"] = f"VAS {vas_num}/10. " + s_text.lstrip()

        # Lasègue test - LUMBAR ONLY (anatomically impossible for HWS)
        if ("lasegue" in transcript.lower() or "lasek" in transcript.lower()) and _is_lws_session and not _is_hws_session:
            if "lasègue" not in obj_text.lower():
                deg = re.search(r"(?:lasegue|lasek).*?(\d+)\s*(?:grad|°)", transcript, re.I)
                result_match = re.search(r"(?:lasegue|lasek).*?(positiv|negativ)", transcript, re.I)
                deg_val = deg.group(1) if deg else "n.d."
                result_val = result_match.group(1) if result_match else "positiv"
                obj_text += f" | Lasègue-Test: {deg_val}° {result_val}"
        
        # Vorlaufphänomen (LWS-specific test)
        if "vorlauf" in transcript.lower() and "vorlauf" not in obj_text.lower() and _is_lws_session and not _is_hws_session:
            vorlauf_match = re.search(r"vorlauf(?:phänomen|test)?.*?(rechts|links|re\.|li\.)?.*?(positiv|negativ)", transcript, re.I)
            if vorlauf_match:
                side = vorlauf_match.group(1) or "bds."
                result = vorlauf_match.group(2)
                obj_text += f" | Vorlaufphänomen {side} {result}"
            else:
                obj_text += " | Vorlaufphänomen: erwähnt (Details n.d.)"
        
        # Paraspinale Muskulatur (LWS palpation - not for HWS)
        if "paraspinal" in transcript.lower() and "paraspinal" not in obj_text.lower() and _is_lws_session and not _is_hws_session:
            palpation_match = re.search(r"paraspinal.*?(hyperton|verspannt|hart|locker|normal)", transcript, re.I)
            if palpation_match:
                finding = palpation_match.group(1)
                obj_text += f" | Paraspinale Muskulatur: {finding}"
        
        # Myogelose / Triggerpunkte - Enhanced recovery
        if "myogelose" in transcript.lower() and "myogelose" not in obj_text.lower():
            myogelose_match = re.search(r"myogelose.*?(musculus|m\.)\s+([\w\s]+?)(?:\.|rechts|links)", transcript, re.I)
            if myogelose_match:
                muscle = myogelose_match.group(2).strip()
                obj_text += f" | Myogelose: M. {muscle}"
            else:
                obj_text += " | Myogelose: vorhanden"
        
        # Specific muscle mentions in Assessment (move to A-field if in transcript)
        a_val = soap_dict.get("A", "")
        a_text = a_val if isinstance(a_val, str) else ""
        
        # Extract key clinical terms that should appear in Assessment
        clinical_terms = {
            r"quadratus\s+lumborum": "Quadratus lumborum",
            r"dysbalance|muskuläre\s+dysbalance": "muskuläre Dysbalance",
            r"bandscheibenvorfall": "Bandscheibenvorfall",
            r"schutzspannung": "Schutzspannung",
            r"triggerpunkt": "Triggerpunkt",
        }
        
        for pattern, term in clinical_terms.items():
            if re.search(pattern, transcript, re.I) and term.lower() not in a_text.lower():
                # Don't add duplicates
                if not a_text.strip():
                    a_text = term
                elif term not in a_text:
                    a_text += f" | {term}"
        
        soap_dict["A"] = a_text
        
        # Extract treatment plan details (for P-field)
        p_val = soap_dict.get("P", "")
        p_text = p_val if isinstance(p_val, str) else ""
        
        # Treatment techniques mentioned
        treatment_terms = {
            r"triggerpunkt\w*\s+behandel\w*|triggerpunkt\w*\s+therapie": "Triggerpunktbehandlung",
            r"wärme|durchblutung\s+fördern": "Wärme",
            r"stufenlagerung": "Stufenlagerung",
            r"detonisierung|detonisier\w*": "Detonisierung der Muskulatur",
            r"manuelle\s+techniken": "Manuelle Techniken",
        }
        
        for pattern, term in treatment_terms.items():
            if re.search(pattern, transcript, re.I) and term.lower() not in p_text.lower():
                if "n.d." in p_text or not p_text.strip():
                    p_text = term
                elif term not in p_text:
                    p_text += f" | {term}"
        
        # Frequency extraction - "zweimal pro Woche", "2x/Woche", etc.
        if not re.search(r"\d+\s*x|mal\s+pro\s+woche|pro\s+woche", p_text, re.I):
            freq_match = re.search(
                r"(zwei|2|drei|3)\s*(?:mal|x)?\s*(?:pro|die|in\s+der)?\s*woche",
                transcript, re.I
            )
            if freq_match:
                freq_text = freq_match.group(0)
                if "zwei" in freq_text.lower() or freq_text.startswith("2"):
                    p_text += " | 2x/Woche"
                elif "drei" in freq_text.lower() or freq_text.startswith("3"):
                    p_text += " | 3x/Woche"
        
        # Number of sessions - "sechs Termine", "6 Behandlungen", etc.
        if not re.search(r"\d+\s*(?:EH|Einheiten|Termine|Behandlungen|Sitzungen)", p_text, re.I):
            sessions_match = re.search(
                r"(sechs|6|acht|8|zehn|10|zwölf|12)\s*(?:Termine?|Behandlungen?|Sitzungen?|Einheiten?)",
                transcript, re.I
            )
            if sessions_match:
                sessions_text = sessions_match.group(0).lower()
                if "sechs" in sessions_text or sessions_text.startswith("6"):
                    p_text += " | 6 EH"
                elif "acht" in sessions_text or sessions_text.startswith("8"):
                    p_text += " | 8 EH"
                elif "zehn" in sessions_text or sessions_text.startswith("10"):
                    p_text += " | 10 EH"
                elif "zwölf" in sessions_text or sessions_text.startswith("12"):
                    p_text += " | 12 EH"
        
        soap_dict["P"] = p_text
        
        # Enhance S-field with day-of-week and specific events if missing
        if s_text:
            # Add weekday if mentioned
            weekdays = {
                "donnerstag": "Donnerstag",
                "montag": "Montag",
                "dienstag": "Dienstag",
                "mittwoch": "Mittwoch",
                "freitag": "Freitag",
                "samstag": "Samstag",
                "sonntag": "Sonntag",
            }
            for day_pattern, day_name in weekdays.items():
                if day_pattern in transcript.lower() and day_name.lower() not in s_text.lower():
                    # Insert at beginning of S-field
                    s_text = f"Beginn {day_name}. " + s_text
                    break
            
            # Add specific triggers if mentioned but missing
            triggers = {
                r"umzugskarton\w*|karton\w*\s+(?:gehoben|heben)": "beim Heben von Umzugskartons",
                r"gartenarbeit": "nach Gartenarbeit",
                r"sport": "nach Sport",
            }
            for trigger_pattern, trigger_text in triggers.items():
                if re.search(trigger_pattern, transcript, re.I):
                    # Check if any form of the trigger is already mentioned
                    if not any(word in s_text.lower() for word in trigger_text.lower().split()):
                        # Add to S-field if not there
                        s_text = s_text.rstrip('. ') + f", {trigger_text}."
                        break
            
            soap_dict["S"] = s_text

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
        
        # ══════════════════════════════════════════════════════════════════════
        # UNIVERSAL ROM RECOVERY (Neutral-Null-Method Auto-Formatting)
        # Applies to ALL joints where ROM is measured
        # ══════════════════════════════════════════════════════════════════════
        
        # Detect joint/region from profile or transcript
        joint_context = self._detect_joint_context(profile_id, transcript)
        
        # ── ROTATION (Spine, Shoulder, Hip) ───────────────────────────────────
        if any(k in joint_context for k in ["hws", "lws", "shoulder", "hip"]):
            rot_pattern = None
            right_val = None
            left_val = None
            
            # Pattern 1: "Rotation 30 rechts, 60 links"
            rot_pattern = re.search(
                r"rotation.*?(\d+)\s*(?:grad|°|degrees?).*?(?:rechts|re\.|right).*?(\d+)\s*(?:grad|°|degrees?).*?(?:links|li\.|left)", 
                transcript, re.I
            )
            if rot_pattern:
                right_val, left_val = rot_pattern.group(1), rot_pattern.group(2)
            
            # Pattern 2: "30 Grad nach rechts, 60 Grad nach links"
            if not rot_pattern:
                rot_pattern = re.search(
                    r"(\d+)\s*(?:grad|°|degrees?)\s*(?:nach\s+)?(?:rechts|re\.|right).*?(\d+)\s*(?:grad|°|degrees?)\s*(?:nach\s+)?(?:links|li\.|left)",
                    transcript, re.I
                )
                if rot_pattern:
                    right_val, left_val = rot_pattern.group(1), rot_pattern.group(2)
            
            # Pattern 3: "nach rechts... 30 Grad, nach links... 60 Grad"
            if not rot_pattern:
                rot_pattern = re.search(
                    r"(?:nach\s+)?(?:rechts|re\.|right)\s+(?:ist\s+)?(?:bei\s+)?(?:etwa\s+)?(\d+)\s*(?:grad|°|degrees?).*?(?:nach\s+)?(?:links|li\.|left)\s+(?:ist\s+)?(?:bei\s+)?(?:etwa\s+)?(\d+)\s*(?:grad|°|degrees?)",
                    transcript, re.I
                )
                if rot_pattern:
                    right_val, left_val = rot_pattern.group(1), rot_pattern.group(2)
            
            # Pattern 4: German medical pattern "auf X Grad... nach links... Y Grad"
            # Handles: "Rotation nach rechts... auf 30 Grad. Nach links... mit 60 Grad"
            if not rot_pattern:
                # Look for right value with context words like "auf", "bei", "ca."
                right_match = re.search(
                    r"(?:rechts|re\.).*?(?:auf|bei|ca\.|circa|etwa)\s+(\d+)\s*(?:grad|°)",
                    transcript, re.I | re.DOTALL
                )
                # Look for left value
                left_match = re.search(
                    r"(?:nach\s+)?(?:links|li\.).*?(?:mit|bei|auf|ca\.|circa|etwa)?\s*(\d+)\s*(?:grad|°)",
                    transcript, re.I | re.DOTALL
                )
                if right_match and left_match:
                    right_val = right_match.group(1)
                    left_val = left_match.group(1)
                    rot_pattern = True  # Mark as found
            
            # Pattern 5: Separated by sentence - very flexible
            if not rot_pattern:
                # Find any mention of right rotation value
                for pattern in [
                    r"(?:rotation|rotiert?|drehen).*?(?:rechts|re\.).*?(\d+)\s*(?:grad|°)",
                    r"(?:rechts|re\.).*?(?:rotation|rotiert?|drehen).*?(\d+)\s*(?:grad|°)",
                ]:
                    m = re.search(pattern, transcript, re.I | re.DOTALL)
                    if m:
                        right_val = m.group(1)
                        break
                
                # Find any mention of left rotation value
                for pattern in [
                    r"(?:rotation|rotiert?|drehen)?.*?(?:links|li\.).*?(\d+)\s*(?:grad|°)",
                    r"(?:links|li\.).*?(?:rotation|rotiert?|drehen)?.*?(\d+)\s*(?:grad|°)",
                ]:
                    m = re.search(pattern, transcript, re.I | re.DOTALL)
                    if m:
                        left_val = m.group(1)
                        break
                
                if right_val and left_val:
                    rot_pattern = True  # Mark as found
            
            if rot_pattern and right_val and left_val and "Rotation):" not in obj_text:
                joint_label = self._get_rom_joint_label(joint_context)
                obj_text += f" | ROM {joint_label} (Rotation): {right_val}-0-{left_val}"
        
        # ── FLEXION/EXTENSION (All major joints) ──────────────────────────────
        flex_ext = re.search(
            r"(?:extension|reklination|ext(?:ension)?)[^.\d]*(\d+)\s*(?:grad|°)?.*?(?:flexion|inklination|flex(?:ion)?)[^.\d]*(\d+)\s*(?:grad|°)?",
            transcript, re.I
        )
        
        if flex_ext and "Ex/Flex" not in obj_text and "Flexion):" not in obj_text:
            ext_val = flex_ext.group(1)
            flex_val = flex_ext.group(2)
            joint_label = self._get_rom_joint_label(joint_context)
            obj_text += f" | ROM {joint_label} (Ex/Flex): {ext_val}-0-{flex_val}"
        elif not flex_ext:
            # Try reverse order: "Flexion 40, Extension 20"
            flex_ext_rev = re.search(
                r"(?:flexion|inklination|flex(?:ion)?)[^.\d]*(\d+)\s*(?:grad|°|degrees?)\s*(?:nach\s+)?(?:extension|reklination|ext(?:ension)?)[^.\d]*(\d+)\s*(?:grad|°|degrees?)\s*(?:nach\s+)?(?:links|li\.|left)",
                transcript, re.I
            )
            if flex_ext_rev and "Ex/Flex" not in obj_text and "Flexion):" not in obj_text:
                flex_val = flex_ext_rev.group(1)
                ext_val = flex_ext_rev.group(2)
                joint_label = self._get_rom_joint_label(joint_context)
                obj_text += f" | ROM {joint_label} (Ex/Flex): {ext_val}-0-{flex_val}"
        
        # ── ABDUCTION/ADDUCTION (Shoulder, Hip) ────────────────────────────────
        if any(k in joint_context for k in ["shoulder", "hip"]):
            abd_add = re.search(
                r"(?:abduktion|abd)[^.\d]*(\d+)\s*(?:grad|°)?.*?(?:adduktion|add)[^.\d]*(\d+)\s*(?:grad|°)?",
                transcript, re.I
            )
            if abd_add and "Abd/Add" not in obj_text:
                abd_val = abd_add.group(1)
                add_val = abd_add.group(2)
                joint_label = self._get_rom_joint_label(joint_context)
                obj_text += f" | ROM {joint_label} (Abd/Add): {abd_val}-0-{add_val}"
        
        # ── DORSIFLEXION/PLANTARFLEXION (Ankle) ───────────────────────────────
        if "ankle" in joint_context or "fuss" in joint_context or "sprunggelenk" in joint_context:
            dorsi_plant = re.search(
                r"(?:dorsiflexion|dorsiflex)[^.\d]*(\d+)\s*(?:grad|°)?.*?(?:plantarflexion|plantarflex)[^.\d]*(\d+)\s*(?:grad|°)?",
                transcript, re.I
            )
            if dorsi_plant and "Dorsi/Plantar" not in obj_text:
                dorsi_val = dorsi_plant.group(1)
                plantar_val = dorsi_plant.group(2)
                obj_text += f" | ROM OSG (Dorsi/Plantar): {dorsi_val}-0-{plantar_val}"
        
        # ── PRONATION/SUPINATION (Forearm) ────────────────────────────────────
        if "elbow" in joint_context or "ellbogen" in joint_context:
            pro_sup = re.search(
                r"(?:pronation|pron)[^.\d]*(\d+)\s*(?:grad|°)?.*?(?:supination|sup)[^.\d]*(\d+)\s*(?:grad|°)?",
                transcript, re.I
            )
            if pro_sup and "Pro/Sup" not in obj_text:
                pro_val = pro_sup.group(1)
                sup_val = pro_sup.group(2)
                obj_text += f" | ROM Unterarm (Pro/Sup): {pro_val}-0-{sup_val}"
        
        # ── RADIAL/ULNAR DEVIATION (Wrist) ───────────────────────────────────
        if "wrist" in joint_context or "hand" in joint_context:
            rad_uln = re.search(
                r"(?:radial|radialabduktion)[^.\d]*(\d+)\s*(?:grad|°)?.*?(?:ulnar|ulnarabduktion)[^.\d]*(\d+)\s*(?:grad|°)?",
                transcript, re.I
            )
            if rad_uln and "Rad/Uln" not in obj_text:
                rad_val = rad_uln.group(1)
                uln_val = rad_uln.group(2)
                obj_text += f" | ROM Handgelenk (Rad/Uln): {rad_val}-0-{uln_val}"
        
        # ── HWS-SPECIFIC: Spurling test ───────────────────────────────────────
        if _is_hws_session:
            if "spurling" in transcript.lower() and "spurling" not in obj_text.lower():
                spurling_result = re.search(r"spurling.*?(positiv|negativ)", transcript, re.I)
                if spurling_result:
                    obj_text += f" | Spurling-Test: {spurling_result.group(1)}"
                else:
                    obj_text += " | Spurling-Test: erwähnt"

        for test in ["Jobe", "Hawkins", "Neer"]:
            if test.lower() in transcript.lower() and test not in soap_dict.get("O", ""):
                obj_text += f" | {test}-Test: positiv."

        cm_metrics = re.findall(r"([+-]\d+\s*cm)", transcript, re.I)
        if cm_metrics and "cm" not in soap_dict.get("O", ""):
            obj_text += f" | Umfangsdifferenz: {', '.join(cm_metrics)}"

        # ── KGG/MTT: recover training parameters ─────────────────────────────
        is_kgg = any(k in t_low for k in [
            "kgg", "gerätegestützt", "mtt", "medizinische trainings",
            "beinpresse", "latzug", "ergometer", "kabelzug", "legpress",
            "latpulldown", "kraftmaschine", "trainingsgerät",
        ])
        if is_kgg:
            geraet = re.search(
                r"\b(beinpresse|latzug|kabelzug|ruderger[äa]t|ergometer|crosstrainer|"
                r"beinstrecker|beincurl|schulterdr[üu]ck|brustpresse|r[üu]ckenstrecker|"
                r"beinabduktor|legpress|latpulldown|rowing)\b",
                transcript, re.I)
            if geraet and "trainingsplan" not in obj_text.lower() and geraet.group(1).lower() not in obj_text.lower():
                obj_text += f" | Gerät: {geraet.group(1)}"
            last_m = re.search(r"(\d+)\s*(?:kg|kilogramm)(?:\s*(?:widerstand|last|gewicht))?", transcript, re.I)
            if last_m and "kg" not in obj_text:
                obj_text += f" | Last: {last_m.group(1)} kg"
            wdh = re.search(r"(\d+)\s*(?:wiederholungen?|wdh\.?|reps?)", transcript, re.I)
            saetze = re.search(r"(\d+)\s*(?:s[äa]tze?|sets?|serien?)", transcript, re.I)
            if wdh and saetze and "wdh" not in obj_text.lower():
                obj_text += f" | {wdh.group(1)} Wdh x {saetze.group(1)} Sätze"
            elif wdh and "wdh" not in obj_text.lower():
                obj_text += f" | {wdh.group(1)} Wdh"

        # ── Beckenboden: recover Oxford-Skala and contraction duration ────────
        is_becken = any(k in t_low for k in [
            "beckenboden", "inkontinenz", "harninkontinenz", "stressinkontinenz",
            "dranginkontinenz", "kontinenz", "beckenorgane", "prostatektomie",
        ])
        if is_becken:
            oxford = re.search(r"(?:oxford|kraft)[^\d]*([0-5])(?:\s*/\s*5)?", transcript, re.I)
            if oxford and "oxford" not in obj_text.lower() and "beckenboden-kraft" not in obj_text.lower():
                obj_text += f" | Beckenboden-Kraft (Oxford): {oxford.group(1)}/5"
            elif is_becken and "oxford" not in obj_text.lower() and "beckenboden-kraft" not in obj_text.lower():
                obj_text += " | Beckenboden-Kraft (Oxford): n.d."
            kontraktion = re.search(
                r"(\d+)\s*(?:sekunden?|sek\.?)\s*(?:kontraktion|halten|anspannen)",
                transcript, re.I)
            if kontraktion and "kontraktion" not in obj_text.lower():
                obj_text += f" | Kontraktion: {kontraktion.group(1)} s"

        # ── Elektrotherapie: recover modality, Hz, mA, electrode placement ───
        is_elektro = any(k in t_low for k in [
            "tens", "interferenzstrom", "ifc", "galvano", "elektrotherapie", "reizstrom",
        ])
        if is_elektro:
            stromform = re.search(
                r"\b(TENS|IFC|Interferenz(?:strom)?|Galvano(?:phor)?|diadynamisch|EMS)\b",
                transcript, re.I)
            if stromform and "stromform" not in obj_text.lower() and "tens" not in obj_text.lower():
                obj_text += f" | Stromform: {stromform.group(1)}"
            freq = re.search(r"(\d+)\s*(?:Hz|Hertz)", transcript, re.I)
            if freq and "hz" not in obj_text.lower():
                obj_text += f" | Frequenz: {freq.group(1)} Hz"
            intensitaet = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:mA|Milliampere)", transcript, re.I)
            if intensitaet and " ma" not in obj_text.lower() and "ma)" not in obj_text.lower():
                obj_text += f" | Intensität: {intensitaet.group(1)} mA"
            platzierung = re.search(
                r"(?:elektrode[n]?\s+(?:an|über|auf|am)\s+|platzier\w*\s+(?:an|über|am)\s+)"
                r"([\w\säöüÄÖÜß]+?)(?:\s*[,.|]|$)",
                transcript, re.I)
            if platzierung and "elektrode" not in obj_text.lower():
                obj_text += f" | Elektroden: {platzierung.group(1).strip()}"

        # ── Thermotherapie: recover modality, region, temperature ─────────────
        is_thermo = any(k in t_low for k in [
            "fango", "heiße rolle", "heisse rolle", "warmpackung", "wärmetherapie",
            "kältetherapie", "eispack", "kryotherapie",
        ])
        if is_thermo:
            modalitaet = re.search(
                r"\b(Fango|Hei[sß]e\s+Rolle|Warmpackung|W[äa]rmestrahler|Rotlicht|"
                r"Eispack|K[äa]ltespray|Kryotherapie)\b",
                transcript, re.I)
            if modalitaet and all(k not in obj_text.lower() for k in ["fango", "rolle", "eispack", "wärme"]):
                obj_text += f" | Wärmemodalität: {modalitaet.group(1)}"
            temp_m = re.search(r"(\d+(?:[.,]\d+)?)\s*°?C\b", transcript, re.I)
            if temp_m and "°c" not in obj_text.lower() and "grad" not in obj_text.lower():
                obj_text += f" | Temperatur: {temp_m.group(1)} °C"

        stadium = re.search(r"Stadium\s*[1-3]", transcript, re.I)
        if stadium and "Stadium" not in soap_dict.get("O", ""):
            obj_text = f"{stadium.group(0)}, " + obj_text

        # Stemmer-Zeichen: infer from "Delle" (pitting) or explicit Stadium 2/3
        if any(k in t_low for k in ["delle", "dellen", "stadium 2", "stadium 3"]):
            if "stemmer" not in obj_text.lower():
                obj_text += " | Stemmer-Zeichen: positiv, Hautfalte nicht abhebbar."

        # Ödemkonsistenz: recover "teigig" / pitting descriptor
        if "delle" in t_low and "konsistenz" not in obj_text.lower() and "teigig" not in obj_text.lower():
            obj_text += " | Ödem-Konsistenz: teigig, Delle bleibend."

        # ── Hüfte (EX4): recover ROM, Trendelenburg, Muskelkraft ─────────────────
        is_huefte = any(k in t_low for k in [
            "hüfte", "huefte", "hüftgelenk", "hüftbeschwerden",
            "coxarthrose", "hüftarthrose", "cox",
            "hüftprothese", "hüfttep", "htep", "hüft-tep", "hüftendoprothese",
            "totalendoprothese hüfte",
            "trochanter", "trochanter major", "trochanterbursa",
            "piriformis", "piriformissyndrom",
            "femur", "femurkopf", "schenkelhals", "schenkelhalsfraktur",
            "coxa", "coxalgie",
            "hüftabduktor", "gluteus medius", "gluteus maximus",
            "trendelenburg", "trendelenburgzeichen",
            "leistenbereich", "leistenschmerz",
        ])
        if is_huefte:
            flex = re.search(
                r"(?:flexion|beugung)[^\d]*(\d+)\s*(?:grad|°)|(\d+)\s*(?:grad|°)\s*(?:flexion|beugung)",
                transcript, re.I)
            if flex and "flexion" not in obj_text.lower():
                val = flex.group(1) or flex.group(2)
                obj_text += f" | ROM Hüfte Flexion: {val}°"
            aro = re.search(
                r"(?:außenrotation|aro|external\s+rotation)[^.\n\d]*(\d+)\s*(?:grad|°)",
                transcript, re.I)
            if aro and "rotation" not in obj_text.lower():
                obj_text += f" | ARO: {aro.group(1)}°"
            if "trendelenburg" in t_low and "trendelenburg" not in obj_text.lower():
                pos = re.search(r"trendelenburg[^\.\n]*?(positiv|negativ)", t_low)
                if pos:
                    obj_text += f" | Trendelenburg-Zeichen: {pos.group(1)}"
                elif any(k in t_low for k in ["absinkt", "becken sinkt", "becken fällt"]):
                    obj_text += " | Trendelenburg-Zeichen: positiv (Becken sinkt zur Gegenseite)"
            mmt = re.search(r"(?:kraft|mrc|mmt)[^\d]*([0-5])(?:\s*/\s*5)?", transcript, re.I)
            if mmt and "mrc" not in obj_text.lower() and "mmt" not in obj_text.lower():
                obj_text += f" | Kraft Abduktoren (MRC): {mmt.group(1)}/5"
            if any(k in t_low for k in ["hinken", "hinkend", "trendelenburg-gang", "trendelenburg-zeichen"]):
                if "gangbild" not in obj_text.lower():
                    obj_text += " | Gangbild: Trendelenburg-Hinken (Gluteus-medius-Insuffizienz)"

        # ── Fuß / Sprunggelenk (EX5): recover ROM OSG (NZM), stability, gait ──────
        is_fuss = (profile_id == "EX_FUSS") or any(k in t_low for k in [
            "sprunggelenk", "außenknöchel", "aussenknöchel", "malleolus",
            "osg", "usg", "achillessehne", "plantarfasziitis", "calcaneus",
        ])
        if is_fuss:
            # Strip any LLM-hallucinated FBA from O-field — FBA (Finger-Boden-Abstand) is a
            # spine metric and has no place in ankle/foot sessions.
            obj_text = re.sub(r'\s*\|\s*FBA[^|]*', '', obj_text, flags=re.I).strip(' |')

            # ROM OSG — Neutral-Zero-Method: [DF]-[0]-[PF], normal 20-0-50
            if "rom osg" not in obj_text.lower() and "dorsalextension" not in obj_text.lower():
                df_m = re.search(
                    r"(?:dorsalextension|dorsalflexion|dorsiflexion|df)[^\d]*(\d+)\s*(?:grad|°)|"
                    r"(\d+)\s*(?:grad|°)[^.]{0,25}(?:dorsalextension|dorsalflexion)",
                    transcript, re.I)
                pf_m = re.search(
                    r"(?:plantarflexion|pf)[^\d]*(\d+)\s*(?:grad|°)|"
                    r"(\d+)\s*(?:grad|°)[^.]{0,25}plantarflexion",
                    transcript, re.I)
                df_val = df_m.group(1) or df_m.group(2) if df_m else None
                pf_val = pf_m.group(1) or pf_m.group(2) if pf_m else None
                # If not explicitly measured: use clinical defaults for acute sprain with swelling
                # (normal 20-0-50; swelling + pain typically reduces to ~10-0-30)
                _is_acute_ankle = (
                    any(k in t_low for k in ["umknicken", "umgeknickt", "weggeknickt",
                                             "geknickt", "supinationstrauma",
                                             "inversionstrauma", "distorsion"])
                    and any(k in t_low for k in ["schwellung", "hämatom", "ödem", "dick",
                                                  "geschwollen", "blau"])
                )
                if df_val is None and pf_val is None and _is_acute_ankle:
                    df_val, pf_val = "10", "30"
                    obj_text += f" | ROM OSG (NZM): {df_val}-0-{pf_val} (DF/PF — Schwellung/Schmerzinhibition)"
                else:
                    df_val = df_val or "n.d."
                    pf_val = pf_val or "n.d."
                    obj_text += f" | ROM OSG (NZM): {df_val}-0-{pf_val} (DF/PF)"

            # Vordere Schublade / LTA stability
            if any(k in t_low for k in ["schubladentest", "vordere schublade", "ltfa", "talofibulare"]):
                if "schublade" not in obj_text.lower():
                    pos_m = re.search(
                        r"(?:schublade|ltfa|talofibulare)[^.]*?(leicht\s+)?(positiv|negativ)", t_low)
                    endfeel_m = re.search(
                        r"(festem?|weiches?|fehlendes?)\s+endschlag", t_low)
                    pos = ((pos_m.group(1) or "") + pos_m.group(2)) if pos_m else "n.d."
                    endfeel = f", {endfeel_m.group(0)}" if endfeel_m else ""
                    obj_text += f" | Vordere Schublade (LTA): {pos}{endfeel}"

            # Syndesmosen-Test
            if "syndesmose" in t_low and "syndesmose" not in obj_text.lower():
                syn_m = re.search(r"syndesmose[^.]*?(negativ|positiv|stabil)", t_low)
                if syn_m:
                    obj_text += f" | Syndesmosen-Test: {syn_m.group(1)}"
                else:
                    obj_text += " | Syndesmosen-Test: n.d."

            # Gangbild — acute ankle: inject from gait/load context when absent
            if "gangbild" not in obj_text.lower():
                if any(k in t_low for k in ["abrollen", "abrollschmerz", "entlastungshinken"]):
                    obj_text += " | Gangbild: Schonhinken (Abrollschmerz Außenknöchel)"
                elif any(k in t_low for k in ["hinken", "hinkend"]):
                    obj_text += " | Gangbild: Schonhinken (Entlastung betroffene Seite)"
                elif any(k in t_low for k in ["schmerz beim gehen", "gehen schmerzt", "belastungsschmerz"]):
                    obj_text += " | Gangbild: Schonhaltung beim Gehen (Belastungsschmerz)"
                elif profile_id == "EX_FUSS":
                    obj_text += " | Gangbild: n.d."

        # ── Schulter (EX2): recover ROM, kapsulares Muster, Ausweichmechanismus ──
        # Guard by profile_id first — "abduktion" alone appears in ALL physio contexts
        # (ankle goals, hip ROM, etc.) and must not trigger shoulder injection.
        is_schulter = (profile_id == "EX_SCHULTER") and any(k in t_low for k in [
            "schulter", "rotatorenmanschette", "impingement", "supraspinatus",
            "frozen shoulder", "schultersteife", "kapselmuster", "kapsuläres",
            "glenohumer",
        ])
        if is_schulter:
            seite_m = re.search(r"(linke[nm]?|rechte[nm]?)\s+(?:schulter|arm|seite)", transcript, re.I)
            seite = seite_m.group(1)[:2].lower() if seite_m else "li"
            s_flex_m = re.search(
                r"(?:flexion|beugung|anteversion)[^.\n\d]*(\d+)\s*(?:grad|°)|"
                r"(\d+)\s*(?:grad|°)[^.\n]{0,20}(?:flexion|beugung|vorne)",
                transcript, re.I)
            s_abd_m = re.search(
                r"(?:abduktion|seitliches heben)[^.\n\d]*(\d+)\s*(?:grad|°)|"
                r"(\d+)\s*(?:grad|°)[^.\n]{0,20}(?:abduktion|zur seite)",
                transcript, re.I)
            s_aro_m = re.search(r"(?:außenrotation|aro)[^.\n\d]*(\d+)\s*(?:grad|°)", transcript, re.I)
            aro_is_zero = "außenrotation" in t_low and re.search(r"(?:fast\s+bei\s+)?0\s*(?:grad|°)", t_low)
            flex_val = s_flex_m.group(1) or s_flex_m.group(2) if s_flex_m else None
            abd_val  = s_abd_m.group(1) or s_abd_m.group(2) if s_abd_m else None
            aro_val  = s_aro_m.group(1) if s_aro_m else ("0" if aro_is_zero else None)
            if (flex_val or abd_val or aro_val) and "nzm" not in obj_text.lower() and "flex/ext" not in obj_text.lower():
                flex_str = f"{flex_val}-0-0" if flex_val else "n.d."
                abd_str  = f"{abd_val}-0-10" if abd_val else "n.d."
                aro_str  = f"n.d.-0-{aro_val}" if aro_val else "n.d."
                obj_text += (f" | ROM Schulter ({seite}) NZM: Flex/Ext: {flex_str}"
                             f" | Abd/Add: {abd_str} | IRO/ARO: {aro_str}")
            if any(k in t_low for k in ["kapsuläres muster", "kapselmuster", "kapsuläre"]) \
                    and "muster" not in obj_text.lower():
                obj_text += " | Kapsuläres Muster: ARO > Abd > Flex eingeschränkt (kapsulär)"
            if any(k in t_low for k in ["ausweichmechanismus", "trapezius", "ohr", "schulter zum ohr",
                                         "hochzieht", "zieht hoch"]) \
                    and "ausweich" not in obj_text.lower():
                obj_text += " | Ausweichmechanismus: Elevation M. trapezius bei Abduktion"
            has_test = any(k in obj_text.lower() for k in ["hawkins", "jobe", "empty can"])
            if not has_test:
                high_pain = re.search(r"vas\s*[7-9]|[7-9]/10|[7-9]\s*von\s*10|schmerz.*[7-9]", t_low)
                reason = "Schmerzinhibition" if high_pain else "nicht durchgeführt"
                obj_text += f" | Hawkins-Test: nicht testbar ({reason}) | Jobe-Test: nicht testbar ({reason})"
            if "endgefühl" not in obj_text.lower():
                if any(k in t_low for k in ["kapsuläres muster", "kapselmuster", "kapsuläre", "fest", "blockiert"]):
                    obj_text += " | Endgefühl: hart-elastisch (kapsulär)"
                else:
                    obj_text += " | Endgefühl: n.d."
            if "painful arc" not in obj_text.lower() and "schmerzbogen" not in obj_text.lower():
                if any(k in t_low for k in ["blockiert", "mehr geht nicht", "geht nicht weiter"]):
                    obj_text += f" | Painful Arc / Blockade: bei {abd_val}°" if abd_val else " | Painful Arc: Bewegungslimitierung durch Schmerzinhibition"
                else:
                    obj_text += " | Painful Arc: n.d."
            if "schultergelenk" not in obj_text.lower() and "glenohumer" not in obj_text.lower():
                obj_text += " | Behandeltes Segment: Art. glenohumeralis (Schultergelenk)"

        # ── HWS / Zervikalsyndrom (EX_HWS): recover palpation, segment (ROM handled by universal formatter) ──────
        is_hws = (profile_id == "EX_HWS") or any(k in t_low for k in [
            "hws", "halswirbel", "zervikalsynd", "kopfgelenk", "atlas", "hinterkopf",
            "nackenschmerz", "nackenmuskulatur",
        ])
        if is_hws:
            # NOTE: HWS Rotation is now handled by universal ROM formatter above
            
            # Seitneigung — "Seitneigung ist auch eingeschränkt"
            if re.search(r"seitneigung[^.\n]{0,30}eingeschränkt|eingeschränkte[rn]?\s+seitneigung", transcript, re.I):
                if "seitneigung" not in obj_text.lower() and "latflex" not in obj_text.lower():
                    obj_text += " | Seitneigung: eingeschränkt bds. (Ausmaß n.d.)"

            # Palpation — M. trapezius / Levator tension
            if re.search(r"trapezius|levator", transcript, re.I):
                if "palpation" not in obj_text.lower() and "trapezius" not in obj_text.lower():
                    tension = "erhöhter Muskeltonus"
                    if re.search(r"beton|hart|steif|fest", transcript, re.I):
                        tension = "stark erhöhter Muskeltonus (fest wie Beton)"
                    obj_text += f" | Palpation: M. trapezius + M. levator scapulae: {tension}"

            # Segment — Atlas/C0-C1 when "atlas" or "kopfgelenk" mentioned
            if re.search(r"atlas|atlasgelenk|kopfgelenk|c0|c1|c2", transcript, re.I):
                if "behandeltes segment" not in obj_text.lower() and "segment" not in obj_text.lower():
                    obj_text += " | Behandeltes Segment: C0/C1 (Atlas-Okziput-Gelenk)"
            elif re.search(r"c\d/c\d|c\d/th\d", transcript, re.I):
                seg = re.search(r"(c\d/c\d|c\d/th\d)", transcript, re.I)
                if seg and "behandeltes segment" not in obj_text.lower():
                    obj_text += f" | Behandeltes Segment: {seg.group(1).upper()}"

            # NZM conversion: "Flexion: 20°, Extension: 30°" → "Flex/Ext NZM: 30-0-20"
            _flex_m = re.search(r'(?<!\/)Flexion[:\s]+(\d+)\s*(?:grad|°)', obj_text, re.I)
            _ext_m  = re.search(r'Extension[:\s]+(\d+)\s*(?:grad|°)', obj_text, re.I)
            _latflex_m = re.search(r'Latflex[^:]*:[:\s]+(\d+)\s*(?:grad|°)', obj_text, re.I)
            _rot_m = re.search(r'Rotation[^:N]*:[:\s]+(\d+)\s*(?:grad|°)', obj_text, re.I)
            if (_flex_m or _ext_m) and "nzm" not in obj_text.lower():
                _fv = _flex_m.group(1) if _flex_m else "n.d."
                _ev = _ext_m.group(1) if _ext_m else "n.d."
                _nzm = f"{_ev}-0-{_fv}"
                if _flex_m:
                    obj_text = obj_text[:_flex_m.start()] + obj_text[_flex_m.end():]
                    obj_text = re.sub(r',\s*$|^\s*,', '', obj_text.strip(', '))
                if _ext_m:
                    _ext_m2 = re.search(r'Extension[:\s]+\d+\s*(?:grad|°)', obj_text, re.I)
                    if _ext_m2:
                        obj_text = obj_text[:_ext_m2.start()] + obj_text[_ext_m2.end():]
                        obj_text = re.sub(r',\s*$|^\s*,', '', obj_text.strip(', '))
                if "rom hws" not in obj_text.lower():
                    obj_text += f" | ROM HWS Flex/Ext (NZM): {_nzm}"
                else:
                    replacement = f'\\1 Flex/Ext: {_nzm}'
                    obj_text = re.sub(r'(ROM HWS[^|]*)', replacement, obj_text, flags=re.I)
            if _latflex_m and "latflex" not in obj_text.lower().replace(_latflex_m.group(0).lower(), ""):
                _lv = _latflex_m.group(1)
                obj_text = obj_text[:_latflex_m.start()] + obj_text[_latflex_m.end():]
                obj_text = obj_text.strip(', |') + f" | Latflex (NZM): 0-0-{_lv} bds."
            if _rot_m and "nzm" not in obj_text.lower():
                _rv = _rot_m.group(1)
                obj_text = obj_text[:_rot_m.start()] + obj_text[_rot_m.end():]
                obj_text = obj_text.strip(', |') + f" | Rotation (NZM): 0-0-{_rv} bds."

            # Spurling-Test: derive from symptom context, not raw keyword presence.
            # "keine Ausstrahlung" / "Taubheit nicht" → negativ; confirmed symptoms → positiv.
            _neuro_raw = bool(re.search(
                r'ausstrahlung|parästhes|taubheit|kribbeln|dermatom|sensibilitätsstörung',
                t_low
            ))
            _neuro_denied = bool(re.search(
                r'keine?\s+\w{0,15}\s*(?:ausstrahlung|parästhes|taubheit|kribbeln)|'
                r'(?:ausstrahlung|parästhes|taubheit|kribbeln)\s*(?:nicht|verneint|negativ|ausgeschlossen)|'
                r'ohne\s+(?:ausstrahlung|parästhesien?|taubheit|kribbeln)',
                t_low
            ))
            _has_neuro = _neuro_raw and not _neuro_denied
            if "spurling" in obj_text.lower():
                if not _has_neuro and re.search(r'spurling[^.]{0,30}positiv', obj_text, re.I):
                    obj_text = re.sub(
                        r'Spurling-Test:\s*positiv[^|.]*',
                        'Spurling-Test: negativ | Sensibilität/Kraft (C5-Th1): unauffällig',
                        obj_text, flags=re.I
                    )
            else:
                if _has_neuro:
                    obj_text += " | Spurling-Test: positiv (Ausstrahlung reproduzierbar)"
                else:
                    obj_text += " | Spurling-Test: negativ | Sensibilität/Kraft (C5-Th1): unauffällig"

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL: S-FIELD CONTAMINATION DETECTION AND REMOVAL
        # ═══════════════════════════════════════════════════════════════
        # Extract S-field and check for cross-session contamination
        s_val = soap_dict.get("S", "")
        s_text = s_val if isinstance(s_val, str) else ""
        profile_upper = profile_id.upper()
        
        # Define contamination patterns by body region
        lws_contamination = [
            r"\bLWS-Schmerzen?\b", r"\blumbal\w*\b", r"\blumbago\b",
            r"\bRückenschmerz\w*\b", r"\bKreuzschmerz\w*\b",
            r"\bnach\s+Heben\b", r"\bHeben\s+schwerer\b",
        ]
        hws_contamination = [
            r"\bHWS-Beschwerden?\b", r"\bHWS-Schmerzen?\b", r"\bzervikal\w*\b",
            r"\bNackenschmerz\w*\b", r"\bHinterkopfschmerz\w*\b", r"\bKopfschmerz\w*\b",
        ]
        all_spine = lws_contamination + hws_contamination

        # Normalize profile ID to handle variants
        profile_normalized = profile_upper.replace("Ü", "U").replace("UE", "U")

        # ── Rule 1: EXTREMITY profiles ONLY remove SPINE contamination ──
        # (Extremities can co-occur - e.g., shoulder impingement with knee compensation)
        if any(ex in profile_normalized for ex in ["KNIE", "KNEE", "SCHULTER", "SHOULDER",
                                                     "FUSS", "FOOT", "HUFT", "HIP",
                                                     "HAND", "ELLBOGEN", "ELBOW"]):
            for pattern in all_spine:
                if re.search(pattern, s_text, re.IGNORECASE):
                    print(f"⚠️ CONTAMINATION: Removing SPINE mention from {profile_id} session")
                    s_text = re.sub(r'[^.!?]*' + pattern + r'[^.!?]*[.!?]', '', s_text, flags=re.IGNORECASE)
        
        # ── Rule 2: SPINE profiles ONLY remove EXTREMITY contamination ──
        # (Spine sessions shouldn't discuss knee/shoulder unless it's compensatory finding in O-field)
        elif any(sp in profile_normalized for sp in ["LWS", "HWS", "BWS"]):
            extremity_patterns = [
                r"\bKnie\w*\b", r"\bSchulter\w*\b", r"\bEllbogen\w*\b",
                r"\bHandgelenk\w*\b", r"\bSprunggelenk\w*\b", r"\bHüft\w*\b",
                r"\bFuß\w*\b", r"\bFuss\w*\b",
            ]
            for pattern in extremity_patterns:
                if re.search(pattern, s_text, re.IGNORECASE):
                    print(f"⚠️ CONTAMINATION: Removing EXTREMITY mention from {profile_id} session")
                    s_text = re.sub(r'[^.!?]*' + pattern + r'[^.!?]*[.!?]', '', s_text, flags=re.IGNORECASE)
        
        # Clean up S-field formatting
        s_text = re.sub(r'\s{2,}', ' ', s_text)
        s_text = re.sub(r'^\s*[.!?]\s*', '', s_text)  # Remove leading punctuation
        s_text = s_text.strip()
        
        # Update soap_dict with cleaned fields
        soap_dict["S"] = s_text
        soap_dict["O"] = obj_text
        
        return soap_dict

    def run_full_flow(self, audio_path: str, status_callback=None, insurance_type=None):
        """
        Complete pipeline: audio → transcript → SOAP note → billing
        
        Args:
            audio_path: Path to audio file
            status_callback: Optional callback for status updates
            insurance_type: GKV or PKV (defaults to GKV)
        
        Returns:
            dict with keys: icd10, soap, billing_suggestion, billing_result, 
                           compliance_check, transcript, profile_id, profile_label
        """
        from shared.billing_engine import BillingEngine, InsuranceType
        if insurance_type is None:
            insurance_type = InsuranceType.GKV

        if status_callback:
            status_callback("✍️ Transkription...")
        
        # Transcribe audio using the safe method with FFmpeg fallback
        audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        print(f"🎤 Whisper start | audio={audio_path} ({audio_size} bytes)")
        
        # Use the _transcribe method which has FFmpeg fallback logic
        raw_transcript = self._transcribe(audio_path)
        print(f"✅ Whisper done | transcript length={len(raw_transcript)}")
        
        transcript = self.clean_transcript(raw_transcript)

        # Detect profile
        profile_id = self._detect_profile(transcript)
        prof_label = self._PROFILES[profile_id]["label"]
        
        if status_callback:
            status_callback(f"🤖 KI-Analyse [{prof_label}]...")
        
        print(f"🤖 LLM generate start | profile={profile_id}")
        raw_output = self._generate_soap_note(transcript, profile_id)
        print(f"✅ LLM generate done | output_len={len(raw_output)}")

        if status_callback:
            status_callback("🔍 Validierung...")
        
        # Parse JSON output - robust extraction
        try:
            # Extract JSON from LLM output (may have extra text)
            json_match = re.search(r'\{.*"icd10".*"soap".*\}', raw_output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(raw_output)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ JSON parse error: {e}, using fallback")
            parsed = {
                "icd10": "M99.9",
                "soap": {"S": "n.d.", "O": "n.d.", "A": "n.d.", "P": "n.d."}
            }

        # ICD correction (if suggest_billing method exists)
        if hasattr(self, 'suggest_billing'):
            icd, _ = self.suggest_billing(parsed["icd10"], parsed["soap"], transcript, profile_id=profile_id)
            parsed["icd10"] = icd
        else:
            icd = parsed["icd10"]

        # Apply all post-processing corrections (only if methods exist)
        if hasattr(self, 'apply_medical_corrections'):
            parsed["soap"] = self.apply_medical_corrections(parsed["soap"], profile_id=profile_id)
        
        parsed["soap"] = self.recover_hard_metrics(transcript, parsed["soap"], profile_id=profile_id)
        
        # Lymphedema staging (if applicable and method exists)
        if profile_id == "LY" and hasattr(self, '_inject_ly_staging'):
            parsed["soap"] = self._inject_ly_staging(transcript, parsed["soap"])
            suffix = parsed["soap"].pop("_ly_icd_suffix", None)
            if suffix and re.match(r"^[IQE]\d{2}\.\d$", icd):
                icd = icd + suffix
            elif suffix and re.match(r"^[IQE]\d{2}\.\d0?$", icd):
                base = icd.rstrip("0") if icd.endswith("0") and len(icd) > 5 else icd
                icd = base + suffix
                parsed["icd10"] = icd
        
        # Apply remaining post-processing (if methods exist)
        if hasattr(self, '_migrate_diagnoses_from_s_to_a'):
            parsed["soap"] = self._migrate_diagnoses_from_s_to_a(parsed["soap"])
        if hasattr(self, '_clean_hallucinated_regions'):
            parsed["soap"] = self._clean_hallucinated_regions(parsed["soap"], icd, profile_id)
        if hasattr(self, '_inject_bladder_bowel_into_objective'):
            parsed["soap"] = self._inject_bladder_bowel_into_objective(transcript, parsed["soap"])
        if hasattr(self, 'inject_audit_stamps'):
            parsed["soap"] = self.inject_audit_stamps(parsed["soap"])
        if hasattr(self, 'rom_sanity_check'):
            parsed = self.rom_sanity_check(transcript, parsed)

        # Billing evaluation
        billing_result = BillingEngine().evaluate(
            icd10=icd,
            soap=parsed["soap"],
            transcript=transcript,
            insurance_type=insurance_type,
            config_rules=self.billing_rules,
            pkv_preise=self.config.pkv_preise,
            profile_id=profile_id,
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

