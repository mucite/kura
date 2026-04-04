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
                # Foot / ankle anatomy
                "fuß", "fuss", "sprunggelenk",
                "osg", "oberes sprunggelenk",
                "usg", "unteres sprunggelenk",
                "außenknöchel", "aussenknöchel", "innenknöchel",
                "malleolus", "malleolus lateralis", "malleolus medialis",
                "knöchel",
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
                # Symptoms / mechanisms
                "fersenschmerz", "ferse",
                "umknicken", "umgeknickt", "umgeknickte",
                "supinationstrauma", "inversionstrauma", "inversionsdistorsion",
                "distorsion sprunggelenk", "distorsion fuß",
                # Tests specific to ankle
                "schubladentest", "vordere schublade", "talarneigung",
                "thompsons test", "wadenkompression",
                # Treatment context
                "lymphtape fuß", "aircast", "knöchelschiene",
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
        style_injection = f"\nBEVORZUGTE CODES DES THERAPEUTEN:\n{learning_notes}\n" if learning_notes else ""
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
10. VERLAUFSDOKUMENTATION: Falls der Therapeut eine Veraenderung zum Vortermin erwaehnt (z.B. "war letzte Woche besser", "VAS gestern 8", "letzte Sitzung noch 7/10"), schreibe den Verlauf direkt nach dem aktuellen VAS-Wert im S-Feld: "VAS 5/10 (Vorsitzung: 8/10, Δ: -3)". Dies ist §106b-Pflicht: Pruefer erwarten messbaren Therapiefortschritt je Sitzung.
11. PROFIL-PARAMETER exakt im O-Feld (als Zahlenwerte, niemals als Prosa-Zusammenfassung): KGG/MTT → Geraet + Last (kg) + Wdh x Saetze; ELEKTRO → Stromform (TENS/IFC/Galvano) + Frequenz (Hz) + Intensitaet (mA) + Elektroden-Platzierung; THERMO/Fango → Modalitaet + Behandlungsregion + Temperatur (°C oder "angenehm warm"); BECKEN → Oxford-Skala (0-5) + Kontraktionsdauer (sek) + Serienzahl.
12. O-FELD NUR BEFUNDERGEBNISSE: Das O-Feld enthaelt AUSSCHLIESSLICH dokumentierte Messwerte und Testresultate — NIEMALS Therapeutenanweisungen ("Heben Sie...", "Legen Sie sich..."), Patientenanfragen ("Mehr geht nicht?") oder Behandlungsschritte ("Ich fixiere jetzt..."). ROM immer im Neutral-Null-Format: z.B. "ROM Schulter (li) NZM: Flex/Ext: 80-0-0 | Abd/Add: 45-0-10 | IRO/ARO: n.d.-0-0". Tests die nicht durchfuehrbar waren: "[Testname]: nicht testbar (Schmerzinhibition)" — nicht weglassen.
13. SMART-ZIEL ist ein FUTURE TARGET, NICHT der aktuelle Befund: "Ziel: Abduktion auf 60° steigern in 6 EH" — NIEMALS aktuelle Einschraenkungswerte als Ziel nennen (falsch: "Ziel: Abduktion bei 45°").
14. KÖRPERREGION-TREUE: S- und O-Feld dokumentieren AUSSCHLIESSLICH Beschwerden und Befunde des behandelten Körperbereichs ({prof["label"]}). Beschwerden aus anderen Körperregionen (z.B. Leiste/Knie/LWS bei Schulter-Profil; Schulter/HWS bei Hüft-Profil) werden NICHT in den Bericht aufgenommen — auch wenn sie im Transkript beiläufig erwähnt werden.
15. POST-OP vs. IDIOPATHISCH: M75.0 (Adhäsive Kapsulitis / Frozen Shoulder) ist eine idiopathische Erkrankung ohne chirurgischen Auslöser. Falls das Transkript "postoperativ" erwähnt UND die Diagnose M75.0 ist: Verwende stattdessen M75.5 (Periarthritis humeroscapularis) oder Z96.6 (Z.n. Schulter-OP) — kombiniere NIEMALS M75.0 mit einem postoperativen Kontext.
16. Jeden Befund und Test genau einmal im O-Feld dokumentieren.
17. O-Feld-Tests: nur echte klinische Untersuchungen (Schubladentest, Lasègue, ROM, Stemmer). Behandlungsschritte und Heimuebungen gehoeren ins P-Feld.

PROFIL-PFLICHTFELDER (diese Felder MUESSEN im O-Feld erscheinen):
{checklist}

SOAP-STRUKTUR — VOLLSTAENDIG AUSSCHREIBEN (kein Kurzhalten, kein Zusammenfassen):

S — Subjektiv (Patientenperspektive):
  • Hauptbeschwerde in den EIGENEN WORTEN des Patienten (direkte Zitate bevorzugt)
  • Schmerzlokalisation exakt (z.B. "{pain_ex}")
  • Schmerzcharakter (ziehend / brennend / stechend / drückend — was der Patient sagt)
  • VAS aktuell x/10; bei Aktivitaet / in Ruhe falls beides genannt
  • Dauer und Verlauf (seit wann, schlechter/besser, Verlauf zur Vorsitzung)
  • Ausloeser / Aggravation / Linderung (was hilft, was verschlimmert)
  • Funktionsziel des Patienten (was moechte er wieder koennen?)
  • Relevanter Kontext: OP-Datum / Wochen postoperativ / Hilfsmittel / Alltagssituation

O — Objektiv (Therapeutenbeobachtung — NUR Befunde, KEINE Interventionen):
  • Inspektion / Gangbild / Haltung (z.B. "{inspection_ex}")
  • Alle Messwerte numerisch: ROM in Grad (Neutral-Null), Kraft MRC 0-5, VAS bei Provokation
  • Alle klinischen Tests mit Ergebnis (z.B. "Trendelenburg-Zeichen: positiv rechts")
  • Palpationsbefund falls genannt (Druckschmerz, Spannung, Schwellung)
  • Profil-Pflichtfelder aus der Checkliste oben (alle mit Zahlenwerten oder "n.d.")
  • KEINE Behandlungsschritte, KEINE Wiederholungen/Saetze in O

A — Assessment (Klinische Einschaetzung des Therapeuten):
  • ICD-10-Code + Diagnosebezeichnung auf Deutsch
  • Klinische Begruendung (warum dieser Code, welche Befunde stuetzen ihn)
  • Differentialdiagnose falls klinisch relevant
  • Funktionsstatus / Stadium (z.B. "6 Wochen postoperativ, Reha-Phase 2")
  • Red-Flag-Ausschluss spezifisch fuer dieses Profil (z.B. "{red_flag_ex}")

P — Plan (Therapieplan dieser Sitzung + Folgeziel):
  • Heilmittel ({prof["label"]}) + konkrete Technik / Uebung heute durchgefuehrt
  • Dosierung / Parameter (Dauer, Wiederholungen, Sets, Widerstand — NUR wenn bekannt)
  • Heimuebungsprogramm falls besprochen
  • SMART-Ziel: spezifisch + messbar + mit Zeitrahmen (z.B. "{smart_goal_ex}")
  • Naechster Behandlungstermin / Frequenz
{kruecken_line}
  | Behandler: n.d.

JSON-OUTPUT (alle Felder Pflicht, auch wenn "n.d." — VOLLSTAENDIGE Saetze, keine Stichwortlisten):
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
            output = self.llm(prompt, max_tokens=2800)
            raw = output["choices"][0]["text"]
        except Exception as e:
            print(f"❌ LLM call failed with: {type(e).__name__}: {e}")
            raw = '{"icd10": "M99.9", "soap": {"S": "n.d.", "O": "n.d.", "A": "n.d. | Red Flags klinisch ausgeschlossen.", "P": "n.d."}}'

        return "{" + raw if not raw.strip().startswith("{") else raw

    # ── Post-processing pipeline ───────────────────────────────────────────────

    def recover_hard_metrics(self, transcript: str, soap_dict: dict, profile_id: str = "KG") -> dict:
        """Safety net: if the therapist SAID it, it MUST appear in O."""
        obj_val = soap_dict.get("O", "")
        obj_text = obj_val if isinstance(obj_val, str) else ""
        t_low = transcript.lower() if isinstance(transcript, str) else ""

        schober = re.search(r"Schober.*?(\d+)\s*(?:zu|bis|-)\s*(\d+)", transcript, re.I)
        if schober and "Schober" not in obj_text:
            obj_text += f" | Schober-Zeichen: {schober.group(1)} - {schober.group(2)}"

        # FBA (Finger-Boden-Abstand) — LWS/spine sessions ONLY.
        # Injecting FBA into extremity sessions (EX_FUSS, EX_KNIE, etc.) causes the LLM
        # to confuse "FBA" with ankle abbreviations ("Fuß-Band-Außen") in the S-field.
        _is_spine_session = profile_id in ("EX_LWS", "MT", "EX_HWS") or any(
            k in t_low for k in ["lws", "lumbal", "isg", "iliosakral", "kreuzschmerz", "bandscheib"])
        fba = re.search(r"(?:finger.boden|fba)[^\d]*(\d+)\s*cm", transcript, re.I)
        if _is_spine_session and fba and "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
            obj_text += f" | FBA: {fba.group(1)} cm"
        elif _is_spine_session and re.search(r"finger.boden|fba", transcript, re.I):
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
            "hüfte", "hüftgelenk", "hüftabduktor", "coxarthrose", "hüftprothese",
            "trochanter", "gluteus", "trendelenburg", "hüft-tep", "htep",
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

        # ── HWS / Zervikalsyndrom (EX_HWS): recover ROM, palpation, segment ──────
        is_hws = (profile_id == "EX_HWS") or any(k in t_low for k in [
            "hws", "halswirbel", "zervikalsynd", "kopfgelenk", "atlas", "hinterkopf",
            "nackenschmerz", "nackenmuskulatur",
        ])
        if is_hws:
            # HWS Rotation — "Rotation nach links ist bei etwa 45 Grad"
            rot_li = re.search(
                r"rotation\s+(?:nach\s+)?links[^.\n\d]*(\d+)\s*(?:grad|°)|"
                r"(\d+)\s*(?:grad|°)[^.\n]{0,30}rotation\s+(?:nach\s+)?links",
                transcript, re.I)
            rot_re = re.search(
                r"rotation\s+(?:nach\s+)?rechts[^.\n\d]*(\d+)\s*(?:grad|°)|"
                r"(\d+)\s*(?:grad|°)[^.\n]{0,30}rotation\s+(?:nach\s+)?rechts",
                transcript, re.I)
            rot_li_val = (rot_li.group(1) or rot_li.group(2)) if rot_li else None
            rot_re_val = (rot_re.group(1) or rot_re.group(2)) if rot_re else None

            # "blockiert" / "schluss" / "eingeschränkt" at a given degree
            block_deg = re.search(
                r"(\d+)\s*(?:grad|°)[^.\n]{0,40}(?:blockier|schluss|eingeschränkt|geht nicht)|"
                r"(?:blockier|schluss|eingeschränkt)[^.\n]{0,40}(\d+)\s*(?:grad|°)",
                transcript, re.I)
            if block_deg and not rot_li_val and not rot_re_val:
                val = block_deg.group(1) or block_deg.group(2)
                rot_li_val = val  # assume left (blocked side mentioned first in transcript)

            if (rot_li_val or rot_re_val) and "rom hws" not in obj_text.lower() and "rotation" not in obj_text.lower():
                li_str = f"0-0-{rot_li_val}" if rot_li_val else "n.d."
                re_str = f"0-0-{rot_re_val}" if rot_re_val else "n.d."
                obj_text += f" | ROM HWS: Rotation li {li_str} / re {re_str} (NZM)"

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

            # Spurling-Test — if not documented, mark as nicht durchgeführt
            if "spurling" not in obj_text.lower():
                obj_text += " | Spurling-Test: nicht durchgeführt"

        # ── Krücke Seitenkontrolle ─────────────────────────────────────────────
        plan_text = soap_dict.get("P", "")
        kruecke_m = re.search(
            r"krücke[n]?\s+(?:auf\s+der\s+|an\s+der\s+)?(?:rechten?|linken?)\s+seite|"
            r"(?:rechten?|linken?)\s+krücke",
            transcript, re.I)
        if kruecke_m:
            kruecke_raw = kruecke_m.group(0).lower()
            kruecke_links = "links" in kruecke_raw
            affected_re = re.search(
                r"(?:schmerzen?|operation|operiert|tep|prothese|beschwerde)\w*\s+(?:im?|am?|der?|des?|auf\s+der)?\s*"
                r"(rechten?|linken?)\s*(?:bein|hüfte|seite|knie|schulter)?",
                transcript, re.I)
            if affected_re:
                affected_links = "link" in affected_re.group(1).lower()
                ipsilateral = (kruecke_links == affected_links)
                if ipsilateral:
                    side_label = "links" if affected_links else "rechts"
                    contra_label = "rechts" if affected_links else "links"
                    warning = (
                        f" ⚠️ KRÜCKEN-SEITE PRÜFEN: Krücke auf der betroffenen Seite ({side_label}) dokumentiert — "
                        f"korrekt ist KONTRALATERAL ({contra_label}), um die betroffene Hüfte zu entlasten."
                    )
                    soap_dict["P"] = plan_text + warning

        soap_dict["O"] = self._dedup_o_field(obj_text)

        # Deduplicate P-field — LLM loop artifacts produce repeated sentences
        p_raw = soap_dict.get("P", "")
        if p_raw:
            soap_dict["P"] = self._dedup_o_field(p_raw)

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

        # ── P-field sanitizer ─────────────────────────────────────────────────

        # 1. Krücken recommendation: only clinically valid for lower-limb profiles.
        _lower_limb_profiles = {"EX_KNIE", "EX_HUefte", "EX_HUFTE", "EX_FUSS", "GER", "POST_OP"}
        if profile_id not in _lower_limb_profiles:
            p_field = soap_dict.get("P", "")
            if isinstance(p_field, str):
                _p_no_kruecken = re.sub(
                    r'[\|,]?\s*Krücken[^|.\n]*(?=[|.]|$)',
                    '', p_field, flags=re.I
                ).strip(' |,.')
                if _p_no_kruecken != p_field:
                    soap_dict["P"] = re.sub(r'\s{2,}', ' ', _p_no_kruecken).strip()

        # 2. Cross-body SMART goal check — two layers:
        #    a) Profile-specific: known wrong-body anatomy in SMART goal
        #    b) Generic: "Ziel: ROM <word>" where <word> absent from all SOAP fields
        _SMART_FORBIDDEN: dict = {
            "EX_HWS":     ["hüftflex", "hüftext", "knieflex", "knieext",
                           "hüfte 0-", "knie 0-", "sprunggelenk"],
            "EX_SCHULTER": ["hüftflex", "hüftext", "knieflex", "hüfte 0-", "knie 0-"],
            "EX_FUSS":    ["hüftflex", "schulterabd", "hüfte 0-"],
            "EX_KNIE":    ["hüftflex", "schulterabd", "schulter 0-"],
        }
        p_field = soap_dict.get("P", "")
        if isinstance(p_field, str):
            p_low = p_field.lower()
            _forbidden_in_p = _SMART_FORBIDDEN.get(profile_id, [])
            if any(f in p_low for f in _forbidden_in_p):
                soap_dict["P"] = re.sub(
                    r'(?:SMART-)?Ziel\s*:?\s*[^|.\n]*(?:Hüftflex|Hüftext|Knieflex|Knieext|'
                    r'Hüfte\s+\d|Knie\s+\d|Sprunggelenk)[^|.\n]*',
                    'Ziel: n.d. — bitte profilkorrektes Funktionsziel ergaenzen',
                    p_field, flags=re.I
                )
                p_field = soap_dict.get("P", "")

            # Generic cross-domain check
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

    _NON_CLINICAL_TEST_RE = re.compile(
        r'^(?:montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag|'
        r'\d+[-–]\d+\s*wochen?|alles\s+klar|bis\s+\w+|verständnis|verstanden|'
        r'langsam\b|übung\s+für\s+heute|rückenlage[\s,].*(?:nase|locker)|'
        r'volle\s+beweglichkeit|lauf-abc\b|einbeinstand(?:\s|,|$)|lauftraining\b)',
        re.I
    )

    _RULE_TEXT_RE = re.compile(
        r'Keine\s+Wiederholungen(?:/Saetze)?|'
        r'Keine\s+Zeitangaben|Keine\s+Heimuebungen|Keine\s+Therapieschritte|'
        r'Keine\s+Patientendialog-Fragmente|Keine\s+Behandlungsschritte|'
        r'KEINE\s+WIEDERHOLUNGEN|KEINE\s+BEHANDLUNGSSCHRITTE',
        re.I
    )

    @staticmethod
    def _strip_rule_text(text: str) -> str:
        m = KuraEngine._RULE_TEXT_RE.search(text)
        if m:
            truncated = text[:m.start()].rstrip(', ')
            return truncated if truncated else text
        return text

    @staticmethod
    def _dedup_o_field(text: str) -> str:
        """Remove duplicate sentences from O-field and strip non-clinical Tests: entries."""
        if not text:
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen: set = set()
        result = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            key = re.sub(r'\s+', ' ', s.lower())
            if re.match(r'^Tests:\s+', s, re.I):
                test_body = s[s.index(':') + 1:].strip()
                test_name = test_body.split(',')[0].strip()
                if KuraEngine._NON_CLINICAL_TEST_RE.match(test_name):
                    continue
            if key not in seen:
                seen.add(key)
                result.append(s)
        return ' '.join(result)

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

    _PROFILE_FORBIDDEN_S: dict = {
        # Terms that should NEVER appear as the primary complaint in the S-field for this profile.
        # Rule: only forbid body regions with ZERO anatomical relationship.
        #   Shoulder ↔ HWS     → related (cervicobrachial) — do NOT forbid each other
        #   Hip     ↔ LWS      → related (hip-spine syndrome, L3–S1 radiculopathy) — do NOT forbid
        #   Hip     ↔ Knie     → related (hip-knee axis, Trendelenburg) — do NOT forbid
        "EX_SCHULTER": [
            # Shoulder has no anatomical relationship to hip, knee, or lumbar spine.
            # HWS is intentionally absent: cervicobrachial syndrome is the #1 differential.
            r"\bleiste\b", r"\boberschenkel\b", r"\bhüfte\b", r"\bhüftgelenk\b",
            r"\bknie\b", r"\bkniegelenk\b", r"\blendenwirbels[äa]ule\b", r"\blws\b",
            r"\bisg\b", r"\bsacroiliakal\b",
        ],
        "EX_KNIE": [
            # Knee has no relation to shoulder or cervical spine.
            # Hip excluded: hip-knee axis compensation is clinically routine.
            # LWS excluded: L3/L4 radiculopathy routinely refers into the knee.
            r"\bschulter\b", r"\brotatorenmanschette\b", r"\bhws\b", r"\bhalswirbel\b",
        ],
        "EX_LWS": [
            # Lumbar has no relation to shoulder.
            # HWS excluded: combined spondylosis (HWS + LWS) is common in older patients.
            # Hip and knee excluded: L3–S1 radiculopathy refers to both.
            r"\bschulter\b", r"\brotatorenmanschette\b", r"\bimpingement\b",
        ],
        "EX_HWS": [
            # Cervical has no relation to hip or knee.
            # Shoulder intentionally absent: cervicobrachial syndrome is the classic co-symptom.
            # LWS excluded: combined degenerative disease occurs in spondylosis patients.
            r"\bhüfte\b", r"\bhüftgelenk\b", r"\bcoxarthrose\b",
            r"\bknie\b", r"\bkniegelenk\b", r"\bgonarthrose\b",
        ],
        "EX_HUefte": [
            # Hip has no relation to shoulder or cervical spine.
            # Knee excluded: hip-knee axis compensation is clinically common.
            # LWS excluded: hip-spine syndrome — hip pain and LWS pain frequently co-occur.
            r"\bschulter\b", r"\brotatorenmanschette\b", r"\bhws\b", r"\bhalswirbel\b",
        ],
        "EX_HUFTE": [  # legacy spelling alias — keep in sync with EX_HUefte
            r"\bschulter\b", r"\brotatorenmanschette\b", r"\bhws\b", r"\bhalswirbel\b",
        ],
        "EX_FUSS": [
            # Ankle/foot has no relation to shoulder or cervical spine.
            # FBA (Finger-Boden-Abstand) is a lumbar spine metric — LLM confuses it with
            # ankle abbreviations ("Fuß-Band-Außen") when it appears in the O-field context.
            r"\bFBA\b", r"finger-boden",
            r"\bschulter\b", r"\bhws\b", r"\bhalswirbel\b",
        ],
    }

    def _clean_hallucinated_regions(self, soap: dict, icd: str, profile_id: str = "KG") -> dict:
        """
        Remove out-of-scope diagnosis terms from the A field and wrong anatomical
        region terms from the S-field. Only removes terms that are NOT negated.
        """
        # ── S-field: strip wrong body-region pain locations ───────────────────
        s_forbidden = self._PROFILE_FORBIDDEN_S.get(profile_id, [])
        if s_forbidden:
            s = soap.get("S", "")
            s_parts = re.split(r'(?<=[.!?|])\s*', s)
            s_clean = []
            for part in s_parts:
                part_stripped = part.strip()
                if not part_stripped:
                    continue
                removed = False
                for pattern in s_forbidden:
                    m = re.search(pattern, part_stripped, re.I)
                    if m:
                        before = part_stripped[:m.start()]
                        nearby = before.split()[-6:]
                        if not self._NEGATION_RE.search(" ".join(nearby)):
                            print(f"[SanityCheck] Removed off-region S term '{pattern}': {part_stripped[:60]}")
                            removed = True
                            break
                if not removed:
                    s_clean.append(part_stripped)
            soap["S"] = " ".join(s_clean).strip()

        # ── Post-op + M75.0 contradiction ────────────────────────────────────
        if profile_id == "EX_SCHULTER" and icd.startswith("M75.0"):
            s = soap.get("S", "")
            if re.search(r'postoperativ|post-op|\bop\b|\boperation\b|wochen\s+postop', s, re.I):
                soap["S"] = re.sub(
                    r'[^|.]*(?:postoperativ|post-op|\bop\b|\boperation\b|wochen\s+postop)[^|.]*[|.]?\s*',
                    '', s, flags=re.I).strip().strip('|').strip()
                print("[SanityCheck] Removed postoperativ from S (M75.0 is idiopathic)")

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

    def suggest_billing(self, icd10: str, soap: dict, transcript: str, profile_id: str = ""):
        codes = self.config.billing_codes
        t_low = transcript.lower() if isinstance(transcript, str) else ""

        plan_val = soap.get("P", "")
        plan_text = plan_val.lower() if isinstance(plan_val, str) else ""

        obj_val = soap.get("O", "")
        obj_text = obj_val.lower() if isinstance(obj_val, str) else ""

        full_text = f"{obj_text} {plan_text} {t_low}"

        is_neuro = any(k in full_text for k in ["bobath", "pnf", "neuro", "zns", "hemiparese", "ataxie", "spastik", "insult", "schlaganfall"])
        _ly_profile = profile_id in ("LY", "LY1", "") or not profile_id
        is_lymph = _ly_profile and any(
            k in full_text for k in ["lymphoedem", "lymphödem", "kpe", "entstauung", "stemmer", "lipödem"]
        )
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
                return res_icd, codes.get("KG_ZNS", "20511")
            return res_icd, codes.get("KG", "20501")

        if is_neuro:
            return res_icd, codes.get("KG_ZNS", "20511")
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
                    if isinstance(value, dict):
                        value = " | ".join(f"{k}: {v}" for k, v in value.items() if v)
                    elif not isinstance(value, str):
                        value = str(value) if value else "N/A"
                    # Flatten embedded JSON object string: {"key": "val"} | rest → key: val | rest
                    if isinstance(value, str) and value.strip().startswith("{"):
                        try:
                            import json as _json
                            _dec = _json.JSONDecoder()
                            _obj, _end = _dec.raw_decode(value.strip())
                            if isinstance(_obj, dict):
                                _flat = " | ".join(f"{k}: {v}" for k, v in _obj.items() if v)
                                _rest = value.strip()[_end:].strip().lstrip("|").strip()
                                value = (_flat + " | " + _rest) if _rest else _flat
                        except Exception:
                            pass
                    value = KuraEngine._strip_rule_text(value) if value else value
                    soap_clean[field] = value.strip() if value else "N/A"

                icd10_val = data.get("icd10", "M99.9")
                # Rescue ICD if LLM embedded it in A-field instead of top-level
                if icd10_val == "M99.9":
                    a_text = soap_clean.get("A", "")
                    _icd_rescue = re.search(r'\b([A-Z]\d{2})\s*\.?\s*(\d+)\b', a_text)
                    if _icd_rescue:
                        icd10_val = _icd_rescue.group(1) + "." + _icd_rescue.group(2)
                return {
                    "icd10": icd10_val,
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

            icd_match = re.search(r'"icd10"\s*:\s*"([A-Z]\d{2}[\.\d\s]*)"', text)
            icd = re.sub(r'\s', '', icd_match.group(1)) if icd_match else "M99.9"

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

        # Strategy 3: Last resort — use n.d. so recover_hard_metrics can still append
        print("⚠️ All JSON parsing failed, using n.d. fallback")
        return {
            "icd10": "M99.9",
            "soap": {
                "S": "n.d.",
                "O": "n.d.",
                "A": "n.d. | Red Flags klinisch ausgeschlossen.",
                "P": "n.d."
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

            if isinstance(value, dict):
                value = " | ".join(f"{k}: {v}" for k, v in value.items() if v)
            elif not isinstance(value, str):
                value = str(value) if value else ""

            # Flatten embedded JSON object string: {"key": "val"} | rest → key: val | rest
            if isinstance(value, str) and value.strip().startswith("{"):
                try:
                    import json as _json
                    _dec = _json.JSONDecoder()
                    _obj, _end = _dec.raw_decode(value.strip())
                    if isinstance(_obj, dict):
                        _flat = " | ".join(f"{k}: {v}" for k, v in _obj.items() if v)
                        _rest = value.strip()[_end:].strip().lstrip("|").strip()
                        value = (_flat + " | " + _rest) if _rest else _flat
                except Exception:
                    pass

            value = KuraEngine._strip_rule_text(value) if value else value
            if not value or value.strip() in ("N/A", "Fehler", "KI-Fehler", "Parsing-Fehler", "{}"):
                if field == "A":
                    value = f"{icd10} | Red Flags klinisch ausgeschlossen."
                else:
                    value = "n.d."

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
        icd, _ = self.suggest_billing(parsed["icd10"], parsed["soap"], transcript, profile_id=profile_id)
        parsed["icd10"] = icd

        parsed["soap"] = self.apply_medical_corrections(parsed["soap"])
        parsed["soap"] = self.recover_hard_metrics(transcript, parsed["soap"], profile_id=profile_id)
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
            profile_id=profile_id,
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
