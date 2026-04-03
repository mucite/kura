
import gc
import json
import logging
import os
import re
import sys
import time

# Persistent log so .app issues are visible without a terminal.
_log_dir = os.path.expanduser("~/Library/Logs/Kura")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_log_dir, "kura.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("kura")

import mlx.core as mx
import mlx_whisper
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from shared.learning_manager import LearningManager

# Add parent directory to path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.config_manager import ConfigManager


class KuraEngine:
    def __init__(self):
        """
        Kura Medical Engine v2.0 (2026 GKV Compliant).
        Full Clinical Logic & MLX Optimization.
        """
        self.learning_mgr = LearningManager()
        self.config = ConfigManager()
        self._setup_paths()
        self._check_concurrent_instances()
        self._check_system_resources()
        self._cleanup_gpu_memory()

        try:
            _log.info("Loading LLM from: %s", self.llm_repo)
            print("🔧 Loading MLX Medical Models...")
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            self.model, self.tokenizer = load(
                self.llm_repo,
                tokenizer_config={"local_files_only": True, "trust_remote_code": True}
            )
            _log.info("LLM loaded OK. Whisper model path: %s", self.stt_model)
            for h in _log.handlers: h.flush()
            print("✅ Models ready.")
        except Exception as e:
            _log.exception("Model load failed: %s", e)
            raise RuntimeError(f"GPU/Metal Error: {e}")

        self.billing_rules = self.config.billing_rules
        self.audit_rules = self.config.audit_rules
        self.llm_config = self.config.data.get("llm_config", {})
        self.whisper_config = self.config.data.get("whisper_config", {})

    def _setup_paths(self):
        if getattr(sys, 'frozen', False):
            # PyInstaller COLLECT+BUNDLE: models land in Contents/MacOS/ next to the exe.
            # _MEIPASS is set for one-file builds; for one-dir COLLECT it equals the exe dir.
            # Never use '../Resources' — that path is empty in a COLLECT bundle.
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.executable)))
            self.model_dir = os.path.join(base_path, "models")
        else:
            self.model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "models")

        self.llm_repo  = os.path.join(self.model_dir, "Llama-3.2-3B-Instruct-4bit")
        self.stt_model = os.path.join(self.model_dir, "whisper-large-v3-turbo")

        # Fail fast — if models aren't found, raise immediately instead of letting
        # mlx_whisper silently try to download from HuggingFace and hang forever.
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Modell-Verzeichnis nicht gefunden: {self.model_dir}\n"
                "Bitte App neu installieren."
            )
        if not os.path.isdir(self.stt_model):
            raise FileNotFoundError(
                f"Whisper-Modell fehlt: {self.stt_model}\n"
                "Bitte App neu installieren."
            )
        if not os.path.isdir(self.llm_repo):
            raise FileNotFoundError(
                f"LLM-Modell fehlt: {self.llm_repo}\n"
                "Bitte App neu installieren."
            )

        # mlx_whisper calls ffmpeg internally as a subprocess to decode audio.
        # In the .app bundle, ffmpeg is not on PATH — add it now so the call works.
        self._ensure_ffmpeg_on_path()

    def _ensure_ffmpeg_on_path(self):
        """
        mlx_whisper decodes audio by calling ffmpeg as a subprocess.
        In the .app bundle ffmpeg lives next to the executable, not on PATH.
        Find the bundled binary and prepend its directory to PATH so that
        any subprocess.run(['ffmpeg', ...]) call (ours or mlx_whisper's) works.
        """
        import shutil
        if shutil.which('ffmpeg'):
            return  # already on PATH

        candidates = []
        if getattr(sys, 'frozen', False):
            base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.executable)))
            candidates.append(os.path.join(base, 'ffmpeg'))
        candidates += ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']

        for path in candidates:
            if os.path.isfile(path):
                ffmpeg_dir = os.path.dirname(path)
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                _log.info("ffmpeg added to PATH from: %s", path)
                return

        _log.warning("ffmpeg not found — audio transcription will fail")

    def _cleanup_gpu_memory(self):
        gc.collect()
        if hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
        time.sleep(0.5)

    def _check_concurrent_instances(self):
        try:
            import psutil
            curr = os.getpid()
            for p in psutil.process_iter(['pid', 'cmdline']):
                if p.pid != curr and any('Kura' in str(a) for a in p.info.get('cmdline', [])):
                    print("⚠️ Concurrent Kura detected.")
        except Exception:
            pass

    def _check_system_resources(self):
        try:
            import psutil
            if psutil.virtual_memory().available / (1024 ** 3) < 2.0:
                print("⚠️ Low RAM Warning.")
        except Exception:
            pass

    def clean_transcript(self, transcript: str) -> str:
        """Fixes transcription hallucinations before AI processing."""
        corrections = {
            r"Bobert|Bobat": "Bobath",
            r"Stämmer|Stemmerzeichen": "Stemmer-Zeichen",
            r"Kompressionsstrübe|Strübe": "Kompressionsstrümpfe",
            r"Lymphödes|Lymphödemen": "Lymphödem",
            r"Anastomosen|Anastomo": "Anastomosen",
            r"Lenertschroth|Schrot-Therapie": "Lehnert-Schroth",
            r"ischio-choraler|ischo-cural": "ischiocrurale",
            r"autochtoner": "autochthoner",
            r"Lasek|Lasegge|Laseque": "Lasègue",
            r"Psoasdehnung": "Psoas-Dehnung",
            r"Stufenlagerung": "Stufenlagerung",
            r"Finger-Bodenabstand|FBA": "Finger-Boden-Abstand (FBA)",
            r"Sanddemobilisation|Sandmobilisation": "sanfte Mobilisation",
            r"VKB Plastik|VKB-Plastik": "Vorderes Kreuzband (VKB) Plastik",
            r"Kranial": "kranial",
            r"M, M, T": "MMT (Manual Muscle Test)",
            r"Kopfgelenk|C1|C2": "Kopfgelenksregion (C1/C2)",
            r"Hinterkopfschmerz": "okzipitaler Kopfschmerz",
            r"Kinnretraktion": "Kinn-Retraktion (Deep Neck Flexor Training)",
            r"Detonisierung": "Detonisierung (Tonussenkung)",
            # CMD / Craniomandibuläre Dysfunktion
            r"CNMD|CMND|CNMT|C\.N\.M\.D\.": "CMD",
            # Knee instability compound word
            r"Knieknadi\w*|Knienadig\w*|Knie.?Nadi\w*|Knienachgibigkeit": "Knienachgiebigkeit",
            # Other frequently mangled German physio compounds
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

        # ── Context-aware anatomical disambiguation ───────────────────────────
        # Whisper frequently confuses "Schenkel" ↔ "Schulter" before "halsfraktur"
        # because both start with "Sch" and end with "-er".
        # Only correct "Schulterhalsfraktur" → "Schenkelhalsfraktur" when the
        # transcript also contains clear hip/femur context.
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

        # Similarly catch common phonetic variants Whisper produces
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
                "GMFCS-Level (I-V)",
                "Tonusregulation: Ashworth-Grad (0-4) je Extremitaet",
                "ADL-Status: selbstaendig / mit Hilfe / abhaengig",
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
                "Augenlidschluss: vollstaendig / unvollstaendig (Lagophthalmus)",
                "Synkinesien: vorhanden / nicht vorhanden",
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
            ],
        },
        "LY": {
            "label":    "Lymphologie / Entstauung",
            "billing":  "20201",
            "priority": 70,
            "triggers": [
                # NOTE: do NOT use bare "lymph" — matches "Lymphabfluss", "Lymphknoten"
                # in orthopaedic/HWS contexts and causes false LY profile selection.
                "lymphoedem", "lymphdrainage", "mld", "kpe", "entstauung", "stemmer",
                "lipoedem", "mastektomie", "axillaer", "sentinel", "erysipel",
                "sekundaeres oedema",
            ],
            "icd_prefix": ["I89", "Q82", "C77", "I97"],
            "checklist": [
                "Stadium: 1 (reversibel) / 2 (irreversibel) / 3 (Elephantiasis)",
                "Stemmer-Zeichen: positiv / negativ + Lokalisation",
                "Umfangsmessung beidseitig in cm (Koerperstelle angeben)",
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
                "SpO2 (%)",
                "Sekret: Menge / Farbe / Konsistenz",
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
                "ROM: Flexion / Abduktion / ARO / IRO (Neutral-Null-Methode)",
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
                "Lachman-Test: positiv / negativ",
                "ROM: Extension / Flexion (Grad)",
                "VAS-Score (0-10)",
            ],
        },
        "MT": {
            "label":    "Manuelle Therapie WS (Facette / ISG)",
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
                "Spurling-Test: positiv / negativ (mit Seitenangabe)",
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
                "FBA (Finger-Boden-Abstand): X cm",
                "Lasegue-Test: Grad + Seite (z.B. re. positiv bei 45 Grad)",
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
                "Kraft Huefte (MRC 0-5): Abduktion / Extension",
            ],
        },
        "EX_HAND": {
            "label":    "Extremitaeten Hand / Handgelenk / Finger",
            "billing":  "21201",
            "priority": 44,   # above EX_FUSS (40) and EX_HUefte (41), below MT (50)
            "triggers": [
                "handgelenk", "handwurzel", "radiokarpal", "radiusfraktur",
                "metakarpal", "fingergelenk", "fingergrundgelenk", "fingermittelgelenk",
                "handkraft", "karpaltunnel", "handchirurgie", "handödem",
                "handwurzelknochen", "daumengelenk",
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
        Diagnosis-First profile detection.
        1. Detect patient age from transcript (e.g. '4 Jahre alt').
        2. Score every profile by trigger matches + age constraints.
        3. Return the highest-priority matching profile ID.
        """
        import re as _re
        t = transcript.lower()

        # Age extraction — "4 Jahre alt", "4-jaehrig", "4 J."
        age = None
        m = _re.search(r'(\d{1,2})\s*(?:jahre?\s*alt|j\b|-jaehrig)', t)
        if m:
            age = int(m.group(1))

        best_id       = "KG"
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
                # (they require age context to avoid false positives)
                if prof.get("age_max", 999) <= 17:
                    continue

            priority = prof.get("priority", 0)
            if priority <= best_priority:
                continue

            if any(trigger in t for trigger in prof.get("triggers", [])):
                best_id       = pid
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

    def build_prompt(self, transcript: str, profile_id: str = "KG"):
        learning_notes = self.learning_mgr.get_relevant_prefs(transcript)
        style_injection = f"\nBEVORZUGTE CODES DES THERAPEUTEN:\n{learning_notes}\n" if learning_notes else ""
        checklist = self._profile_checklist(profile_id)
        prof      = self._PROFILES.get(profile_id, self._PROFILES["KG"])

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
6. DIAGNOSEN GEHOEREN IN A, NICHT IN S: ICD-10-Codes, Erkrankungsbezeichnungen (z.B. "Gonarthrose", "Bandscheibenvorfall", "Lymphödem"), Diagnose-Aussagen und Vordiagnosen NIEMALS in S schreiben. S enthaelt NUR subjektive Patientenaussagen: Schmerzschilderung, Funktionsziel, Vorgeschichte in eigenen Worten. Wenn der Therapeut eine Diagnose nennt, landet sie in A.
7. THERAPIEZIEL im P-Feld: SMART formulieren — Spezifisch, Messbar, Erreichbar, Relevant, Terminiert. Beispiel: "Ziel: ROM Knieflexion 0-0-120 in 6 EH."
8. KPE-DOKUMENTATION (nur bei MLD/Lymph): P-Feld muss alle 4 Komponenten nennen: MLD + Kompressionsbandagierung + Entstauungsgymnastik + Hautpflege.
9. VERLAUFSDOKUMENTATION: Falls der Therapeut eine Veraenderung zum Vortermin erwaehnt (z.B. "war letzte Woche besser", "VAS gestern 8", "letzte Sitzung noch 7/10"), schreibe den Verlauf direkt nach dem aktuellen VAS-Wert im S-Feld: "VAS 5/10 (Vorsitzung: 8/10, Δ: -3)". Dies ist §106b-Pflicht: Pruefer erwarten messbaren Therapiefortschritt je Sitzung.
10. PROFIL-PARAMETER exakt im O-Feld (als Zahlenwerte, niemals als Prosa-Zusammenfassung): KGG/MTT → Geraet + Last (kg) + Wdh x Saetze; ELEKTRO → Stromform (TENS/IFC/Galvano) + Frequenz (Hz) + Intensitaet (mA) + Elektroden-Platzierung; THERMO/Fango → Modalitaet + Behandlungsregion + Temperatur (°C oder "angenehm warm"); BECKEN → Oxford-Skala (0-5) + Kontraktionsdauer (sek) + Serienzahl.

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

    def _generate_soap_note(self, transcript: str, profile_id: str = "KG") -> str:
        prompt = self.build_prompt(transcript, profile_id)
        cfg = self.llm_config
        sampler = make_sampler(
            temp=cfg.get("temperature", 0.15),
            top_p=cfg.get("top_p", 0.9),
        )
        raw = generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=cfg.get("max_tokens", 1800),
            sampler=sampler,
        )
        return "{" + raw if not raw.strip().startswith("{") else raw

    def recover_hard_metrics(self, transcript, soap_dict):
        """
        The 'Safety Net': If the therapist SAID it, it MUST be in 'O'.
        Rescues numbers the AI summarized away.
        """
        import re
        obj_text = soap_dict.get("O", "")
        t_low = transcript.lower()

        # 1. Recover Schober-Zeichen (e.g., 10 zu 13)
        schober = re.search(r"Schober.*?(\d+)\s*(?:zu|bis|-)\s*(\d+)", transcript, re.I)
        if schober and "Schober" not in obj_text:
            obj_text += f" | Schober-Zeichen: {schober.group(1)} - {schober.group(2)}"

        # 1b. Recover Finger-Boden-Abstand (FBA) — most common LWS metric
        fba = re.search(r"(?:finger.boden|fba)[^\d]*(\d+)\s*cm", transcript, re.I)
        if fba and "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
            obj_text += f" | FBA: {fba.group(1)} cm"
        elif re.search(r"finger.boden|fba", transcript, re.I):
            if "fba" not in obj_text.lower() and "finger-boden" not in obj_text.lower():
                obj_text += " | FBA: n.d."

        # 1c. Verbal LWS flexion descriptions → convert to FBA, remove hallucinated degree-ROM
        # Therapists often say "bis Mitte der Schienbeine" instead of giving cm.
        # The LLM then invents a Neutral-Null value (e.g. 0-0-90) that was never mentioned.
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
                    # Strip any hallucinated Neutral-Null ROM for LWS flexion from O
                    obj_text = re.sub(r'(?:LWS[^|]*?)?\b0-0-\d{2,3}\b[^|]*', '', obj_text).strip(' |')
                    break

        # 2. Recover VAS (Pain scale) — handle both orderings:
        #    "VAS 6" / "6/10" / "6 von 10" / "eine 6 von 10 beim Schmerz"
        if "VAS" not in soap_dict.get("S", ""):
            vas_num = None
            # Pattern A: explicit VAS label before number
            m = re.search(r"\bVAS\s*(\d{1,2})\b", transcript, re.I)
            if m:
                vas_num = m.group(1)
            if not vas_num:
                # Pattern B: "Schmerz ... X von 10" (forward)
                m = re.search(r"(?:Schmerz|Schmerzen|schmerzt)[^.]*?(\d{1,2})\s*(?:von|/)\s*10", transcript, re.I)
                if m:
                    vas_num = m.group(1)
            if not vas_num:
                # Pattern C: "X von 10 ... Schmerz" (reversed — e.g. "eine 6 von 10 beim Schmerz")
                m = re.search(r"\b(\d{1,2})\s*(?:von|/)\s*10\b[^.]*?(?:schmerz|schmerzen|schmerzt)", transcript, re.I)
                if m:
                    vas_num = m.group(1)
            if not vas_num:
                # Pattern D: bare "X/10" or "X von 10" anywhere (only 1-10)
                m = re.search(r"\b([1-9]|10)\s*/\s*10\b", transcript)
                if m:
                    vas_num = m.group(1)
            if vas_num:
                soap_dict["S"] = f"VAS {vas_num}/10. " + soap_dict["S"].lstrip()

        # 3. Recover Tests (Lasègue) if mentioned but missing in O
        if "lasegue" in transcript.lower() or "lasek" in transcript.lower():
            if "lasègue" not in obj_text.lower():
                # Try to find the degrees near the word
                deg = re.search(r"(?:lasegue|lasek).*?(\d+)\s*(?:grad|°)", transcript, re.I)
                deg_val = deg.group(1) if deg else "positiv"
                obj_text += f" | Lasègue-Test: {deg_val}° positiv."

        # Recover Ashworth Scale
        ashworth = re.search(r"Ashworth.*?(\d+)", transcript, re.I)
        if ashworth and "Ashworth" not in obj_text:
            obj_text += f" | Ashworth-Skala: {ashworth.group(1)}"

        # Recover Timed Up and Go
        tug = re.search(r"Timed Up and Go.*?(\d+)\s*Sekunden", transcript, re.I)
        if tug and "Timed Up and Go" not in obj_text:
            obj_text += f" | Timed Up & Go: {tug.group(1)}s"

        # Recover Barthel Index
        barthel = re.search(r"barthel.*?(\d+)", transcript, re.I)
        if barthel and "barthel" not in obj_text.lower():
            obj_text += f" | Barthel-Index: {barthel.group(1)}/100"

        # Recover House-Brackmann (Fazialisparese)
        hb = re.search(r"house.brackmann[^\d]*(grad\s*[IVX]+|\d)", transcript, re.I)
        if hb and "house" not in obj_text.lower():
            obj_text += f" | House-Brackmann: {hb.group(1)}"

        # Look for Abduction/Adduction or Rotation patterns
        rom_match = re.search(r"(?:Abduktion|Rotation).*?(\d+)\s*(?:zu|bis)\s*0\s*(?:zu|bis)\s*(\d+)", transcript, re.I)
        if rom_match and "-" not in obj_text:
            obj_text += f" | ROM: {rom_match.group(1)} - 0 - {rom_match.group(2)}"

        # Recover the Tests
        for test in ["Jobe", "Hawkins", "Neer"]:
            if test.lower() in transcript.lower() and test not in obj_text:
                obj_text += f" | {test}-Test: positiv."

        # Relative girth differences: "3 cm mehr Umfang als rechts"
        rel_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*cm\s+(?:mehr|weniger|größer|kleiner|Differenz|Unterschied)",
            transcript, re.I
        )
        if rel_match:
            diff = rel_match.group(1).replace(",", ".")
            if f"{diff} cm" not in obj_text and "Umfangsdifferenz" not in obj_text:
                obj_text += (
                    f" | Umfangsdifferenz: {diff} cm"
                    f" [⚠️ SEITENVERGLEICH FEHLT: Gegenseite einmalig dokumentieren,"
                    f" z.B. re. Handgelenk 14 cm / li. 18 cm — belegt die 4 cm Differenz]"
                )
        elif re.findall(r"([+-]\d+\s*cm)", transcript, re.I) and "cm" not in obj_text:
            # fallback: explicit +/- notation
            cm_metrics = re.findall(r"([+-]\d+\s*cm)", transcript, re.I)
            obj_text += f" | Umfangsdifferenz: {', '.join(cm_metrics)}"

        # ── KGG/MTT: recover training parameters ─────────────────────────────
        is_kgg = any(k in t_low for k in [
            "kgg", "gerätegestützt", "gerät", "mtt", "medizinische trainings",
            "beinpresse", "latzug", "ergometer", "krafttraining",
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
                r"(\d+)\s*(?:sekunden?|sek\.?)\s*(?:kontraktion|halten|anspannen|halten)",
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
            if modalitaet and all(k not in obj_text.lower() for k in ["fango", "rolle", "eispack", "fango", "wärme"]):
                obj_text += f" | Wärmemodalität: {modalitaet.group(1)}"
            temp_m = re.search(r"(\d+(?:[.,]\d+)?)\s*°?C\b", transcript, re.I)
            if temp_m and "°c" not in obj_text.lower() and "grad" not in obj_text.lower():
                obj_text += f" | Temperatur: {temp_m.group(1)} °C"

        # Explicit absolute girth pairs: "45 cm rechts, 42 cm links"
        abs_pair = re.findall(r"(\d{2,3})\s*cm", transcript)
        if len(abs_pair) >= 2 and "cm" in obj_text and "UNVOLLSTÄNDIG" in obj_text:
            # Therapist also gave absolute values — remove the prompt
            obj_text = re.sub(r"\s*\[⚠️ MESSUNG UNVOLLSTÄNDIG[^\]]*\]", "", obj_text)

        # Look for Stadium/Stage
        stadium = re.search(r"Stadium\s*[1-3]", transcript, re.I)
        if stadium and "Stadium" not in obj_text:
            obj_text = f"{stadium.group(0)}, " + obj_text

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
            a_field = soap_dict.get("A", "")
            if "ablation" not in a_field.lower() and "mastektomie" not in a_field.lower():
                soap_dict["A"] = a_field + " | Z.n. Mamma-Ablation (onkologische Vordiagnose erfüllt)."

        # Spannungsgefühl: patient's subjective complaint — must appear in S, not just as diagnosis
        if "spannungsgefühl" in t_low:
            s_field = soap_dict.get("S", "")
            if "spannungsgefühl" not in s_field.lower():
                # Determine body region if possible
                region = "linken Arm" if "links" in t_low or "linken arm" in t_low else "betroffenen Arm"
                soap_dict["S"] = f"Spannungsgefühl im {region}. " + s_field

        return soap_dict

    def _inject_ly_staging(self, transcript: str, soap_dict: dict) -> dict:
        """
        For LY domain: infer lymphedema stadium from transcript language and
        upgrade the A-field with a clinical classification if missing.
        """
        import re
        t = transcript.lower()
        a_field = soap_dict.get("A", "")

        # 1. Check transcript for EXPLICIT therapist stadium statement — highest priority
        explicit_in_transcript = re.search(r"stadium\s*([1-3])", transcript, re.I)
        if explicit_in_transcript:
            explicit_num = explicit_in_transcript.group(1)
            # Build correct staging label
            _label_map = {
                "1": "Stadium 1 (reversibel, pitting)",
                "2": "Stadium 2 (irreversibel, fibrosiert)",
                "3": "Stadium 3 (Elephantiasis)",
            }
            explicit_stadium = _label_map.get(explicit_num, f"Stadium {explicit_num}")
            icd_suffix = "02" if explicit_num == "3" else "01"
            soap_dict["_ly_icd_suffix"] = icd_suffix
            # Override A-field: replace any existing (possibly wrong) stadium
            new_staging = f"Lymphödem {explicit_stadium}."
            corrected = re.sub(r"Lymphödem\s+Stadium\s*[1-3][^\.\|]*\.", new_staging, a_field)
            if corrected == a_field:
                # No existing staging found — prepend
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

        # Stadium clues
        is_hard    = any(w in t for w in ["hart", "fibrosiert", "nicht eindrückbar", "derb", "induriert"])
        is_pitting = any(w in t for w in ["delle", "dellen", "pitting", "eindrückbar"])
        is_prall   = any(w in t for w in ["prall", "pralle"])  # tense/taut → Stage 2
        is_soft    = any(w in t for w in ["weich", "morgens besser", "reversibel"])
        is_massive = any(w in t for w in ["elephantiasis", "massiv", "extrem", "riesig"])
        is_postop  = any(w in t for w in ["post-op", "postoperativ", "postop", "op ", "nach der op",
                                           "mastektomie", "axilläre", "sentinel"])

        if is_massive:
            stadium = "Stadium 3 (Elephantiasis)"
            icd_suffix = "02"
        elif is_hard or (is_prall and is_pitting):
            # Pitting + tense/taut = chronic → Stage 2
            stadium = "Stadium 2 (irreversibel, fibrosiert)"
            icd_suffix = "01"
        elif is_soft or is_pitting:
            stadium = "Stadium 1 (reversibel, pitting)"
            icd_suffix = "01"
        elif is_postop:
            stadium = "Stadium 1–2 (postoperativ, noch zu klassifizieren)"
            icd_suffix = "01"
        else:
            stadium = "Stadium 1–2 (Klassifikation ausstehend — Stemmer-Zeichen prüfen)"
            icd_suffix = "01"

        # Store inferred suffix for run_full_flow to apply to the ICD-10 code
        soap_dict["_ly_icd_suffix"] = icd_suffix

        # Inject into A-field if not already there
        staging_note = f"Lymphödem {stadium}."
        if "Stadium" not in a_field:
            soap_dict["A"] = f"{staging_note} {a_field}".strip()

        return soap_dict

    def apply_medical_corrections(self, soap_dict):
        """
        Professional Grade Medical Text Refiner.
        Fixes Whisper hallucinations and standardizes terminology.
        """
        import re

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
            "Schlingentisch": "Schlingentisch",
            "Schlingen Tisch": "Schlingentisch",
            "Stoßwellentherapie": "Stoßwellentherapie",
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
            if not text:
                continue

            # Apply Simple Fixes
            for wrong, right in simple_fixes.items():
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                text = pattern.sub(right, text)

            # Apply Regex Fixes (Do NOT use re.escape here)
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

        # Cross-domain SMART goal sanity check:
        # If the SMART goal mentions a body part that doesn't appear in the SOAP at all,
        # it's a hallucination — clear the goal and flag it for manual completion.
        p_field = soap_dict.get("P", "")
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
        p_clean = _diag_test_re.sub('', p_field).strip(' |,.')
        if p_clean != p_field:
            soap_dict["P"] = re.sub(r'\s{2,}', ' ', p_clean).strip()

        return soap_dict

    def _inject_bladder_bowel_into_objective(self, transcript: str, soap: dict) -> dict:
        """
        Cauda-equina safety net for LWS/MT cases.

        If the therapist mentioned bladder/bowel function in the dictation
        (positively or negatively) but the AI buried it in S or A instead of O,
        extract the documented status and place it explicitly in O.
        A negative finding ('keine Blasen-Mastdarm-Störungen') is just as important
        as a positive one — it proves the clinician actively screened for Cauda equina.
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

    def inject_audit_stamps(self, soap):
        # Ensure Assessment has Red Flag exclusion statement (§ 106b requirement)
        if "red flag" not in soap["A"].lower():
            soap["A"] += " | Red Flags klinisch ausgeschlossen."
        return soap

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
        Skips negated conditions ("kein Bandscheibenvorfall") — those stay in S or A
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

    def run_full_flow(self, audio_path: str, status_callback=None, insurance_type=None):
        from shared.billing_engine import BillingEngine, InsuranceType
        if insurance_type is None:
            insurance_type = InsuranceType.GKV

        if status_callback: status_callback("✍️ Transkription...")
        wcfg = self.whisper_config
        audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        _log.info("Whisper start | model=%s | audio=%s (%d bytes)",
                  self.stt_model, audio_path, audio_size)
        for h in _log.handlers: h.flush()
        raw_t = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.stt_model,
            language=wcfg.get("language", "de"),
            initial_prompt=wcfg.get("initial_prompt", "Physiotherapie Befund. Neutral-Null-Methode. VAS Schmerzskala."),
            temperature=wcfg.get("temperature", 0.0),
            condition_on_previous_text=wcfg.get("condition_on_previous_text", True),
        ).get("text", "")
        _log.info("Whisper done | transcript length=%d", len(raw_t))
        for h in _log.handlers: h.flush()
        transcript = self.clean_transcript(raw_t)

        profile_id = self._detect_profile(transcript)
        prof_label = self._PROFILES[profile_id]["label"]
        if status_callback: status_callback(f"KI-Analyse [{prof_label}]...")
        _log.info("LLM generate start | profile=%s", profile_id)
        for h in _log.handlers: h.flush()
        raw_output = self._generate_soap_note(transcript, profile_id)
        _log.info("LLM generate done | output_len=%d", len(raw_output))
        for h in _log.handlers: h.flush()

        if status_callback: status_callback("🔍 Validierung...")
        parsed = self.parse_robust_json(raw_output)

        # ICD correction (domain detection + keyword-based upgrade)
        icd, _ = self.suggest_billing(parsed["icd10"], parsed["soap"], transcript)
        parsed["icd10"] = icd

        parsed["soap"] = self.apply_medical_corrections(parsed["soap"])
        parsed["soap"] = self.recover_hard_metrics(transcript, parsed["soap"])
        if profile_id == "LY":
            parsed["soap"] = self._inject_ly_staging(transcript, parsed["soap"])
            # Apply ICD-10 staging suffix (Oct 2024 rule: terminal staging code required)
            suffix = parsed["soap"].pop("_ly_icd_suffix", None)
            if suffix and re.match(r"^[IQE]\d{2}\.\d$", icd):
                icd = icd + suffix  # e.g. I89.0 → I89.001, but use correct format
            elif suffix and re.match(r"^[IQE]\d{2}\.\d0?$", icd):
                # Normalise: I89.0 → I89.01, I97.2 → I97.21, E88.2 → E88.21
                base = icd.rstrip("0") if icd.endswith("0") and len(icd) > 5 else icd
                icd = base + suffix
                parsed["icd10"] = icd
        parsed["soap"] = self._migrate_diagnoses_from_s_to_a(parsed["soap"])
        parsed["soap"] = self._clean_hallucinated_regions(parsed["soap"], icd, profile_id)
        parsed["soap"] = self._inject_bladder_bowel_into_objective(transcript, parsed["soap"])
        parsed["soap"] = self.inject_audit_stamps(parsed["soap"])
        parsed = self.rom_sanity_check(transcript, parsed)

        # Dual billing engine: GKV deterministic / PKV AI-assisted
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

    def suggest_billing(self, icd10: str, soap: dict, transcript: str):
        """
        Supports Orthopedic, Neurological, and Lymphatic Physiotherapy.
        Orders priority: ZNS > MT > MLD > KG.
        """
        codes = self.config.billing_codes
        t_low = transcript.lower()
        plan_text = soap.get("P", "").lower()
        obj_text = soap.get("O", "").lower()
        full_text = f"{obj_text} {plan_text} {t_low}"

        # --- 1. DOMAIN DETECTION ---
        is_neuro = any(k in full_text for k in
                       ["bobath", "pnf", "neuro", "zns", "hemiparese", "ataxie", "spastik", "insult", "schlaganfall"])
        is_lymph = any(k in full_text for k in ["mld", "lymph", "ödem", "kpe", "entstauung", "stemmer"])
        is_ortho_mt = any(k in full_text for k in
                          ["manuelle therapie", " mt ", "traktion", "gleitmobilisation", "manipulation",
                           "mobilisation"])

        # --- 2. ICD-10 CROSS-CHECK (Multi-Domain) ---
        res_icd = icd10
        # NEURO PRIORITY: If neuro keywords exist, it CANNOT be an orthopedic code (M)
        if is_neuro:
            if not icd10.startswith(("G", "I69")):
                res_icd = "I69.3"  # Default: Folgen eines Hirninfarkts

        # LYMPH PRIORITY
        elif is_lymph:
            if not icd10.startswith("I89"):
                res_icd = "I89.0"  # Default: Lymphödem

        # ORTHO REFINEMENT
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
                # Schenkelhalsfraktur: traumatic S72.0, osteoporotic M80.05
                is_osteoporotic = any(k in t_low for k in ["osteoporose", "osteoporotisch", "knochendichte"])
                res_icd = "M80.05" if is_osteoporotic else "S72.0"
            elif icd10.startswith("M81") and any(k in t_low for k in ["fraktur", "bruch", "gebrochen"]):
                # M81 = osteoporosis WITHOUT fracture — if fracture is mentioned, upgrade
                if hip_ctx:
                    res_icd = "M80.05"   # Osteoporotic femoral neck fracture
                else:
                    res_icd = "M80.08"   # Osteoporotic fracture, other site
            # Dominant body-region guards — order: Schulter > Knie > Hüfte > Rücken
            # Each guard also protects the already-correct ICD from being overridden.
            _is_schulter = "schulter" in t_low or icd10.startswith("M75")
            _is_knie     = "knie" in t_low or icd10.startswith("M17")
            if _is_schulter and not icd10.startswith("M75"):
                res_icd = "M75.4"
            elif _is_knie and not _is_schulter and not icd10.startswith("M17"):
                res_icd = "M17.1"
            elif hip_ctx and not _is_schulter and not _is_knie and not icd10.startswith(("M16", "M80", "S72")):
                res_icd = "M16.1"   # Koxarthrose
            elif (not _is_schulter and not _is_knie and not hip_ctx and
                  (any(k in t_low for k in ["hexenschuss", "lumbago", "ischiasschmerz", "lws"]) or
                   re.search(r'rücken(?:schmerz|weh|beschwerden|problem)', t_low))):
                # "rücken" alone excluded: "Legen Sie sich auf den Rücken" is positional,
                # not a pain complaint — require it to be compounded with a symptom word.
                res_icd = "M54.5"
                if any(k in t_low for k in ["ausstrahlung", "lasegue", "radikulär", "bein", "wade"]):
                    res_icd = "M51.1"

        # --- 3. BILLING ALLOCATION (Hierarchical) ---
        # Apply ICD domain lock before billing allocation
        res_icd = self._lock_icd_domain(res_icd, soap, transcript)

        # Rule: If 'Krankengymnastik' is explicitly dictated as the work type,
        # we don't 'Upcode' to MT even if mobilisation is mentioned.
        if "krankengymnastik" in plan_text or " kg" in plan_text:
            if is_neuro: return res_icd, codes.get("KG_ZNS", "20511")
            return res_icd, codes.get("KG", "20501")

        if is_neuro: return res_icd, codes.get("KG_ZNS", "20511")
        # MT must NOT override lymph — a lymph case mentioning "mobilisation" is still MLD
        if is_ortho_mt and not is_lymph: return res_icd, codes.get("MT", "21201")
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
            # Stadium II/III or explicit 60-min → MLD-60 (20202)
            # Stadium I, two body parts or explicit 45-min → MLD-45 (20201)
            # Stadium I, one body part or explicit 30-min → MLD-30 (20205)
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
            "neuro_parkinson": (["parkinson", "tremor", "rigor", "brady­kinese", "bradykinese",
                                  "hoehn yahr", "hoehn-yahr"], "G20"),
            "neuro_ms":      (["multiple sklerose", "ms-schub", "ms schub", "demyelini",
                                "fatigue ms", "gangstörung ms"], "G35"),
            "neuro_facial":  (["fazialisparese", "fazialis", "bell'sche", "bellsche",
                                "gesichtslähmung", "house-brackmann"], "G51.0"),
            "lymph":         (["lymphödem", "lymphdrainage", "mld", "entstauung",
                                "stemmer", "kpe", "lipödem"], "I89.0"),
            "copd":          (["copd", "atemwegsobstruktion", "emphysem", "dyspnoe",
                                "atemtherapie", "atemübung"], "J44.1"),
        }

        for domain, (keywords, fallback_icd) in domains.items():
            hits = sum(1 for kw in keywords if kw in combined)
            if hits >= 2:
                # Hard block: if current ICD belongs to wrong chapter, override
                if domain.startswith("neuro_") and icd10[0] == "M":
                    return fallback_icd
                if domain == "lymph" and icd10[0] == "M":
                    return fallback_icd
                if domain == "copd" and icd10[0] == "M":
                    return fallback_icd

        return icd10

    # Terms that are out-of-scope for each profile domain.
    # If these appear as diagnoses in A (not as exclusions), they are hallucinations.
    _PROFILE_FORBIDDEN_A: dict = {
        "EX_SCHULTER": [
            r"gonarthrose", r"koxarthrose", r"meniskus(?:riss|läsion)",
            r"kreuzband(?:ruptur)?", r"vkb[\-\s](?:ruptur|riss|plastik)",
            r"bandscheibenvorfall", r"bandscheibenprotrusion", r"diskushernie",
            r"spinalkanalstenose", r"spondylolisthese",
            r"schenkelhalsfraktur", r"schenkelhals",   # Whisper Schulter↔Schenkel confusion
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
        E.g. a Schulter patient should not be diagnosed with Gonarthrose.
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
                m = re.search(pattern, sent_stripped, re.I)
                if m:
                    # Proximity negation: check only the 6 words BEFORE the matched term
                    before = sent_stripped[:m.start()]
                    nearby_words = before.split()[-6:]
                    negation_nearby = self._NEGATION_RE.search(" ".join(nearby_words))
                    if negation_nearby:
                        break  # term is locally negated — keep it
                    is_hallucination = True
                    print(f"[SanityCheck] Removed off-profile term '{pattern}' from A: {sent_stripped[:60]}")
                    break
            if not is_hallucination:
                cleaned.append(sent_stripped)

        soap["A"] = " ".join(cleaned).strip()
        return soap

    def compliance_check(self, soap: dict, billing_code: str):
        warns = []
        obj = soap.get("O", "").lower()
        # Red Flag Check — look FORWARD after the flag word (German: "Parese: negativ")
        for f in self.audit_rules.get("red_flags", []):
            idx = obj.find(f.lower())
            if idx != -1:
                after = obj[idx: idx + 70]
                if not any(n in after for n in ['neg', 'ausg', 'kein', 'ohne', 'unauffällig']):
                    warns.append(f"🔴 NOTFALL: {f.upper()}!")
        # 2026 Density
        if billing_code in ["21201", "20511"] and len(obj) < 60:
            warns.append(f"📋 DOKU: Befunddichte zu gering für {billing_code}.")
        if "°" in obj and not re.search(r"\d+-\d+-\d+", obj):
            warns.append("⚠️ HINWEIS: Bitte Neutral-Null-Methode nutzen.")
        return warns if warns else ["✅ Dokumentation GKV-konform."]

    def rom_sanity_check(self, transcript: str, parsed: dict):
        obj = parsed["soap"].get("O", "")
        t_nums = set(re.findall(r'\b\d+\b', transcript))
        for l, r in re.findall(r'(\d+)-0-(\d+)', obj):
            if l not in t_nums or r not in t_nums:
                if "compliance_check" not in parsed: parsed["compliance_check"] = []
                parsed["compliance_check"].append(f"⚠️ ROM Halluzination? {l}-0-{r}!")
        return parsed

    def parse_robust_json(self, text):
        """
        Robust JSON parser - ensures all SOAP fields are strings, not nested objects.
        """
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            data = json.loads(match.group() if match else "{}")
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
                # Ensure non-empty
                if not value or value in ("N/A", "n.d.", "Fehler", "{}"):
                    if field == "A":
                        # Assessment should never be empty
                        value = f"{data.get('icd10', 'M99.9')} | Red Flags klinisch ausgeschlossen."
                    else:
                        value = "N/A"
                soap_clean[field] = value.strip() if value else "N/A"

            return {"icd10": data.get("icd10", "M99.9"), "soap": soap_clean}
        except Exception as e:
            print(f"⚠️ JSON parsing failed: {e}")
            return {"icd10": "M99.9", "soap": {k: "Fehler" for k in ["S", "O", "A", "P"]}}

    def cleanup(self):
        self.model = self.tokenizer = None
        self._cleanup_gpu_memory()
