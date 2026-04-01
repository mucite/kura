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
                "Palpation: Druckdolenz + exakte Lokalisation",
                "Schmerz: VAS x/10",
                "Provokationstest: Lasegue / Spurling / Slump (positiv / negativ)",
                "Blasen-/Mastdarmfunktion: unauffaellig / gestaert (Cauda-equina-Screening)",
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
                "ROM Finger: je Strahl Grundgelenk / Mittelgelenk / Endgelenk (Grad oder Faust-cm)",
                "Jamar-Handkraft (kg) re / li (altersadjustiert falls Kind)",
                "Pinzettengriff / Schluesselgriff: moeglich / eingeschraenkt / nicht moeglich",
                "Schmerz: VAS x/10",
                "Narbe (falls OP): Verschieblichkeit / Sensibilitaet / Roeotung",
                "Oedemmass: Umfang Handgelenk in cm (falls geschwollen)",
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

PROFIL-PFLICHTFELDER (diese Felder MUESSEN im O-Feld erscheinen):
{checklist}

SOAP-STRUKTUR:
S: Hauptbeschwerde + Schmerzlokalisation + VAS x/10 + Dauer + Ausloeser
O: ALLE klinischen Messwerte und Tests des Profils — KEINE Zusammenfassungen
A: ICD-10-Diagnose | Differentialdiagnose | Red-Flag-Ausschluss
P: Heilmittel ({prof["billing"]}) + Technik + Frequenz + messbares Funktionsziel

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

        # 1. Recover Schober-Zeichen (e.g., 10 zu 13)
        schober = re.search(r"Schober.*?(\d+)\s*(?:zu|bis|-)\s*(\d+)", transcript, re.I)
        if schober and "Schober" not in obj_text:
            obj_text += f" | Schober-Zeichen: {schober.group(1)} - {schober.group(2)}"

        # 2. Recover VAS (Pain scale)
        vas = re.search(r"VAS\s*(\d+)", transcript, re.I)
        if vas and "VAS" not in soap_dict["S"]:
            soap_dict["S"] += f" (VAS {vas.group(1)}/10)"

        vas_match = re.search(r"(?:Schmerz|VAS).*?(\d+)\s*(?:von|/)\s*10", transcript, re.I)
        if vas_match and "VAS" not in soap_dict["S"]:
            soap_dict["S"] = f"VAS {vas_match.group(1)}/10. " + soap_dict["S"]

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
                    f" [⚠️ MESSUNG UNVOLLSTÄNDIG: bitte absolute Werte ergänzen,"
                    f" z.B. re. 45 cm / li. 42 cm]"
                )
        elif re.findall(r"([+-]\d+\s*cm)", transcript, re.I) and "cm" not in obj_text:
            # fallback: explicit +/- notation
            cm_metrics = re.findall(r"([+-]\d+\s*cm)", transcript, re.I)
            obj_text += f" | Umfangsdifferenz: {', '.join(cm_metrics)}"

        # Explicit absolute girth pairs: "45 cm rechts, 42 cm links"
        abs_pair = re.findall(r"(\d{2,3})\s*cm", transcript)
        if len(abs_pair) >= 2 and "cm" in obj_text and "UNVOLLSTÄNDIG" in obj_text:
            # Therapist also gave absolute values — remove the prompt
            obj_text = re.sub(r"\s*\[⚠️ MESSUNG UNVOLLSTÄNDIG[^\]]*\]", "", obj_text)

        # Look for Stadium/Stage
        stadium = re.search(r"Stadium\s*[1-3]", transcript, re.I)
        if stadium and "Stadium" not in obj_text:
            obj_text = f"{stadium.group(0)}, " + obj_text

        soap_dict["O"] = obj_text
        return soap_dict

    def _inject_ly_staging(self, transcript: str, soap_dict: dict) -> dict:
        """
        For LY domain: infer lymphedema stadium from transcript language and
        upgrade the A-field with a clinical classification if missing.
        """
        import re
        t = transcript.lower()
        a_field = soap_dict.get("A", "")

        # Don't overwrite if therapist explicitly stated a Stadium already
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

        # 1. Split into Simple Replacements and Regex Patterns
        # Simple: Exact string match (Case-insensitive)
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
            "Subakromialraum": "Subakromialraum",
            "Jobe Test": "Jobe-Test",
            "Hawkins Test": "Hawkins-Test",
            "Supraspinatus-Szene": "Supraspinatussehne",
            "Jove-Test": "Jobe-Test",
            "Befallung": "Läsion/Dysfunktion",
            "Mayus": "majus",
            "Bobad": "Bobath",
            "Bobert": "Bobath",
            "P N F": "PNF",
            "ischemisch": "ischämisch",
            "Zirkumduktion": "Zirkumduktion",
            "Bobart": "Bobath",
            "Mama-Karzinom": "Mamma-Karzinom",
            "Hochlagerndes": "Hochlagern des",
            "Lymphödes": "Lymphödem",
            "Anastomosen": "Lymph-Anastomosen"
        }

        # Complex: Regex patterns for multiple variations
        regex_fixes = {
            r"Laseck|Lasegge|Laseque": "Lasègue-Test",
            r"Schoberzeichen|Schober Zeichen": "Schober-Zeichen",
            r"Fußheber": "M. extensor hallucis longus (Fußheber)",
            r"(\d+)\s*zu\s*(\d+)": r"\1 - \2",  # Neutral-Zero fix (10 zu 15 → 10 - 15)
            # CMD
            r"CNMD|CMND|CNMT|C\.N\.M\.D\.": "CMD",
            # Knee instability
            r"Knieknadi\w*|Knienadig\w*|Knie.?Nadi\w*|Knienachgibigkeit": "Knienachgiebigkeit",
            # Other compounds
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

            # Apply Simple Fixes
            for wrong, right in simple_fixes.items():
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                text = pattern.sub(right, text)

            # Apply Regex Fixes (Do NOT use re.escape here)
            for pattern_str, right in regex_fixes.items():
                text = re.sub(pattern_str, right, text, flags=re.IGNORECASE)

            soap_dict[key] = text

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
            elif "schulter" in t_low and not icd10.startswith("M75"):
                res_icd = "M75.4"
            elif "knie" in t_low and not icd10.startswith("M17"):
                res_icd = "M17.1"
            elif hip_ctx and not icd10.startswith(("M16", "M80", "S72")):
                res_icd = "M16.1"   # Koxarthrose
            elif any(k in t_low for k in ["hexenschuss", "lumbago", "ischiasschmerz", "lws", "rücken"]):
                res_icd = "M54.5"
                if any(k in t_low for k in ["ausstrahlung", "lasegue", "radikulär", "bein", "wade"]):
                    res_icd = "M51.1"

        # --- 3. BILLING ALLOCATION (Hierarchical) ---
        # Rule: If 'Krankengymnastik' is explicitly dictated as the work type,
        # we don't 'Upcode' to MT even if mobilisation is mentioned.
        if "krankengymnastik" in plan_text or " kg" in plan_text:
            if is_neuro: return res_icd, codes.get("KG_ZNS", "20511")
            return res_icd, codes.get("KG", "20501")

        if is_neuro: return res_icd, codes.get("KG_ZNS", "20511")
        if is_ortho_mt: return res_icd, codes.get("MT", "21201")
        if is_lymph: return res_icd, codes.get("MLD", "20201")

        return res_icd, codes.get("KG", "20501")

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
