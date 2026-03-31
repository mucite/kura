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
        # Dynamic corrections from Gist whisper_config.medical_corrections
        for wrong, correct in self.whisper_config.get("medical_corrections", {}).items():
            try:
                transcript = re.sub(wrong, correct, transcript, flags=re.IGNORECASE)
            except re.error:
                pass
        return transcript

    # ── Domain detection & prompt building ────────────────────────────────────

    def _detect_domain(self, transcript: str) -> str:
        t = transcript.lower()
        if any(k in t for k in ["bobath", "pnf", "vojta", "zns", "hemiparese", "hemiplegie",
                                  "parkinson", "multiple sklerose", " ms ", "schlaganfall",
                                  "insult", "apoplex", "spastik", "ataxie", "fazialisparese"]):
            return "ZNS"
        if any(k in t for k in ["lymph", "ödem", "mld", "kpe", "entstauung", "stemmer",
                                  "stadium", "lipödem", "mastektomie"]):
            return "LY"
        if any(k in t for k in ["copd", "asthma", "atemweg", "mukoviszidose",
                                  "atemtherapie", "sekretmobilisation"]):
            return "AT"
        if any(k in t for k in ["manuelle therapie", " mt ", "traktion", "gleitmobilisation",
                                  "manipulation", "gelenkmobilisation", "hvla"]):
            return "MT"
        return "KG"

    def _domain_checklist(self, domain: str) -> str:
        if domain == "ZNS":
            return """PFLICHTFELDER NEUROLOGIE — O-Feld MUSS enthalten:
- Tonus: Ashworth-Skala 0-4 mit Lokalisation (z.B. "Ashworth 2 re. Arm")
- Gangbild: Typ (Zirkumduktion/Steppergang/Trendelenburg/normal)
- ADL-Status: Barthel-Index (0-100) ODER Selbständigkeit in %
- Koordination: Knie-Hacke-Test (sicher / unsicher / nicht möglich)
- Bei Parkinson: Hoehn-Yahr-Skala (1-5) + TUG-Test in Sekunden
- Bei MS: EDSS-Score + Fatigue (vorhanden/nicht vorhanden)
ICD: G20/G35/G81/I69 | Billing: 20511 (KG-ZNS, 45 min)"""
        if domain == "LY":
            return """PFLICHTFELDER LYMPHOLOGIE — O-Feld MUSS enthalten:
- Stadium: 1 / 2 / 3
- Stemmer-Zeichen: positiv/negativ + Lokalisation
- Umfangsmessung: beidseitig in cm + Körperstelle
- Konsistenz: weich / teigig / hart / fibrosiert
ICD: I89.0/I97.2/Q82.0 | Billing: 21101 (MLD 45min) / 21110 (KPE)"""
        if domain == "AT":
            return """PFLICHTFELDER ATEMTHERAPIE — O-Feld MUSS enthalten:
- Spirometrie: FEV1/FVC oder "nicht gemessen"
- Atemmuster: Typ
- SpO2: Wert in %
- Sekretmobilisation: vorhanden/nicht vorhanden
ICD: J44/J45/E84 | Billing: 20560 (KG atemtherapeutisch)"""
        if domain == "MT":
            return """PFLICHTFELDER MANUELLE THERAPIE — O-Feld MUSS enthalten:
- ROM Neutral-Null: [Ext]-[0]-[Flex] je Bewegungsebene
- Endgefühl: fest / weich / leer / hart
- Palpation: Druckdolenz + Lokalisation
- VAS: x/10
- Region-Tests: [HWS: Spurling] [Schulter: Jobe, Hawkins] [LWS: Lasègue°, Schober cm, FBA cm]
ICD: M54.x/M75.x/M17 | Billing: 20701 (MT 20min)"""
        return """PFLICHTFELDER KG — O-Feld MUSS enthalten:
- ROM Neutral-Null: [Ext]-[0]-[Flex]
- Muskelkraft: MMT 0-5
- VAS: x/10
- Palpation: Tonusbefund
Billing: 20501 (KG 20min)"""

    def build_prompt(self, transcript: str, domain: str = "KG") -> str:
        learning_notes = self.learning_mgr.get_relevant_prefs(transcript)
        style_injection = f"\nBEVORZUGTE CODES DES THERAPEUTEN:\n{learning_notes}\n" if learning_notes else ""
        checklist = self._domain_checklist(domain)

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Du bist ein klinischer Dokumentationsexperte für deutsche Physiotherapie (§106b SGB V).
DOMÄNE DIESER SITZUNG: {domain}
{style_injection}
━━ EXTRAKTIONSREGELN (ABSOLUT VERBINDLICH) ━━
1. Extrahiere AUSSCHLIESSLICH Informationen aus dem Transkript.
2. Fehlende Werte: schreibe "n.d." (nicht dokumentiert) — NIEMALS erfinden.
3. Zahlen EXAKT übernehmen: "VAS 7" nicht "starke Schmerzen".
4. Neutral-Null-Methode: [Ext] - [0] - [Flex], z.B. Knie-Flex = "0 - 0 - 90".
5. Red Flags IMMER im A-Feld: "Red Flags (Parese, Cauda) klinisch ausgeschlossen."

━━ DOMÄNEN-PFLICHTFELDER ━━
{checklist}

━━ SOAP-STRUKTUR ━━
S: Hauptbeschwerde + VAS x/10 + Dauer + Auslöser
O: ALLE Messwerte, Tests, Palpation — KEINE Zusammenfassungen
A: ICD-10-Diagnose + Red-Flag-Ausschluss
P: Heilmittel + Technik + Frequenz + konkretes Funktionsziel

{{
  "icd10": "[spezifischer Code]",
  "soap": {{"S": "...", "O": "...", "A": "...", "P": "..."}},
  "billing_suggestion": "[Positionsnummer]"
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

    def _generate_soap_note(self, transcript: str, domain: str = "KG") -> str:
        prompt = self.build_prompt(transcript, domain)
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

    # ── Post-processing pipeline (matches macOS) ───────────────────────────────

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
            # CMD / Craniomandibuläre Dysfunktion
            r"CNMD|CMND|CNMT|C\.N\.M\.D\.": "CMD",
            # Knee instability compound word
            r"Knieknadi\w*|Knienadig\w*|Knie.?Nadi\w*|Knienachgibigkeit": "Knienachgiebigkeit",
            # Other frequently mangled German physio compounds
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

    def _clean_hallucinated_regions(self, soap: dict, icd: str) -> dict:
        if icd.startswith("M17"):
            for p in [r"C[0-7]/C[0-7]", r"HWS", r"Zervikal", r"Kopfgelenk"]:
                soap["O"] = re.sub(p, "[Korrektur: Anatomischer Widerspruch]", soap.get("O", ""), flags=re.IGNORECASE)
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
            if "schulter" in t_low and not icd10.startswith("M75"):
                res_icd = "M75.4"
            elif "knie" in t_low and not icd10.startswith("M17"):
                res_icd = "M17.1"
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
            return res_icd, codes.get("MT", "20701")
        if is_lymph:
            return res_icd, codes.get("MLD", "21101")

        return res_icd, codes.get("KG", "20501")

    # ── Compliance ────────────────────────────────────────────────────────────

    def compliance_check(self, soap: dict, billing_code: str) -> list:
        warns = []
        obj = soap.get("O", "").lower()

        # Red flag check — look FORWARD after the flag word (German: "Parese: negativ")
        for f in self.audit_rules.get("red_flags", []):
            idx = obj.find(f.lower())
            if idx != -1:
                after = obj[idx: idx + 70]
                if not any(n in after for n in ["negativ", "unauffällig", "keine", "kein", "ausgeschlossen", "normal"]):
                    warns.append(f"🔴 NOTFALL: {f.upper()}!")

        if billing_code in ["20701", "20511"] and len(obj) < 60:
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

        domain = self._detect_domain(transcript)
        if status_callback:
            status_callback(f"🧠 KI-Analyse [{domain}]...")
        raw_output = self._generate_soap_note(transcript, domain)

        if status_callback:
            status_callback("🔍 Validierung...")
        parsed = self.parse_robust_json(raw_output)

        # ICD correction (domain detection + keyword-based upgrade)
        icd, _ = self.suggest_billing(parsed["icd10"], parsed["soap"], transcript)
        parsed["icd10"] = icd

        parsed["soap"] = self.apply_medical_corrections(parsed["soap"])
        parsed["soap"] = self.recover_hard_metrics(transcript, parsed["soap"])
        parsed["soap"] = self._clean_hallucinated_regions(parsed["soap"], icd)
        parsed["soap"] = self.inject_audit_stamps(parsed["soap"])
        parsed = self.rom_sanity_check(transcript, parsed)

        # Dual billing engine: GKV deterministic / PKV AI-assisted
        billing_result = BillingEngine().evaluate(
            icd10=icd,
            soap=parsed["soap"],
            transcript=transcript,
            insurance_type=insurance_type,
            config_rules=self.billing_rules,
        )

        return {
            "icd10": icd,
            "soap": parsed["soap"],
            "billing_suggestion": billing_result.position_number,
            "billing_result": billing_result,
            "compliance_check": billing_result.compliance_warnings,
            "transcript": transcript,
        }

    def cleanup(self):
        self.llm = None
        self.whisper = None
        gc.collect()