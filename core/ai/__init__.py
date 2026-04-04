"""
Core AI Engine - Platform-Agnostic Base
========================================
Shared logic for Whisper STT + Llama NLP across macOS and Windows.
"""
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable

logger = logging.getLogger("kura.ai.engine")


class AIEngineBase(ABC):
    """
    Abstract base class for Kura AI Engine.

    Platform-specific implementations (MLX for macOS, PyTorch for Windows)
    inherit from this and implement the abstract methods.
    """

    def __init__(self):
        self.learning_mgr = None  # Set by subclass
        self.config = None  # Set by subclass
        self.model = None  # Set by subclass
        self.tokenizer = None  # Set by subclass

    @abstractmethod
    def transcribe_audio(self, audio_path: str, language: str = "de") -> str:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_path: Path to audio file (WAV, 16kHz mono)
            language: Language code (default: "de" for German)

        Returns:
            Transcribed text
        """
        pass

    @abstractmethod
    def generate_soap(
        self,
        transcript: str,
        profile_id: str,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate SOAP note from transcript using LLM.

        Args:
            transcript: Cleaned transcript
            profile_id: Diagnosis profile ID
            status_callback: Optional callback for progress updates

        Returns:
            JSON string with SOAP structure
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Release GPU/Metal memory and cleanup resources."""
        pass

    # ========================================================================
    # SHARED METHODS (Platform-independent)
    # ========================================================================

    def clean_transcript(self, transcript: str) -> str:
        """
        Fix common transcription hallucinations.
        Platform-independent preprocessing.
        """
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
            r"Finger-Bodenabstand|FBA": "Finger-Boden-Abstand (FBA)",
            r"Sanddemobilisation|Sandmobilisation": "sanfte Mobilisation",
            r"VKB Plastik|VKB-Plastik": "Vorderes Kreuzband (VKB) Plastik",
            r"M, M, T": "MMT (Manual Muscle Test)",
            r"CNMD|CMND|CNMT|C\.N\.M\.D\.": "CMD",
            r"Knieknadi\w*|Knienadig\w*|Knie.?Nadi\w*|Knienachgibigkeit": "Knienachgiebigkeit",
            r"Rotatorenmanschete\b|Rotatoren.?Manschette": "Rotatorenmanschette",
            r"Impingmentsyndrom|Impingement.?Syndrom": "Impingementsyndrom",
            r"Plantarfasciitis|Plantar.?Fasziitis": "Plantarfasziitis",
            r"Karpaltunnel.?Syndrom": "Karpaltunnelsyndrom",
            r"Propriozepzion|Propioception|Propiozeption": "Propriozeption",
            r"Tendinapathie|Tendinopatie": "Tendinopathie",
        }

        for pattern, replacement in corrections.items():
            transcript = re.sub(pattern, replacement, transcript, flags=re.IGNORECASE)

        # Context-aware anatomical disambiguation
        if re.search(r'schulterhalsfraktur', transcript, re.I):
            hip_context = any(k in transcript.lower() for k in [
                'hüft', 'huefte', 'femur', 'schenkel', 'oberschenkel',
                'tep', 'hüft-tep', 'hüftprothese', 'schenkelhals',
            ])
            if hip_context:
                transcript = re.sub(
                    r'schulterhalsfraktur', 'Schenkelhalsfraktur',
                    transcript, flags=re.I
                )

        logger.debug(f"Transcript cleaned: {len(transcript)} characters")
        return transcript

    def detect_profile(self, transcript: str, profiles: Dict) -> str:
        """
        Detect medical diagnosis profile from transcript.
        Platform-independent profile matching.

        Args:
            transcript: Cleaned transcript
            profiles: Profile configuration dict

        Returns:
            Profile ID (e.g., "EX_SCHULTER", "ZNS_ADULT")
        """
        t = transcript.lower()

        # Age extraction
        age = None
        age_match = re.search(r'(\d{1,2})\s*(?:jahre?\s*alt|j\b|-jaehrig)', t)
        if age_match:
            age = int(age_match.group(1))

        best_id = "KG"
        best_score = -1

        for profile_id, profile in profiles.items():
            if profile_id == "KG":
                continue  # Fallback only

            # Age constraints
            if age is not None:
                age_max = profile.get("age_max")
                age_min = profile.get("age_min")
                if age_max and age > age_max:
                    continue
                if age_min and age < age_min:
                    continue
            else:
                # No age detected - skip pediatric profiles
                if profile.get("age_max", 999) <= 17:
                    continue

            # Match triggers
            triggers = profile.get("triggers", [])
            match_count = sum(1 for trigger in triggers if trigger in t)

            if match_count == 0:
                continue

            # Score = priority * 1000 + match_count
            priority = profile.get("priority", 0)
            score = priority * 1000 + match_count

            if score > best_score:
                best_id = profile_id
                best_score = score

        logger.info(f"Profile detected: {best_id} (score: {best_score})")
        return best_id

    def extract_icd10(self, text: str) -> Optional[str]:
        """
        Extract ICD-10 code from LLM output.
        Platform-independent.

        Prefers codes that appear before any "Differentialdiagnose" section
        to avoid returning a differential-diagnosis code instead of the
        primary diagnosis code.

        Args:
            text: LLM output text

        Returns:
            ICD-10 code or None
        """
        icd_pattern = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,2})?)\b')

        # Find the start of the differential-diagnosis section, if any
        dd_match = re.search(
            r'(?:differentialdiagnose|differential\s*diagnose|dd\s*:|differenzialdiagnose)',
            text, re.I
        )
        dd_pos = dd_match.start() if dd_match else len(text)

        # Prefer codes that appear before the differential section
        pre_dd = [m.group(1) for m in icd_pattern.finditer(text) if m.start() < dd_pos]
        if pre_dd:
            logger.debug(f"ICD-10 extracted (pre-DD): {pre_dd[0]}")
            return pre_dd[0]

        # Fallback: first code anywhere in the text
        all_codes = icd_pattern.findall(text)
        if all_codes:
            logger.debug(f"ICD-10 extracted (fallback): {all_codes[0]}")
            return all_codes[0]

        logger.warning("No ICD-10 code found in LLM output")
        return None

    def run_full_flow(
        self,
        audio_path: str,
        status_callback: Optional[Callable[[str], None]] = None,
        insurance_type = None
    ) -> Dict:
        """
        Complete AI workflow: Audio → Transcript → SOAP → Billing.

        Args:
            audio_path: Path to recorded audio file
            status_callback: Optional callback for progress updates
            insurance_type: Insurance type (GKV/PKV/BG)

        Returns:
            Complete result dictionary with SOAP, ICD-10, billing
        """
        from core.logging import PerformanceLogger

        try:
            # Step 1: Transcription
            with PerformanceLogger("Whisper STT"):
                if status_callback:
                    status_callback("🎤 Transkription...")

                raw_transcript = self.transcribe_audio(audio_path)
                logger.info(f"Transcription complete: {len(raw_transcript)} chars")

            # Step 2: Cleaning
            transcript = self.clean_transcript(raw_transcript)

            # Step 3: Profile detection
            if hasattr(self, 'profiles'):
                profile_id = self.detect_profile(transcript, self.profiles)
            else:
                profile_id = "KG"  # Fallback if no profiles loaded

            logger.info(f"Using profile: {profile_id}")

            # Step 4: SOAP generation
            with PerformanceLogger("LLM SOAP Generation"):
                if status_callback:
                    status_callback("🧠 KI-Analyse...")

                soap_json = self.generate_soap(transcript, profile_id, status_callback)

            # Step 5: Parse result
            import json
            result = json.loads(soap_json) if isinstance(soap_json, str) else soap_json

            # Step 6: Billing audit
            if status_callback:
                status_callback("✓ Prüfung...")

            icd10 = result.get('icd10', 'N/A')
            soap = result.get('soap', {})

            # Billing engine integration (if available)
            billing_result = None
            if insurance_type and hasattr(self, 'billing_engine'):
                from shared.billing_engine import BillingEngine

                engine = BillingEngine()
                billing_result = engine.evaluate(
                    icd10=icd10,
                    soap=soap,
                    transcript=transcript,
                    insurance_type=insurance_type,
                    config_rules=self.config.billing_rules if self.config else {},
                    profile_id=profile_id
                )

                result['billing_result'] = billing_result

            result['transcript'] = transcript
            result['profile_id'] = profile_id

            logger.info("AI workflow completed successfully")
            return result

        except Exception as e:
            logger.error(f"AI workflow failed: {e}", exc_info=True)
            raise

