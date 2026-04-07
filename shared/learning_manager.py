# shared/learning_manager.py
"""
Learning Manager — accumulates accepted sessions and improves SOAP generation
over time via few-shot example injection.

Two complementary mechanisms:
  1. ICD preferences  – which code this therapist prefers for a given topic
  2. Few-shot examples – inject the most similar accepted session into the
                         prompt so the LLM mirrors the therapist's style

Data is stored in ~/Documents/Kura/:
  practice_memory.json      – ICD preferences (legacy + new)
  training_data.jsonl       – accepted transcript → SOAP pairs (one JSON per line)
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from ._compat import fix_windows_encoding

fix_windows_encoding()

logger = logging.getLogger("kura.learning")

# Maximum number of sessions kept in memory for retrieval (oldest pruned first)
_MAX_EXAMPLES = 500
# Number of few-shot examples injected into the prompt (1 is safest for GGUF context limits)
_INJECT_COUNT = 1


def _keyword_overlap(a: str, b: str) -> int:
    """Count shared content words between two strings (simple similarity proxy)."""
    stop = {"der", "die", "das", "und", "mit", "bei", "in", "an", "auf", "ist",
            "hat", "von", "zu", "im", "ein", "eine", "den", "des", "dem",
            "ich", "sie", "er", "es", "wir", "hier", "auch", "nach", "wie"}
    words_a = {w for w in re.findall(r'\b\w{4,}\b', a.lower()) if w not in stop}
    words_b = {w for w in re.findall(r'\b\w{4,}\b', b.lower()) if w not in stop}
    return len(words_a & words_b)


class LearningManager:
    def __init__(self):
        self.memory_dir = os.path.expanduser("~/Documents/Kura")
        os.makedirs(self.memory_dir, exist_ok=True)

        self.memory_path = os.path.join(self.memory_dir, "practice_memory.json")
        self.training_path = os.path.join(self.memory_dir, "training_data.jsonl")

        self.memory = self._load_memory()
        self._examples: list[dict] = []   # in-memory cache of training examples
        self._examples_loaded = False

    # ── Preferences (ICD, terminology) ───────────────────────────────────────

    def _load_memory(self) -> dict:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load practice memory, starting fresh: {e}")
        return {"icd_preferences": {}, "terminology": {}}

    def _save_memory(self) -> None:
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Learning memory save error: {e}")

    def log_correction(self, transcript: str, ai_icd: str, user_icd: str) -> None:
        """Record that the therapist changed an ICD code — used as a preference signal."""
        if ai_icd == user_icd:
            return

        keywords = [
            "skoliose", "impingement", "gonarthrose", "bandscheibe", "lymph",
            "hws", "lws", "schulter", "hüfte", "knie", "sprunggelenk",
            "frozen shoulder", "karpaltunnel", "plantarfasziitis", "tennisarm",
        ]
        t_low = transcript.lower()
        matched = next((k for k in keywords if k in t_low), None)

        if matched:
            self.memory.setdefault("icd_preferences", {})[matched] = user_icd
            logger.debug(f"ICD preference saved: '{matched}' → {user_icd}")
            self._save_memory()

    def get_relevant_prefs(self, transcript: str) -> str:
        """Return ICD/style preferences relevant to this transcript for prompt injection."""
        t_low = transcript.lower()
        prefs = []
        for kw, code in self.memory.get("icd_preferences", {}).items():
            if kw in t_low:
                prefs.append(f"Für das Thema '{kw.capitalize()}' bevorzugt dieser Therapeut den Code {code}.")
        return "\n".join(prefs)

    # ── Session examples (few-shot) ───────────────────────────────────────────

    def _load_examples(self) -> None:
        """Lazy-load training examples from disk into memory cache."""
        if self._examples_loaded:
            return
        self._examples = []
        if not os.path.exists(self.training_path):
            self._examples_loaded = True
            return
        try:
            with open(self.training_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._examples.append(json.loads(line))
                    except Exception:
                        pass
            logger.debug(f"Loaded {len(self._examples)} training examples")
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
        self._examples_loaded = True

    def log_session(
        self,
        transcript: str,
        soap: dict,
        icd10: str,
        profile_id: str = "KG",
        was_corrected: bool = False,
    ) -> None:
        """
        Persist an accepted (optionally corrected) session as a training example.

        Call this whenever the therapist clicks Save — pass was_corrected=True if
        they edited the AI output before saving.

        Args:
            transcript: The cleaned session transcript.
            soap: Final SOAP dict {"S": ..., "O": ..., "A": ..., "P": ...}.
            icd10: Final ICD-10 code after any corrections.
            profile_id: Diagnosis profile used (e.g. "EX_HWS").
            was_corrected: Whether the therapist edited the AI output.
        """
        if not transcript or not soap:
            return

        example = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "profile_id": profile_id,
            "icd10": icd10,
            "was_corrected": was_corrected,
            "transcript": transcript[:2000],   # cap to keep file manageable
            "soap": soap,
        }

        # Append to JSONL file
        try:
            with open(self.training_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.debug(f"Session logged (corrected={was_corrected}, icd={icd10})")
        except Exception as e:
            logger.error(f"Could not save training example: {e}")
            return

        # Update in-memory cache
        self._load_examples()    # ensure cache is warm
        self._examples.append(example)

        # Prune oldest entries beyond the cap (in-memory only; file keeps all)
        if len(self._examples) > _MAX_EXAMPLES:
            self._examples = self._examples[-_MAX_EXAMPLES:]

    def get_few_shot_examples(self, transcript: str, profile_id: str = "") -> list[dict]:
        """
        Return the _INJECT_COUNT most similar past sessions for use in the prompt.

        Similarity = keyword overlap, with strong preference for same profile_id.
        Incompatible profiles (e.g., spine vs extremity) are excluded to prevent
        context contamination.

        Returns:
            List of example dicts with keys: transcript, soap, icd10.
        """
        self._load_examples()
        if not self._examples:
            return []

        # Define incompatible profile groups to prevent contamination
        _profile_groups = {
            "SPINE": {"EX_HWS", "EX_LWS", "EX_BWS"},
            "EXTREMITY": {"EX_SCHULTER", "EX_KNIE", "EX_HUefte", "EX_HUFTE", "EX_FUSS", "EX_ELLBOGEN"},
            "SPECIAL": {"LY", "AT", "GEB", "PAED", "NEURO", "GER"},
        }

        # Determine current profile group
        current_group = None
        for group, profiles in _profile_groups.items():
            if profile_id in profiles:
                current_group = group
                break

        scored = []
        for ex in self._examples:
            ex_profile = ex.get("profile_id", "")

            # STRICT FILTER: Exclude examples from incompatible profile groups
            if current_group:
                ex_group = None
                for group, profiles in _profile_groups.items():
                    if ex_profile in profiles:
                        ex_group = group
                        break
                # Skip if from different group (prevents spine tests in extremity sessions)
                if ex_group and ex_group != current_group:
                    continue

            score = _keyword_overlap(transcript, ex.get("transcript", ""))

            # STRONG preference for exact profile match (increased from +10 to +50)
            if profile_id and ex_profile == profile_id:
                score += 50
            # MEDIUM preference for same group but different profile
            elif current_group and ex_group == current_group:
                score += 20

            if ex.get("was_corrected"):
                score += 5
            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ex for score, ex in scored[:_INJECT_COUNT] if score > 0]
        return top

    def format_few_shot_block(self, transcript: str, profile_id: str = "") -> str:
        """
        Return a formatted few-shot block ready to inject into the system prompt.

        Returns empty string if no relevant examples exist yet.

        CRITICAL: Only shows O/A/P fields from past sessions to prevent
        context contamination. The S-field MUST come from the current transcript only.
        """
        examples = self.get_few_shot_examples(transcript, profile_id)
        if not examples:
            return ""

        lines = ["BEISPIEL AUS VERGANGENEN SITZUNGEN (Stil dieses Therapeuten):"]
        lines.append("⚠️ ACHTUNG: Das S-Feld (Subjektiv) MUSS aus dem AKTUELLEN Transkript kommen!")
        lines.append("⚠️ Die folgenden Beispiele zeigen nur O/A/P zur Stil-Orientierung.\n")
        for i, ex in enumerate(examples, 1):
            soap = ex.get("soap", {})
            lines.append(f"--- Beispiel {i} (Profil: {ex.get('profile_id', 'n.d.')}) ---")
            # REMOVED: S-field to prevent contamination
            lines.append(f"O: {soap.get('O', '')}")
            lines.append(f"A: {soap.get('A', '')}  |  ICD-10: {ex.get('icd10', '')}")
            lines.append(f"P: {soap.get('P', '')}")
        lines.append("--- Ende Beispiel ---")
        lines.append("Übernehme den Detailgrad, die Terminologie und den Stil des obigen Beispiels.")
        lines.append("⚠️ WICHTIG: Erstelle das S-Feld NUR aus dem AKTUELLEN Transkript, NICHT aus dem Beispiel!")

        return "\n".join(lines)

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary of accumulated learning data."""
        self._load_examples()
        corrected = sum(1 for e in self._examples if e.get("was_corrected"))
        profiles: dict[str, int] = {}
        for e in self._examples:
            pid = e.get("profile_id", "KG")
            profiles[pid] = profiles.get(pid, 0) + 1
        return {
            "total_sessions": len(self._examples),
            "corrected_sessions": corrected,
            "icd_preferences": len(self.memory.get("icd_preferences", {})),
            "profiles": profiles,
        }

    def export_training_data(self, output_path: Optional[str] = None) -> str:
        """
        Export training data in instruction-tuning format (Alpaca/Unsloth compatible).

        Each record:  {"instruction": <system>, "input": <transcript>, "output": <soap_json>}

        Returns the path to the exported file.
        """
        self._load_examples()
        if not self._examples:
            logger.warning("No training examples to export")
            return ""

        out_path = output_path or os.path.join(self.memory_dir, "finetune_dataset.jsonl")
        system_msg = (
            "Du bist ein klinischer Dokumentationsexperte für deutsche Physiotherapie. "
            "Erstelle aus dem Transkript einen strukturierten SOAP-Befund als JSON."
        )
        count = 0
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                for ex in self._examples:
                    record = {
                        "instruction": system_msg,
                        "input": ex.get("transcript", ""),
                        "output": json.dumps(
                            {"soap": ex.get("soap", {}), "icd10": ex.get("icd10", "")},
                            ensure_ascii=False
                        ),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            logger.info(f"Exported {count} training examples to {out_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ""

        return out_path
