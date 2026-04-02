# shared/learning_manager.py
import json
import os


class LearningManager:
    def __init__(self):
        self.memory_dir = os.path.expanduser("~/Documents/Kura")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.memory_path = os.path.join(self.memory_dir, "practice_memory.json")
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"icd_preferences": {}}

    def log_correction(self, transcript, ai_icd, user_icd):
        """Detects the medical topic and saves the practitioner's preferred ICD."""
        if ai_icd == user_icd:
            return

        keywords = ["skoliose", "impingement", "gonarthrose", "bandscheibe", "lymph"]
        t_low = transcript.lower()

        for k in keywords:
            if k in t_low:
                self.memory["icd_preferences"][k] = user_icd
                break

        try:
            with open(self.memory_path, "w") as f:
                if f is not None:
                    json.dump(self.memory, f, indent=2)
        except Exception as mem_err:
            print(f"Learning memory save error: {mem_err}")

    def get_relevant_prefs(self, transcript):
        """Returns a string of preferences to inject into the AI prompt."""
        t_low = transcript.lower()
        prefs = []
        for kw, code in self.memory["icd_preferences"].items():
            if kw in t_low:
                prefs.append(f"Für das Thema '{kw.capitalize()}' bevorzugt dieser Therapeut den Code {code}.")
        return "\n".join(prefs)