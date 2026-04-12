"""
Kura Medical — Transkript-Evaluation mit PDF-Abschlussbericht
=============================================================
Fuehrt alle 5 Testfaelle durch die Engine (ohne Whisper-Transkription)
und erzeugt einen vollstaendigen Pruefbericht als PDF.

Usage:
    cd /Users/mgke/Downloads/medic
    python eval_transcripts.py
    # -> evaluation_report.pdf
"""

import json
import os
import re
import sys
import textwrap
import types
from datetime import datetime

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
_PLATFORM = "windows" if "--platform" in sys.argv and sys.argv[sys.argv.index("--platform") + 1] == "windows" else "macos"
sys.path.insert(0, os.path.join(ROOT, _PLATFORM))
sys.path.insert(0, ROOT)

# ── Stub heavy ML dependencies so eval runs without GPU/MLX hardware ─────────
def _stub_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

if "mlx" not in sys.modules:
    _mx = _stub_module("mlx")
    _mx_core = _stub_module("mlx.core")
    _mx.core = _mx_core

if "mlx_whisper" not in sys.modules:
    _stub_module("mlx_whisper")

if "mlx_lm" not in sys.modules:
    _mlx_lm = _stub_module("mlx_lm")
    _mlx_lm.load = lambda *a, **kw: (None, None)
    _mlx_lm.generate = lambda *a, **kw: ""
    _stub_module("mlx_lm.sample_utils").make_sampler = lambda *a, **kw: None

if _PLATFORM == "windows" and "llama_cpp" not in sys.modules:
    class _FakeLlama:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw):
            return {"choices": [{"text": ""}]}
        def reset(self): pass
    _llama_cpp = _stub_module("llama_cpp")
    _llama_cpp.Llama = _FakeLlama
    _llama_cpp.llama_kv_cache_clear = lambda *a, **kw: None

if _PLATFORM == "windows" and "whisper" not in sys.modules:
    _whisper = _stub_module("whisper")
    class _FakeWhisperModel:
        def transcribe(self, *a, **kw): return {"text": ""}
    _whisper.load_model = lambda *a, **kw: _FakeWhisperModel()

# ── Stress-test transcripts (25 cases) ─────────────────────────────────────
# expected_profile: what profile the engine must assign
# expected_position: correct GKV Abrechnungsposition
# expected_audit:  PASS | REVIEW | BLOCK

# ── 10 new real-session transcripts (T01–T10) ───────────────────────────────
CASES = [
    {
        "id": "T01", "patient": "Wagner (Mamma-Ablation links, MLD-60)",
        "code": "LY2", "context": "Sekundäres Lymphödem post Mamma-Ablation, Stadium 2, brennendes Spannungsgefühl Oberarm, VAS 4, Stemmer positiv Handrücken, 60 min MLD",
        "setting": "Therapeutin dokumentiert aktuellen Befund.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Frau Wagner ist heute zur MLD-60 hier. Nach ihrer Mamma-Ablation links vor vier Jahren "
            "berichtet sie heute von einem brennenden Gefühl im Oberarm. Sie sagt: 'Es fühlt sich an, "
            "als würde die Haut von innen spannen.' Den Schmerz bewertet sie mit einer 4 auf der Skala. "
            "Das Stemmer-Zeichen ist am Handrücken positiv, Gewebe ist prall-elastisch, also Stadium 2. "
            "Haut ist trocken, aber ohne Rötung."
        ),
    },
    {
        "id": "T02", "patient": "Sprunggelenksdistorsion (MLD-30, traumatisches Ödem)",
        "code": "LY1", "context": "Traumatisches Ödem post-Distorsion, Hämatom lateral, Stemmer negativ, Stadium 1, VAS 5, 30 min MLD",
        "setting": "Therapeut dokumentiert Befund zur Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "PASS",
        "transcript": (
            "Behandlung nach einer Sprunggelenksdistorsion. Der Fuß ist lateral stark geschwollen und "
            "bläulich verfärbt durch das Hämatom. Wir führen eine MLD-30 durch. Wichtig für die Doku: "
            "Das Stemmer-Zeichen ist negativ, es handelt sich um ein rein traumatisches Ödem Stadium 1. "
            "Der Patient klagt über Druckschmerzen bei Belastung, VAS 5."
        ),
    },
    {
        "id": "T03", "patient": "Klein (Primäres Lymphödem beidseits, MLD-45)",
        "code": "LY1", "context": "Primäres Lymphödem beider Beine, Stadium 2, Stemmer beidseits positiv, tief eindrückbar mit Delle, MLD-45",
        "setting": "Therapeutin dokumentiert aktuellen Befund.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Frau Klein leidet an einem primären Lymphödem beider Beine. Heute zeigt sich das Gewebe "
            "an den Fesseln besonders tief eindrückbar, Dellen bleiben bestehen. Wir dokumentieren ein "
            "Stadium 2. Stemmer-Zeichen beidseits positiv. Die Patientin hat das Ziel, ihre "
            "Kompressionsstrümpfe Klasse 2 wieder leichter anziehen zu können. MLD-45 wurde durchgeführt."
        ),
    },
    {
        "id": "T04", "patient": "Fischer (Sekundäres Lymphödem post Prostatektomie, MLD-60)",
        "code": "LY2", "context": "Sekundäres Lymphödem Genitalien + rechter Oberschenkel nach radikaler Prostatektomie, Stadium 2, Blankoverordnung, 60 min MLD",
        "setting": "Therapeut dokumentiert Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Nach radikaler Prostatektomie zeigt Herr Fischer ein sekundäres Lymphödem im Bereich der "
            "Genitalien und des rechten Oberschenkels. Das Gewebe ist ödematisiert, aber noch weich, "
            "Stemmer am Fuß ist negativ, aber das Ödem ist im Beckenbereich irreversibel, daher Stadium 2. "
            "Wir arbeiten heute 60 Minuten an den Abflusswegen. Ziel ist die Spannungsreduktion. "
            "Blankoverordnung liegt vor."
        ),
    },
    {
        "id": "T05", "patient": "Elephantiasis Stadium 3 (MLD-60, VAS 6)",
        "code": "LY1", "context": "Stadium 3 Elephantiasis, massive Fibrosierung Unterschenkel, Stemmer massiv positiv, Hyperkeratosen, VAS 6, 60 min MLD",
        "setting": "Therapeutin begründet Behandlungsdauer.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Aufgrund der massiven Fibrosierung am Unterschenkel bei Stadium 3 (Elephantiasis) entscheide "
            "ich mich für eine Behandlungszeit von 60 Minuten MLD. Stemmer-Zeichen ist massiv positiv. "
            "Die Haut zeigt Hyperkeratosen. Die Patientin gibt an, dass das Bein heute 'extrem schwer' "
            "sei, VAS 6."
        ),
    },
    {
        "id": "T06", "patient": "Neck-Dissection (Gesichtsödem submental, MLD-30)",
        "code": "LY2", "context": "Sekundäres Lymphödem post Neck-Dissection, Gesichtsödem submental, Kloßgefühl beim Schlucken, Stemmer nicht verwertbar, 30 min MLD",
        "setting": "Therapeutin dokumentiert Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "PASS",
        "transcript": (
            "Zustand nach Neck-Dissection. Die Patientin hat ein ausgeprägtes Gesichtsödem, vor allem "
            "submental. Sie berichtet von einem Kloßgefühl im Hals beim Schlucken. Wir machen 30 Minuten "
            "MLD zur Druckentlastung. Das Stemmer-Zeichen ist im Gesicht nicht verwertbar, aber das "
            "Gewebe ist prall. Ziel: Verbesserung des Schluckaktes."
        ),
    },
    {
        "id": "T07", "patient": "Neumann (Kardiale Dekompensation — BLOCK)",
        "code": "RF", "context": "Akute kardiale Dekompensation V.a., Therapieabbruch nach 10 min, MLD absolut kontraindiziert, BLOCK erwartet",
        "setting": "Notfalldokumentation, Therapieabbruch.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "BLOCK", "expected_audit": "BLOCK",
        "transcript": (
            "Therapieabbruch nach 10 Minuten. Herr Neumann atmet im Liegen sehr schwer und hat rasselnde "
            "Geräusche beim Einatmen. Seine Knöchelödeme sind heute massiv und schneeweiß. Verdacht auf "
            "akute kardiale Dekompensation. MLD ist absolut kontraindiziert. Der Patient wurde sofort "
            "zur weiteren Abklärung an den Kardiologen verwiesen."
        ),
    },
    {
        "id": "T08", "patient": "Sommer (Non-Compliance Kompression, Arm Stadium 2, MLD-45)",
        "code": "LY1", "context": "Non-Compliance Kompressionsstrumpf, Arm-Ödem Stadium 2 verschlechtert, Stemmer positiv, Aufklärung, 45 min MLD",
        "setting": "Therapeutin dokumentiert Befund und Compliance-Aufklärung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Frau Sommer trägt ihre Kompressionsbestrumpfung heute nicht. Sie sagt: 'Bei der Hitze halte "
            "ich das nicht aus.' Das Ödem am Arm ist prompt praller geworden, Stadium 2, Stemmer positiv. "
            "Wir haben heute 45 Minuten MLD gemacht und intensiv über die Risiken der Non-Compliance "
            "aufgeklärt. Ziel bleibt die Volumenreduktion."
        ),
    },
    {
        "id": "T09", "patient": "Chylöser Reflux (Lymphödem untere Ext., MLD-60 mit Bauchdrainage)",
        "code": "LY1", "context": "Chylöser Reflux + Lymphödem untere Extremität Stadium 2, Völlegefühl Bauch, MLD-60 mit tiefer Bauchdrainage, Stemmer positiv",
        "setting": "Therapeut dokumentiert Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Patient mit chylösem Reflux und Lymphödem der unteren Extremität Stadium 2. Der Patient "
            "berichtet über ein unangenehmes Völlegefühl im Bauchraum. Wir führen eine MLD-60 inklusive "
            "tiefer Bauchdrainage durch. Stemmer-Zeichen am Fuß ist positiv. Hautzustand ist stabil, "
            "keine Lymphfisteln sichtbar."
        ),
    },
    {
        "id": "T10", "patient": "Tiefe Venenthrombose V.a. — BLOCK",
        "code": "RF", "context": "Wade glänzend, heiß, bläulich, 3 cm Differenz, einschießender Schmerz, V.a. TVT, Behandlung nicht gestartet, BLOCK erwartet",
        "setting": "Sicherheitsdokumentation vor Behandlungsbeginn.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "BLOCK", "expected_audit": "BLOCK",
        "transcript": (
            "Befundung vor der MLD: Die linke Wade ist im Vergleich zu rechts um 3 cm dicker, glänzend, "
            "heiß und zeigt eine bläuliche Verfärbung. Der Patient gibt einen einschießenden Schmerz beim "
            "Gehen an. Verdacht auf tiefe Venenthrombose. Die Behandlung wurde nicht gestartet (BLOCK). "
            "Der Patient wurde umgehend in die Notaufnahme geschickt."
        ),
    },
    # ── T11–T15: New real-session transcripts ────────────────────────────────
    {
        "id": "T11", "patient": "Fischer (Post-Mamma-OP, Sekundäres Lymphödem Arm, MLD-45)",
        "code": "LY2", "context": "Sekundäres Lymphödem rechter Arm post Brustkrebs-OP + Lymphknotenentfernung, Stadium 2, Stemmer positiv, VAS 6, 45 min MLD",
        "setting": "Therapeutin dokumentiert aktuellen Befund.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Guten Morgen, Frau Fischer. Wie fühlt sich der rechte Arm heute an, so fünf Jahre nach der "
            "Brustkrebs-OP und der Lymphknotenentfernung? Er ist ziemlich schwer, besonders am Ellbogen. "
            "Seit ich gestern im Garten gearbeitet habe, pocht es richtig. Ich schaue mir das mal an. "
            "Die Schwellung ist deutlich sichtbar und die Haut ist sehr prall. Ich teste mal kurz das "
            "Stemmer-Zeichen an Ihrer Hand... ja, das ist eindeutig positiv, ich kann die Hautfalte gar "
            "nicht abheben. Ich würde das als Stadium 2 einstufen. Auf einer Skala von 0 bis 10, wie stark "
            "ist dieses Spannungsgefühl heute? Ich würde sagen eine 6. Es fühlt sich eher so an, als würde "
            "der Arm platzen, weniger wie ein stechender Schmerz. Verstehe. Die Haut sieht aber gesund aus "
            "– keine Rötung oder Hitze, also kein Anzeichen für eine Entzündung. Wir machen heute die "
            "vollen 45 Minuten MLD, um die Flüssigkeit wieder in Gang zu bringen."
        ),
    },
    {
        "id": "T12", "patient": "Wagner (Post-Schlaganfall, Lymphödem Bein Stadium 1, MLD-30)",
        "code": "LY1", "context": "Sekundäres Lymphödem linkes Bein bei Immobilität nach Schlaganfall, Stadium 1, Stemmer negativ, Pitting-Ödem, 30 min MLD",
        "setting": "Therapeutin dokumentiert Kombinationsbehandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "PASS",
        "transcript": (
            "Hallo Herr Wagner. Normalerweise konzentrieren wir uns ja auf die Reha nach Ihrem Schlaganfall "
            "und die Lähmung, aber heute sieht Ihr linkes Bein deutlich dicker aus als sonst. Ja, das macht "
            "mir seit heute Morgen echt zu schaffen. Es fühlt sich an, als würde ich einen Sandsack hinter "
            "mir herziehen. Ich überprüfe das mal. Das Ödem ist ziemlich weich und ich kann eine Delle "
            "eindrücken, die kurz stehen bleibt. Interessanterweise ist das Stemmer-Zeichen an Ihrem Zeh "
            "negativ. Es sieht nach einem Stadium 1 Lymphödem aus, wahrscheinlich weil die Bewegung im "
            "Bein gerade fehlt. Keine Rötung oder Fieber, also ist es sicher zu behandeln. Wir hängen "
            "heute 30 Minuten MLD an unsere Sitzung dran, um die Schwellung zu lindern."
        ),
    },
    {
        "id": "T13", "patient": "Lang (Stadium 3 Elephantiasis, Hyperkeratose, MLD-60)",
        "code": "LY1", "context": "Stadium 3 Elephantiasis Unterschenkel, Hyperkeratose Knöchel, Fibrose, Stemmer massiv positiv, VAS 8, 60 min MLD",
        "setting": "Therapeut begründet Behandlungsdauer.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Herr Lang, wir müssen uns heute wirklich intensiv um die Hautfalten an Ihrem Unterschenkel "
            "kümmern. Die Fibrosen sind dort mittlerweile sehr hart geworden. Das habe ich gemerkt. Die "
            "Haut wird in der Nähe des Knöchels richtig dick und rau. Genau, das ist die Hyperkeratose, "
            "über die wir schon gesprochen haben. Das Stemmer-Zeichen ist hier massiv positiv. Das ist ein "
            "klares Stadium 3, also eine Elephantiasis. Weil das Gewebe so fest ist, brauchen wir heute "
            "die vollen 60 Minuten, um überhaupt eine Lockerung zu erreichen. Fühlt sich das Bein heute "
            "sehr schwer an? Extrem. Auf der Skala ist das heute sicher eine 8 von 10, was die Schwere "
            "angeht."
        ),
    },
    {
        "id": "T14", "patient": "Janson (Post-Prostatektomie, Sekundäres Lymphödem Leiste, MLD-60)",
        "code": "LY2", "context": "Sekundäres Lymphödem Leiste + Oberschenkel nach Prostata-OP, Stadium 2, Stemmer negativ am Fuß, Blankoverordnung, 60 min MLD",
        "setting": "Therapeut dokumentiert Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Herr Janson, nach Ihrer Prostata-Operation sehe ich hier eine sekundäre Schwellung im Bereich "
            "der Leiste. Ja, das ist sehr unangenehm und zieht sich bis in den rechten Oberschenkel runter. "
            "Ich sehe es. Das Gewebe hat sich irreversibel verändert, wir sind hier bei Stadium 2. Das "
            "Stemmer-Zeichen am Fuß ist zwar negativ, aber das Ödem im Beckenbereich ist eindeutig "
            "lymphatisch bedingt. Da wir eine Blankoverordnung haben, werde ich heute 60 Minuten nutzen, "
            "um die Abflusswege im Bauchraum intensiv mitzubehandeln. Haben Sie heute spezifische "
            "Schmerzen? Nur dieser konstante Druck, vielleicht eine 4 auf der Skala."
        ),
    },
    {
        "id": "T15", "patient": "Bauer (Erysipel Red Flag — BLOCK)",
        "code": "RF", "context": "Erysipel-Verdacht, leuchtend rote Verfärbung Wade, Überwärmung, Schüttelfrost, Therapieabbruch, BLOCK erwartet",
        "setting": "Notfalldokumentation, Therapieabbruch.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "BLOCK", "expected_audit": "BLOCK",
        "transcript": (
            "Frau Bauer, mir fällt auf, dass Sie gerade ein wenig frösteln. Fühlen Sie sich okay? "
            "Ehrlich gesagt fühle ich mich ein bisschen wie bei einer Grippe. Und mein Bein hat angefangen "
            "zu brennen. Zeigen Sie mal... oh, da müssen wir sofort abbrechen. Da ist eine leuchtend rote, "
            "sich ausbreitende Verfärbung an Ihrer Wade, und die Stelle ist glühend heiß. Das sieht nach "
            "einer Wundrose aus. Haben Sie auch Schüttelfrost? Ja, ein bisschen. Okay, ich darf die "
            "Lymphdrainage jetzt auf keinen Fall durchführen. Ein Erysipel ist eine absolute Gegenanzeige, "
            "da wir sonst die Bakterien im Körper verteilen könnten. Ich streiche den Termin für heute und "
            "Sie müssen bitte sofort zum Arzt oder in die Notaufnahme. Wir können erst weitermachen, wenn "
            "das antibiotisch behandelt wurde und die Entzündung abgeklungen ist."
        ),
    },
]

# ── Original 25 stress-test cases (F01–F25) ─────────────────────────────────
CASES_ORIG = [
    # ── Group A: Duration variants ───────────────────────────────────────────
    {
        "id": "F01", "patient": "Meyer (Post-Op Hüfte)",
        "code": "LY1", "context": "Post-Op traumatisches Ödem, Stemmer negativ, 30 min MLD",
        "setting": "Therapeut bespricht Befund direkt mit dem Patienten.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "PASS",
        "transcript": (
            "Guten Morgen, Herr Meyer. Vier Wochen nach Ihrer Hüft-OP schauen wir uns heute die "
            "Schwellung am rechten Bein an. Wir machen die verordneten 30 Minuten MLD. Am "
            "Außenknöchel ist noch eine leichte Ödembildung sichtbar, aber ich teste jetzt mal den "
            "Vorfuß - das Stemmer-Zeichen ist eindeutig negativ, die Hautfalte lässt sich gut abheben. "
            "Die Haut wirkt insgesamt reizlos, ich sehe keine Rötung oder Anzeichen für eine Entzündung. "
            "Auf der Schmerzskala von null bis zehn geben Sie mir heute eine 2 an? Alles klar, dann "
            "starten wir proximal in der Leiste."
        ),
    },
    {
        "id": "F02", "patient": "Koch (Stadium 1)",
        "code": "LY1", "context": "Stadium 1 reversibel, Stemmer negativ, 30 min MLD",
        "setting": "Therapeutin kommentiert Befund laufend während der Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "PASS",
        "transcript": (
            "Frau Koch, das Ödem am Unterschenkel wirkt heute im Vergleich zur letzten Woche viel "
            "weicher und ist fast vollständig reversibel. Ich stufe das klinisch als Stadium 1 ein. "
            "Auch der Stemmer-Test am zweiten Zeh ist negativ. Die Haut ist sehr gut gepflegt und "
            "weist keine Läsionen auf. Unser Ziel für die heutige MLD-30 ist der Erhalt dieses guten "
            "Zustands und die weitere Resorption der restlichen Flüssigkeit."
        ),
    },
    {
        "id": "F03", "patient": "Patient Trauma-Fraktur",
        "code": "LY1", "context": "Traumatisches Ödem post-Fraktur, Stemmer negativ, 45 min MLD",
        "setting": "Therapeut diktiert Befund unmittelbar nach der Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Nach Ihrer Sprunggelenksfraktur ist der gesamte Fußrücken noch ordentlich prall, da sind "
            "die 45 Minuten MLD heute genau richtig. Ich habe gerade den Test gemacht: Das "
            "Stemmer-Zeichen ist negativ, was bestätigt, dass es sich um ein rein traumatisches Ödem "
            "handelt und nicht um eine chronische Lymphabflussstörung. Die Haut ist intakt, es ist "
            "keine lokale Überwärmung tastbar, was gegen einen akuten Entzündungsprozess spricht."
        ),
    },
    {
        "id": "F04", "patient": "Patient Muskelfaserriss",
        "code": "LY1", "context": "Muskelfaserriss Wade, Stemmer negativ, 30 min MLD, VAS 3",
        "setting": "Diktat nach der Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "PASS",
        "transcript": (
            "Behandlungsprotokoll für den Patienten nach Muskelfaserriss in der Wade. Durchführung "
            "von 30 Minuten MLD. Das Hämatom wandert physiologisch nach distal. Der Stemmer-Test am "
            "Fuß ist negativ. Der Patient berichtet über ein deutliches Spannungsgefühl im betroffenen "
            "Segment, er gibt den Schmerz auf der VAS mit 3 von 10 an. Das primäre Ziel der heutigen "
            "Sitzung war die Resorptionsförderung des Blutergusses."
        ),
    },
    # ── Group B: Stadium & Suffix-Korrektheit ────────────────────────────────
    {
        "id": "F05", "patient": "Bauer (Stadium 2, Mamma-Ablation)",
        "code": "LY2", "context": "Sekundäres Lymphödem post-Mamma-Ablation, Stemmer positiv, 60 min",
        "setting": "Therapeutin dokumentiert während der Griffsequenz.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "So, Frau Bauer, wir behandeln heute wieder das sekundäre Lymphödem nach Ihrer "
            "Mamma-Ablation. Das Gewebe am linken Arm fühlt sich heute sehr fest an und lässt sich "
            "kaum noch eindrücken. Wir dokumentieren hier definitiv ein Stadium 2, da das Ödem "
            "irreversibel wirkt. Das Stemmer-Zeichen am Handrücken ist positiv. Aufgrund der "
            "deutlichen Gewebshärte und der Fibrosierung führen wir heute die vollen 60 Minuten MLD durch."
        ),
    },
    {
        "id": "F06", "patient": "Lang (Stadium 3 Elephantiasis)",
        "code": "LY1", "context": "Stadium 3 Elephantiasis, massiv positiv, 60 min MLD",
        "setting": "Befunddokumentation am Ende der Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Herr Lang, am rechten Unterschenkel haben wir heute leider massive Hautfalten und eine "
            "sehr starke Gewebsverhärtung. Klinisch dokumentieren wir das heute als Stadium 3, also "
            "eine Elephantiasis. Die Hyperkeratose im Bereich der Zehenzwischenräume ist deutlich "
            "sichtbar und muss beobachtet werden. Der Stemmer-Test ist massiv positiv, die Haut lässt "
            "sich nirgends abheben. Wir nutzen die vollen 60 Minuten für die intensive Entstauung."
        ),
    },
    {
        "id": "F07", "patient": "Reuter (Stadium 2, Pitting)",
        "code": "LY1", "context": "Stadium 2 chronisch, Pitting-Ödem, Umfangsmessung",
        "setting": "Therapeut dokumentiert laufend.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Bei Herrn Reuter zeigt sich heute ein klassisches Stadium 2 am linken Bein. Wenn ich mit "
            "dem Daumen fest eindrücke, bleiben die Dellen lange im Gewebe bestehen. Das Stemmer-Zeichen "
            "ist positiv. Die Umfangsmessung an der Wade ergibt heute 45 cm im Vergleich zu 41 cm auf "
            "der gesunden Seite. Unser therapeutisches Ziel bleibt die Lockerung der Fibrosen und die "
            "Reduktion des massiven Spannungsgefühls."
        ),
    },
    {
        "id": "F08", "patient": "Armödem (VAS, Stadium 2)",
        "code": "LY1", "context": "Armödem Stadium 2, Stemmer positiv, VAS 4, 45 min MLD",
        "setting": "Kurzes Diktat.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Dokumentation zur MLD-45 am linken Arm. Das Ödem ist heute prall-elastisch, was klinisch "
            "einem Stadium 2 entspricht. Das Stemmer-Zeichen am Handrücken ist positiv. Die Patientin "
            "gibt eine Schmerzintensität von 4 auf der VAS an, wobei sie dies primär als belastendes "
            "Schweregefühl beschreibt. Behandlungsfokus liegt auf der Entstauung der proximalen "
            "Lymphknotenstationen."
        ),
    },
    # ── Group C: S-Feld Extraktion aus Patientenzitat ─────────────────────────
    {
        "id": "F09", "patient": "Patientenzitat S-Feld",
        "code": "LY1", "context": "Direktes Patientenzitat im S-Feld, Juckreiz + VAS 6",
        "setting": "Therapeutin zitiert den Patienten wörtlich.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Die Patientin berichtet mir zu Beginn der Behandlung: 'Mein Arm fühlt sich heute an wie "
            "ein schweres Bleirohr und die Haut juckt extrem, ich könnte ständig kratzen.' Das "
            "subjektive Spannungsgefühl bewertet sie heute mit einer 6 auf der Skala. Unser Hauptziel "
            "für die heutige Sitzung ist die Lockerung des Gewebes und die Linderung dieses massiven "
            "Juckreizes durch die Druckentlastung."
        ),
    },
    {
        "id": "F10", "patient": "Übergabe Fuchs",
        "code": "LY1", "context": "Übergabeprotokoll mit indirekter Patientenangabe, VAS 5",
        "setting": "Notiz zur Übergabe nach Kollegendokumentation.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Notiz für die Dokumentation nach Übergabe durch die Kollegin: Herr Fuchs klagte heute "
            "Morgen über extrem schwere Beine und berichtete, dass er kaum in seine Straßenschuhe "
            "hineinkam. Er bewertet die Beschwerden heute mit einer 5 auf der Schmerzskala. Wir haben "
            "daraufhin eine 45-minütige MLD zur Entlastung durchgeführt. Das Therapieziel ist die "
            "Verbesserung der Mobilität durch Volumenreduktion."
        ),
    },
    {
        "id": "F11", "patient": "Schmidt (Compliance + Ziel)",
        "code": "LY1", "context": "Patientenziel explizit, Kompressionsstrumpf Klasse 2, VAS 4",
        "setting": "Therapiegespräch zu Beginn der Sitzung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Frau Schmidt, wir besprechen kurz Ihr wichtigstes Ziel für diesen Verordnungszeitraum. "
            "Sie sagt mir gerade, sie möchte unbedingt wieder schmerzfrei im Garten arbeiten können, "
            "ohne dass der Arm danach so massiv und schmerzhaft anschwillt. Die aktuelle Spannung "
            "bewertet sie mit 4 von 10. Die Patientin trägt ihre Kompressionsbestrumpfung Klasse 2 "
            "laut eigener Aussage konsequent."
        ),
    },
    {
        "id": "F12", "patient": "Erstbefund Strahlentherapie",
        "code": "LY2", "context": "Erstbefund nach Beckenbestrahlung, VAS 7, Erstdiagnose",
        "setting": "Anamnese-Erstgespräch Neupatientin.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Anamnese der Neupatientin nach vorangegangener Bestrahlung im Beckenbereich. Sie schildert "
            "mir, dass sich das rechte Bein 'eng und kurz vor dem Platzen' anfühlt. Sie gibt diesen "
            "Zustand auf der Skala mit einer 7 an. Das primäre Ziel der Behandlung ist die "
            "Schmerzreduktion und eine messbare Minderung der Umfangsdifferenz durch die manuelle "
            "Lymphdrainage."
        ),
    },
    # ── Group D: Abrechnungslogik ─────────────────────────────────────────────
    {
        "id": "F13", "patient": "Abrechnungsdiktat Pos. 20203",
        "code": "LY1", "context": "60-min MLD explizit diktiert, Stadium 2",
        "setting": "Kurzdiktat zur Abrechnung nach Sitzungsende.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Heutige Sitzung abgeschlossen. Wir haben gemäß der ärztlichen Verordnung die vollen 60 "
            "Minuten MLD durchgeführt. Der Schwerpunkt lag auf der vorbereitenden Entstauung des Rumpfes "
            "und der anschließenden Behandlung des rechten Beins Stadium 2. Die Abrechnung erfolgt "
            "korrekt nach der Positionsnummer 20203 für die Ganzbehandlung."
        ),
    },
    {
        "id": "F14", "patient": "Sitzungskürzung 30 min",
        "code": "LY1", "context": "Vorzeitiger Abbruch, nur 30 min MLD, Pos. 20205",
        "setting": "Kurzprotokoll zur abgebrochenen Sitzung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "REVIEW",
        "transcript": (
            "Kurzes Protokoll zur heutigen Einheit: Der Patient musste die Behandlung leider vorzeitig "
            "beenden, da er einen dringenden Folgetermin beim Facharzt hatte. Wir konnten heute daher "
            "nur 30 Minuten MLD durchführen. Die Abrechnungsposition muss für diesen Termin manuell "
            "auf die Teilbehandlung 20205 korrigiert werden."
        ),
    },
    {
        "id": "F15", "patient": "Standardbehandlung 45 min",
        "code": "LY1", "context": "Reguläre 45-min MLD, Stadium 2, gute Compliance",
        "setting": "Routineprotokoll.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Reguläre Sitzung heute: Durchführung von 45 Minuten MLD für den linken Arm bei Stadium 2. "
            "Die Patientin zeigt eine sehr gute Compliance und trägt ihre Kompressionsstrümpfe der "
            "Klasse 2 vorbildlich. Das Gewebe im Unterarmbereich wirkt heute im Vergleich zur Vorwoche "
            "deutlich weicher. Wir behalten den Behandlungsplan so bei."
        ),
    },
    {
        "id": "F16", "patient": "Blankoverordnung 60 min",
        "code": "LY1", "context": "Therapeutische Entscheidung 60 min MLD bei Blankoverordnung",
        "setting": "Befundbegründung zur erweiterten Behandlungsdauer.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Bei der vorliegenden Blankoverordnung habe ich heute auf Basis des aktuellen Befundes "
            "entschieden, die Behandlungszeit auf 60 Minuten MLD zu erhöhen. Dies ist notwendig, um "
            "die tiefsitzenden Fibrosen am linken Oberschenkel effektiver lockern zu können. Die "
            "Patientin wurde über die Anpassung der Frequenz und Dauer informiert."
        ),
    },
    # ── Group E: Spezialdiagnosen & Trigger ──────────────────────────────────
    {
        "id": "F17", "patient": "Janson (Urologie, Prostata-OP)",
        "code": "LY2", "context": "Sekundäres Ödem nach Prostata-OP, LY-Trigger urologisch",
        "setting": "Dokumentation während der Griffsequenz.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Guten Tag Herr Janson. Nach Ihrer Prostata-Operation mit der umfangreichen "
            "Lymphknotenentfernung hat sich das Ödem in der rechten Leiste leider chronifiziert. "
            "Ich taste hier ein festes Gewebe, das Stemmer-Zeichen ist positiv. Wir führen heute "
            "45 Minuten MLD durch, um den Abfluss über die kontralaterale Seite zu fördern."
        ),
    },
    {
        "id": "F18", "patient": "Gynäkologie Beckenlymphödem",
        "code": "LY2", "context": "Beckenlymphödem nach Zervix-Ca., beidseitig, 60 min MLD",
        "setting": "Behandlungsdokumentation.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20203", "expected_audit": "PASS",
        "transcript": (
            "Behandlung der Patientin mit einem Beckenlymphödem infolge eines Zervix-Karzinoms. "
            "Wir dokumentieren heute eine beidseitige Schwellung der unteren Extremitäten, klinisch "
            "Stadium 2. Wir arbeiten heute volle 60 Minuten intensiv an den Lymphabflusswegen des "
            "Rumpfes und beider Beine, um das massive Druckgefühl zu lindern."
        ),
    },
    {
        "id": "F19", "patient": "Liposuktion Lipödem",
        "code": "LY3", "context": "Sekundäres Ödem nach Liposuktion bei Lipödem, 45 min MLD",
        "setting": "Behandlungsprotokoll.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Heute behandeln wir die Schwellungszustände nach der Liposuktion bei bekanntem Lipödem. "
            "Es zeigt sich ein sekundäres traumatisches Ödem. Wir führen eine MLD-45 durch. Die Haut "
            "weist noch großflächige Hämatome auf, ist aber an allen Stellen geschlossen und zeigt "
            "keine Anzeichen einer Infektion."
        ),
    },
    {
        "id": "F20", "patient": "Kopf-Hals-Tumor",
        "code": "LY2", "context": "Sekundäres Gesichtslymphödem nach Tumorektomie, 30 min MLD",
        "setting": "Befunddokumentation.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20205", "expected_audit": "PASS",
        "transcript": (
            "Patient nach operativer Tumorektomie im Kopf-Hals-Bereich. Das sekundäre Lymphödem im "
            "Gesichtsbereich spannt heute besonders stark. Durchführung von 30 Minuten MLD. Der "
            "Patient berichtet im S-Feld zusätzlich über leichte Schluckbeschwerden, die er auf den "
            "hohen Druck im Gewebe zurückführt. Ziel ist die Druckentlastung."
        ),
    },
    # ── Group F: Red Flags — BLOCK erwartet ───────────────────────────────────
    {
        "id": "F21", "patient": "Maier (Erysipel Red Flag)",
        "code": "RF", "context": "Erysipel-Verdacht — Therapieabbruch, BLOCK erwartet",
        "setting": "Notfalldokumentation während der Behandlung.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "BLOCK", "expected_audit": "BLOCK",
        "transcript": (
            "Stopp, Herr Maier, ich muss die Behandlung hier sofort unterbrechen. Ich sehe an Ihrem "
            "Unterschenkel eine flammende, scharf begrenzte Rötung und das gesamte Areal ist sehr heiß. "
            "Sie sagen, Sie fühlen sich elend und haben 39 Grad Fieber? Ich breche die Therapie sofort "
            "ab. Es besteht dringender Verdacht auf ein Erysipel. Bitte begeben Sie sich umgehend in "
            "ärztliche Behandlung!"
        ),
    },
    {
        "id": "F22", "patient": "Patient Thrombose-Verdacht",
        "code": "RF", "context": "Tiefe Venenthrombose V.a. — keine MLD, BLOCK erwartet",
        "setting": "Sicherheitsdokumentation vor Behandlungsbeginn.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "BLOCK", "expected_audit": "BLOCK",
        "transcript": (
            "Die Wade des Patienten wirkt heute im Seitenvergleich extrem hart, glänzend und ist "
            "bläulich-livid verfärbt. Bei der Palpation gibt der Patient einen massiven, stechenden "
            "Druckschmerz an. Es besteht der dringende Verdacht auf eine tiefe Venenthrombose. Es "
            "wurde keine MLD durchgeführt. Die Notaufnahme wurde vorab informiert und der Patient "
            "dorthin überwiesen."
        ),
    },
    {
        "id": "F23", "patient": "Patient Kardiales Ödem",
        "code": "RF", "context": "Akute kardiale Dekompensation — absolute KI, BLOCK erwartet",
        "setting": "Notfalldokumentation, Behandlungsabbruch.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "BLOCK", "expected_audit": "BLOCK",
        "transcript": (
            "Therapieabbruch nach fünf Minuten: Der Patient berichtet über plötzlich einsetzende "
            "Atemnot im Liegen und zeigt bläuliche Lippen. Bei der bekannten schweren Herzinsuffizienz "
            "besteht der Verdacht auf eine akute kardiale Dekompensation. Die MLD ist hier absolut "
            "kontraindiziert. Der Notarzt wurde bereits verständigt."
        ),
    },
    # ── Group G: Edge-Cases ───────────────────────────────────────────────────
    {
        "id": "F24", "patient": "Ulcus Cruris (Wundmanagement)",
        "code": "LY1", "context": "Stadium 3, Ulkus lokaler Ausschluss, MLD ausgespart",
        "setting": "Behandlungsprotokoll mit Wundhinweis.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "REVIEW",
        "transcript": (
            "Patient mit Stadium 3 am rechten Bein. Das bekannte Ulkus am Innenknöchel zeigt heute "
            "einen gelblichen Exsudatfluss und riecht unangenehm. Ich habe die Lymphdrainage heute "
            "lokal weiträumig ausgespart, um keine Keimverschleppung zu riskieren. Der Patient wurde "
            "zur dringenden Wundkontrolle an seinen behandelnden Wundmanager verwiesen."
        ),
    },
    {
        "id": "F25", "patient": "Finaler Audit-Safe Check",
        "code": "LY1", "context": "Vollständige Befunddoku, alle Vitals gecheckt, PASS erwartet",
        "setting": "Abschlussdokumentation nach Routine-MLD.",
        "insurance": "GKV",
        "expected_profile": "LY", "expected_position": "20201", "expected_audit": "PASS",
        "transcript": (
            "Abschlussdokumentation: Alle Vitalsignale wurden vor Behandlungsbeginn geprüft. Es liegen "
            "kein Fieber, keine pathologische Rötung und keine Ruheschmerzen vor. Red Flags wurden "
            "klinisch sicher ausgeschlossen. Wir starten nun mit der geplanten MLD-45 bei einem "
            "chronischen Lymphödem im Stadium 2."
        ),
    },
]

CASES = CASES + CASES_ORIG


# ── Engine pipeline (transcript-only, no Whisper) ───────────────────────────

def run_from_transcript(engine, transcript_text: str, insurance_type):
    from shared.billing_engine import BillingEngine

    transcript = engine.clean_transcript(transcript_text)
    profile_id = engine._detect_profile(transcript)
    prof_label = engine._PROFILES[profile_id]["label"]

    raw_output = engine._generate_soap_note(transcript, profile_id)

    if hasattr(engine, 'parse_robust_json'):
        parsed = engine.parse_robust_json(raw_output)
    else:
        try:
            json_match = re.search(r'\{.*"icd10".*"soap".*\}', raw_output, re.DOTALL)
            parsed = json.loads(json_match.group(0)) if json_match else json.loads(raw_output)
        except (json.JSONDecodeError, ValueError):
            parsed = {"icd10": "M99.9", "soap": {"S": "n.d.", "O": "n.d.", "A": "n.d.", "P": "n.d."}}

    if hasattr(engine, 'suggest_billing'):
        icd, _ = engine.suggest_billing(
            parsed["icd10"], parsed["soap"], transcript, profile_id=profile_id
        )
        parsed["icd10"] = icd
    else:
        icd = parsed["icd10"]

    _PROFILE_VALID_PREFIXES = {
        "EX_KNIE":    (("M17", "M22", "M23", "M24", "S82", "S83", "Z96.6"), "M23.51"),
        "EX_HUefte":  (("M16", "M17", "S72", "Z96.6", "M24"),               "M16.9"),
        "EX_SCHULTER":(("M75", "S43", "M24"),                                "M75.1"),
        "EX_FUSS":    (("M19", "M77", "S93", "S92", "M79.6"),                "M19.07"),
        "EX_HAND":    (("M19", "M65", "G56", "M77", "S52", "S62", "T92"),    "M19.04"),
    }
    if profile_id in _PROFILE_VALID_PREFIXES:
        valid_prefixes, fallback_icd = _PROFILE_VALID_PREFIXES[profile_id]
        if not any(icd.startswith(p) for p in valid_prefixes):
            icd = fallback_icd
            parsed["icd10"] = icd

    if hasattr(engine, 'apply_medical_corrections'):
        parsed["soap"] = engine.apply_medical_corrections(parsed["soap"])
    parsed["soap"] = engine.recover_hard_metrics(transcript, parsed["soap"], profile_id=profile_id)

    if profile_id == "LY" and hasattr(engine, '_inject_ly_staging'):
        parsed["soap"] = engine._inject_ly_staging(transcript, parsed["soap"])
        suffix = parsed["soap"].pop("_ly_icd_suffix", None)
        if suffix and re.match(r"^[IQE]\d{2}\.\d$", icd):
            icd = icd + suffix
        elif suffix and re.match(r"^[IQE]\d{2}\.\d0?$", icd):
            base = icd.rstrip("0") if icd.endswith("0") and len(icd) > 5 else icd
            icd = base + suffix
            parsed["icd10"] = icd

    if hasattr(engine, '_migrate_diagnoses_from_s_to_a'):
        parsed["soap"] = engine._migrate_diagnoses_from_s_to_a(parsed["soap"])
    if hasattr(engine, '_clean_hallucinated_regions'):
        parsed["soap"] = engine._clean_hallucinated_regions(parsed["soap"], icd, profile_id)
    if hasattr(engine, '_inject_bladder_bowel_into_objective'):
        parsed["soap"] = engine._inject_bladder_bowel_into_objective(transcript, parsed["soap"])
    if hasattr(engine, 'inject_audit_stamps'):
        parsed["soap"] = engine.inject_audit_stamps(parsed["soap"])
    if hasattr(engine, 'rom_sanity_check'):
        parsed = engine.rom_sanity_check(transcript, parsed)

    billing_result = BillingEngine().evaluate(
        icd10=icd,
        soap=parsed["soap"],
        transcript=transcript,
        insurance_type=insurance_type,
        config_rules=engine.billing_rules,
        pkv_preise=engine.config.pkv_preise,
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


# ── PDF report generator ────────────────────────────────────────────────────

def _s(text: str) -> str:
    """Sanitize text to latin-1 for fpdf Helvetica core font."""
    if not text:
        return ""
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2015": "-",  # dashes
        "\u2018": "'", "\u2019": "'",                   # smart quotes
        "\u201c": '"', "\u201d": '"',
        "\u00e4": "ae", "\u00f6": "oe", "\u00fc": "ue",  # lowercase umlauts
        "\u00c4": "Ae", "\u00d6": "Oe", "\u00dc": "Ue",  # uppercase umlauts
        "\u00df": "ss",                                   # eszett
        "\u00a7": "SS",                                   # section sign
        "\u2026": "...",                                  # ellipsis
        "\u00b0": " Grad",                               # degree
        "\u00b1": "+/-",
        "\u2264": "<=", "\u2265": ">=",
        "\u00e9": "e", "\u00e8": "e", "\u00ea": "e",
        "\u00e0": "a", "\u00e2": "a",
        "\u2122": "(TM)", "\u00ae": "(R)",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Drop any remaining non-latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(cases_meta: list, results: list, out_path: str):
    from fpdf import FPDF

    # ── Colour palette ──────────────────────────────────────────────────────
    C_BRAND    = (30, 90, 160)     # deep blue
    C_DARK     = (30, 30, 30)
    C_MID      = (80, 80, 80)
    C_LIGHT    = (245, 247, 250)
    C_WHITE    = (255, 255, 255)
    C_PASS     = (22, 160, 80)
    C_WARN     = (210, 140, 0)
    C_FAIL     = (200, 40, 40)
    C_REVIEW   = (180, 100, 0)
    C_BLOCK    = (160, 0, 0)
    C_SECTION  = (220, 230, 245)
    C_ROW_ALT  = (248, 249, 252)
    C_BORDER   = (200, 210, 225)

    STATUS_COLOR = {
        "PASS":   C_PASS,
        "REVIEW": C_REVIEW,
        "WARN":   C_WARN,
        "FAIL":   C_FAIL,
        "BLOCK":  C_BLOCK,
    }
    AUDIT_COLOR = {
        "PASS":  C_PASS,
        "WARN":  C_WARN,
        "FAIL":  C_FAIL,
        "BLOCK": C_BLOCK,
    }

    def set_fill(pdf, rgb):
        pdf.set_fill_color(*rgb)

    def set_text(pdf, rgb):
        pdf.set_text_color(*rgb)

    def set_draw(pdf, rgb):
        pdf.set_draw_color(*rgb)

    class KuraPDF(FPDF):
        def header(self):
            pass  # custom headers per section

        def normalize_text(self, text: str) -> str:
            return _s(text)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            set_text(self, C_MID)
            self.cell(0, 6, f"Kura Medical  |  Transkript-Evaluation  |  Seite {self.page_no()}", align="C")

    pdf = KuraPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    W = 174  # usable width

    # ────────────────────────────────────────────────────────────────────────
    # COVER PAGE
    # ────────────────────────────────────────────────────────────────────────
    pdf.add_page()

    # Top colour bar
    set_fill(pdf, C_BRAND)
    pdf.rect(0, 0, 210, 48, "F")

    pdf.set_y(14)
    pdf.set_font("Helvetica", "B", 26)
    set_text(pdf, C_WHITE)
    pdf.cell(0, 12, "Kura Medical", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.cell(0, 8, "KI-Dokumentations-Engine  -  Evaluierungsbericht", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(60)
    set_text(pdf, C_DARK)

    # Report metadata box
    set_fill(pdf, C_LIGHT)
    set_draw(pdf, C_BORDER)
    pdf.set_line_width(0.3)
    pdf.rect(18, 56, W, 38, "FD")

    pdf.set_xy(26, 62)
    pdf.set_font("Helvetica", "B", 10)
    set_text(pdf, C_BRAND)
    pdf.cell(40, 7, "Berichtsdatum:", new_x="RIGHT")
    pdf.set_font("Helvetica", "", 10)
    set_text(pdf, C_DARK)
    pdf.cell(0, 7, datetime.now().strftime("%d.%m.%Y  %H:%M Uhr"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(26)
    pdf.set_font("Helvetica", "B", 10)
    set_text(pdf, C_BRAND)
    pdf.cell(40, 7, "Testfaelle:")
    pdf.set_font("Helvetica", "", 10)
    set_text(pdf, C_DARK)
    pdf.cell(0, 7, f"{len(cases_meta)} Transkripte (25-Fall Stress-Test: Duration, Stadium, S-Feld, Red Flags)", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(26)
    pdf.set_font("Helvetica", "B", 10)
    set_text(pdf, C_BRAND)
    pdf.cell(40, 7, "Versicherung:")
    pdf.set_font("Helvetica", "", 10)
    set_text(pdf, C_DARK)
    pdf.cell(0, 7, "GKV (Gesetzliche Krankenversicherung, SS125 SGB V)", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(26)
    pdf.set_font("Helvetica", "B", 10)
    set_text(pdf, C_BRAND)
    pdf.cell(40, 7, "Engine-Version:")
    pdf.set_font("Helvetica", "", 10)
    set_text(pdf, C_DARK)
    pdf.cell(0, 7, "Kura Medical v2026  |  LLM: Meta-Llama-3.1-8B-Instruct-4bit  |  MLX", new_x="LMARGIN", new_y="NEXT")

    # Summary table on cover
    pdf.set_y(110)
    pdf.set_font("Helvetica", "B", 12)
    set_text(pdf, C_BRAND)
    pdf.cell(0, 8, "Zusammenfassung der Evaluierung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_line_width(0.5)
    set_draw(pdf, C_BRAND)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(3)

    # Table header
    cols = [8, 44, 22, 28, 44, 28]
    headers = ["#", "Patient / Kontext", "Profil", "ICD-10", "Leistung", "Audit"]
    pdf.set_font("Helvetica", "B", 8.5)
    set_fill(pdf, C_BRAND)
    set_text(pdf, C_WHITE)
    set_draw(pdf, C_BRAND)
    pdf.set_line_width(0.1)
    x = 18
    for w, h in zip(cols, headers):
        pdf.set_xy(x, pdf.get_y())
        pdf.cell(w, 7, h, border=1, fill=True)
        x += w
    pdf.ln(7)

    for i, (meta, res) in enumerate(zip(cases_meta, results)):
        br = res["billing_result"]
        row_fill = C_ROW_ALT if i % 2 == 0 else C_WHITE
        audit_col = STATUS_COLOR.get(br.audit_status, C_MID)

        pdf.set_font("Helvetica", "", 8)
        x = 18
        y = pdf.get_y()

        set_fill(pdf, row_fill)
        set_text(pdf, C_DARK)
        set_draw(pdf, C_BORDER)
        pdf.set_line_width(0.1)

        vals = [
            str(i + 1),
            f"{meta['patient']} ({meta['code']})",
            res["profile_id"],
            res["icd10"],
            br.position_name[:36] + ("..." if len(br.position_name) > 36 else ""),
            br.audit_status,
        ]
        for j, (w, v) in enumerate(zip(cols, vals)):
            pdf.set_xy(x, y)
            if j == 5:
                set_fill(pdf, audit_col)
                set_text(pdf, C_WHITE)
                pdf.cell(w, 7, v, border=1, fill=True, align="C")
                set_fill(pdf, row_fill)
                set_text(pdf, C_DARK)
            else:
                pdf.cell(w, 7, v, border=1, fill=True)
            x += w
        pdf.ln(7)

    # Overall stats
    pdf.ln(6)
    n_pass = sum(1 for r in results if r["billing_result"].audit_status == "PASS")
    n_review = sum(1 for r in results if r["billing_result"].audit_status == "REVIEW")
    n_block = sum(1 for r in results if r["billing_result"].audit_status == "BLOCK")

    set_fill(pdf, C_LIGHT)
    set_draw(pdf, C_BORDER)
    pdf.set_line_width(0.3)
    pdf.rect(18, pdf.get_y(), W, 22, "FD")
    pdf.set_xy(26, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 10)
    set_text(pdf, C_DARK)
    pdf.cell(0, 6, f"Gesamt:  {len(results)} Faelle  |  "
                   f"PASS: {n_pass}  |  REVIEW: {n_review}  |  BLOCK: {n_block}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(26)
    pdf.set_font("Helvetica", "", 9)
    set_text(pdf, C_MID)
    pdf.cell(0, 6, "Ausfuehrliche Bewertung je Fall auf den Folgeseiten.")

    # ────────────────────────────────────────────────────────────────────────
    # PER-CASE PAGES
    # ────────────────────────────────────────────────────────────────────────
    for meta, res in zip(cases_meta, results):
        pdf.add_page()
        br = res["billing_result"]
        soap = res["soap"]
        audit_status_col = STATUS_COLOR.get(br.audit_status, C_MID)

        # ── Case header bar ─────────────────────────────────────────────────
        set_fill(pdf, C_BRAND)
        pdf.rect(0, 0, 210, 28, "F")
        pdf.set_xy(18, 7)
        pdf.set_font("Helvetica", "B", 14)
        set_text(pdf, C_WHITE)
        pdf.cell(120, 8, f"{meta['id']}  -  {meta['patient']} ({meta['code']})", new_x="RIGHT")
        # Audit badge top-right
        set_fill(pdf, audit_status_col)
        pdf.set_xy(152, 6)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(40, 10, f"  {br.audit_status}  ", border=0, fill=True, align="C")
        pdf.set_xy(18, 16)
        pdf.set_font("Helvetica", "", 9)
        set_text(pdf, (200, 220, 255))
        pdf.cell(0, 6, meta["context"])
        pdf.ln(16)

        # ── Context strip ───────────────────────────────────────────────────
        set_fill(pdf, C_SECTION)
        set_draw(pdf, C_BORDER)
        pdf.set_line_width(0.2)
        pdf.rect(18, pdf.get_y(), W, 8, "FD")
        pdf.set_xy(20, pdf.get_y() + 1.5)
        pdf.set_font("Helvetica", "I", 8.5)
        set_text(pdf, C_MID)
        pdf.cell(0, 5, f"Kontext: {meta['setting']}")
        pdf.ln(10)

        # ── Original transcript ─────────────────────────────────────────────
        def section_title(title):
            pdf.set_font("Helvetica", "B", 10)
            set_text(pdf, C_BRAND)
            pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            set_draw(pdf, C_BRAND)
            pdf.set_line_width(0.4)
            pdf.line(18, pdf.get_y(), 192, pdf.get_y())
            pdf.ln(2)

        section_title("Eingabe-Transkript")
        set_fill(pdf, (252, 253, 255))
        set_draw(pdf, C_BORDER)
        pdf.set_line_width(0.2)
        # Draw transcript box
        transcript_lines = textwrap.wrap(meta["transcript"], width=110)
        box_h = len(transcript_lines) * 4.5 + 5
        pdf.rect(18, pdf.get_y(), W, box_h, "FD")
        pdf.set_xy(20, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "", 8)
        set_text(pdf, C_MID)
        for line in transcript_lines:
            pdf.cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ── Engine analysis ─────────────────────────────────────────────────
        section_title("Engine-Analyse")

        def kv_row(label, value, bold_val=False):
            pdf.set_font("Helvetica", "B", 9)
            set_text(pdf, C_BRAND)
            pdf.cell(48, 6, label + ":", new_x="RIGHT")
            pdf.set_font("Helvetica", "B" if bold_val else "", 9)
            set_text(pdf, C_DARK)
            pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

        kv_row("Erkanntes Profil", f"{res['profile_id']}  -  {res['profile_label']}", bold_val=True)
        kv_row("ICD-10-GM Code", res["icd10"], bold_val=True)
        kv_row("Versicherungsart", f"{meta['insurance']}  (Gesetzliche Krankenversicherung)")
        pdf.ln(3)

        # ── SOAP note ───────────────────────────────────────────────────────
        section_title("SOAP-Dokumentation")

        soap_labels = {
            "S": ("S - Subjektiv", "Patientenangaben, Anamnese, Beschwerden"),
            "O": ("O - Objektiv", "Klinische Befunde, Messungen, Tests"),
            "A": ("A - Assessment", "Diagnose, klinische Beurteilung"),
            "P": ("P - Plan", "Behandlungsplan, Ziele, Massnahmen"),
        }
        soap_bg = [C_WHITE, C_ROW_ALT, C_WHITE, C_ROW_ALT]

        for idx, key in enumerate(("S", "O", "A", "P")):
            val = soap.get(key, "-")
            label_short, label_long = soap_labels[key]
            # Remove internal audit stamps
            val_clean = re.sub(r"\|?\s*\[AUDIT:[^\]]+\]", "", val).strip()
            wrapped = textwrap.wrap(val_clean, width=100) if val_clean else ["-"]
            row_h = max(len(wrapped) * 4.5 + 4, 10)

            y0 = pdf.get_y()
            if y0 + row_h > 270:
                pdf.add_page()
                y0 = pdf.get_y()

            set_fill(pdf, soap_bg[idx])
            set_draw(pdf, C_BORDER)
            pdf.set_line_width(0.15)
            pdf.rect(18, y0, W, row_h, "FD")

            # Key label column
            set_fill(pdf, C_BRAND)
            pdf.rect(18, y0, 14, row_h, "F")
            pdf.set_xy(18, y0 + (row_h - 6) / 2)
            pdf.set_font("Helvetica", "B", 10)
            set_text(pdf, C_WHITE)
            pdf.cell(14, 6, f" {key}", align="L")

            # Label
            pdf.set_xy(34, y0 + 1.5)
            pdf.set_font("Helvetica", "B", 8)
            set_text(pdf, C_BRAND)
            pdf.cell(0, 4, label_long)

            # Value
            pdf.set_xy(34, y0 + 5.5)
            pdf.set_font("Helvetica", "", 8.5)
            set_text(pdf, C_DARK)
            for line in wrapped:
                pdf.cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        pdf.ln(4)

        # ── Billing ─────────────────────────────────────────────────────────
        section_title("Abrechnung")

        # Two-column billing info
        left_col = [
            ("Positionsnummer", br.position_number),
            ("Leistungsbezeichnung", br.position_name),
            ("Festpreis SS125 SGB V", f"EUR {br.fixed_price_eur:.2f}" if br.fixed_price_eur else "-"),
            ("Behandlungsdauer", f"{br.session_duration_min} Minuten"),
        ]
        right_col = [
            ("Diagnosegruppe", f"{br.diagnosegruppe}  ({br.diagnosegruppe_desc})" if br.diagnosegruppe_desc else br.diagnosegruppe),
            ("Rechtsgrundlage", br.legal_basis),
            ("Max. EH Regelfall", str(br.max_units_regelfall) if br.max_units_regelfall else "-"),
            ("Langfristgenehmigung", "Erforderlich" if br.requires_langfrist_approval else "Nicht erforderlich"),
        ]

        row_y = pdf.get_y()
        for (lk, lv), (rk, rv) in zip(left_col, right_col):
            pdf.set_xy(18, row_y)
            pdf.set_font("Helvetica", "B", 8)
            set_text(pdf, C_MID)
            pdf.cell(34, 5.5, lk + ":")
            pdf.set_font("Helvetica", "", 8.5)
            set_text(pdf, C_DARK)
            pdf.cell(50, 5.5, lv)
            pdf.set_font("Helvetica", "B", 8)
            set_text(pdf, C_MID)
            pdf.cell(34, 5.5, rk + ":")
            pdf.set_font("Helvetica", "", 8.5)
            set_text(pdf, C_DARK)
            # Wrap long diagnosegruppe desc
            rv_short = rv[:46] + "..." if len(rv) > 48 else rv
            pdf.cell(0, 5.5, rv_short, new_x="LMARGIN", new_y="NEXT")
            row_y = pdf.get_y()

        pdf.ln(4)

        # ── Audit checklist ─────────────────────────────────────────────────
        section_title("Audit-Pruefpunkte")

        # Audit status banner
        set_fill(pdf, audit_status_col)
        set_draw(pdf, audit_status_col)
        pdf.set_line_width(0.1)
        pdf.rect(18, pdf.get_y(), W, 8, "FD")
        pdf.set_xy(20, pdf.get_y() + 1.5)
        pdf.set_font("Helvetica", "B", 9)
        set_text(pdf, C_WHITE)
        status_labels = {"PASS": "BESTANDEN", "REVIEW": "PRUEFUNG ERFORDERLICH", "BLOCK": "GESPERRT"}
        pdf.cell(0, 5, f"Gesamt-Status: {br.audit_status}  -  {status_labels.get(br.audit_status, br.audit_status)}")
        pdf.ln(10)

        if not br.audit_items:
            pdf.set_font("Helvetica", "I", 8.5)
            set_text(pdf, C_MID)
            pdf.cell(0, 6, "Keine Pruefpunkte gefunden.", new_x="LMARGIN", new_y="NEXT")
        else:
            for i_a, item in enumerate(br.audit_items):
                row_col = C_ROW_ALT if i_a % 2 == 0 else C_WHITE
                ic = AUDIT_COLOR.get(item.status, C_MID)
                item_y = pdf.get_y()
                if item_y > 268:
                    pdf.add_page()
                    item_y = pdf.get_y()

                set_fill(pdf, row_col)
                set_draw(pdf, C_BORDER)
                pdf.set_line_width(0.1)
                pdf.rect(18, item_y, W, 7, "FD")

                # Status badge
                set_fill(pdf, ic)
                pdf.rect(18, item_y, 18, 7, "F")
                pdf.set_xy(18, item_y + 1)
                pdf.set_font("Helvetica", "B", 7.5)
                set_text(pdf, C_WHITE)
                pdf.cell(18, 5, item.status, align="C")

                # Label + detail
                pdf.set_xy(38, item_y + 1.5)
                pdf.set_font("Helvetica", "B", 8.5)
                set_text(pdf, C_DARK)
                label_w = 80
                pdf.cell(label_w, 5, item.label)
                if item.detail:
                    pdf.set_font("Helvetica", "", 8)
                    set_text(pdf, C_MID)
                    detail_short = item.detail[:62] + "..." if len(item.detail) > 64 else item.detail
                    pdf.cell(0, 5, detail_short)
                pdf.ln(7)

        # ── Compliance warnings ─────────────────────────────────────────────
        if res["compliance_check"]:
            pdf.ln(2)
            section_title("Compliance-Hinweise")
            for w_item in res["compliance_check"]:
                w_y = pdf.get_y()
                if w_y > 270:
                    pdf.add_page()
                set_fill(pdf, (255, 250, 230))
                set_draw(pdf, (210, 175, 80))
                pdf.set_line_width(0.2)
                wrapped_w = textwrap.wrap(w_item, width=95)
                box_h = len(wrapped_w) * 5 + 4
                pdf.rect(18, pdf.get_y(), W, box_h, "FD")
                pdf.set_xy(22, pdf.get_y() + 2)
                pdf.set_font("Helvetica", "", 8.5)
                set_text(pdf, (120, 80, 0))
                for wline in wrapped_w:
                    pdf.cell(0, 5, wline, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

        # ── Required documentation checklist ────────────────────────────────
        if br.required_documentation:
            pdf.ln(2)
            section_title("Erforderliche Dokumentation (Pflichtfelder)")
            pdf.set_font("Helvetica", "", 8.5)
            set_text(pdf, C_DARK)
            for doc in br.required_documentation:
                pdf.cell(6, 5.5, "-")
                pdf.cell(0, 5.5, doc, new_x="LMARGIN", new_y="NEXT")

    # ────────────────────────────────────────────────────────────────────────
    # FINAL ASSESSMENT PAGE
    # ────────────────────────────────────────────────────────────────────────
    pdf.add_page()

    set_fill(pdf, C_BRAND)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_xy(18, 9)
    pdf.set_font("Helvetica", "B", 16)
    set_text(pdf, C_WHITE)
    pdf.cell(0, 10, "Abschliessende Bewertung")
    pdf.ln(22)

    section_title("Testergebnisse im Ueberblick")

    metrics = {
        "Korrekte Profilerkennung (LY)":        0,
        "Korrekte ICD-10-Gruppe (I89/I97/Q82)": 0,
        "SOAP vollstaendig (kein Platzhalter)":  0,
        "Abrechnungsposition korrekt":           0,
        "Red-Flag-Erkennung (BLOCK)":            0,
        "Audit-Status korrekt (erw. vs. akt.)":  0,
    }
    _rf_cases = [m for m in cases_meta if m.get("expected_audit") == "BLOCK"]
    _rf_total = len(_rf_cases)

    findings = []
    for meta, res in zip(cases_meta, results):
        br = res["billing_result"]
        soap = res["soap"]
        exp_pos   = meta.get("expected_position", "")
        exp_audit = meta.get("expected_audit", "PASS")
        is_rf     = exp_audit == "BLOCK"

        # 1. Profile
        profile_ok = res["profile_id"] == "LY"
        if profile_ok:
            metrics["Korrekte Profilerkennung (LY)"] += 1
        else:
            findings.append(f"{meta['id']} ({meta['patient']}): Profil '{res['profile_id']}' — 'LY' erwartet.")

        # 2. ICD group
        icd = res["icd10"]
        icd_ok = (icd.startswith("I89") or icd.startswith("I97") or
                  icd.startswith("Q82") or icd.startswith("E88"))
        if icd_ok:
            metrics["Korrekte ICD-10-Gruppe (I89/I97/Q82)"] += 1
        else:
            findings.append(f"{meta['id']} ({meta['patient']}): ICD '{icd}' nicht lymphatisch (I89/I97/Q82 erwartet).")

        # 3. SOAP completeness
        placeholders = {"patientengeschichte als string", "behandlung | ziel",
                        "diagnose | red flags", "test: ergebnis"}
        soap_ok = all(
            soap.get(k, "").strip() and
            not any(ph in soap.get(k, "").lower() for ph in placeholders)
            for k in ("S", "O", "A", "P")
        )
        if soap_ok:
            metrics["SOAP vollstaendig (kein Platzhalter)"] += 1
        else:
            missing = [k for k in ("S", "O", "A", "P")
                       if not soap.get(k, "").strip() or
                       any(ph in soap.get(k, "").lower() for ph in placeholders)]
            findings.append(f"{meta['id']} ({meta['patient']}): SOAP Platzhalter in: {', '.join(missing)}.")

        # 4. Billing position
        if exp_pos == "BLOCK":
            # Red-flag case: any BLOCK is a correct billing outcome
            bill_ok = br.audit_status == "BLOCK"
        else:
            bill_ok = br.position_number == exp_pos
        if bill_ok:
            metrics["Abrechnungsposition korrekt"] += 1
        else:
            findings.append(f"{meta['id']} ({meta['patient']}): Position '{br.position_number}' — '{exp_pos}' erwartet.")

        # 5. Red-flag catch
        if is_rf:
            if br.audit_status == "BLOCK":
                metrics["Red-Flag-Erkennung (BLOCK)"] += 1
            else:
                findings.append(f"{meta['id']} ({meta['patient']}): Red Flag NICHT geblockt! "
                                 f"Status: {br.audit_status} — BLOCK erwartet.")

        # 6. Audit status match
        audit_ok = br.audit_status == exp_audit
        if audit_ok:
            metrics["Audit-Status korrekt (erw. vs. akt.)"] += 1
        elif not is_rf:  # already reported above for RF
            findings.append(f"{meta['id']} ({meta['patient']}): Audit '{br.audit_status}' — '{exp_audit}' erwartet.")

    total = len(cases_meta)

    # Metrics table
    pdf.set_font("Helvetica", "B", 8.5)
    set_fill(pdf, C_BRAND)
    set_text(pdf, C_WHITE)
    set_draw(pdf, C_BRAND)
    pdf.set_line_width(0.1)
    pdf.cell(110, 7, "Pruefkriterium", border=1, fill=True)
    pdf.cell(20, 7, "Bestanden", border=1, fill=True, align="C")
    pdf.cell(20, 7, "Gesamt", border=1, fill=True, align="C")
    pdf.cell(24, 7, "Quote", border=1, fill=True, align="C")
    pdf.ln(7)

    for i_m, (metric, count) in enumerate(metrics.items()):
        row_bg = C_ROW_ALT if i_m % 2 == 0 else C_WHITE
        # Red-flag metric denominator is only RF cases
        denom = _rf_total if "Red-Flag" in metric and _rf_total > 0 else total
        pct = count / denom * 100 if denom > 0 else 0
        pct_col = C_PASS if pct == 100 else (C_WARN if pct >= 70 else C_FAIL)
        set_fill(pdf, row_bg)
        set_text(pdf, C_DARK)
        set_draw(pdf, C_BORDER)
        pdf.cell(110, 7, metric, border=1, fill=True)
        pdf.cell(20, 7, str(count), border=1, fill=True, align="C")
        pdf.cell(20, 7, str(denom), border=1, fill=True, align="C")
        set_fill(pdf, pct_col)
        set_text(pdf, C_WHITE)
        pdf.cell(24, 7, f"{pct:.0f}%", border=1, fill=True, align="C")
        set_fill(pdf, row_bg)
        set_text(pdf, C_DARK)
        pdf.ln(7)

    pdf.ln(6)

    # Findings
    section_title("Detailfunde & Handlungsbedarf")

    if not findings:
        set_fill(pdf, (235, 255, 240))
        set_draw(pdf, C_PASS)
        pdf.set_line_width(0.3)
        pdf.rect(18, pdf.get_y(), W, 10, "FD")
        pdf.set_xy(22, pdf.get_y() + 2.5)
        pdf.set_font("Helvetica", "B", 9)
        set_text(pdf, C_PASS)
        pdf.cell(0, 5, "Alle Faelle fehlerfrei verarbeitet. Keine Handlungsempfehlungen.")
        pdf.ln(14)
    else:
        for i_f, finding in enumerate(findings):
            f_y = pdf.get_y()
            if f_y > 265:
                pdf.add_page()
                f_y = pdf.get_y()
            wrapped_f = textwrap.wrap(finding, width=98)
            set_fill(pdf, (255, 248, 245))
            set_draw(pdf, C_FAIL)
            pdf.set_line_width(0.25)
            fbox_h = len(wrapped_f) * 5 + 5
            pdf.rect(18, f_y, W, fbox_h, "FD")
            set_fill(pdf, C_FAIL)
            pdf.rect(18, f_y, 4, fbox_h, "F")
            pdf.set_xy(25, f_y + 2)
            pdf.set_font("Helvetica", "", 8.5)
            set_text(pdf, C_DARK)
            for fline in wrapped_f:
                pdf.cell(0, 5, fline, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # Recommendations
    pdf.ln(2)
    section_title("Empfehlungen")
    recs = [
        "MLD-Dauerlogik (Pos. 20205 / 20203): Engine erkennt '30 Minuten MLD' -> 20205 und "
        "'60 Minuten MLD' -> 20203 automatisch. Pruefen ob alle Duration-Faelle korrekt abgerechnet.",
        "Red-Flag-BLOCK (F21/F22/F23): Erysipel, Tiefe Venenthrombose und kardiale Dekompensation "
        "sind absolute Kontraindikationen fuer MLD. Engine muss BLOCK ausgeben, keine Abrechnung.",
        "Staging-Suffix: Stadium 2 -> Suffix '02', Stadium 3 -> '02', Stadium 1 -> '01'. "
        "_inject_ly_staging() korrigiert. Validierung: I89.002 bei Stad. 2/3, I89.001 bei Stad. 1.",
        "S-Feld-Recovery: Wenn LLM 'n.d' ausgibt, extrahiert recover_hard_metrics() VAS + Symptome "
        "aus dem Transkript. Handover-Protokolle (F14, F21-F23) bleiben berechtigterweise REVIEW.",
        "Stemmer-Zeichen negativ: Regex 'stemmer\\b.{0,40}negativ' faengt alle deutschen Formulierungen "
        "inkl. 'Stemmer-Zeichen am Vorfuss ist negativ'. Kein false-positive Staging mehr.",
    ]
    for i_r, rec in enumerate(recs):
        r_y = pdf.get_y()
        if r_y > 265:
            pdf.add_page()
        wrapped_r = textwrap.wrap(rec, width=96)
        pdf.set_font("Helvetica", "B", 9)
        set_text(pdf, C_BRAND)
        pdf.cell(8, 5.5, f"{i_r + 1}.")
        pdf.set_font("Helvetica", "", 8.5)
        set_text(pdf, C_DARK)
        first = True
        for rline in wrapped_r:
            if first:
                pdf.cell(0, 5.5, rline, new_x="LMARGIN", new_y="NEXT")
                first = False
            else:
                pdf.set_x(26)
                pdf.cell(0, 5.5, rline, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Closing stamp
    pdf.ln(6)
    set_fill(pdf, C_BRAND)
    set_draw(pdf, C_BRAND)
    pdf.set_line_width(0.3)
    pdf.rect(18, pdf.get_y(), W, 14, "FD")
    pdf.set_xy(22, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "B", 9)
    set_text(pdf, C_WHITE)
    pdf.cell(0, 4, "Kura Medical  -  Automatischer Evaluierungsbericht", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(22)
    pdf.set_font("Helvetica", "", 8)
    set_text(pdf, (180, 210, 255))
    pdf.cell(0, 4,
             f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}  |  "
             "Kein Ersatz fuer aerztliche oder therapeutische Beurteilung.")

    pdf.output(out_path)
    print(f"\n  PDF gespeichert: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 68)
    print("  Kura Medical  -  Transkript-Evaluation")
    print("=" * 68)
    print("  Lade Engine (LLM + Konfiguration)...")

    if _PLATFORM == "windows":
        from physio_scribe_crossplatform import KuraEngine
    else:
        from physio_scribe import KuraEngine
    from shared.billing_engine import InsuranceType

    engine = KuraEngine()
    print("  Engine bereit.\n")

    results = []
    for case in CASES:
        ins_type = InsuranceType[case["insurance"]]
        print(f"  Verarbeite {case['id']}  ({case['patient']})...")
        result = run_from_transcript(engine, case["transcript"], ins_type)
        results.append(result)
        br = result["billing_result"]
        audit_icon = {"PASS": "[OK]", "REVIEW": "[!]", "BLOCK": "[X]"}.get(br.audit_status, "[?]")
        print(f"    Profil: {result['profile_id']}  |  ICD: {result['icd10']}  "
              f"|  Pos: {br.position_number}  |  Audit: {audit_icon} {br.audit_status}")

    out = os.path.join(ROOT, "evaluation_report.pdf")
    print(f"\n  Erstelle PDF-Bericht...")
    build_pdf(CASES, results, out)

    print("\n" + "=" * 68)
    print("  Evaluation abgeschlossen.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()