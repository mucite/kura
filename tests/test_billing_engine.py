"""
Unit tests for the Kura Dual Billing Engine.
Covers GKV, PKV, BG dispatch, ICD lookup, audit logic, and formatting.
"""
import pytest

from shared.billing_engine import (
    _GKV_PRICES,
    AuditItem,
    BillingEngine,
    BillingResult,
    InsuranceType,
    _BGEngine,
    _check_doc,
    _GKVEngine,
    _match_dg,
    _PKVEngine,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _soap_lws_complete():
    """Minimal complete SOAP for M54.5/WS1b that passes all audit checks."""
    return {
        "S": "Patient klagt über Lumbago seit zwei Wochen.",
        "O": (
            "ROM LWS: 60 - 0 - 30. Lasègue negativ bds. FBA 20 cm. "
            "Segmentbefund L4/L5 blockiert. "
            "Blasen-/Mastdarmfunktion unauffällig."
        ),
        "A": "Lumbales Schmerzsyndrom M54.5. Red Flags ausgeschlossen.",
        "P": "MT L4/L5 Traktion und Gleitmobilisation. Heimübungen.",
    }


def _soap_knie_complete():
    return {
        "S": "Knieschmerzen rechts seit Sturz.",
        "O": "ROM Knie: 100 - 0 - 10. Kraft MMT 4/5. Gangbild verändert, Schonhinken.",
        "A": "Gonarthrose M17. Red Flags ausgeschlossen.",
        "P": "KG Knie Kräftigung und Koordination.",
    }


# ── AuditItem ─────────────────────────────────────────────────────────────────

class TestAuditItem:
    def test_icon_pass(self):
        assert AuditItem("X", "X", "PASS").icon == "[PASS]"

    def test_icon_fail(self):
        assert AuditItem("X", "X", "FAIL").icon == "[FEHLT]"

    def test_icon_warn(self):
        assert AuditItem("X", "X", "WARN").icon == "[WARN]"

    def test_icon_block(self):
        assert AuditItem("X", "X", "BLOCK").icon == "[STOP]"

    def test_icon_unknown_status(self):
        assert AuditItem("X", "X", "CUSTOM").icon == "[?]"

    def test_str_with_detail(self):
        item = AuditItem("X", "Label", "PASS", "some detail")
        s = str(item)
        assert "Label" in s
        assert "some detail" in s

    def test_str_without_detail(self):
        item = AuditItem("X", "Label", "PASS")
        s = str(item)
        assert "Label" in s
        assert s.endswith("Label")


# ── ICD → Diagnosegruppe mapping ──────────────────────────────────────────────

class TestMatchDg:
    def test_lws_maps_to_ws1b(self):
        assert _match_dg("M54.5") == "WS1b"

    def test_hws_maps_to_ws1a(self):
        assert _match_dg("M54.2") == "WS1a"

    def test_gonarthrose_maps_to_ex3(self):
        assert _match_dg("M17") == "EX3"

    def test_schulter_maps_to_ex2(self):
        assert _match_dg("M75") == "EX2"

    def test_hemiplegie_maps_to_zns1(self):
        assert _match_dg("G81") == "ZNS1"

    def test_lymphoedem_maps_to_ly_group(self):
        result = _match_dg("I89.0")
        assert result in ("LY1", "LY2")

    def test_unknown_icd_returns_none(self):
        assert _match_dg("Z99.9") is None

    def test_case_insensitive(self):
        assert _match_dg("m54.5") == _match_dg("M54.5")

    def test_specific_prefix_wins_over_generic(self):
        # M54.5 is more specific than M54 — should route to WS1b, not WS1a
        assert _match_dg("M54.5") == "WS1b"
        assert _match_dg("M54.2") == "WS1a"

    def test_hip_icd(self):
        assert _match_dg("M16") == "EX4"

    def test_copd(self):
        assert _match_dg("J44") == "AT1"

    def test_parkinson(self):
        assert _match_dg("G20") == "ZNS3"


# ── Documentation checkers ────────────────────────────────────────────────────

class TestCheckDoc:
    def test_rom_neutral_null_spaced(self):
        assert _check_doc("ROM (Neutral-Null)", {"O": "ROM LWS: 60 - 0 - 30"})

    def test_rom_neutral_null_compact(self):
        assert _check_doc("ROM (Neutral-Null)", {"O": "Flexion 90-0-10"})

    def test_rom_missing(self):
        assert not _check_doc("ROM (Neutral-Null)", {"O": "Patient hat Schmerzen"})

    def test_schober(self):
        assert _check_doc("Schober-Zeichen", {"O": "Schober-Zeichen positiv"})

    def test_lasegue(self):
        assert _check_doc("Lasègue", {"O": "Lasègue negativ"})

    def test_lasegue_ascii(self):
        assert _check_doc("Lasègue", {"O": "Lasegue negativ"})

    def test_vas_with_schmerz(self):
        assert _check_doc("Schmerz (VAS)", {"S": "Schmerzen 6/10"})

    def test_vas_prefix(self):
        assert _check_doc("Schmerz (VAS)", {"S": "VAS 7"})

    def test_vas_nrs(self):
        assert _check_doc("Schmerz (VAS)", {"S": "NRS 5"})

    def test_segment_l4_l5(self):
        assert _check_doc("Behandeltes Segment", {"O": "Segmentbefund L4/L5"})

    def test_segment_c5_c6(self):
        assert _check_doc("Behandeltes Segment", {"O": "Behandlung C5/C6"})

    def test_fba(self):
        assert _check_doc("FBA (Finger-Boden-Abstand)", {"O": "FBA 15 cm"})

    def test_barthel(self):
        assert _check_doc("Barthel-Index", {"O": "Barthel-Index 60"})

    def test_stemmer(self):
        assert _check_doc("Stemmer-Zeichen", {"O": "Stemmer positiv"})

    def test_doc_not_present(self):
        assert not _check_doc("Barthel-Index", {"O": "ROM unauffällig"})

    def test_fallback_word_match(self):
        # Fallback: any word >3 chars from doc_name found in text
        assert _check_doc("Spirometrie", {"O": "Spirometrie durchgeführt"})


# ── GKV Engine ────────────────────────────────────────────────────────────────

class TestGKVEngine:
    def test_returns_billing_result(self):
        result = _GKVEngine().evaluate("M54.5", _soap_lws_complete(), "", {})
        assert isinstance(result, BillingResult)

    def test_insurance_type_is_gkv(self):
        result = _GKVEngine().evaluate("M54.5", _soap_lws_complete(), "", {})
        assert result.insurance_type == InsuranceType.GKV

    def test_lws_routes_to_mt(self):
        result = _GKVEngine().evaluate("M54.5", _soap_lws_complete(), "", {})
        assert result.position_number == "21201"

    def test_gonarthrose_routes_to_kg(self):
        result = _GKVEngine().evaluate("M17", _soap_knie_complete(), "", {})
        assert result.position_number == "20501"
        assert result.diagnosegruppe == "EX3"

    def test_gkv_price_populated(self):
        result = _GKVEngine().evaluate("M54.5", _soap_lws_complete(), "", {})
        assert result.fixed_price_eur is not None
        assert result.fixed_price_eur > 0

    def test_mt_price_matches_price_table(self):
        result = _GKVEngine().evaluate("M54.5", _soap_lws_complete(), "", {})
        assert result.fixed_price_eur == pytest.approx(_GKV_PRICES["21201"])

    def test_clean_soap_passes_audit(self):
        result = _GKVEngine().evaluate("M54.5", _soap_lws_complete(), "", {})
        assert result.audit_status == "PASS"
        assert result.risk_level == "OK"

    def test_missing_assessment_fails(self):
        soap = {**_soap_lws_complete(), "A": ""}
        result = _GKVEngine().evaluate("M54.5", soap, "", {})
        codes = [a.code for a in result.audit_items]
        assert "SOAP_A" in codes

    def test_short_obj_for_mt_fails(self):
        # O must be < 60 chars to trigger OBJ_DENSITY for position 21201
        soap = {**_soap_lws_complete(), "O": "Segmentbefund L4/L5. Blasen-/Mastdarmfunktion o.b."}
        assert len(soap["O"]) < 60
        result = _GKVEngine().evaluate("M54.5", soap, "", {})
        codes = [a.code for a in result.audit_items]
        assert "OBJ_DENSITY" in codes

    def test_red_flag_not_negated_blocks(self):
        # Kraftverlust in S, no negation word within 120 chars of the flag.
        # O is long enough to push A (which has "ausgeschlossen") past the 120-char window.
        soap_block = {
            "S": "Kraftverlust linksseitig akut aufgetreten, Beinschwäche",
            "O": "Patient kommt gehend. Bewegungsausmaß eingeschränkt. Muskeltonus seitengleich.",
            "A": "Mögliche neurologische Ursache. Red Flags ausgeschlossen.",
            "P": "Überweisung Neurologie.",
        }
        result = _GKVEngine().evaluate("M54.5", soap_block, "", {})
        statuses = {a.status for a in result.audit_items}
        assert "BLOCK" in statuses

    def test_red_flag_negated_does_not_block(self):
        soap = {
            "S": "Kein Kraftverlust",
            "O": "ROM: 60 - 0 - 30. Segmentbefund L4/L5. Blasen-/Mastdarmfunktion unauffällig.",
            "A": "M54.5 Lumbago. Red Flags ausgeschlossen. Kein Kraftverlust.",
            "P": "MT L4/L5.",
        }
        result = _GKVEngine().evaluate("M54.5", soap, "", {})
        assert result.risk_level != "BLOCK"

    def test_profile_override_kgg(self):
        soap = {
            "S": "Rückentraining.",
            "O": "Trainingsplan: Beinpresse 60 kg 3×15. Krafttest MRC 4/5. Therapieziel: Rumpfstabilität.",
            "A": "LWS-Beschwerden. Red Flags ausgeschlossen.",
            "P": "KGG Krafttraining.",
        }
        result = _GKVEngine().evaluate("M54.5", soap, "", {}, profile_id="KGG")
        assert result.position_number == "20507"
        assert result.diagnosegruppe == "KGG"

    def test_zns_requires_langfrist(self):
        soap = {
            "S": "Hemiplegie nach Schlaganfall.",
            "O": "Barthel-Index 40. Ashworth-Skala 2. Ganganalyse: Trendelenburg. ADL stark eingeschränkt.",
            "A": "Hemiplegie G81. Red Flags ausgeschlossen.",
            "P": "KG-ZNS Bobath.",
        }
        result = _GKVEngine().evaluate("G81", soap, "", {})
        assert result.requires_langfrist_approval is True

    def test_mt_segment_missing_fails(self):
        soap = {
            "S": "LWS-Schmerzen.",
            "O": "ROM LWS: 60 - 0 - 30. Lasègue negativ. FBA 20 cm. Blasen-/Mastdarmfunktion unauffällig.",
            "A": "M54.5. Red Flags ausgeschlossen.",
            "P": "MT.",
        }
        result = _GKVEngine().evaluate("M54.5", soap, "", {})
        mt_seg = next((a for a in result.audit_items if a.code == "MT_SEGMENT"), None)
        assert mt_seg is not None
        assert mt_seg.status == "FAIL"

    def test_regelfall_populated(self):
        result = _GKVEngine().evaluate("M54.5", _soap_lws_complete(), "", {})
        assert result.max_units_regelfall == 6

    def test_unknown_icd_fallback(self):
        soap = {
            "S": "LWS-Beschwerden.",
            "O": "ROM: 60 - 0 - 30. Segmentbefund L4/L5. FBA 15 cm. Lasègue negativ. Blasen-/Mastdarmfunktion unauffällig.",
            "A": "Unspezifisch. Red Flags ausgeschlossen.",
            "P": "MT.",
        }
        # Z99.9 is unknown — should not crash, uses fallback DG
        result = _GKVEngine().evaluate("Z99.9", soap, "", {})
        assert isinstance(result, BillingResult)
        assert result.position_number  # some position assigned


# ── PKV Engine ────────────────────────────────────────────────────────────────

class TestPKVEngine:
    def test_insurance_type_is_pkv(self):
        soap = {"S": "", "O": "ROM: 60-0-30", "A": "M54.5", "P": ""}
        result = _PKVEngine().evaluate("M54.5", soap, "")
        assert result.insurance_type == InsuranceType.PKV

    def test_price_range_populated(self):
        soap = {"S": "", "O": "", "A": "M54.5", "P": ""}
        result = _PKVEngine().evaluate("M54.5", soap, "")
        lo, hi = result.price_range_eur
        assert hi > lo > 0

    def test_praxispreis_used_when_provided(self):
        soap = {"S": "", "O": "", "A": "M54.5", "P": ""}
        result = _PKVEngine().evaluate("M54.5", soap, "", pkv_preise={"21201": 75.00})
        assert result.pkv_praxispreis_eur == 75.00

    def test_no_praxispreis_emits_warn(self):
        soap = {"S": "", "O": "", "A": "M54.5", "P": ""}
        result = _PKVEngine().evaluate("M54.5", soap, "")
        warn_codes = [a.code for a in result.audit_items if a.status == "WARN"]
        assert "PKV_INFO" in warn_codes

    def test_likelihood_hoch_for_complete_soap(self):
        soap = {
            "S": "Schmerz VAS 6.",
            "O": "ROM: 60 - 0 - 30. Segmentbefund L4/L5. Befund ausführlich dokumentiert, Kraft 4/5, SLR neg.",
            "A": "M54.5",
            "P": "MT Mobilisation, Traktion, Hausübungen zur Stabilisation.",
        }
        result = _PKVEngine().evaluate("M54.5", soap, "")
        assert result.reimbursement_likelihood in ("HOCH", "MITTEL")

    def test_profile_override_pkv(self):
        soap = {"S": "", "O": "", "A": "M54.5", "P": ""}
        result = _PKVEngine().evaluate("M54.5", soap, "", profile_id="KGG")
        assert result.position_number == "20507"


# ── BG Engine ─────────────────────────────────────────────────────────────────

class TestBGEngine:
    def _bg_soap(self):
        return {
            "S": "Arbeitsunfall beim Heben.",
            "O": (
                "D-Arzt-Bericht vorhanden. Unfallhergang dokumentiert. "
                "BG-Fallnummer 12345. Erstbehandlung heute. "
                "Segmentbefund L4/L5. Blasen-/Mastdarmfunktion unauffällig. "
                "ROM LWS: 50 - 0 - 20. Lasègue negativ. FBA 25 cm."
            ),
            "A": "M54.5 Arbeitsunfall. Unfallhergang: Sturz. Red Flags ausgeschlossen.",
            "P": "MT L4/L5.",
        }

    def test_insurance_type_is_bg(self):
        result = _BGEngine().evaluate("M54.5", self._bg_soap(), "", {})
        assert result.insurance_type == InsuranceType.BG

    def test_bg_surcharge_applied(self):
        result = _BGEngine().evaluate("M54.5", self._bg_soap(), "", {})
        assert result.bg_surcharge_pct > 0
        base = _GKV_PRICES.get("21201", 0)
        assert result.fixed_price_eur > base

    def test_bg_extra_docs_listed(self):
        result = _BGEngine().evaluate("M54.5", self._bg_soap(), "", {})
        assert len(result.bg_extra_docs) > 0

    def test_bg_missing_d_arzt_flagged(self):
        soap = {
            "S": "Arbeitsunfall.",
            "O": "Segmentbefund L4/L5. Blasen-/Mastdarmfunktion unauffällig.",
            "A": "M54.5. Red Flags ausgeschlossen.",
            "P": "MT.",
        }
        result = _BGEngine().evaluate("M54.5", soap, "", {})
        bg_codes = [a.code for a in result.audit_items if a.code.startswith("BG_")]
        assert len(bg_codes) > 0

    def test_bg_legal_basis_mentions_dguv(self):
        result = _BGEngine().evaluate("M54.5", self._bg_soap(), "", {})
        assert "DGUV" in result.legal_basis


# ── BillingEngine dispatcher ──────────────────────────────────────────────────

class TestBillingEngineDispatcher:
    def test_gkv_dispatch(self):
        engine = BillingEngine()
        result = engine.evaluate("M54.5", _soap_lws_complete(), "", InsuranceType.GKV)
        assert result.insurance_type == InsuranceType.GKV

    def test_pkv_dispatch(self):
        engine = BillingEngine()
        soap = {"S": "", "O": "", "A": "M54.5", "P": ""}
        result = engine.evaluate("M54.5", soap, "", InsuranceType.PKV)
        assert result.insurance_type == InsuranceType.PKV

    def test_bg_dispatch(self):
        engine = BillingEngine()
        soap = {
            "S": "",
            "O": "Segmentbefund L4/L5. Blasen-/Mastdarmfunktion unauffällig.",
            "A": "M54.5. Red Flags ausgeschlossen.",
            "P": "",
        }
        result = engine.evaluate("M54.5", soap, "", InsuranceType.BG)
        assert result.insurance_type == InsuranceType.BG

    def test_default_is_gkv(self):
        engine = BillingEngine()
        result = engine.evaluate("M54.5", _soap_lws_complete(), "")
        assert result.insurance_type == InsuranceType.GKV

    def test_none_config_rules_ok(self):
        engine = BillingEngine()
        result = engine.evaluate("M54.5", _soap_lws_complete(), "", config_rules=None)
        assert isinstance(result, BillingResult)


# ── BillingResult formatting ──────────────────────────────────────────────────

class TestBillingResultFormat:
    def _gkv_result(self):
        return BillingResult(
            insurance_type=InsuranceType.GKV,
            position_number="21201",
            position_name="Manuelle Therapie",
            diagnosegruppe="WS1b",
            diagnosegruppe_desc="LWS/ISG",
            legal_basis="§125 SGB V",
            session_duration_min=20,
            risk_level="OK",
            fixed_price_eur=35.59,
            max_units_regelfall=6,
        )

    def test_billing_line_contains_position(self):
        line = self._gkv_result().format_billing_line()
        assert "21201" in line

    def test_billing_line_contains_insurance_type(self):
        line = self._gkv_result().format_billing_line()
        assert "GKV" in line

    def test_billing_line_contains_price(self):
        line = self._gkv_result().format_billing_line()
        assert "35.59" in line

    def test_billing_line_langfrist_warning(self):
        result = self._gkv_result()
        result.requires_langfrist_approval = True
        line = result.format_billing_line()
        assert "Langfrist" in line

    def test_billing_line_max_units(self):
        line = self._gkv_result().format_billing_line()
        assert "6" in line

    def test_format_audit_report_has_status(self):
        result = self._gkv_result()
        result.audit_items = [AuditItem("T", "Test", "PASS")]
        report = result.format_audit_report()
        assert "STATUS" in report

    def test_format_audit_report_pass_icon(self):
        result = self._gkv_result()
        result.audit_items = []
        result.audit_status = "PASS"
        report = result.format_audit_report()
        assert "✅" in report

    def test_pkv_billing_line_uses_praxispreis(self):
        result = BillingResult(
            insurance_type=InsuranceType.PKV,
            position_number="21201",
            position_name="MT (PKV)",
            diagnosegruppe="WS1b",
            diagnosegruppe_desc="LWS/ISG",
            legal_basis="PKV",
            session_duration_min=20,
            risk_level="OK",
            pkv_praxispreis_eur=75.00,
        )
        line = result.format_billing_line()
        assert "75.00" in line
        assert "Praxispreis" in line

    def test_pkv_billing_line_uses_range_when_no_praxispreis(self):
        result = BillingResult(
            insurance_type=InsuranceType.PKV,
            position_number="21201",
            position_name="MT (PKV)",
            diagnosegruppe="WS1b",
            diagnosegruppe_desc="LWS",
            legal_basis="PKV",
            session_duration_min=20,
            risk_level="OK",
            price_range_eur=(38.0, 90.0),
        )
        line = result.format_billing_line()
        assert "38" in line
        assert "GebüTh" in line


# ── GKV price table sanity ────────────────────────────────────────────────────

class TestGKVPrices:
    def test_prices_loaded(self):
        assert len(_GKV_PRICES) > 0

    def test_mt_price(self):
        assert _GKV_PRICES["21201"] == pytest.approx(35.59)

    def test_kg_price(self):
        assert _GKV_PRICES["20501"] == pytest.approx(29.63)

    def test_kgg_price(self):
        assert _GKV_PRICES["20507"] == pytest.approx(55.81)

    def test_all_prices_positive(self):
        for pos, price in _GKV_PRICES.items():
            assert price > 0, f"Non-positive price for {pos}: {price}"
