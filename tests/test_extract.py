"""Offline tests for src/extract.py: the model is replaced by canned responses."""

import json

import pytest

import src.extract as ex
from src.corpus import build, load_manifest

FAIR_RULE = {
    "category": "algorithmic_rent_setting",
    "jurisdiction": "NJ",
    "title": "FAIR Act: municipalities may not enact conflicting ordinances",
    "requirement": "Cities may not adopt ordinances that conflict with the FAIR Act.",
    "key_value": None,
    "coverage": {"all_residential_rentals": True, "min_units": None, "max_units": None,
                 "built_on_or_before": None, "built_after": None, "cutoff_basis": None,
                 "rolling_age_years": None, "owner_condition": None,
                 "excluded_property_types": [], "notes": None},
    "exemptions": None,
    "penalty": None,
    "citation": "P.L.2026, c.43, s.6",
    "chunk_id": "D069-c003",
    "quoted_span": "A municipality shall be prohibited from enacting an ordinance that conflicts with this act.",
    "legal_status": "enacted",
    "enacted_date": "2026-07-20",
    "effective_date": "2027-07-20",  # wrong on purpose: the formula below must win
    "effective_date_formula": "This act shall take effect on the first day of the twelfth month next following the date of enactment.",
    "date_notes": None,
    "interaction": "May preempt local algorithmic-pricing ordinances.",
    "confidence": 0.9,
}


@pytest.fixture(scope="module")
def corpus():
    docs, chunks = build()
    return {d.doc_id: d for d in docs}, chunks


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    """Queue canned responses; record the params of each call."""
    monkeypatch.setattr(ex, "CACHE_DIR", tmp_path / "llm")
    calls, queue = [], []

    def fake_call(client, params):
        calls.append(params)
        rules = queue.pop(0)
        return {"text": json.dumps({"rules": rules}), "content": [{"type": "text", "text": "..."}],
                "stop_reason": "end_turn", "usage": {}, "model": ex.MODEL, "request_id": "req_test"}

    monkeypatch.setattr(ex, "call_model", fake_call)
    return calls, queue


def _run(corpus, doc_id, use_cache=True):
    docs, chunks = corpus
    jur = sorted({r["jurisdictions"] for r in load_manifest()})
    return ex.extract_doc(None, docs[doc_id], [c for c in chunks if c.doc_id == doc_id], jur,
                          "high", True, use_cache)


def test_verified_rule_becomes_schema_record_with_computed_date(corpus, fake_model):
    calls, queue = fake_model
    queue.append([dict(FAIR_RULE)])
    audit = _run(corpus, "D069")
    assert len(audit["rules"]) == 1 and not audit["rejected"] and len(calls) == 1

    docs, _ = corpus
    rec = ex.finalize_rule(audit["rules"][0], docs["D069"], "2026-10-01")
    assert rec["effective_date"] == "2027-07-01"
    assert rec["status"] == "not_yet_effective"
    assert rec["conflict_flag"] and "2027-07-20" in rec["conflict_note"]
    assert rec["source_url"].startswith("https://pub.njleg") and rec["retrieved_at"].startswith("2026-10-01")
    recs = ex.assign_ids([rec])
    assert ex.validate(recs) == []
    assert ex.finalize_rule(audit["rules"][0], docs["D069"], "2027-07-02")["status"] == "in_force"


def test_request_shape(corpus, fake_model):
    calls, queue = fake_model
    queue.append([])
    _run(corpus, "D069")
    p = calls[0]
    assert p["model"] == "claude-opus-5-5"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["fallbacks"] == "default" and p["betas"] == ["server-side-fallback-2026-07-01"]
    assert "[D069-c001]" in p["messages"][0]["content"]


def test_paraphrased_quote_gets_one_repair_then_rejection(corpus, fake_model):
    calls, queue = fake_model
    bad = dict(FAIR_RULE, quoted_span="Towns cannot pass ordinances conflicting with the FAIR Act.")
    queue.append([bad])            # first answer: paraphrase
    queue.append([dict(bad)])      # repair answer: still a paraphrase
    audit = _run(corpus, "D069")
    assert len(calls) == 2 and "not found verbatim" in calls[1]["messages"][-1]["content"]
    assert audit["rules"] == [] and len(audit["rejected"]) == 1


def test_repair_can_fix_quote(corpus, fake_model):
    calls, queue = fake_model
    queue.append([dict(FAIR_RULE, quoted_span="Municipalities may not enact conflicting ordinances.")])
    queue.append([dict(FAIR_RULE)])
    audit = _run(corpus, "D069")
    assert len(audit["rules"]) == 1 and audit["rejected"] == []


def test_responses_are_cached(corpus, fake_model):
    calls, queue = fake_model
    queue.append([dict(FAIR_RULE)])
    _run(corpus, "D069")
    _run(corpus, "D069")  # served from cache: no second call
    assert len(calls) == 1


def test_dedupe_prefers_official_source():
    base = {"jurisdiction": "CA", "category": "security_deposits", "citation": "Cal. Civ. Code § 1950.5",
            "key_value": "1 month", "confidence": 0.9}
    a = dict(base, source_type="secondary (law firm / news / mirror)", source_doc_id="D099")
    b = dict(base, source_type="official", source_doc_id="D025", confidence=0.8)
    out = ex.dedupe([a, b])
    assert len(out) == 1 and out[0]["source_doc_id"] == "D025" and out[0]["also_supported_by"] == ["D099"]


def test_dry_run_needs_no_api_key(capsys):
    assert ex.main(["--dry-run"]) == 0
    assert "estimated cost" in capsys.readouterr().out
