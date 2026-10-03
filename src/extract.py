"""LLM rule extraction (Module A, step 2).

For each captured document, send its cleaned text (split into labelled sections
by src/corpus.py) to Claude and get back structured candidate rules. Then, in
plain code:

  * check every quoted_span is verbatim in the raw source text (one repair round
    for misses, then reject),
  * compute effective dates from statutory formulas (src/dates.py),
  * compute status as of the query date,
  * dedupe, assign team_rule_ids and validate against the challenge schema.

Every API response is cached under cache/llm/, so re-runs are free and the demo
works offline. Every document's outcome is logged to out/extraction_audit.jsonl.

Usage:
  python -m src.extract --dry-run                # cost estimate, no API calls
  python -m src.extract --docs D069,D022         # a few documents
  python -m src.extract                          # whole corpus -> out/rules.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from src.corpus import OUT_DIR, ROOT, Chunk, Doc, build, load_manifest, locate_quote
from src.dates import resolve_effective_date, status_as_of

MODEL = "claude-opus-5-5"
PROMPT_VERSION = "extract-v1"
DEFAULT_AS_OF = "2026-10-01"
CACHE_DIR = ROOT / "cache" / "llm"
SCHEMA_PATH = ROOT / "schema" / "rule_record.schema.json"

CATEGORIES = [
    "rent_increase_limits", "just_cause_eviction", "security_deposits",
    "application_screening_fees", "screening_restrictions", "algorithmic_rent_setting",
]
OFFICIAL_RANK = {"official": 0, "official city-linked policy": 1, "code publisher": 2}

# --------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You extract housing-law rules from official legal texts for a research prototype that tells renters and housing providers which rules apply at an address. Accuracy and traceability matter more than coverage: every rule you report will be shown to users with your quoted source text, and a reviewer will check it.

You will receive one document from a fixed corpus: its metadata, then its text split into sections labelled [chunk_id]. Extract every distinct rule in these six categories:

- rent_increase_limits: caps or formulas limiting rent increases (annual allowable increase, CPI-based caps, rent control coverage), including bans on local rent control.
- just_cause_eviction: limits on terminating tenancies (allowed causes, notice periods, relocation assistance, required notices to tenants about their rights when a tenancy is ended).
- security_deposits: maximum deposit amounts, interest, return deadlines, deposit exceptions.
- application_screening_fees: caps on application or screening fees, allowed upfront charges, receipts and refunds.
- screening_restrictions: limits on using criminal history, source of income / vouchers, credit or eviction history in tenant screening, and timing rules for those checks.
- algorithmic_rent_setting: bans or limits on using or selling algorithmic / coordinated pricing software to set rents or occupancy.

What counts as a rule: a binding requirement, prohibition or limit that a law imposes (or that a pending bill or ballot measure would impose). Do not extract definitions, findings, purpose statements, enforcement-agency housekeeping, or general advice, except as context for a rule. Do extract a pending bill or failed ballot measure as a rule with the matching legal_status, because users must be told it is not law. Extract one record per distinct obligation; split a section that sets two different limits (e.g. a deposit cap and an interest requirement) into two records.

Field guidance:
- jurisdiction: the government whose law it is. Usually the document's jurisdiction; a city web page that describes a STATE law should use the state code.
- quoted_span: copy one to three consecutive sentences EXACTLY as they appear in the document text, character for character (you may skip line breaks). It must state the rule itself, not a heading. Never paraphrase, never stitch non-adjacent text together, never add ellipses. A program checks it against the source and rejects anything not found verbatim.
- chunk_id: the [chunk_id] label of the section the quote comes from.
- citation: the official cite as the document states it (e.g. "Cal. Civ. Code § 1947.12", "N.J.S.A. 46:8-21.2", "P.L.2026, c.43, s.4", "BMC 13.63.030", "LAMC 151.06"). If the document gives no section number, cite the document title and section heading. Never invent section numbers.
- requirement: one or two plain-language sentences a renter could understand.
- key_value: the headline number or formula ("1 month's rent", "lesser of 5% + CPI or 10%", "60 days' notice"), else null.
- coverage: who and what the rule covers. Use null for anything the text does not state. Year/date cutoffs go in built_on_or_before / built_after with cutoff_basis saying whether the law keys on the certificate of occupancy, first construction, or a rolling age (rolling_age_years, e.g. 15 for "issued more than 15 years ago"). owner_condition describes owner-type tests (e.g. "natural person owning no more than two properties with no more than four units total").
- exemptions: plain-language summary of exemptions, or null.
- penalty: remedies or penalties for violations, or null.
- legal_status: "enacted" for adopted laws and ordinances (whether or not yet in effect), "pending_bill" for bills or proposals not yet enacted, "failed" for measures that were defeated, struck or withdrawn, "unknown" if the text does not say.
- enacted_date: the date the law was approved/signed/adopted, YYYY-MM-DD, if stated.
- effective_date: only if the document states a calendar date the rule takes or took effect (YYYY-MM-DD, or YYYY-MM / YYYY if that is all it says). Do not compute dates yourself.
- effective_date_formula: if the effective date is stated as a formula relative to enactment (e.g. "on the first day of the twelfth month next following the date of enactment"), copy that phrase exactly; else null.
- date_notes: anything that makes the effective date uncertain, e.g. two different published dates, a phased start, or an operative date that differs from the enactment date; else null.
- interaction: how this rule relates to other levels of law if the text says so (e.g. "does not apply to units covered by a stricter local rent control ordinance", "municipalities may not enact conflicting ordinances"), else null.
- confidence: your confidence 0-1 that the record is accurate and complete.

If the document contains no rule in the six categories, return an empty list. Do not use outside knowledge to add rules or facts that are not in this document."""


def _nullable(t: dict) -> dict:
    return {"anyOf": [t, {"type": "null"}]}


def output_schema(jurisdictions: list[str]) -> dict:
    s, i = {"type": "string"}, {"type": "integer"}
    coverage = {
        "type": "object",
        "properties": {
            "all_residential_rentals": _nullable({"type": "boolean"}),
            "min_units": _nullable(i),
            "max_units": _nullable(i),
            "built_on_or_before": _nullable(s),
            "built_after": _nullable(s),
            "cutoff_basis": _nullable({"type": "string", "enum": [
                "certificate_of_occupancy", "first_built", "rolling_age", "other"]}),
            "rolling_age_years": _nullable(i),
            "owner_condition": _nullable(s),
            "excluded_property_types": {"type": "array", "items": s},
            "notes": _nullable(s),
        },
        "required": ["all_residential_rentals", "min_units", "max_units", "built_on_or_before",
                     "built_after", "cutoff_basis", "rolling_age_years", "owner_condition",
                     "excluded_property_types", "notes"],
        "additionalProperties": False,
    }
    rule = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "jurisdiction": {"type": "string", "enum": jurisdictions},
            "title": s,
            "requirement": s,
            "key_value": _nullable(s),
            "coverage": coverage,
            "exemptions": _nullable(s),
            "penalty": _nullable(s),
            "citation": s,
            "chunk_id": s,
            "quoted_span": s,
            "legal_status": {"type": "string", "enum": ["enacted", "pending_bill", "failed", "unknown"]},
            "enacted_date": _nullable(s),
            "effective_date": _nullable(s),
            "effective_date_formula": _nullable(s),
            "date_notes": _nullable(s),
            "interaction": _nullable(s),
            "confidence": {"type": "number"},
        },
        "required": ["category", "jurisdiction", "title", "requirement", "key_value", "coverage",
                     "exemptions", "penalty", "citation", "chunk_id", "quoted_span", "legal_status",
                     "enacted_date", "effective_date", "effective_date_formula", "date_notes",
                     "interaction", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"rules": {"type": "array", "items": rule}},
        "required": ["rules"],
        "additionalProperties": False,
    }


def render_document(doc: Doc, chunks: list[Chunk]) -> str:
    parts = [
        f"doc_id: {doc.doc_id}",
        f"jurisdiction (from corpus manifest): {doc.jurisdiction}",
        f"source_type: {doc.source_type}",
        f"source_url: {doc.source_url}",
        f"retrieved: {doc.retrieved_at}",
        "",
        "Document text:",
    ]
    for c in chunks:
        parts.append(f"\n[{c.chunk_id}] {c.heading}\n{c.text}")
    parts.append("\nExtract the rules from this document.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- API + cache

def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def call_model(client, params: dict) -> dict:
    """One streamed Messages API call. Returns text, stop_reason, usage, model, request_id."""
    with client.beta.messages.stream(**params) as stream:
        msg = stream.get_final_message()
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return {
        "text": text,
        "content": [b.to_dict() for b in msg.content],
        "stop_reason": msg.stop_reason,
        "usage": msg.usage.to_dict() if msg.usage else None,
        "model": msg.model,
        "request_id": getattr(msg, "_request_id", None),
    }


def cached_call(client, params: dict, use_cache: bool = True) -> tuple[dict, bool]:
    key = _cache_key({"v": PROMPT_VERSION, **{k: v for k, v in params.items() if k != "max_tokens"}})
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text()), True
    result = call_model(client, params)
    if result["stop_reason"] == "end_turn":  # never cache refusals or truncated output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cache_key": key, **result}, indent=1))
    return result, False


def build_params(user_text: str, schema: dict, effort: str, fallback: bool,
                 history: list | None = None) -> dict:
    messages = (history or []) + [{"role": "user", "content": user_text}]
    params = {
        "model": MODEL,
        "max_tokens": 64000,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    }
    if fallback:
        # Server-side fallback: if a safety classifier declines, the API retries on a fallback model.
        params["betas"] = ["server-side-fallback-2026-07-01"]
        params["fallbacks"] = "default"
    return params


# --------------------------------------------------------------------------- per-document extraction

def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def verify_quotes(doc: Doc, rules: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rules into (verified, failed). Verified rules get the verbatim raw quote."""
    ok, bad = [], []
    for r in rules:
        span = locate_quote(doc.raw, r["quoted_span"])
        if span and span[1] - span[0] >= 20:
            r["quoted_span"] = _squash(doc.raw[span[0]:span[1]])
            r["_raw_offsets"] = list(span)
            ok.append(r)
        else:
            bad.append(r)
    return ok, bad


def extract_doc(client, doc: Doc, chunks: list[Chunk], jurisdictions: list[str], effort: str,
                fallback: bool, use_cache: bool = True) -> dict:
    """Extract, verify and (once) repair the rules of one document. Returns an audit entry."""
    schema = output_schema(jurisdictions)
    user_text = render_document(doc, chunks)
    params = build_params(user_text, schema, effort, fallback)
    audit = {"doc_id": doc.doc_id, "model": MODEL, "prompt_version": PROMPT_VERSION, "effort": effort,
             "n_chunks": len(chunks), "calls": []}

    res, hit = cached_call(client, params, use_cache)
    audit["calls"].append({"kind": "extract", "cache_hit": hit, "stop_reason": res["stop_reason"],
                           "served_model": res.get("model"), "request_id": res.get("request_id"),
                           "usage": res.get("usage")})
    if res["stop_reason"] != "end_turn":
        audit.update(error=f"stop_reason={res['stop_reason']}", rules=[], rejected=[])
        return audit

    rules = json.loads(res["text"])["rules"]
    ok, bad = verify_quotes(doc, rules)

    if bad:  # one repair round: show the misses and ask for verbatim quotes
        lines = [f"{i}. {r['title']} ({r['citation']}): {r['quoted_span']!r}" for i, r in enumerate(bad, 1)]
        repair_text = (
            "These quoted_span values were not found verbatim in the document text, so the rules were "
            "rejected:\n" + "\n".join(lines) + "\n\nFor each of these rules, return the full rule record again "
            "with a quoted_span copied exactly from the document text. Drop any rule that the document "
            "does not actually support. Return only these rules."
        )
        history = params["messages"] + [{"role": "assistant", "content": res["content"]}]
        rparams = build_params(repair_text, schema, effort, fallback, history=history)
        rres, rhit = cached_call(client, rparams, use_cache)
        audit["calls"].append({"kind": "repair", "cache_hit": rhit, "stop_reason": rres["stop_reason"],
                               "served_model": rres.get("model"), "request_id": rres.get("request_id"),
                               "usage": rres.get("usage")})
        if rres["stop_reason"] == "end_turn":
            fixed, bad = verify_quotes(doc, json.loads(rres["text"])["rules"])
            ok += fixed

    for r in ok:
        r["_doc_id"] = doc.doc_id
    audit["rules"] = ok
    audit["rejected"] = [{"title": r["title"], "citation": r["citation"], "quoted_span": r["quoted_span"],
                          "reason": "quote not found verbatim in source"} for r in bad]
    return audit


# --------------------------------------------------------------------------- assembly

def finalize_rule(r: dict, doc: Doc, as_of: str) -> dict:
    """Turn a verified LLM rule into a challenge-schema record (minus team_rule_id)."""
    eff, eff_rule = r["effective_date"], None
    computed, how = resolve_effective_date(r["effective_date_formula"], r["enacted_date"])
    date_notes = r["date_notes"]
    if computed:
        if eff and eff != computed:
            date_notes = (date_notes + " " if date_notes else "") + \
                f"Model-stated date {eff} differs from computed {computed}; using computed."
        eff, eff_rule = computed, how
    if eff and not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", eff):
        date_notes = (date_notes + " " if date_notes else "") + f"Unparsed effective date: {eff!r}."
        eff = None

    status = status_as_of(r["legal_status"], eff, as_of)
    cov = r["coverage"]
    return {
        "team_rule_id": None,
        "jurisdiction": r["jurisdiction"],
        "level": "state" if re.fullmatch(r"[A-Z]{2}", r["jurisdiction"]) else "city",
        "category": r["category"],
        "status": status,
        "title": r["title"],
        "requirement": r["requirement"],
        "key_value": r["key_value"],
        "coverage_conditions": cov,
        "exemptions": r["exemptions"],
        "overrides": [],
        "interaction": r["interaction"],
        "effective_date": eff,
        "citation": r["citation"],
        "source_doc_id": doc.doc_id,
        "source_url": doc.source_url,
        "quoted_span": r["quoted_span"],
        "confidence": round(float(r["confidence"]), 2) if r["confidence"] is not None else None,
        "conflict_flag": bool(date_notes),
        "conflict_note": date_notes,
        # Extra fields (the schema allows them): provenance and inputs for Modules B and C.
        "retrieved_at": doc.retrieved_at,
        "source_type": doc.source_type,
        "legal_status": r["legal_status"],
        "enacted_date": r["enacted_date"],
        "effective_date_formula": r["effective_date_formula"],
        "effective_date_rule": eff_rule,
        "penalty": r["penalty"],
        "as_of": as_of,
        "provenance": {"chunk_id": r["chunk_id"], "raw_offsets": r["_raw_offsets"],
                       "model": MODEL, "prompt_version": PROMPT_VERSION},
    }


def _dedupe_key(rec: dict) -> tuple:
    cite = re.sub(r"[^a-z0-9.:-]", "", rec["citation"].lower())
    return (rec["jurisdiction"], rec["category"], cite, (rec["key_value"] or "").lower())


def dedupe(records: list[dict]) -> list[dict]:
    """Merge records of the same rule from several documents, preferring official sources."""
    best: dict[tuple, dict] = {}
    for rec in records:
        k = _dedupe_key(rec)
        cur = best.get(k)
        rank = (OFFICIAL_RANK.get(rec["source_type"], 9), -(rec["confidence"] or 0))
        if cur is None:
            rec["also_supported_by"] = []
            best[k] = rec
        else:
            cur_rank = (OFFICIAL_RANK.get(cur["source_type"], 9), -(cur["confidence"] or 0))
            loser, winner = (cur, rec) if rank < cur_rank else (rec, cur)
            winner.setdefault("also_supported_by", [])
            winner["also_supported_by"] = sorted(set(
                winner["also_supported_by"] + loser.get("also_supported_by", []) + [loser["source_doc_id"]]))
            best[k] = winner
    return list(best.values())


def assign_ids(records: list[dict]) -> list[dict]:
    records.sort(key=lambda r: (r["level"] != "state", r["jurisdiction"], r["category"],
                                r["citation"], r["source_doc_id"], r["quoted_span"]))
    for n, r in enumerate(records, 1):
        r["team_rule_id"] = f"r-{n:04d}"
    return records


def validate(records: list[dict]) -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text())
    v = jsonschema.Draft202012Validator(schema)
    errors = []
    for r in records:
        for e in v.iter_errors(r):
            errors.append(f"{r['team_rule_id']}: {'/'.join(map(str, e.path))}: {e.message}")
    return errors


# --------------------------------------------------------------------------- CLI

def estimate(docs: list[Doc], by_doc: dict) -> None:
    sys_tok = len(SYSTEM_PROMPT) // 4
    in_tok = sum(len(render_document(d, by_doc[d.doc_id])) // 4 for d in docs) + sys_tok * len(docs)
    out_tok = 6000 * len(docs)  # rough: thinking + JSON per document
    cost = in_tok * 4 / 1e6 + out_tok * 20 / 1e6
    print(f"{len(docs)} documents, ~{in_tok:,} input tokens, ~{out_tok:,} output tokens (rough)")
    print(f"estimated cost on {MODEL}: ~${cost:.2f} for a cold run (+ repair calls); cached re-runs are free")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", help="comma-separated doc_ids (default: all captured documents)")
    ap.add_argument("--as-of", default=DEFAULT_AS_OF)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached responses")
    ap.add_argument("--no-fallback", action="store_true", help="omit the server-side refusal fallback")
    ap.add_argument("--dry-run", action="store_true", help="print a cost estimate and exit")
    args = ap.parse_args(argv)

    docs, chunks = build()
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    # Documents with no relevant chunk carry no rules; skip them.
    docs = [d for d in docs if any(c.relevant for c in by_doc[d.doc_id])]
    if args.docs:
        wanted = set(args.docs.split(","))
        docs = [d for d in docs if d.doc_id in wanted]
    jurisdictions = sorted({r["jurisdictions"] for r in load_manifest()})

    if args.dry_run:
        estimate(docs, by_doc)
        return 0

    import anthropic
    client = anthropic.Anthropic()

    audits, started = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_doc, client, d, by_doc[d.doc_id], jurisdictions, args.effort,
                            not args.no_fallback, not args.no_cache): d for d in docs}
        for f in as_completed(futs):
            d = futs[f]
            try:
                a = f.result()
            except Exception as e:  # keep going; the audit log records the failure
                a = {"doc_id": d.doc_id, "error": f"{type(e).__name__}: {e}", "rules": [], "rejected": []}
            audits.append(a)
            print(f"{d.doc_id}: {len(a['rules'])} rules, {len(a['rejected'])} rejected"
                  + (f"  ERROR {a['error']}" if a.get("error") else ""), flush=True)

    doc_index = {d.doc_id: d for d in docs}
    records = [finalize_rule(r, doc_index[a["doc_id"]], args.as_of) for a in audits for r in a["rules"]]
    records = assign_ids(dedupe(records))
    errors = validate(records)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "rules.json"
    if args.docs:
        out_path = OUT_DIR / "rules.partial.json"  # never overwrite the full run with a subset
    out_path.write_text(json.dumps({"as_of": args.as_of, "rules": records}, indent=2, ensure_ascii=False))
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(OUT_DIR / "extraction_audit.jsonl", "a", encoding="utf-8") as f:
        for a in sorted(audits, key=lambda a: a["doc_id"]):
            entry = {k: v for k, v in a.items() if k != "rules"}
            entry.update(run_at=run_at, n_rules=len(a["rules"]))
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    n_rej = sum(len(a["rejected"]) for a in audits)
    n_err = sum(1 for a in audits if a.get("error"))
    print(f"\n{len(records)} rules after dedupe -> {out_path}  ({n_rej} rejected quotes, "
          f"{n_err} documents with errors, {time.time() - started:.0f}s)")
    if errors:
        print(f"{len(errors)} schema errors:\n  " + "\n  ".join(errors[:20]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
