# Rental Housing Law Navigator — Build Plan

*Not legal advice. Prototype for the MIT AI Hackathon.*

## 0. What we must ship

| Deliverable | Content | Priority |
|---|---|---|
| `out/rules.json` | Rule records from automated extraction, validated against `schema/rule_record.schema.json` | P0 (Module A) |
| `out/lookups.json` | All 500 addresses → `[{team_rule_id, result, explanation, conflict_flag}]`, `as_of` 2026-10-01 | P0 (Module B) |
| `out/changes.json` | T1–T5 → `affected_address_ids`, `conflict_flag_address_ids`, `notes` | P1 (Module C) |
| Demo UI | Address + as-of date → cited rules, "Not legal advice" banner, audit view | P1 |
| Method note | One page: pipeline, assumptions, known gaps | P1 |
| Stretch | Spanish view, confidence display, new jurisdiction | P2 |

## 1. Findings from the starter pack (read before coding)

**Corpus**
- 87 docs: 54 have text (`corpus/text/`); 33 are link-only (no text, so no quoted span possible).
- Link-only items that matter: **Hoboken (D032–34), Newark (D070–72), LA RSO code (D038)** ordinances, **Jersey City algorithmic ban (D035, D037)**, CA Civ. Code mirrors (D017–D020), NJ statute mirrors (D061–64), the MA ballot ruling (D059) and D056 (403).
- Test T2 needs `HOB-ALG-01` / `JC-ALG-01`, but no text is supplied for either. **Decision needed (§7).**
- Every text file starts with `SOURCE:` and `RETRIEVED:` lines. Parse those for `source_url` and the retrieval date.
- Pages are scraped HTML with a lot of boilerplate (nav menus, "Skip to Main Content"). Strip it before chunking.

**Key dates in the text**
- CA AB 325 (D022): chaptered 10/06/25, so effective 2026-01-01 under CA's default rule. T1 checks 2025-12-31 vs 2026-01-02.
- NJ FAIR Act (D069): approved 2026-07-20; §9 says it takes effect "on the first day of the twelfth month next following the date of enactment", which is **2027-07-01**. §6(b) ("A municipality shall be prohibited from enacting an ordinance that conflicts with this act") is the textual basis for the T3 conflict flag.
- MA S.2983 / H.5222 (D045–47): reported out of committee, still pending, so status `pending`.
- MA rent-control ballot question 25-21: only link-only news (D059). Record it as `failed`. Base the "no rent cap" claim on G.L. c.40P (D048, the state ban on local rent control), which has text.
- Open questions to surface (bonus): Berkeley ch. 13.63 effective date (D001 vs D002), LA RSO formula date, CA screening-fee cap figure.

**Addresses (`data/sample_addresses.csv`, 500 rows)**
- Boston rows use neighborhood mailing names (Dorchester, Roxbury, East Boston, Allston, …). All of them are the City of Boston.
- San Diego has one `San Ysidro` row, which is inside the City of San Diego.
- **All 80 LA rows say "Los Angeles", but they come from LA *County* parcels.** The zip prefixes (910, 913, 914, 916) suggest some may be outside the City of LA (for example San Fernando, Burbank or Pasadena), so we must geocode.
- **NJ `zip` is unreliable**: 11219, 100xx and 787xx are NY/TX zips, probably owner mailing addresses. Geocode NJ without the zip.
- Missing facts: year built (NJ 106, CA 98, MA 8); units (all of Newark/JC/Berkeley, 39 of 40 Hoboken, Boston `A/` rows, 3 LA). Two buildings were built in 1978, so the LA cutoff is `unknown` for them.
- There are no owner names, so owner-type exceptions are `unknown`, or "cannot apply" when units > 4.

## 2. Architecture

```
corpus/text/*.txt ──► [A] extract ──► rules.json (+ rules_audit.jsonl)
                                         │
data/sample_addresses.csv ─► [B1] geocode (Census, cached) ─► addresses_resolved.csv
                                         │
                         [B2] coverage engine (deterministic, 3-valued) ─► lookups.json
                                         │
                         [C] as-of runner + diff over T1–T5 ─► changes.json
                                         │
                                  [UI] Streamlit / FastAPI demo
```

**Rule of thumb:** the LLM reads the law; plain code decides coverage. The LLM extracts structured predicates, and a deterministic evaluator applies them to building facts. That keeps answers reproducible and auditable.

Suggested layout:
```
src/
  corpus.py        # load manifest + text, strip boilerplate, split into sections
  extract.py       # LLM extraction → candidate records (cached by doc sha256)
  verify.py        # verbatim quoted_span check, schema validation, dedupe
  geocode.py       # Census batch geocoder + cache + fallbacks
  coverage.py      # predicate evaluator: applies / unknown / superseded / ...
  lookup.py        # build lookups.json for any as_of date
  changes.py       # T1–T5
  app.py           # demo UI
out/               # rules.json, lookups.json, changes.json, audit logs
cache/             # LLM responses, geocoder responses (commit these for a reproducible demo)
tests/             # schema + T1–T5 assertions
```

## 3. Module A: rule extraction (hours 1–6)

1. **Load and clean.** Parse the `SOURCE`/`RETRIEVED` headers, drop navigation boilerplate, and split into sections (§ headings, numbered sections, "SEC." markers). Some docs are large (D067 is 160 KB, D049 is 56 KB), so chunk them to about 8–12k tokens.
2. **Relevance pass (cheap model).** For each chunk, decide whether it states a rule in one of the 6 categories. Skip the rest.
3. **Extraction pass (strong model, structured output / tool schema).** Produce one record per rule with every schema field, plus a machine-readable `coverage_conditions` object:
   ```json
   {"min_units": 2, "co_on_or_before": "1978-10-01", "built_before_rolling_years": 15,
    "owner_type": null, "use": "residential", "excludes": ["owner_occupied_duplex"]}
   ```
   Also extract `penalty` (required by the brief) into `requirement` or an extra field.
4. **Verify.** `quoted_span` must be an **exact substring** of the cleaned source (after normalizing whitespace). Reject or retry otherwise. Validate against the JSON schema. Dedupe the same rule found in several docs, keeping the official source over secondary ones.
5. **Status and effective date.** Compute `status` against the as-of date in code, from the extracted `enacted_date`, `effective_date` and `bill_status` (pending/failed). The as-of runner needs this anyway.
6. **Interactions.** Fill `overrides`/`interaction` by pairing rules in the same category and state (for example, local rent control supersedes CA Civ. 1947.12 where it applies, and the FAIR Act may preempt JC/HOB bans). Use an LLM pass over the pairs, then human-review the list.
7. **Test-ID aliases.** Map our records to `CA-ALG-01`, `HOB-ALG-01`, `JC-ALG-01`, `NJ-ALG-01`, `MA-ALG-P1/P2`, `MA-RENT-P1` in a small `test_aliases.json` (a mapping, not hand-coded rules).
8. **Audit log.** For every record, log the doc_id, sha256, chunk offsets, model, prompt hash and timestamp to `out/rules_audit.jsonl`.

Self-check: a coverage matrix of jurisdiction × category, showing where we have zero rules. Expect gaps where the source is link-only.

## 4. Module B: address lookup (hours 6–11)

**B1. Jurisdiction resolution**
- Census Geocoder batch endpoint (`/geocoder/geographies/addressbatch`, `benchmark=Public_AR_Current`, `vintage=Current_Current`). Read the **Incorporated Places** layer for the legal city, plus state and county.
- Send `street, postal_city, state` and **omit the NJ zip**. Handle `1031-1035 CLINTON ST` style ranges by retrying with the first number.
- Fallbacks, each tagged with lower confidence: (1) the single-line geocoder; (2) a postal-city map (Boston neighborhoods → Boston, San Ysidro → San Diego); (3) otherwise `jurisdiction_unknown`, so only state rules apply and city rules are `unknown`.
- Cache the results to `cache/geocode.csv` and commit them, so the demo works offline.
- Stack per address: `[state, county, city]`.

**B2. Coverage engine (3-valued logic)**
- Each predicate returns `True`, `False` or `Unknown`. AND and OR follow Kleene logic.
- Results:
  - The rule's jurisdiction is not in the stack: omit the rule.
  - Status `pending`: `pending`.
  - Effective date after as_of: `not_yet_effective`.
  - Coverage `Unknown`: `unknown`.
  - Coverage `True` and a stricter applicable local rule exists in the same category: `superseded`.
  - Coverage `True` otherwise: `applies`.
  - `failed` rules are never reported as applying (T5).
- Cutoff logic: CO-date cutoffs compare against `year_built`. Year < cutoff year gives True, year > cutoff year gives False, and the cutoff year itself or a missing year gives Unknown. This covers SF 1979-06-13 and LA 1978-10-01.
- Exceptions: when units > 4, the CA small-landlord deposit exception "cannot apply", so the result is `applies` with that explanation. When units are missing, the result is `unknown`.
- `explanation`: a templated sentence naming the deciding fact ("Built 1927, before the 1978-10-01 RSO cutoff; 32 units"). Optionally polish the wording with an LLM, but never decide coverage with it.
- `conflict_flag`: true when the rule has `conflict_flag` or an unresolved interaction with another rule in this address's stack.

## 5. Module C: change tracking (hours 11–18)

`lookup.run(as_of)` is a pure function of (rules, addresses, date). Then:

| Test | Method | Expected affected set |
|---|---|---|
| T1 | run(2025-12-31) vs run(2026-01-02) for CA-ALG-01 | every CA address (all 250 CA rows: LA 80, SF 80, SD 50, Berkeley 40; the rule is statewide, so this includes any LA row geocoded outside the city) |
| T2 | as_of 2026-10-01: HOB-ALG-01 / JC-ALG-01 by **geocoded** city | Hoboken and JC rows only, no Newark rows |
| T3 | run(2026-10-01) vs run(2027-07-02) for NJ-ALG-01 | all 140 NJ rows; conflict flags on the 90 JC and Hoboken rows (by geocoded city) |
| T4 | rules with status `pending`, evaluated as if enacted | all 110 MA rows (Boston 60, Cambridge 50) |
| T5 | MA-RENT-P1 status `failed` | empty set; assert no rent-cap rule `applies` to any MA address |

Write these as `pytest` assertions so we notice when an extraction change breaks a test.

## 6. Demo, validation, polish (hours 11–23)

- **UI (Streamlit is fastest):**
  - Inputs: an address picker or search, and an as-of date (default 2026-10-01).
  - The resolved stack is shown with the geocoder match and its confidence.
  - Rules are grouped by category, each with a status badge, plain-language text, key value, quoted span, citation, source link and retrieval date.
  - A persistent "Not legal advice" banner, and an "Audit" tab showing the reasoning trace and log line.
  - A T1–T5 page with before/after counts.
  - An "Open questions" panel listing the conflicting effective dates.
- **Validation run:**
  - Schema-validate all three JSON files.
  - All 500 addresses are present in lookups and in every test's universe.
  - Every quoted span passes the verbatim check.
  - Hand spot-check about 30 addresses: 3 per city, chosen to include missing-fact rows and the 1978/1979 rows.
- **Spanish (stretch):** cache an LLM translation of `requirement` and `explanation`. Keep citations and quotes in the original language.
- **Method note:** pipeline diagram, the LLM/code boundary, how we handle unknowns, and known gaps (the link-only sources).

## 7. Decisions for the team

1. **Link-only ordinances (Hoboken, Newark, LA RSO code, JC algorithmic ban).** The guide lets us *read* code sites freely but forbids bulk scraping, and D032–34 and D070–72 are marked `check-terms`. Recommendation: ask the organizers. If they allow it, fetch only those few specific pages once into `corpus_extra/` (with URL, retrieval timestamp and sha256) and run them through the same pipeline. Otherwise, build those rules from the official city pages we do have (D036 for JC) where possible. If no source text exists, emit no record (rather than invent one) and explain the gap in `changes.json` notes.
2. **Stack:** Python 3.11, `jsonschema`, `pydantic`, `requests`, `pandas`, `streamlit`, and the Claude API (structured output via tool use). Cache every LLM call.
3. **Team split (4 people):** (a) extraction and verification; (b) geocoding and the coverage engine; (c) as-of runner, change tests and validation; (d) UI, method note and demo script. With fewer people, merge (c) into (b) and (d) into (a) after hour 11.

## 8. Timeline (from the brief)

| Hours | Focus | Exit criterion |
|---|---|---|
| 0–1 | Kickoff, read the pack, settle §7 decisions | Repo scaffolded, API keys work |
| 1–6 | Module A | `rules.json` validates; coverage matrix reviewed |
| 6–11 | Geocoding and coverage engine | `addresses_resolved.csv` with 500 rows; first `lookups.json` |
| 11–18 | Lookups, citations, T1–T5, UI | `changes.json`; pytest T1–T5 green; UI shows one address end to end |
| 18–23 | Validation, fixes, demo prep | Spot-check done, method note written, demo rehearsed offline from caches |
| 23–24 | Demo | Not legal advice |
