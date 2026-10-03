"""Corpus loading, cleaning and chunking (Module A, step 1).

Reads corpus/corpus_manifest.csv and corpus/text/*.txt, removes web and PDF
boilerplate, re-joins hard-wrapped lines, splits each document into sections
and packs them into chunks sized for LLM extraction.

The raw text is never modified. Every clean character keeps a pointer back to
its position in the raw file, so each chunk carries raw offsets and any quote
can be checked against (and copied from) the original text.

Usage:  python -m src.corpus            # writes out/chunks.jsonl, out/corpus_report.csv, out/clean/
"""

from __future__ import annotations

import csv
import json
import re
import sys
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
OUT_DIR = ROOT / "out"

MIN_CHUNK_WORDS = 250
MAX_CHUNK_WORDS = 2000

# Exact lines (case-insensitive) that are always navigation chrome.
STOP_LINES = {
    "skip to main content", "skip to content", "main navigation", "menu", "search",
    "back", "print", "share", "toggle", "page sections", "quick links", "home",
    "close", "open", "here's how you know", "lock", "a lock", "secure", "or",
    "website feedback", "announcements", "select code", "all", "code:", "section:",
    "article:", "official websites use .gov", "translate", "english", "español",
}
STOP_PATTERNS = [
    re.compile(r"^an official website of", re.I),
    re.compile(r"^secure \.gov websites use https", re.I),
    re.compile(r"^a \.\w+ website belongs to", re.I),
    re.compile(r"^source:\s*https?://", re.I),
    re.compile(r"^retrieved:\s*\d{4}-", re.I),
]
# Enumerators such as "(a)", "1.", "iv)" are kept even when short or repeated.
ENUMERATOR = re.compile(r"^\(?[0-9a-zA-Z]{1,4}[\).]$")

# Section headings found in statutes, ordinances and bills.
HEADING_PATTERNS = [
    re.compile(r"^§+\s*\d"),                                   # § 1947.12
    re.compile(r"^(SECTION|Section|SEC\.|Sec\.)\s*\d+[\dA-Za-z.-]*(\.|:|\s+[A-Z(]|$)"),  # Section 1. / SEC. 2 (not "Section 8 means")
    re.compile(r"^\d{1,3}(\.\d{1,4}){1,3}\.?\s+[A-Z]"),        # 13.63.030 Use and sale...
    re.compile(r"^C\.\s?\d+[A-Z]?:\d+[A-Z]?-\d+"),             # C.46:8-54 (NJ codified)
    re.compile(r"^\d{1,2}\.\s{1,6}(\(?[a-z]\)?\.?\s+)?[A-Z“\"(]"),  # 3.    As used in this act
    re.compile(r"^(ARTICLE|Article|CHAPTER|Chapter|DIVISION|Division)\s+[0-9IVX]+"),
]

CATEGORY_KEYWORDS = {
    "rent_increase_limits": [
        "rent increase", "rent control", "rent stabiliz", "allowable increase",
        "general adjustment", "consumer price index", "cpi", "maximum allowable rent",
        "rent cap", "increase the rent", "increase in rent", "rent ceiling",
    ],
    "just_cause_eviction": [
        "just cause", "evict", "notice to quit", "relocation", "terminate the tenancy",
        "termination of tenancy", "at-fault", "no-fault", "for cause",
    ],
    "security_deposits": ["security deposit", "deposit"],
    "application_screening_fees": [
        "application fee", "screening fee", "tenant screening", "credit report",
        "application screening", "processing fee",
    ],
    "screening_restrictions": [
        "criminal", "conviction", "source of income", "voucher", "section 8",
        "background check", "fair chance", "arrest", "cori",
    ],
    "algorithmic_rent_setting": [
        "algorithm", "pricing software", "coordinated pricing", "rent-setting",
        "rent setting", "realpage", "price-fixing", "price fixing",
    ],
}
DATE_SIGNALS = ["effective", "take effect", "operative", "enacted", "approved", "chaptered"]
# Binding language: a chunk with no category keyword can still hold a rule (e.g. the
# FAIR Act's core ban speaks of "rental price" and "coordinating", not "algorithm").
OBLIGATION = re.compile(r"\b(shall|must|may not|unlawful|prohibit\w*|required to|is a violation)\b", re.I)


# --------------------------------------------------------------------------- data

@dataclass
class Doc:
    doc_id: str
    jurisdiction: str
    source_url: str
    source_type: str
    retrieved_at: str
    sha256: str
    path: Path
    raw: str = ""
    clean: str = ""
    cmap: array = field(default_factory=lambda: array("i"))  # clean index -> raw index (-1 = inserted)
    paragraphs: list = field(default_factory=list)            # (clean_start, clean_end)

    @property
    def domain(self) -> str:
        return urlparse(self.source_url).netloc.removeprefix("www.")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    jurisdiction: str
    source_url: str
    source_type: str
    retrieved_at: str
    heading: str
    text: str
    raw_start: int
    raw_end: int
    n_words: int
    keyword_hits: dict
    date_signal: bool
    relevant: bool              # worth sending to the LLM extractor (keyword, obligation or date signal)


# --------------------------------------------------------------------------- loading

def load_manifest(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    with open(corpus_dir / "corpus_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_docs(corpus_dir: Path = CORPUS_DIR) -> list[Doc]:
    """Documents whose text was captured (status 'ok')."""
    docs = []
    for row in load_manifest(corpus_dir):
        if row["status"] != "ok" or not row["text_file"]:
            continue
        path = corpus_dir / row["text_file"]
        raw = path.read_text(encoding="utf-8")
        url, retrieved = _parse_header(raw)
        docs.append(Doc(
            doc_id=row["doc_id"],
            jurisdiction=row["jurisdictions"],
            source_url=url or row["url"],
            source_type=row["source_type"],
            retrieved_at=retrieved or row["retrieved_at"],
            sha256=row["sha256"],
            path=path,
            raw=raw,
        ))
    return docs


def _parse_header(raw: str) -> tuple[str | None, str | None]:
    url = re.search(r"^SOURCE:\s*(\S+)", raw, re.M)
    ret = re.search(r"^RETRIEVED:\s*(.+)$", raw, re.M)
    return (url.group(1) if url else None, ret.group(1).strip() if ret else None)


# --------------------------------------------------------------------------- cleaning

def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _lines_with_offsets(raw: str):
    """Yield (stripped_text, raw_start) for each non-empty line."""
    pos = 0
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped:
            yield stripped, pos + (len(line) - len(line.lstrip()))
        pos += len(line) + 1


def domain_boilerplate(docs: list[Doc], min_docs: int = 3, min_share: float = 0.5,
                       max_len: int = 150) -> dict[str, set]:
    """Short lines that recur in most documents from the same website.

    A line counts as boilerplate when it appears in at least `min_docs` documents
    AND in at least `min_share` of that website's documents. The share test keeps
    content shared by a few similar pages (e.g. bill-history rows on 3 of 13
    malegislature.gov pages) while still removing site-wide navigation.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    n_docs: Counter = Counter()
    for d in docs:
        seen = {_norm_line(t) for t, _ in _lines_with_offsets(d.raw) if len(t) < max_len}
        counts[d.domain].update(seen)
        n_docs[d.domain] += 1
    return {dom: {k for k, n in c.items()
                  if n >= max(min_docs, min_share * n_docs[dom]) and not ENUMERATOR.match(k)}
            for dom, c in counts.items()}


def _is_heading(text: str) -> bool:
    return any(p.match(text) for p in HEADING_PATTERNS)


DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(19|20)\d{2}-\d{2}-\d{2}\b|"
                  r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}, (19|20)\d{2}", re.I)


def _drop_line(text: str, key: str, boiler: set, repeats: Counter) -> bool:
    if key in STOP_LINES or any(p.match(text) for p in STOP_PATTERNS):
        return True
    if ENUMERATOR.match(text) or DATE.search(text):  # dates carry status/effective-date facts
        return False
    if key in boiler and not _is_heading(text):
        return True
    if re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", text, re.I):  # page numbers: "25", "Page 3 of 9"
        return True
    if re.fullmatch(r"[A-Z]{2,5}", text):                      # code-site menus: FIN, FGC, CIV
        return True
    if len(text) < 80 and repeats[key] >= 3 and not _is_heading(text):  # repeated PDF headers/footers
        return True
    return False


def clean_doc(doc: Doc, boiler: set) -> None:
    """Fill doc.clean, doc.cmap and doc.paragraphs from doc.raw."""
    lines = list(_lines_with_offsets(doc.raw))
    repeats = Counter(_norm_line(t) for t, _ in lines)
    wrap = _wrap_width(lines)

    out: list[str] = []
    cmap = array("i")
    paragraphs: list[list[int]] = []
    prev_text = ""

    def emit(s: str, raw_idx: int | None):
        # Append s, collapsing whitespace runs; raw_idx=None marks inserted separators.
        for i, ch in enumerate(s):
            if ch.isspace():
                if out and out[-1] in " \n":
                    continue
                ch = " "
            out.append(ch)
            cmap.append(-1 if raw_idx is None else raw_idx + i)

    for text, raw_start in lines:
        key = _norm_line(text)
        if any(c.isalnum() for c in text) and _drop_line(text, key, boiler, repeats):
            continue
        if not any(c.isalnum() for c in text):                # lone "." or "," closes the previous line
            if paragraphs:
                emit(text, raw_start)
                prev_text += text
                paragraphs[-1][1] = len(out)
            continue
        if paragraphs and _continues(prev_text, text, wrap):
            emit(" ", None)
            emit(text, raw_start)
            prev_text = text
            paragraphs[-1][1] = len(out)
        else:
            if out:
                out.append("\n")
                cmap.append(-1)
            start = len(out)
            emit(text, raw_start)
            paragraphs.append([start, len(out)])
            prev_text = text

    doc.clean = "".join(out)
    doc.cmap = cmap
    doc.paragraphs = [tuple(p) for p in paragraphs]


def _wrap_width(lines) -> int:
    """Typical line width of a hard-wrapped document (PDF/statute text); 0 if not wrapped.

    Hard-wrapped text has many lines of similar length that end mid-sentence.
    Web pages have one line per paragraph or list item, so they return 0.
    """
    lengths = sorted(len(t) for t, _ in lines if len(t) >= 30)
    if len(lengths) < 10:
        return 0
    open_ended = sum(1 for t, _ in lines if len(t) >= 30 and not re.search(r"[.!?:;]['\"”’)]?$", t))
    if open_ended < 0.4 * len(lengths):
        return 0
    return lengths[int(0.75 * (len(lengths) - 1))]


def _continues(prev: str, cur: str, wrap: int = 0) -> bool:
    """Is `cur` a hard-wrapped continuation of `prev`?"""
    if _is_heading(cur) or ENUMERATOR.match(cur.split(" ")[0]):
        return False
    if re.search(r"[.!?:;]['\"”’)]?$", prev):
        return False
    if cur[0].islower() or cur[0] in ",;:)]%":
        return True
    if prev[-1] in ",(-–—/&" or re.search(r"\b(the|of|and|or|to|a|an|in|on|by|for|with|as|that|than|under)$", prev):
        return True
    # "...the Division on Civil" / "Rights." -- only in hard-wrapped documents
    return bool(wrap) and prev[-1].isalpha() and len(prev) >= 0.7 * wrap


# --------------------------------------------------------------------------- offsets & quotes

def raw_span(doc: Doc, clean_start: int, clean_end: int) -> tuple[int, int]:
    """Map a clean-text span to the smallest raw span covering it."""
    idx = [doc.cmap[i] for i in range(clean_start, clean_end) if doc.cmap[i] >= 0]
    if not idx:
        return (-1, -1)
    return (idx[0], idx[-1] + 1)


_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "})


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    chars, idx = [], []
    for i, ch in enumerate(s.translate(_QUOTE_CHARS)):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch.lower())
        idx.append(i)
    return "".join(chars), idx


def locate_quote(raw: str, quote: str) -> tuple[int, int] | None:
    """Find `quote` in `raw`, ignoring whitespace, case and curly-vs-straight quotes.

    Returns the raw (start, end) or None. Use raw[start:end] as the verbatim quoted_span.
    """
    norm_raw, idx = _normalize_with_map(raw)
    norm_q, _ = _normalize_with_map(quote.strip())
    if not norm_q:
        return None
    pos = norm_raw.find(norm_q)
    if pos < 0:
        return None
    return idx[pos], idx[pos + len(norm_q) - 1] + 1


# --------------------------------------------------------------------------- sections & chunks

def split_sections(doc: Doc) -> list[tuple[str, list[tuple[int, int]]]]:
    """Group paragraphs into (heading, paragraphs) sections."""
    paras = doc.paragraphs
    texts = [doc.clean[s:e] for s, e in paras]
    marked = [_is_heading(t) for t in texts]

    if sum(marked) < 3:  # web pages: short title-like line followed by prose
        for i, t in enumerate(texts):
            words = t.split()
            nxt = texts[i + 1] if i + 1 < len(texts) else ""
            marked[i] = (
                1 <= len(words) <= 10 and t[0].isupper()
                and not re.search(r"[.,;]$", t) and len(nxt.split()) >= 12
            )

    sections: list[tuple[str, list]] = []
    for p, t, is_head in zip(paras, texts, marked):
        if is_head or not sections:
            sections.append((t[:120] if is_head else "(preamble)", [p]))
        else:
            sections[-1][1].append(p)
    return sections


def _words(doc: Doc, paras) -> int:
    return sum(len(doc.clean[s:e].split()) for s, e in paras)


def chunk_doc(doc: Doc) -> list[Chunk]:
    # Pack small sections together; split oversized ones at paragraph boundaries.
    groups: list[tuple[str, list]] = []
    for heading, paras in split_sections(doc):
        if _words(doc, paras) > MAX_CHUNK_WORDS:
            buf: list = []
            for p in paras:
                if buf and _words(doc, buf + [p]) > MAX_CHUNK_WORDS:
                    groups.append((heading, buf))
                    buf = [buf[-1]]  # one-paragraph overlap
                buf.append(p)
            if buf:
                groups.append((heading, buf))
        elif groups and _words(doc, groups[-1][1]) < MIN_CHUNK_WORDS \
                and _words(doc, groups[-1][1] + paras) <= MAX_CHUNK_WORDS:
            groups[-1] = (groups[-1][0], groups[-1][1] + paras)
        else:
            groups.append((heading, list(paras)))

    chunks = []
    for n, (heading, paras) in enumerate(groups, 1):
        cs, ce = paras[0][0], paras[-1][1]
        text = doc.clean[cs:ce]
        rs, re_ = raw_span(doc, cs, ce)
        low = text.lower()
        hits = {cat: sum(low.count(k) for k in kws) for cat, kws in CATEGORY_KEYWORDS.items()}
        hits = {k: v for k, v in hits.items() if v}
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}-c{n:03d}",
            doc_id=doc.doc_id,
            jurisdiction=doc.jurisdiction,
            source_url=doc.source_url,
            source_type=doc.source_type,
            retrieved_at=doc.retrieved_at,
            heading=heading,
            text=text,
            raw_start=rs,
            raw_end=re_,
            n_words=len(text.split()),
            keyword_hits=hits,
            date_signal=any(s in low for s in DATE_SIGNALS),
            relevant=bool(hits) or bool(OBLIGATION.search(text)) or any(s in low for s in DATE_SIGNALS),
        ))
    return chunks


# --------------------------------------------------------------------------- pipeline

def build(corpus_dir: Path = CORPUS_DIR) -> tuple[list[Doc], list[Chunk]]:
    docs = load_docs(corpus_dir)
    boiler = domain_boilerplate(docs)
    chunks: list[Chunk] = []
    for d in docs:
        clean_doc(d, boiler.get(d.domain, set()))
        chunks.extend(chunk_doc(d))
    return docs, chunks


def write_outputs(docs: list[Doc], chunks: list[Chunk], out_dir: Path = OUT_DIR) -> None:
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    with open(out_dir / "corpus_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "jurisdiction", "domain", "raw_chars", "clean_chars", "kept_pct",
                    "paragraphs", "chunks", "relevant_chunks", "categories"])
        for d in docs:
            cs = by_doc[d.doc_id]
            cats = Counter(k for c in cs for k in c.keyword_hits)
            w.writerow([d.doc_id, d.jurisdiction, d.domain, len(d.raw), len(d.clean),
                        round(100 * len(d.clean) / max(len(d.raw), 1)), len(d.paragraphs),
                        len(cs), sum(c.relevant for c in cs), ";".join(sorted(cats))])
    for d in docs:
        (out_dir / "clean" / f"{d.doc_id}.txt").write_text(
            f"SOURCE: {d.source_url}\nRETRIEVED: {d.retrieved_at}\n\n{d.clean}\n", encoding="utf-8")


def main() -> int:
    docs, chunks = build()
    write_outputs(docs, chunks)
    raw = sum(len(d.raw) for d in docs)
    clean = sum(len(d.clean) for d in docs)
    rel = [c for c in chunks if c.relevant]
    print(f"{len(docs)} docs | {raw:,} raw chars -> {clean:,} clean ({100 * clean // raw}%)")
    print(f"{len(chunks)} chunks, {len(rel)} relevant ({sum(c.n_words for c in rel):,} words to extract)")
    print(f"wrote {OUT_DIR / 'chunks.jsonl'}, {OUT_DIR / 'corpus_report.csv'}, {OUT_DIR / 'clean'}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
