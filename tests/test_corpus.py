"""Checks for src/corpus.py: offsets stay faithful and key legal facts survive cleaning."""

import re

import pytest

from src.corpus import build, locate_quote


@pytest.fixture(scope="module")
def corpus():
    docs, chunks = build()
    return {d.doc_id: d for d in docs}, chunks


def _squash(s):
    return re.sub(r"\s+", " ", s).strip()


def test_every_captured_doc_is_chunked(corpus):
    docs, chunks = corpus
    assert len(docs) == 54
    assert {c.doc_id for c in chunks} == set(docs)


def test_chunks_map_back_to_raw_text(corpus):
    docs, chunks = corpus
    for c in chunks:
        raw = docs[c.doc_id].raw
        assert 0 <= c.raw_start < c.raw_end <= len(raw), c.chunk_id
        # The span starts and ends on the chunk's first and last characters (dropped
        # boilerplate such as PDF page headers may sit in between).
        assert raw[c.raw_start] == c.text[0], c.chunk_id
        assert raw[c.raw_end - 1] == c.text[-1], c.chunk_id


def test_cmap_points_at_same_characters(corpus):
    docs, _ = corpus
    for d in docs.values():
        for i, r in enumerate(d.cmap):
            if r >= 0 and not d.clean[i].isspace():
                assert d.raw[r] == d.clean[i], (d.doc_id, i)


def test_locate_quote_is_whitespace_and_quote_insensitive(corpus):
    docs, _ = corpus
    raw = docs["D069"].raw
    span = locate_quote(raw, "A municipality shall be prohibited from enacting an ordinance that conflicts with this act.")
    assert span is not None
    assert "conflicts with this act" in raw[span[0]:span[1]]
    assert locate_quote(raw, "A municipality may adopt any ordinance it likes.") is None


@pytest.mark.parametrize("doc_id, phrase", [
    ("D069", "take effect on the first day of the twelfth month next following the date of enactment"),
    ("D069", "approved July 20, 2026"),
    ("D046", "Referred to Senate Committee on Ways and Means"),
    ("D046", "3/12/2026"),
    ("D041", "first built on or before October 1, 1978"),
    ("D081", "went into effect on October 14, 2024"),
    ("D001", "13.63.030"),
])
def test_key_facts_survive_cleaning(corpus, doc_id, phrase):
    docs, _ = corpus
    assert phrase.lower() in _squash(docs[doc_id].clean).lower()


def test_navigation_boilerplate_removed(corpus):
    docs, _ = corpus
    assert "skip to main content" not in docs["D041"].clean.lower()
    assert "\nFGC\n" not in docs["D023"].clean


def test_algorithmic_docs_flagged_relevant(corpus):
    _, chunks = corpus
    alg_docs = {c.doc_id for c in chunks if "algorithmic_rent_setting" in c.keyword_hits}
    assert {"D001", "D022", "D045", "D046", "D069", "D081"} <= alg_docs


def test_fair_act_core_ban_is_relevant(corpus):
    _, chunks = corpus
    fair = [c for c in chunks if c.doc_id == "D069"]
    assert all(c.relevant for c in fair)


def test_section_8_definition_is_not_a_heading(corpus):
    _, chunks = corpus
    assert not any(c.heading.startswith("Section 8 means") for c in chunks)


def test_header_lines_not_in_clean_text(corpus):
    docs, _ = corpus
    for d in docs.values():
        assert "RETRIEVED:" not in d.clean and "SOURCE: http" not in d.clean, d.doc_id
