"""Effective-date arithmetic and as-of status.

Statutes often state their effective date as a formula ("shall take effect on the
first day of the twelfth month next following the date of enactment"). The LLM
quotes the formula; this module computes the date deterministically so the
answer is reproducible and auditable.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "eighteenth": 18,
    "twentieth": 20, "thirtieth": 30, "sixtieth": 60, "ninetieth": 90,
    "one hundred eightieth": 180,
}


def _num(word: str) -> int | None:
    word = word.lower().strip()
    if word in ORDINALS:
        return ORDINALS[word]
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)?", word)
    return int(m.group(1)) if m else None


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return date(d.year + y, m + 1, 1)


def parse_date(s: str | None) -> date | None:
    """'2026-07-20' -> date; partial '2026-03' / '2026' -> first day of that period."""
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", s.strip())
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))


def resolve_effective_date(formula: str | None, enacted: str | None) -> tuple[str | None, str | None]:
    """Compute an effective date from a statutory formula and the enactment date.

    Returns (iso_date, rule_used) or (None, None) when the formula isn't recognised.
    """
    if not formula:
        return None, None
    f = re.sub(r"\s+", " ", formula.lower())
    enacted_d = parse_date(enacted)

    m = re.search(r"first day of the ([a-z ]+?|\d+(?:st|nd|rd|th)?) (?:calendar )?month (?:next )?following", f)
    if m and enacted_d and _num(m.group(1)):
        n = _num(m.group(1))
        return _add_months(enacted_d, n).isoformat(), f"first day of month {n} after enactment"

    m = re.search(r"(?:on the )?([a-z ]+?|\d+(?:st|nd|rd|th)?) day (?:next )?(?:following|after) (?:the date of )?(?:enactment|approval)", f)
    if m and enacted_d and _num(m.group(1)):
        n = _num(m.group(1))
        return (enacted_d + timedelta(days=n)).isoformat(), f"day {n} after enactment"

    m = re.search(r"(\d+) days (?:following|after) (?:the date of )?(?:enactment|approval)", f)
    if m and enacted_d:
        n = int(m.group(1))
        return (enacted_d + timedelta(days=n)).isoformat(), f"{n} days after enactment"

    if re.search(r"take effect immediately", f) and enacted_d:
        return enacted_d.isoformat(), "immediately on enactment"

    if re.search(r"january 1 of the (?:calendar )?year next following", f) and enacted_d:
        return date(enacted_d.year + 1, 1, 1).isoformat(), "January 1 of the next year"

    return None, None


def status_as_of(legal_status: str, effective_date: str | None, as_of: str | date) -> str:
    """Schema status for a rule on a query date.

    legal_status: 'enacted' | 'pending_bill' | 'failed' | 'unknown'
    """
    if legal_status == "failed":
        return "failed"
    if legal_status == "pending_bill":
        return "pending"
    as_of_d = as_of if isinstance(as_of, date) else parse_date(as_of)
    eff = parse_date(effective_date)
    if eff and as_of_d and eff > as_of_d:
        return "not_yet_effective"
    return "in_force"
