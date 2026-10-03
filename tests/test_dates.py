from src.dates import resolve_effective_date, status_as_of


def test_fair_act_twelfth_month_formula():
    # NJ FAIR Act (D069): approved 2026-07-20, "first day of the twelfth month next following the date of enactment"
    d, how = resolve_effective_date(
        "shall take effect on the first day of the twelfth month next following the date of enactment",
        "2026-07-20")
    assert d == "2027-07-01" and "month 12" in how


def test_other_formulas():
    assert resolve_effective_date("on the 90th day following enactment", "2026-01-20")[0] == "2026-04-20"
    assert resolve_effective_date("This act shall take effect immediately.", "2026-01-20")[0] == "2026-01-20"
    assert resolve_effective_date("January 1 of the year next following enactment", "2026-01-20")[0] == "2027-01-01"
    assert resolve_effective_date("on a date set by the director", "2026-01-20") == (None, None)
    assert resolve_effective_date("first day of the fourth month next following", None) == (None, None)


def test_status_as_of_change_tests():
    # T1: CA AB 325, effective 2026-01-01
    assert status_as_of("enacted", "2026-01-01", "2025-12-31") == "not_yet_effective"
    assert status_as_of("enacted", "2026-01-01", "2026-01-02") == "in_force"
    # T3: FAIR Act
    assert status_as_of("enacted", "2027-07-01", "2026-10-01") == "not_yet_effective"
    assert status_as_of("enacted", "2027-07-01", "2027-07-02") == "in_force"
    # T4 / T5
    assert status_as_of("pending_bill", None, "2026-10-01") == "pending"
    assert status_as_of("failed", "2026-01-01", "2026-10-01") == "failed"
    # Partial dates compare from the start of the period
    assert status_as_of("enacted", "2026-03", "2026-02-28") == "not_yet_effective"
