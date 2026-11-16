from truthlens.live_verdict import build_stances, decide_live_verdict

LABELS = ["SUPPORTED", "REFUTED", "UNKNOWN"]

FACT_CHECKERS = {"snopes.com"}


def snip(domain, stance, confidence=0.9):
    return {"domain": domain, "stance": stance, "confidence": confidence}


def verdict(snippets):
    return decide_live_verdict(snippets, FACT_CHECKERS)["verdict"]


def test_no_snippets_is_insufficient():
    assert verdict([]) == "INSUFFICIENT_EVIDENCE"


def test_one_supporting_domain_is_not_enough():
    assert verdict([snip("bbc.com", "SUPPORTED")]) == "INSUFFICIENT_EVIDENCE"


def test_same_domain_twice_counts_once():
    assert (
        verdict([snip("bbc.com", "SUPPORTED"), snip("bbc.com", "SUPPORTED")])
        == "INSUFFICIENT_EVIDENCE"
    )


def test_two_independent_domains_support():
    assert verdict([snip("bbc.com", "SUPPORTED"), snip("reuters.com", "SUPPORTED")]) == "SUPPORTED"


def test_fact_checker_refutation_is_enough():
    assert verdict([snip("snopes.com", "REFUTED")]) == "REFUTED"


def test_non_fact_checkers_cannot_refute():
    assert verdict([snip("bbc.com", "REFUTED")]) == "INSUFFICIENT_EVIDENCE"
    assert (
        verdict([snip("bbc.com", "REFUTED"), snip("reuters.com", "REFUTED")])
        == "INSUFFICIENT_EVIDENCE"
    )


def test_conflict_is_insufficient():
    snippets = [
        snip("bbc.com", "SUPPORTED"),
        snip("reuters.com", "SUPPORTED"),
        snip("snopes.com", "REFUTED"),
    ]
    assert verdict(snippets) == "INSUFFICIENT_EVIDENCE"


def test_low_confidence_stance_is_ignored():
    snippets = [snip("bbc.com", "SUPPORTED", 0.5), snip("reuters.com", "SUPPORTED", 0.6)]
    assert verdict(snippets) == "INSUFFICIENT_EVIDENCE"


def test_build_stances_marks_unrelated_snippet_unknown():
    snippets = [{"domain": "bbc.com"}, {"domain": "reuters.com"}]
    probs = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]
    stances = build_stances(snippets, probs, [0.7, 0.3], LABELS, 0.55)
    assert stances[0]["stance"] == "SUPPORTED"
    assert stances[1]["stance"] == "UNKNOWN"
    assert stances[1]["domain"] == "reuters.com"
