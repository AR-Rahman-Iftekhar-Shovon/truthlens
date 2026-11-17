from truthlens.report_card import format_card

EVIDENCE = [{"text": "Q: x <b> A: y", "source_url": "http://a"}, {"text": "z", "source_url": ""}]
PROBS = {"SUPPORTED": 0.1, "REFUTED": 0.8, "UNKNOWN": 0.1}


def test_card_shows_verdict_sources_and_escapes_html():
    card = format_card("claim", "REFUTED", 0.8, PROBS, EVIDENCE, 0.6, None)
    assert "False" in card and 'href="http://a"' in card
    assert "x <b>" not in card and "x &lt;b&gt;" in card


def test_card_explains_abstention():
    card = format_card("claim", "INSUFFICIENT_EVIDENCE", 0.5, PROBS, EVIDENCE, 0.2, "weak match")
    assert "Not enough evidence" in card and "Reason: weak match." in card


def test_live_card_lists_sources_that_took_a_side():
    from truthlens.report_card import format_live_card

    stances = [
        {
            "stance": "REFUTED",
            "confidence": 0.9,
            "snippet": "s1",
            "url": "https://x.org/a",
            "domain": "x.org",
        },
        {
            "stance": "UNKNOWN",
            "confidence": 0.9,
            "snippet": "s2",
            "url": "https://y.org",
            "domain": "y.org",
        },
    ]
    result = {
        "claim": "c",
        "verdict": "REFUTED",
        "reason": "a fact-checking site",
        "stances": stances,
    }
    card = format_live_card(result)
    assert "x.org" in card and "y.org" not in card and "1 of 2 read" in card
