import urllib.error

import pytest

from truthlens import live_search
from truthlens.live_search import domain_of, load_trusted_domains, parse_results, search_evidence


def test_domain_of_strips_www():
    assert domain_of("https://www.Reuters.com/a/b") == "reuters.com"


def test_load_trusted_domains_sorted_unique():
    domains = load_trusted_domains()
    assert domains == sorted(set(domains))
    assert "reuters.com" in domains
    assert all("." in domain and " " not in domain for domain in domains)


def test_parse_results_ignores_answer_and_missing_urls():
    data = {
        "answer": "AI summary",
        "results": [
            {"title": "T", "url": "https://www.bbc.com/x", "content": "snippet"},
            {"title": "no url", "content": "x"},
        ],
    }
    results = parse_results(data)
    assert results == [
        {"title": "T", "url": "https://www.bbc.com/x", "domain": "bbc.com", "snippet": "snippet"}
    ]


def test_search_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        search_evidence("claim", ["bbc.com"], post=lambda url, payload: {})


def test_search_payload_has_allowlist_and_no_answer(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    seen = {}

    def fake_post(url, payload):
        seen.update(payload)
        return {"results": [{"url": "https://bbc.com/a", "content": "c"}]}

    results = search_evidence("claim", ["bbc.com"], post=fake_post)
    assert seen["include_answer"] is False
    assert seen["include_domains"] == ["bbc.com"]
    assert results[0]["domain"] == "bbc.com"


def test_search_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(live_search.time, "sleep", lambda s: None)
    calls = []

    def flaky_post(url, payload):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 503, "unavailable", None, None)
        return {"results": []}

    assert search_evidence("claim", ["bbc.com"], post=flaky_post) == []
    assert len(calls) == 2
