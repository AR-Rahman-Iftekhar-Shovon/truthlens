import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlparse

import yaml

from truthlens.config import PROJECT_ROOT

TAVILY_URL = "https://api.tavily.com/search"
RETRY_CODES = (429, 500, 502, 503, 504)
DOMAIN_GROUPS = ("fact_checkers", "news_agencies")


def _load_groups(name: str, groups: tuple[str, ...]) -> list[str]:
    with open(PROJECT_ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return sorted({domain for group in groups for domain in config.get(group, [])})


def load_trusted_domains(name: str = "trusted_domains") -> list[str]:
    return _load_groups(name, DOMAIN_GROUPS)


def load_fact_checkers(name: str = "trusted_domains") -> set[str]:
    return set(_load_groups(name, ("fact_checkers",)))


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def post_json(url: str, payload: dict, timeout: int = 45) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_results(data: dict) -> list[dict]:
    return [
        {
            "title": item.get("title", ""),
            "url": item["url"],
            "domain": domain_of(item["url"]),
            "snippet": item.get("content", ""),
        }
        for item in data.get("results", [])
        if item.get("url")
    ]


def search_evidence(
    query: str,
    domains: list[str],
    max_results: int = 5,
    post: Callable[[str, dict], dict] = post_json,
    attempts: int = 3,
) -> list[dict]:
    """Raw snippets and URLs from allowlisted domains only. The AI answer is never requested."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_domains": domains,
    }
    for attempt in range(1, attempts + 1):
        try:
            return parse_results(post(TAVILY_URL, payload))
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_CODES or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
        time.sleep(2**attempt)
    return []
