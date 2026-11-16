import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
MIN_WORDS = 5


def split_sentences(text: str, min_words: int = MIN_WORDS) -> list[str]:
    parts = (part.strip() for part in SENTENCE_END.split(" ".join(text.split())))
    return [part for part in parts if len(part.split()) >= min_words]


def select_claims(
    sentences: list[str], scores: list[float], top_n: int = 3, min_score: float = 0.5
) -> list[dict]:
    """Highest check-worthy sentences first; sentences below min_score are dropped."""
    ranked = sorted(zip(sentences, scores, strict=True), key=lambda pair: -pair[1])
    return [{"claim": s, "score": float(p)} for s, p in ranked[:top_n] if p >= min_score]
