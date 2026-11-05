import pandas as pd

from truthlens.bm25_retrieval import retrieve, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Q: Who's there? A: Tom-3") == ["q", "who", "s", "there", "a", "tom", "3"]


def test_retrieve_ranks_matching_passage_first():
    claims = pd.DataFrame({"claim_id": ["c1"], "claim": ["the moon landing was staged"]})
    evidence = pd.DataFrame(
        {
            "evidence_id": ["e1", "e2", "e3"],
            "text": [
                "Q: tax rates A: they went up",
                "Q: moon landing A: it was filmed in a studio, staged",
                "Q: weather A: sunny",
            ],
        }
    )

    assert retrieve(claims, evidence, k=2)["c1"][0] == "e2"
