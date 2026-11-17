import pandas as pd

from truthlens.archive_search import search_archive


def test_search_archive_ranks_matching_claim_with_notes():
    claims = pd.DataFrame(
        {
            "claim_id": ["a", "b"],
            "claim": ["Masks stop covid spread", "Taxes rose in 2019"],
            "label": ["Supported", "Refuted"],
            "label_mapped": ["SUPPORTED", "REFUTED"],
            "claim_date": ["1-1-2020", "2-2-2020"],
        }
    )
    evidence = pd.DataFrame(
        {"claim_id": ["a"], "question": ["Do masks work?"], "answer": ["Yes"], "source_url": ["u"]}
    )
    results = search_archive("covid masks", claims, evidence)
    assert results[0]["claim"] == "Masks stop covid spread"
    assert results[0]["notes"][0]["answer"] == "Yes"
    assert len(results) == 1
