from truthlens.averitec_corpus import build_tables


def test_build_tables_skips_unanswerable_and_maps_labels():
    claims = [
        {
            "claim": "X happened",
            "label": "Not Enough Evidence",
            "claim_date": "01-01-2020",
            "justification": "secret",
            "questions": [
                {
                    "question": "Who said it",
                    "answers": [{"answer": "A", "answer_type": "Extractive", "source_url": "u"}],
                },
                {
                    "question": "Any proof",
                    "answers": [{"answer": "No answer", "answer_type": "Unanswerable"}],
                },
            ],
        }
    ]

    claim_df, evidence_df = build_tables(claims, "dev")

    assert list(claim_df["label_mapped"]) == ["UNKNOWN"]
    assert list(claim_df["n_evidence"]) == [1]
    assert list(evidence_df["evidence_id"]) == ["dev-0-0-0"]
    assert "justification" not in claim_df.columns
