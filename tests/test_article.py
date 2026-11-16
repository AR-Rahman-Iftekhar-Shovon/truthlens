from truthlens.article import select_claims, split_sentences


def test_split_sentences_drops_short_fragments():
    text = "The minister said taxes rose by 5 percent last year. Wow! He also cited new jobs data."
    assert split_sentences(text) == [
        "The minister said taxes rose by 5 percent last year.",
        "He also cited new jobs data.",
    ]


def test_split_sentences_handles_newlines():
    assert (
        len(split_sentences("First long sentence is here.\n\nSecond long sentence is here.")) == 2
    )


def test_select_claims_orders_by_score_and_filters():
    sentences = ["a b c d e", "f g h i j", "k l m n o"]
    chosen = select_claims(sentences, [0.2, 0.9, 0.7], top_n=2, min_score=0.5)
    assert [c["claim"] for c in chosen] == ["f g h i j", "k l m n o"]
    assert select_claims(sentences, [0.1, 0.2, 0.3]) == []
