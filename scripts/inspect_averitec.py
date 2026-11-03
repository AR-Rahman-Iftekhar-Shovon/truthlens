import json
from collections import Counter

from truthlens.config import PROJECT_ROOT, load_config


def load_claims(name: str) -> list[dict]:
    path = PROJECT_ROOT / load_config()["paths"]["data_raw"] / "averitec" / name
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    for name in ("train.json", "dev.json"):
        claims = load_claims(name)
        print(f"\n== {name}: {len(claims)} claims")
        print(dict(Counter(claim["label"] for claim in claims)))

        question_counts = sorted(len(claim["questions"]) for claim in claims)
        print(
            "questions per claim (min, median, max):",
            question_counts[0],
            question_counts[len(question_counts) // 2],
            question_counts[-1],
        )

        answers = [a for claim in claims for q in claim["questions"] for a in q["answers"]]
        print("total answers:", len(answers))
        print("answer types:", dict(Counter(a.get("answer_type") for a in answers)))
        print("answers without source_url:", sum(1 for a in answers if not a.get("source_url")))

    dev = load_claims("dev.json")
    print("\nanswer keys:", sorted(dev[0]["questions"][0]["answers"][0]))
    print("\nfirst dev claim, questions:")
    print(json.dumps(dev[0]["questions"], indent=1)[:1500])


if __name__ == "__main__":
    main()
