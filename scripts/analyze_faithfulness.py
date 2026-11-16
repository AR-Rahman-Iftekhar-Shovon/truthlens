import argparse
import json

import torch
from sentence_transformers import SentenceTransformer

from truthlens.analysis_dump import max_similarity
from truthlens.averitec import gold_by_claim, load_averitec_tables
from truthlens.bm25_retrieval import retrieve as bm25_retrieve
from truthlens.claim_pipeline import (
    CONFIDENCE_THRESHOLD,
    DENSE_MODEL,
    DEPTH,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from truthlens.config import PROJECT_ROOT
from truthlens.dense_retrieval import retrieve as dense_retrieve
from truthlens.faithfulness import claim_faithfulness, drop_one, summarize_faithfulness
from truthlens.hybrid_retrieval import fuse_rankings
from truthlens.live_pipeline import verifier_probabilities_fn
from truthlens.report import abstention_reason
from truthlens.verification_data import LABELS, format_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    claims, evidence = load_averitec_tables()
    dev = claims[(claims["split"] == "dev") & claims["label_mapped"].isin(LABELS)]
    queries = dev[["claim_id", "claim"]]
    claim_texts = dict(zip(queries["claim_id"], queries["claim"]))
    gold = gold_by_claim(evidence)
    by_id = evidence.set_index("evidence_id")

    dense_model = SentenceTransformer(DENSE_MODEL)
    bm25 = bm25_retrieve(queries, evidence, k=DEPTH)
    dense = dense_retrieve(queries, evidence, dense_model, k=DEPTH)
    hybrid = fuse_rankings(bm25, dense, top_n=TOP_K)
    texts = {cid: [by_id.loc[e, "text"] for e in ids] for cid, ids in hybrid.items()}
    similarity = max_similarity(dense_model, claim_texts, texts)

    probabilities = verifier_probabilities_fn(args.seed, torch.device("cuda"))
    ids = list(queries["claim_id"])
    batch_claims, batch_evidence, spans = [], [], {}
    for cid in ids:
        variants = [texts[cid], []] + drop_one(texts[cid])
        spans[cid] = (len(batch_claims), len(variants))
        batch_claims += [claim_texts[cid]] * len(variants)
        batch_evidence += [format_evidence(v, TOP_K) for v in variants]
    all_probs = probabilities(batch_claims, batch_evidence)

    rows = []
    for cid in ids:
        start, count = spans[cid]
        block = all_probs[start : start + count]
        full = block[0]
        label = LABELS[int(full.argmax())]
        reason = abstention_reason(
            label, float(full.max()), similarity[cid], CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD
        )
        is_gold = [e in gold.get(cid, set()) for e in hybrid[cid]]
        row = claim_faithfulness(full, block[1], block[2:], is_gold)
        rows.append({**row, "answered": reason is None})

    result = {
        "seed": args.seed,
        "all_dev_claims": summarize_faithfulness(rows),
        "answered_only": summarize_faithfulness([r for r in rows if r["answered"]]),
    }
    print(json.dumps(result, indent=2))
    path = PROJECT_ROOT / "experiments" / f"faithfulness_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
