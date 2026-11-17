# TruthLens

Evidence-based claim verification in Python. TruthLens does not map text to "fake/real". It retrieves evidence for a claim, judges the claim against that evidence (SUPPORTED / REFUTED / UNKNOWN), calibrates its confidence, and **abstains** (INSUFFICIENT_EVIDENCE) when the evidence is weak.

This is a research and engineering project, not a production fact-checker. Read the limitations below before trusting any output.

## Pipeline

```
article --> sentence split --> check-worthy detector --> claims
claim --> BM25 + dense retrieval --> reciprocal rank fusion (top 5)
      --> verifier (claim + evidence) --> temperature-scaled probabilities
      --> abstention gates (confidence >= 0.75, claim-evidence similarity >= 0.55, not UNKNOWN)
      --> report: assessment, evidence with source URLs, probabilities, uncertainty
```

An experimental second path searches the live web (Tavily) over an allowlist of trusted domains. It has low coverage (see results).

## Results (all numbers come from our own runs; most use one seed or a small N)

| Component | Result |
|---|---|
| LIAR (text only), test | DistilRoBERTa macro-F1 0.635 +- 0.012 (3 seeds) vs TF-IDF baseline 0.622: no clear gain |
| ClaimBuster check-worthy, test | DistilRoBERTa F1 0.780 +- 0.009 (3 seeds) vs baseline 0.676 |
| Retrieval, 500 dev claims | MRR: BM25 0.725, dense 0.786, hybrid 0.782 (closed pool of 9453 passages) |
| Reranker (cross-encoder) | Lowered MRR and Recall@1/5, dropped |
| Verifier, dev accuracy (3 seeds) | claim only 0.647, + hybrid evidence 0.722, gold evidence 0.729 |
| Calibration (seed 42) | ECE 0.141 -> 0.038 after temperature scaling (T = 1.80) |
| Abstention (seed 42, dense evidence) | confidence >= 0.75: coverage 0.42, accuracy on answered 0.84 |
| Misleading evidence | accuracy falls to 0.37; the similarity gate cuts wrongly answered cases but does not stop strongly related misleading evidence |
| Faithfulness (lite, seed 44) | most influential passage is a gold passage in 39% of claims vs 35% chance (within noise) |
| Live web verification, 20 custom claims | coverage 0.15 (3 of 20 answered), all 3 correct, 0 wrong answers |

Differences in macro-F1 between verifier variants are within seed noise (the UNKNOWN class has only 35 dev claims). Evidence adds about 4-8 accuracy points over claim-only.

## Limitations

- Retrieval and verification are evaluated on a closed pool of 9453 passages built from AVeriTeC human question-answer evidence (about 2018-2021). This is easier than open-web retrieval. New or local claims usually end as INSUFFICIENT_EVIDENCE.
- AVeriTeC dev (500 claims) is the only evaluation set for retrieval and verification; many results use a single seed.
- The verifier is fooled by misleading but related evidence and mishandles Yes/No answers. It does not transfer to web snippets (live accuracy was poor with this verifier; a zero-shot NLI model gave only low coverage).
- The "Why" line in the report is a template. The faithfulness study found no strong evidence that the model relies on the gold passage, so it is not a causal explanation.
- The live evaluation is 20 claims written by us, and its settings were chosen after seeing the same claims. Treat it as an indication, not a measurement.
- The check-worthy detector is trained on US presidential debates, not news. Its use on articles was only checked by hand on one example.
- Not done: Bangla/Banglish support, paired significance tests, manual error categories, human faithfulness judgement.

## Quick start

Training, embeddings and the demo need torch. The local development machine has no GPU; GPU work was run on Kaggle (T4).

```bash
pip install -e .
pytest -v
```

Data and weights are not in the repository. Recreate them (on Kaggle or any GPU machine):

```bash
python scripts/download_averitec.py && python scripts/build_averitec_corpus.py
python scripts/train_verifier.py --seed 44
python scripts/download_claimbuster.py && python scripts/prepare_claimbuster.py
python scripts/train_transformer.py --dataset claimbuster --seed 42
```

Use it (the scripts expect `PYTHONPATH=src` unless the package is installed):

```bash
python scripts/verify_claim.py --claim "Wearing face masks will stop the spread of covid 19"
python scripts/verify_article.py --file article.txt
python scripts/demo.py                      # Gradio web demo
TAVILY_API_KEY=... python scripts/verify_live.py --stance nli --claim "..."   # experimental
```

Inference falls back to CPU when no GPU is available; this path was written but not benchmarked.

## Repository layout

| Path | Contents |
|---|---|
| `src/truthlens/` | library code: retrieval, verification, calibration, decision layer, report, live search, analysis |
| `scripts/` | command-line entry points (training, evaluation, analysis, demo) |
| `tests/` | unit tests for everything that runs without torch |
| `configs/` | base config (seed 42, paths) and the trusted-domain allowlist |
| `experiments/` | small result JSON files |
| `docs/` | dataset notes, design decisions, related work log |

## Data and licenses

LIAR, ClaimBuster (CC-BY-4.0) and AVeriTeC (cc-by-nc-4.0 in the current dataset README; an older README said apache-2.0) are downloaded by scripts and never committed. Check each dataset's license before reuse. FEVER was surveyed but not used; FakeNewsNet was excluded.

## Reproducibility

Base seed 42 (seeds 43 and 44 for repeats). GPU runs are not bit-identical, so small run-to-run differences are expected. The AVeriTeC download script prints the sha256 of each source file so it can be compared with the hashes recorded in our runs. `docs/related_work.md` logs the published methods that influenced design decisions.
