# Results

All numbers come from our own runs. Unless stated, validation is used for choices and AVeriTeC dev is the evaluation set. Test sets were evaluated once. "+-" is the standard deviation over seeds (42-44), not a confidence interval. Small result files are in `experiments/`.

## 1. LIAR claim-text classifier (Phases 2-3)

Text only, speaker metadata dropped. Binary label: true / mostly-true / half-true = true.

| System | Validation macro-F1 | Test macro-F1 (evaluated once, 1283 rows) |
|---|---|---|
| Majority class | 0.342 | - |
| TF-IDF + LogisticRegression (C = 0.3) | 0.629 | 0.622 |
| DistilRoBERTa, 3 seeds | 0.644 +- 0.017 | 0.635 +- 0.012 (0.624 / 0.634 / 0.648) |

Conclusion: no clear gain from the Transformer. The validation standard error is about 0.014, and the seeds differ in class bias.

## 2. ClaimBuster check-worthy detector (Phase 4)

Split by debate. Positive class = check-worthy.

| System | Val check-worthy F1 | Test F1 / macro-F1 | Gold, 10 unseen debates F1 / macro-F1 |
|---|---|---|---|
| TF-IDF + LogReg (C = 3) | 0.613 | 0.676 / 0.773 | 0.773 / 0.859 |
| DistilRoBERTa, 3 seeds | 0.743 +- 0.004 | 0.780 +- 0.009 / 0.848 +- 0.008 | 0.893 +- 0.009 / 0.935 +- 0.005 |

The data are US presidential debates. Transfer to news articles was not measured.

## 3. Retrieval (Phases 6-9)

500 dev claims, closed pool of 9453 passages, gold = the claim's own evidence (about 2.8 per claim).

| System | MRR | R@1 | R@5 | R@10 | R@50 |
|---|---|---|---|---|---|
| BM25 | 0.725 | 0.332 | 0.611 | 0.676 | 0.794 |
| Dense (all-MiniLM-L6-v2) | 0.786 | 0.371 | 0.683 | 0.762 | 0.873 |
| Hybrid (RRF, k = 60) | 0.782 | 0.366 | 0.670 | 0.748 | 0.880 |
| Dense + zero-shot cross-encoder | 0.781 | 0.355 | 0.682 | 0.759 | 0.873 |
| Dense + fine-tuned cross-encoder | 0.721 | 0.318 | 0.652 | 0.763 | 0.873 |
| Hybrid + fine-tuned cross-encoder | 0.667 | 0.276 | 0.633 | 0.767 | 0.880 |

Gold passages found in the top 50: only BM25 3.4%, only dense 11.3%, both 72.7%, neither 12.7%. The reranker lowers the top ranks and was dropped. No significance test was run. The pool is closed, so these numbers are easier than open-web retrieval.

## 4. Verifier ablation (Phase 10 and 16)

DistilRoBERTa, 3 labels, 462 dev claims (SUPPORTED 122, REFUTED 305, UNKNOWN 35; 38 conflicting claims excluded). Majority-class accuracy is 0.660. Mean +- std over seeds 42-44.

| Evidence given to the verifier | Accuracy | Macro-F1 |
|---|---|---|
| none (claim only) | 0.647 +- 0.013 | 0.477 +- 0.045 |
| BM25 top 5 | 0.711 +- 0.017 | 0.486 +- 0.040 |
| dense top 5 | 0.691 +- 0.005 | 0.484 +- 0.019 |
| hybrid top 5 (final system) | 0.722 +- 0.011 | 0.506 +- 0.055 |
| dense + zero-shot rerank | 0.706 +- 0.002 | 0.506 +- 0.023 |
| gold evidence | 0.729 +- 0.004 | 0.593 +- 0.016 |

Evidence adds about 4-8 accuracy points. Macro-F1 differences between variants are within noise: UNKNOWN has only 35 dev claims and its F1 ranged from 0.087 to 0.413 across runs. Hybrid was chosen for the end-task result even though dense had equal or better retrieval metrics.

## 5. Calibration and abstention (Phases 11-12)

Seed 42, evidence from dense top 5, temperature fitted on a held-out 15% of train.

- Temperature T = 1.80. Dev ECE 0.141 -> 0.038, NLL 0.855 -> 0.745, accuracy unchanged at 0.697.
- Risk-coverage with calibrated confidence: coverage 1.0 gives accuracy 0.697; 0.8 gives 0.751; 0.6 gives 0.827; 0.4 gives 0.854; 0.2 gives 0.837 (n = 92, noisy).
- Operating points chosen on validation by a target accuracy on answered claims:

| Target | Rule | Dev coverage | Dev accuracy on answered |
|---|---|---|---|
| 0.75 | confidence >= 0.60 | 0.716 | 0.792 |
| 0.80 (the rule used by the pipeline) | confidence >= 0.75 | 0.424 | 0.842 |
| 0.85 | confidence >= 0.75 and top-1 similarity >= 0.65 | 0.307 | 0.824 (missed the target on dev) |

A decision layer tuned on macro-F1 instead gave almost no gain (0.5045 -> 0.5077) and was replaced.

## 6. Robustness (Phase 13)

Dev, seed 42, confidence threshold 0.75, T = 1.80.

| Condition | Accuracy | Macro-F1 | Coverage | Accuracy on answered |
|---|---|---|---|---|
| clean | 0.697 | 0.505 | 0.424 | 0.842 |
| claim typos | 0.693 | 0.499 | 0.433 | 0.840 |
| sensational wording | 0.699 | 0.412 | 0.539 | 0.827 |
| noisy text | 0.686 | 0.492 | 0.435 | 0.846 |
| misleading evidence | 0.374 | 0.288 | 0.266 | 0.333 |
| mixed (3 real + 2 misleading) | 0.641 | 0.447 | 0.398 | 0.815 |
| random evidence | 0.548 | 0.292 | 0.288 | 0.699 |
| no evidence | 0.253 | 0.201 | 0.006 | 3 claims |

The misleading set uses evidence of the most similar other claims with the opposite verdict (mean TF-IDF similarity only 0.22), so it is a weak attack. A claim-evidence similarity gate of 0.55 cut answered misleading claims from 26.6% to 5.4% and random evidence from 28.8% to 0.2%, at a cost of clean coverage 0.424 -> 0.381 with the same accuracy 0.841. It does nothing for mixed evidence and would not stop strongly related misleading evidence. Single seed.

## 7. Error analysis (Phase 15)

462 dev claims: correct 322 (69.7%), low-confidence flip 60 (13.0%), insufficient-vs-verdict 35 (7.6%), overconfident flip 26 (5.6%), retrieval failure 19 (4.1%). Accuracy is 0.698 when a gold passage is in the top 5 (n = 400) and 0.694 when not (n = 62), so the verifier, not retrieval, is the bottleneck. The categories are automatic; manual categories (ambiguous claim, outdated evidence) were not done.

## 8. Evidence faithfulness, lite (Phase 14)

Seed 44, 462 dev claims, hybrid top 5, leave-one-out removal of passages (`experiments/faithfulness_seed44.json`).

| Measure | All claims | Answered only (n = 197) |
|---|---|---|
| Verdict changes when all evidence is removed | 76% | 72% |
| Verdict flips when any single passage is removed | 25% | 11% |
| Most influential passage is a gold passage | 38.7% (chance 35.1%) | 43.0% (chance 35.3%) |
| Mean probability drop, gold vs other passage | 0.025 vs 0.020 | 0.038 vs 0.024 |

The "no evidence" input is out of distribution (the model never saw empty evidence in training). The gap to chance is within noise for all claims (standard error about 0.025) and borderline for answered claims (about 0.037). No significance test, one seed, no human judgement. The model depends on having evidence but only weakly on which passage is the gold one, so the template "Why" line is not a causal explanation.

## 9. Live web verification (experimental)

20 custom claims written by us (10 REFUTED from Rumor Scanner and BOOM, 8 SUPPORTED stable facts, 2 invented unverifiable claims; `experiments/custom_eval_claims.json`). Search is Tavily over an allowlist of trusted domains. SUPPORTED needs at least 2 independent domains with stance confidence >= 0.75; REFUTED needs a fact-checking site. One run each (`experiments/live_eval_summary.json`).

| Stance model | Search | REFUTED rule | Coverage | Answered, correct | Wrong answers |
|---|---|---|---|---|---|
| Existing verifier | basic, 5 results | fact-checker or 2 domains | 0.10 | 2 of 2 | 0 |
| Zero-shot NLI (deberta-v3-small) | basic, 5 results | fact-checker or 2 domains | 0.05 | 1 of 1 | 0 |
| Zero-shot NLI | advanced, 10 results | fact-checker or 2 domains | 0.25 | 4 of 5 | 1 |
| Zero-shot NLI (final) | advanced, 10 results | fact-checker only | 0.15 | 3 of 3 | 0 |

The wrong answer in row 3 was an invented claim "refuted" by three trusted domains whose snippets were on the same topic but did not contradict it; the 2-domain REFUTED rule was removed for that reason. Search depth, number of results and the REFUTED rule were chosen after seeing these same 20 claims, so the numbers are optimistic. The verifier trained on AVeriTeC question-answer evidence does not transfer to web snippets (best SUPPORTED confidence on true claims was only 0.35-0.62).

## 10. Article input

The check-worthy detector retrained for the demo (seed 42) reached validation macro-F1 0.841 and check-worthy F1 0.746, consistent with the earlier runs. One hand-written 5-sentence article was passed through the pipeline: the detector kept 3 factual sentences and dropped 2 non-claims, and all 3 claims ended as INSUFFICIENT_EVIDENCE because the closed pool had no matching evidence. This is a smoke test (n = 1), not an evaluation.
