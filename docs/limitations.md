# Limitations

What TruthLens does not show, and what was not verified. See `results.md` for the numbers.

## Evaluation design
- **Closed evidence pool.** Retrieval and verification are evaluated on 9453 passages built from the human question-answer evidence of AVeriTeC train and dev claims (about 2018-2021). Annotators wrote the questions after seeing the claim, which inflates lexical overlap. Open-web retrieval is harder, so these numbers are optimistic.
- **One evaluation set.** AVeriTeC dev (500 claims) is the only set for retrieval and verification, and it was also used to choose between retrieval variants (a mild selection effect).
- **Few seeds.** The verifier ablation uses 3 seeds; calibration, abstention, robustness and the faithfulness study use one seed. GPU runs are not bit-identical.
- **Small classes.** UNKNOWN has only 35 dev claims, so macro-F1 and UNKNOWN F1 are very noisy. Differences in macro-F1 between verifier variants are within noise.
- **No significance tests.** Retrieval and verification comparisons were not tested for statistical significance.

## Models and components
- The Transformer classifier on LIAR did not clearly beat a TF-IDF baseline.
- The verifier mishandles Yes/No (Boolean) answers, becomes less balanced under sensational wording, and is fooled by misleading evidence (accuracy 0.37 on the misleading set). The similarity gate reduces this but would not stop strongly related misleading evidence, and does nothing for mixed evidence.
- The similarity gate can pass evidence from the wrong country or topic if the wording is similar (a South African schools passage scored 0.56 for a generic "government built 500 schools" claim; only low confidence stopped the answer).
- The reranker was a negative result and is not part of the pipeline.
- Time-sensitive claims are unsupported: the corpus ends around 2021, and there is no date handling.

## Explanations
- The "Why" line in the report is a template. The lite faithfulness study (one seed, no human judgement) found the most influential passage is a gold passage only slightly more often than chance, so the cited evidence is shown as context, not as proof of why the model decided.
- Removing all evidence is out of distribution for the verifier, so that measurement partly reflects unfamiliar input.

## Claim extraction (article input)
- The check-worthy detector is trained on US presidential debates, not news. Its quality on articles was not measured; one hand-written article was checked.
- Extracted claims keep attribution phrases ("the minister said ..."). The verifier cannot check who said something.
- Sentence splitting is a simple regex and drops fragments shorter than five words.

## Live web verification (experimental)
- Evaluated on 20 claims written by us, one run each. Search depth, number of results and the REFUTED rule were chosen after seeing the same claims, so the result is optimistic. Final coverage was 0.15.
- Stance judgement is the bottleneck. The AVeriTeC-trained verifier does not transfer to web snippets; the zero-shot NLI model is uncalibrated, is often neutral on relevant text and can be confidently wrong on topically related text. Several trusted domains agreeing does not make their stance errors independent.
- Only fact-checking sites can produce REFUTED. Absence of evidence is never treated as falsity.
- The trusted-domain list is a short hand-made allowlist, not a source-credibility model. A large language model as stance judge was not tested.
- Image and video claims cannot be settled from text snippets.

## Data and licensing
- AVeriTeC is cc-by-nc-4.0 in its current README (an older README said apache-2.0); non-commercial terms apply. LIAR license was not verified. ClaimBuster is CC-BY-4.0. Check each license before redistributing data, models or a hosted demo.
- AVeriTeC `justification` and `fact_checking_article` are never used as input (label leakage). Conflicting-evidence and cherry-picking claims are excluded from the three-class scheme and never counted as REFUTED.

## Engineering
- CPU inference was written but not benchmarked; all GPU runs were on a Kaggle T4. Model weights and processed data are not in the repository and must be recreated with the scripts.
- Modules that import torch cannot be unit tested without it. Their logic is kept in torch-free modules where possible; the torch pipelines were checked by running them, not by tests.
- Some Kaggle-written result files were lost between sessions; the numbers recorded in `experiments/` and `results.md` come from the pasted run outputs.

## Not done
Bangla and Banglish support, paired significance tests, manual error categories (ambiguous claim, outdated evidence), multi-seed robustness and faithfulness runs, human faithfulness judgement, a custom evaluation set larger than 20 claims, and hosting the demo for non-technical users.
