# Data Decisions

## LIAR (Phase 1)

- Input: raw LIAR TSV files (sha256 recorded in docs/dataset_survey.md).
- Text-only setting. Dropped speaker credit-count columns because they include the
  current statement and leak the label. Also dropped speaker, party, job_title, state and
  context for the main experiments; metadata may return later as a separate ablation.
- Duplicates: normalized statements (lowercase, collapsed whitespace). Duplicates inside train
  are removed (first kept). Train rows that also appear in validation or test are removed from
  train. Validation and test are left unchanged to stay comparable with the published splits.
- Labels: original 6-class `label` is kept. `binary_label` groups true, mostly-true, half-true
  as `true` and barely-true, false, pants-fire as `false` (common convention; half-true is
  ambiguous and is the main weakness of this mapping).
- Output: data/processed/liar/{train,validation,test}.csv (not committed to Git).
- Duplicate groups in train with conflicting original labels: 6. The first occurrence is kept.
  This is arbitrary but affects only 6 groups; revisit if label noise turns out to matter.
- Reproducibility: processed CSVs are written with `\n` line endings so hashes match across
  operating systems. Row counts and SHA256 hashes are recorded in docs/liar_manifest.json.

  ## Verification label scheme (Phases 10-12, provisional)

TruthLens verification labels: SUPPORTED, REFUTED, UNKNOWN.

| Source label | TruthLens label |
|--------------|-----------------|
| FEVER SUPPORTS | SUPPORTED |
| FEVER REFUTES | REFUTED |
| FEVER NOT ENOUGH INFO | UNKNOWN |
| AVeriTeC Supported | SUPPORTED |
| AVeriTeC Refuted | REFUTED |
| AVeriTeC Not Enough Evidence | UNKNOWN |
| AVeriTeC Conflicting Evidence/Cherrypicking | not mapped; evaluated separately, never counted as REFUTED |

LIAR is not mapped to this scheme. Datasets are not concatenated into one training set.
## ClaimBuster (Phase 4)

- Source: Zenodo 3609356 (CC-BY-4.0). `crowdsourced.csv` (22501 rows) is split by debate (`File_id`); `groundtruth.csv` (1032 rows) is a separate expert gold set, never trained on.
- Binary label: Verdict == 1 (check-worthy). Verdict -1 (non-factual) and 0 (unimportant factual) are not check-worthy. The meaning of the verdict codes was inferred from samples.
- Splits (rows / debates): train 15384 / 23, validation 3550 / 5, test 3464 / 5, gold 1031 / 30. Twenty gold debates also appear in train debates, so the headline gold number uses the 10 unseen debates (342 rows); the all-gold number is optimistic.
- Test and gold were each evaluated once. Domain: US presidential debates, not news.

## AVeriTeC evidence corpus (Phases 5-13)

- Files: `train.json` (3068 claims) and `dev.json` (500 claims) only. Knowledge-store archives (dev alone 11.5 GB) are not used.
- Corpus: human-annotated question-answer evidence with Unanswerable answers dropped: 3568 claims, 9453 passages, text "Q: ... A: ...". The pool is closed and includes the evidence of the claims being evaluated.
- Never used as input: `justification` and `fact_checking_article` (label leakage).
- Splits for training: 15% of train claims held out for epoch selection and temperature fitting; dev is the only evaluation set.
- License: cc-by-nc-4.0 in the current dataset README (an older README said apache-2.0). Non-commercial terms apply.

## Custom live evaluation set

- `experiments/custom_eval_claims.json`: 20 claims written by us (10 REFUTED from Rumor Scanner and BOOM Bangladesh, 8 SUPPORTED stable facts with primary sources, 2 invented claims with no evidence). Labels were supplied by the project owner and checked only partly. It is used to measure the experimental live path, not for training.
