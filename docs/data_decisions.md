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