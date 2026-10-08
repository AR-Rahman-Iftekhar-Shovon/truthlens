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