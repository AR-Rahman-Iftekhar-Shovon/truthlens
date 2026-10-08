# Dataset Survey

Checked on 2026-09-29 from papers and dataset cards. Licenses must be re-checked
on the official pages before the Phase 1 download. No data is committed to Git.

## Candidates

| Dataset | Task | Size | Labels | License (as found) |
|---------|------|------|--------|--------------------|
| LIAR | Claim-level truthfulness classification | 12.8K statements (10269 / 1284 / 1283) | 6: pants-fire, false, barely-true, half-true, mostly-true, true | Unknown / unspecified on TFDS card |
| FakeNewsNet | Article-level fake news detection (PolitiFact, GossipCop) | PolitiFact 432 fake / 624 real; GossipCop 5323 fake / 16817 real | fake / real | Not verified yet |
| FEVER | Claim verification against Wikipedia | 185,445 claims | SUPPORTS / REFUTES / NOT ENOUGH INFO | cc-by-sa-3.0 and gpl-3.0 (HF card) |
| AVeriTeC | Real-world claim verification with web evidence | 4,568 claims (train 3,068 / dev 500 in original release) | Supported / Refuted / Conflicting Evidence-Cherrypicking / Not Enough Evidence | CC-BY-NC-4.0 |

## Notes and risks (to verify in Phase 1)

- LIAR: short claims with speaker metadata. Check whether speaker, party and
  speaker-history count columns leak the label; the text-only setting is the honest baseline.
- FakeNewsNet: repository ships URLs and tweet IDs, article text must be crawled.
  Links decay, so the usable size will be smaller than the published size.
  Check source leakage between fake and real articles.
- FEVER: claims are synthetic (edited Wikipedia sentences) and the corpus is
  Wikipedia only. Good for verification/NLI and retrieval sanity checks, not for
  real-world misinformation style.
- AVeriTeC: real claims, temporal split, evidence as question-answer pairs.
  The knowledge store is large; check its download size before planning storage.
  Non-commercial license is acceptable for this portfolio project.

## Roles (decided after literature notes, see docs/paper_notes.md)

| Dataset | Role in TruthLens | Not used for |
|---------|-------------------|--------------|
| LIAR | Claim-level classifier baseline and Transformer classifier (Phases 2-3) | Evidence verification: its six labels are truthfulness ratings, not evidence-based verdicts |
| FEVER | Controlled retrieval and claim-verification experiments (Phases 6-10), Wikipedia as evidence | Real-world open-web evidence |
| AVeriTeC | Main real-world evidence and verification benchmark (Phases 5-12) | |
| FakeNewsNet | Excluded from the core pipeline | Article-level and social-context modelling; full data cannot be redistributed |

Datasets are never merged row by row. Each is used in its own role and reported separately.

## Open questions

- Handling of AVeriTeC "Conflicting Evidence/Cherrypicking" (see docs/data_decisions.md).
- How a classifier trained on LIAR behaves on AVeriTeC claims (domain shift). Measured in
  Phases 11 and 13; whether the classifier signal helps at all is decided by the Phase 16 ablation.
- AVeriTeC knowledge store download size, to be checked before Phase 5.
- LIAR and FakeNewsNet licenses to be confirmed on the official pages.

## Dataset versions

| Dataset | Source | File | SHA256 |
|---------|--------|------|--------|
| LIAR | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip | liar_dataset.zip | 611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7 |