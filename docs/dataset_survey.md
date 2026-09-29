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

## Provisional stage mapping (final decision after inspection in Phase 1)

| Stage | Candidate |
|-------|-----------|
| Classical baseline and Transformer classifier (Phases 2-3) | LIAR, text-only first |
| Evidence corpus and retrieval evaluation (Phases 5-9) | AVeriTeC knowledge store, FEVER for sanity checks |
| Claim-evidence verification (Phase 10) | FEVER (NLI-style training), AVeriTeC (real-world evaluation) |
| Bangla / Banglish (Phase 17) | Not surveyed yet, deliberately postponed |

## Open questions

- How to map AVeriTeC "Conflicting Evidence" and "Not Enough Evidence" to our UNKNOWN label.
- Whether LIAR's six labels should be collapsed, and if so how.

## Dataset versions

| Dataset | Source | File | SHA256 |
|---------|--------|------|--------|
| LIAR | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip | liar_dataset.zip | 611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7 |