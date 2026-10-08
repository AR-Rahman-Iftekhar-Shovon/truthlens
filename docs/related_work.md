# Related Work Log

Running log of published methods that influenced or were compared against a design decision in TruthLens.
Add an entry at the moment the decision is made, not later.

## Entry format

### <Paper name> (<authors>, <year>)
- **Phase / component:** where it applies in TruthLens
- **Borrowed:** what we took from the paper
- **Diverged:** what we changed
- **Why:** reason for the decision

## Entries

### FEVER: a large-scale dataset for Fact Extraction and VERification (Thorne et al., 2018)
- **Phase / component:** Phase 10 claim-evidence verification, retrieval sanity checks
- **Borrowed:** three-way verification scheme including a NOT ENOUGH INFO class (provisional)
- **Diverged:** we plan a final decision layer that can abstain, not only a three-way label
- **Why:** the project must not force a verdict when evidence is insufficient

### AVeriTeC (Schlichtkrull et al., 2023)
- **Phase / component:** Phases 5 and 10, evidence corpus and verification
- **Borrowed:** real-world claims, web evidence with a knowledge store for offline retrieval, and a separate class for conflicting evidence (provisional)
- **Diverged:** to be decided; we will not implement the question-answer evidence format unless retrieval results justify it
- **Why:** real claims and offline evidence avoid dependence on external search APIs during development

### "Liar, Liar Pants on Fire" (Wang, 2017)
- **Phase / component:** Phases 1-3, claim classification baseline
- **Borrowed:** the six-label PolitiFact dataset and its official splits
- **Diverged:** text-only setting; speaker metadata and credit-history counts dropped because of leakage risk; duplicates removed from train
- **Why:** the project needs a claim classifier signal that does not rely on speaker history

### A Survey on Automated Fact-Checking (Guo, Schlichtkrull, Vlachos; arXiv 2021)
- **Phase / component:** overall pipeline design, Phases 4-10
- **Borrowed:** framing of fact-checking as claim detection, evidence retrieval and verdict prediction (provisional)
- **Diverged:** we add a calibrated abstention layer on top of the verdict
- **Why:** the system must not force a verdict when evidence is insufficient

### A Systematic Survey of Claim Verification (Zerong et al., Findings of EMNLP 2025)
- **Phase / component:** label scheme and dataset selection, Phases 5 and 10
- **Borrowed:** three-way verification labels (supported, refuted, not enough evidence) as the common scheme, per notes in docs/paper_notes.md
- **Diverged:** the third class is called UNKNOWN, and AVeriTeC conflicting evidence is kept out of the three-way scheme
- **Why:** conflicting evidence is not the same as refuted

### FakeNewsNet (Shu et al.; arXiv 2018)
- **Phase / component:** dataset selection, Phase 1
- **Borrowed:** nothing in the core pipeline
- **Diverged:** excluded from the system
- **Why:** full data cannot be redistributed, and its social-context structure does not fit evidence-based verification