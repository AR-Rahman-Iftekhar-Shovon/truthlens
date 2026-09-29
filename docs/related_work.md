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