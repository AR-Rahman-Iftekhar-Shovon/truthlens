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

### RoBERTa (Liu et al., 2019) and DistilBERT (Sanh et al., 2019)
- **Phase / component:** Phases 3, 4 and 10, text classifier, check-worthy detector and verifier
- **Borrowed:** the pretrained `distilroberta-base` encoder (RoBERTa pretraining, distilled with the DistilBERT recipe) fine-tuned for sequence classification
- **Diverged:** we did not move to roberta-base; on LIAR the DistilRoBERTa model showed no clear gain over a TF-IDF baseline, so a larger model was not justified
- **Why:** the project prefers the smallest model that measurably helps, and training must fit on a free Kaggle T4

### BM25 (Robertson and Zaragoza, 2009)
- **Phase / component:** Phase 7, lexical retrieval
- **Borrowed:** the BM25 ranking function, used through the `rank-bm25` package with its default parameters
- **Diverged:** nothing tuned; simple lowercase alphanumeric tokenization and no stop-word list
- **Why:** a cheap lexical baseline that the dense and hybrid systems must beat; it also found about 3% of gold passages that dense retrieval missed in the top 50

### Sentence-BERT (Reimers and Gurevych, 2019) and MiniLM (Wang et al., 2020)
- **Phase / component:** Phase 6 dense retrieval, and the claim-evidence similarity gate
- **Borrowed:** bi-encoder embeddings with cosine similarity, using the `all-MiniLM-L6-v2` sentence-transformers model
- **Diverged:** used zero-shot, no fine-tuning; the same model also scores claim-evidence similarity for abstention
- **Why:** small (about 22M parameters) and fast enough to embed the whole pool on a free GPU

### Cross-encoder re-ranking (Nogueira and Cho, 2019; MS MARCO, Bajaj et al., 2016)
- **Phase / component:** Phase 9, reranking
- **Borrowed:** re-scoring top candidates with a BERT-style cross-encoder; the zero-shot model was `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Diverged:** we also fine-tuned it on AVeriTeC train claims; neither version helped
- **Why:** both versions lowered MRR and Recall@1/5 on dev, so the reranker was dropped and the result is kept as a negative finding

### Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, 2009)
- **Phase / component:** Phase 8, hybrid retrieval
- **Borrowed:** the fusion score sum of 1 / (k + rank) with k = 60 over BM25 and dense rankings
- **Diverged:** nothing in the formula; we fix depth 100 per system and break ties by evidence id
- **Why:** RRF needs no score normalization. Dense retrieval alone had slightly better retrieval metrics, but hybrid evidence gave better verifier accuracy over three seeds (0.722 vs 0.691), so hybrid is used

### Temperature scaling (Guo et al., "On Calibration of Modern Neural Networks", 2017)
- **Phase / component:** Phase 12, calibration and abstention
- **Borrowed:** a single temperature fitted on held-out data by minimizing negative log-likelihood, and ECE with 10 bins
- **Diverged:** the temperature is fitted on validation probabilities that use retrieved (not gold) evidence, matching test-time conditions
- **Why:** it keeps accuracy unchanged and lowered dev ECE from 0.141 to 0.038 (seed 42); the abstention threshold is only meaningful on calibrated confidence

### ClaimBuster (Hassan et al., 2017) and its crowdsourced benchmark (Arslan et al., 2020)
- **Phase / component:** Phase 4, claim extraction as check-worthy sentence detection
- **Borrowed:** the check-worthy / not check-worthy framing and the public crowdsourced and expert-labelled debate sentences
- **Diverged:** we split by debate to avoid leakage, kept the expert set separate, and fine-tuned DistilRoBERTa instead of the original feature-based classifier
- **Why:** the data is US presidential debates, not news, so use on articles is unmeasured
