# Model cards

Short cards for the models used or tested in TruthLens. Metrics are in `results.md`; limits in `limitations.md`. Weights are not committed; scripts recreate them.

## 1. Verifier (claim + evidence -> SUPPORTED / REFUTED / UNKNOWN)
- **Model:** `distilroberta-base`, sequence classification with 3 labels. Script: `scripts/train_verifier.py`.
- **Input:** the claim paired with the top-5 evidence passages joined by spaces, max length 384.
- **Training data:** AVeriTeC train, 3 labels (conflicting evidence excluded). Trained on gold human question-answer evidence; 15% of train claims held out for epoch selection and temperature fitting.
- **Training:** AdamW (weight decay 0.01), lr 2e-5, batch 16, 4 epochs, linear schedule with 10% warmup, fp16 autocast, gradient clipping 1.0, class weights proportional to 1/sqrt(frequency), best epoch by validation macro-F1. Seeds 42-44.
- **Output:** probabilities, temperature-scaled (T = 1.80 for seed 42). The pipeline abstains when the top label is UNKNOWN, confidence < 0.75, or claim-evidence similarity < 0.55.
- **Intended use:** research on abstaining claim verification over a fixed evidence pool.
- **Not for:** judging web text, new or time-sensitive claims, or any decision with real consequences. It was not evaluated on open-web evidence (live use was poor), it can be fooled by misleading related evidence, and it mishandles Yes/No evidence.
- **Results:** dev accuracy 0.722 +- 0.011 with hybrid evidence (claim only 0.647); calibrated ECE 0.038.

## 2. Check-worthy detector (sentence -> check-worthy or not)
- **Model:** `distilroberta-base`, 2 labels, max length 128. Script: `scripts/train_transformer.py --dataset claimbuster`.
- **Training data:** ClaimBuster crowdsourced sentences (CC-BY-4.0), split by debate (train 23 debates, validation 5, test 5); expert "gold" set kept separate. Label: Verdict == 1.
- **Training:** lr 2e-5, batch 32, 4 epochs, best epoch by validation macro-F1.
- **Results:** test check-worthy F1 0.780 +- 0.009 (3 seeds); the model reused in the demo (seed 42, retrained) had validation F1 0.746.
- **Limits:** US presidential debates, not news. On articles it was only checked by hand on one example.

## 3. Dense retriever and similarity gate
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, used zero-shot with normalized embeddings (cosine similarity). About 22M parameters.
- **Use:** dense retrieval over the 9453-passage pool; claim-evidence similarity for the abstention gate (0.55); filtering live snippets.
- **Limits:** not fine-tuned. The gate cannot tell wrong-country or wrong-topic evidence apart when wording is similar.

## 4. Lexical retriever
- **Model:** BM25 (`rank-bm25`, library defaults), lowercase alphanumeric tokens, no stop words. Fused with the dense ranking by reciprocal rank fusion (k = 60, depth 100, top 5 to the verifier).

## 5. Zero-shot NLI stance model (live path only)
- **Model:** `cross-encoder/nli-deberta-v3-small`, not trained or tuned by us. Snippet = premise, claim = hypothesis; entailment -> SUPPORTED, contradiction -> REFUTED, neutral -> UNKNOWN.
- **Limits:** probabilities are not calibrated. It is often neutral on relevant text and can be confidently wrong on text that is only topically related. Live coverage with the final rules was 0.15 on 20 claims.

## 6. LIAR text classifiers (baseline experiments)
- **Models:** TF-IDF + LogisticRegression (1-2 grams, class_weight balanced) and DistilRoBERTa (max length 64, lr 2e-5, batch 32, 4 epochs). Text only; speaker metadata dropped because of leakage.
- **Results:** test macro-F1 0.622 (baseline) vs 0.635 +- 0.012 (Transformer): no clear gain. Not part of the final pipeline.

## 7. Cross-encoder reranker (dropped)
- **Models:** `cross-encoder/ms-marco-MiniLM-L-6-v2` zero-shot and fine-tuned on AVeriTeC train claims. Both lowered MRR and Recall@1/5 and are not used.
