# Paper Notes

Notes from reading dataset-related parts of papers. Only record what the paper states.
Section/table references are included so each claim can be checked directly.

## 1. A Systematic Survey of Claim Verification (Zerong et al., 2025)

- **Link / ID:** https://aclanthology.org/2025.findings-emnlp.1170/ — `2025.findings-emnlp.1170`
- **Datasets it uses or introduces:** This is a survey, not a new dataset paper. It covers 198 claim-verification (CV) papers published January 2022–March 2025; 65 of those papers create new CV corpora. **Source: §3.1–§3.2.**
- **Why those datasets (reason given):** The survey studies corpus construction and the relationship between claims, references, labels and justifications, and how these affect system development. **Source: §4, especially §4.1.**
- **Size, labels, evidence source:** Among the 65 newly created corpora, 12 have <=1,000 instances, 20 have 1,000–10,000, and 33 have >10,000; 52 are text-only and 13 multimodal. Most CV corpora use three labels—Supported, Refuted and NEI; 17 use binary True/False; others add labels such as Partially Supported and Conflicting Evidence/Cherry-picking. The majority include some form of justification. **Source: §4.2 and §4.1.**
- **License (if stated):** not stated for the surveyed corpora as a whole.
- **Limitations the authors mention:** Corpus construction has issues involving veracity annotation, evidence selection, claim decomposition and representing real-world multi-source reasoning. The survey also reports that many corpora have relatively few evidence-bearing sentences per claim. **Source: §4.1–§4.2 and the corpus-property discussion.**
- **Relevance to TruthLens (which phase):** Dataset-selection/literature phase. It supports treating claim classification and evidence-based verification as related but distinct problems.
- **Could it be combined with LIAR / AVeriTeC / FEVER? Label mapping needed:** Yes only with explicit mapping. The survey itself shows that label schemes vary; therefore direct row-level concatenation is not justified merely because datasets are all called claim-verification datasets. **Source: §4.1.**

## 2. A Survey on Automated Fact-Checking (Guo, Schlichtkrull & Vlachos, 2021/2022)

- **Link / ID:** https://arxiv.org/abs/2108.11896
- **Datasets it uses or introduces:** Survey; it does not introduce a new dataset. It reviews LIAR, FEVER, FakeNewsNet and many others. **Source: §3 and Tables 1–3.**
- **Why those datasets (reason given):** The survey organizes automated fact-checking around three stages—claim detection, evidence retrieval, and claim verification—and analyzes datasets by input, evidence, verdict and justification. **Source: §3, Figure 2, and Table 2.**
- **Size, labels, evidence source:** Table 2 lists LIAR as 12,836 statements, with metadata and 6 classes; FEVER as 185,445 statements, text evidence, 3 classes, Wikipedia; FakeNewsNet as 23,196 articles, metadata, 2 classes, fact-check sources. **Source: Table 2.**
- **License (if stated):** not stated in the dataset tables.
- **Limitations the authors mention:** The survey emphasizes that datasets use different definitions and that evidence is an important distinguishing factor. It also discusses challenges including dataset bias/artifacts, evidence retrieval, multilinguality, multimodality and faithfulness. **Source: §3 and the discussion following Tables 1–3.**
- **Relevance to TruthLens (which phase):** Conceptual architecture and dataset-selection phase.
- **Could it be combined with LIAR / AVeriTeC / FEVER? Label mapping needed:** Yes, but only after defining a common target label/task. LIAR's 6 classes are not equivalent to FEVER's 3 evidence-verification classes. **Source: Table 2.**

## 3. AVeriTeC (Schlichtkrull, Guo & Vlachos, 2023)

- **Link / ID:** https://arxiv.org/abs/2305.13117 — `2305.13117`
- **Datasets it uses or introduces:** AVeriTeC, a new real-world claim-verification dataset with 4,568 claims from fact-checks by 50 organizations. **Source: Abstract; §4; §5/Table 2.**
- **Why those datasets (reason given):** The authors wanted real-world claims with realistic web evidence, question-answer decomposition and textual justifications, while addressing context dependence, evidence insufficiency and temporal leakage in earlier resources. **Source: §1, especially lines/discussion around the three limitations; Table 1 in §2 compares prior datasets.**
- **Size, labels, evidence source:** 4,568 claims. Train/dev/test = 3,068/500/1,000. Labels: Supported, Refuted, Conflicting Evidence/Cherry-picking, Not Enough Evidence. Evidence is represented through question-answer pairs backed by web sources; the dataset uses open-web evidence and caches evidence pages. **Source: §3; §4; §5/Table 2.**
- **License (if stated):** Dataset and baseline: CC-BY-NC-4.0. The Google FactCheck Claim Search API is stated as CC-BY-4.0. **Source: §1.**
- **Limitations the authors mention:** The dataset is somewhat unbalanced, with most claims refuted, because journalists tend to select false/misleading claims. Temporal-leakage protection is approximate because publication dates can be unavailable/incorrectly inferred; about 6% of development answers came from fact-checking domains. **Source: §5; §4 (Question Generation & Answering); §8.**
- **Relevance to TruthLens (which phase):** Main evidence-based verification/evaluation phase: evidence retrieval, evidence sufficiency, verdict and justification.
- **Could it be combined with LIAR / AVeriTeC / FEVER? Label mapping needed:** Do not row-wise merge. AVeriTeC's four labels have distinct meanings; especially Conflicting Evidence/Cherry-picking is not automatically Refuted. **Source: §3 and Table 2.**

## 4. FEVER (Thorne, Vlachos, Christodoulopoulos & Mittal, 2018)

- **Link / ID:** https://aclanthology.org/N18-1074/ — `N18-1074`
- **Datasets it uses or introduces:** FEVER, a new publicly available claim-verification dataset. **Source: Abstract; dataset construction section.**
- **Why those datasets (reason given):** The authors introduce FEVER as a testbed for verification against textual sources. Claims are generated by altering sentences extracted from Wikipedia and then verified by annotators. **Source: Abstract; dataset construction discussion.**
- **Size, labels, evidence source:** 185,445 claims. Labels: Supported, Refuted, NotEnoughInfo. For Supported/Refuted claims, annotators recorded the sentence(s) forming the necessary evidence. Evidence source: Wikipedia. **Source: Abstract and dataset construction section.**
- **License (if stated):** not stated in the paper's dataset description.
- **Limitations the authors mention:** The paper demonstrates that evidence retrieval/verification is difficult: the best system accuracy with correct evidence was 31.87%, while ignoring evidence gave 50.91%. The paper therefore treats FEVER as a challenging testbed. **Source: Abstract and evaluation discussion.**
- **Relevance to TruthLens (which phase):** Controlled evidence retrieval and claim-verification training/benchmark.
- **Could it be combined with LIAR / AVeriTeC / FEVER? Label mapping needed:** FEVER labels map naturally to TruthLens' three-way scheme: Supported → SUPPORTED; Refuted → REFUTED; NotEnoughInfo → UNKNOWN. **Source: FEVER label definition.**

## 5. FakeNewsNet (Shu, Mahudeswaran, Wang, Lee & Liu, 2019 version)

- **Link / ID:** https://arxiv.org/abs/1809.01286
- **Datasets it uses or introduces:** FakeNewsNet repository with two datasets/domains, PolitiFact and GossipCop, containing news content, social context and spatiotemporal information. **Source: §1; §3; Table 1.**
- **Why those datasets (reason given):** The authors argue that existing datasets did not combine news content, social context and spatiotemporal information, and they wanted a repository supporting fake-news detection, propagation/evolution and mitigation studies. **Source: §1–§2.**
- **Size, labels, evidence source:** Table 2 reports PolitiFact: 432 fake / 624 real news articles; GossipCop: 5,323 fake / 16,817 real. Ground-truth labels come from fact-checking sources: PolitiFact journalists/domain experts; GossipCop ratings are used for fake stories and E! Online is used for real entertainment news. **Source: §3; Table 2.**
- **License (if stated):** not stated in the paper.
- **Limitations the authors mention:** The paper says existing datasets such as LIAR mostly contain short statements rather than complete news articles; it also identifies missing temporal/social information in earlier datasets. **Source: §2 and Table 1.** The paper itself does not state the current repository's later privacy/copyright distribution restriction, so that should not be attributed to this paper.
- **Relevance to TruthLens (which phase):** Optional article-level/social-context experiment. It is not a direct replacement for claim-level evidence verification.
- **Could it be combined with LIAR / AVeriTeC / FEVER? Label mapping needed:** Direct combination is not recommended: FakeNewsNet is article-level binary fake/real data with social context, whereas FEVER/AVeriTeC are claim-level verification datasets. **Source: §3; Table 1–2.**

## Optional: LIAR (Wang, 2017)

- **Link / ID:** https://arxiv.org/abs/1705.00648
- **Datasets it uses or introduces:** LIAR, a publicly available benchmark with 12,836 manually labeled short statements from PolitiFact.com. **Source: §2; Table 1.**
- **Why those datasets (reason given):** The author argues that previous public datasets were too small for machine-learning benchmarks and introduces LIAR as a substantially larger resource. **Source: §2.**
- **Size, labels, evidence source:** Train/validation/test = 10,269/1,284/1,283. Six labels: pants-fire, false, barely-true, half-true, mostly-true, true. The statements come from PolitiFact and include analysis reports/source links, but the dataset does **not provide structured evidence annotations comparable to FEVER/AVeriTeC**. **Source: §2, especially the dataset description and Table 1.**
- **License (if stated):** not stated in the paper.
- **Limitations the authors mention:** The author notes that fact-checking verdicts require extensive journalism training and discusses the need to handle speaker credit-history carefully because the current statement's label is included in that metadata. **Source: §2.**
- **Relevance to TruthLens (which phase):** Baseline claim-classification signal / initial experiments.
- **Could it be combined with LIAR / AVeriTeC / FEVER? Label mapping needed:** It should not be directly row-merged with evidence datasets. Its six labels need an explicit mapping if used as a 3-way TruthLens classifier.

## TruthLens label mapping

Target labels: **SUPPORTED / REFUTED / UNKNOWN**

| Source | Original label | TruthLens label | Rule |
|---|---|---|---|
| FEVER | Supported | SUPPORTED | Direct |
| FEVER | Refuted | REFUTED | Direct |
| FEVER | NotEnoughInfo | UNKNOWN | Direct |
| AVeriTeC | Supported | SUPPORTED | Direct |
| AVeriTeC | Refuted | REFUTED | Direct |
| AVeriTeC | Not Enough Evidence | UNKNOWN | Direct |
| AVeriTeC | Conflicting Evidence / Cherry-picking | UNKNOWN | Do not equate conflict/cherry-picking with outright refutation |
| LIAR | true / mostly-true | SUPPORTED* | Coarse mapping; loses fine-grained truthfulness information |
| LIAR | half-true / barely-true | UNKNOWN* | Conservative mapping because these are not equivalent to evidence-based support |
| LIAR | false / pants-fire | REFUTED* | Coarse mapping; still not equivalent to an evidence-based refutation |

`*` These LIAR mappings are a **TruthLens design decision**, not a label mapping stated by Wang (2017). They should therefore be treated as an experiment/design choice and validated in the later ablation.

## Combining datasets: my own conclusion

- **Which datasets seem to be used most in recent work?**
  - The 2025 systematic survey reports that most recent CV corpora use **Supported / Refuted / NEI**. **Source: Zerong et al., §4.1.**
  - FEVER is a major controlled benchmark; AVeriTeC is designed for real-world open-web evidence verification. **Source: Guo et al., §3/Table 2; Schlichtkrull et al., §2/Table 1.**

- **Which label schemes conflict, and how would I map them?**
  - LIAR = six fine-grained truthfulness levels. **Source: Wang (2017), §2.**
  - FEVER = Supported / Refuted / NotEnoughInfo. **Source: Thorne et al. (2018), dataset construction section/Abstract.**
  - AVeriTeC = Supported / Refuted / Conflicting Evidence/Cherry-picking / Not Enough Evidence. **Source: Schlichtkrull et al. (2023), §3.**
  - TruthLens can use a three-way output: **SUPPORTED / REFUTED / UNKNOWN**.
  - FEVER and the non-conflict AVeriTeC labels map directly. LIAR requires a coarse mapping, so information is lost. AVeriTeC's conflicting/cherry-picking class is conservatively mapped to UNKNOWN for the first version.

- **What license or contamination risks did I notice?**
  - AVeriTeC dataset/baseline: **CC-BY-NC-4.0**. **Source: Schlichtkrull et al., §1.**
  - AVeriTeC explicitly addresses temporal leakage and says its guarantee is approximate because publication dates can be missing/incorrect. **Source: §4; §8.**
  - FEVER uses Wikipedia as its evidence source, so it represents a closed-source evidence setting rather than open-web retrieval. **Source: FEVER dataset description; AVeriTeC §2/Table 1.**
  - LIAR contains PolitiFact analysis reports/source links, but it does not provide structured evidence annotations comparable to FEVER/AVeriTeC. **Source: Wang §2; Schlichtkrull et al. §2/Table 1.**
  - FakeNewsNet's current repository distribution restrictions are **not stated in the cited paper**, so I will not treat them as a paper-derived limitation here.

- **What should TruthLens use, and what should it leave out?**
  - **LIAR:** use as a baseline/classifier signal, not as the main evidence dataset.
  - **FEVER:** use for controlled evidence retrieval + verification.
  - **AVeriTeC:** use as the main real-world open-web evidence benchmark/evaluation.
  - **FakeNewsNet:** leave out of the core TruthLens pipeline unless an article-level/social-context module is added.
  - **Do not row-wise merge LIAR + FEVER + AVeriTeC.** Use them in separate roles and report dataset-specific results.

- **Open question for the later ablation:** LIAR does not provide structured evidence annotations like FEVER/AVeriTeC. Therefore, an experiment that combines a LIAR-trained classifier signal with evidence verification should be evaluated on AVeriTeC (and potentially FEVER), not assumed to transfer. A LIAR-trained classifier may face **domain/task shift** when applied to AVeriTeC's real-world claims. Whether that classifier signal actually improves TruthLens must be tested in the later ablation rather than assumed.
