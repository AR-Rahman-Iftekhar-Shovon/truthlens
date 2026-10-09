# Experiment Notes

## Phase 2: TF-IDF baselines on LIAR (validation only)

Frozen baseline: TF-IDF (1-2 grams, min_df=2, sublinear_tf) + Logistic Regression,
class_weight=balanced.

- Binary: C=0.3 (validation macro-F1 0.6291)
- 6-class: C=3.0 (validation macro-F1 0.267)

Chosen on validation. The test set has not been used. It will be evaluated once in Phase 3,
together with the Transformer.

Differences between Logistic Regression and Linear SVM, and between neighbouring C values,
are within about one standard error (about 0.014 for binary accuracy), so they are not
treated as real improvements. Training prints a harmless-looking scipy OptimizeWarning
(`iprint`); cause not investigated.

### Top features (binary, C=0.3, trained on train only)

Towards 'true':
(paste)

Towards 'false':
(paste)

### My observations

* The features pushing towards `true` include numerical and quantitative terms such as `percent`, `million`, `000`, and `more than`. Some economic and general terms, such as `debt`, `average`, and `countries`, are also present. The list contains a specific place name, `georgia`.
* The features pushing towards `false` are dominated by political names, locations, and policy-related terms, including `obama`, `scott walker`, `wisconsin`, `obamacare`, `medicare`, and `stimulus`.
* The feature lists suggest that the baseline may be learning topic-specific associations in addition to general language patterns. The `true` list contains several quantitative expressions, while the `false` list contains many political and policy-related terms.
* These associations do not establish that the model understands truthfulness or that any individual feature causes a prediction. They reflect patterns learned from the training data.

### What this means for later phases

* The baseline may perform less reliably on new topics or domains if the associations learned from the training data do not generalize.
* Phase 13 should evaluate the frozen baseline on an appropriate domain-shifted dataset to investigate whether its predictions depend heavily on familiar topics, names, or vocabulary.
* The baseline's feature weights provide an initial interpretability check, not proof of generalization or a complete explanation of its decisions. The test set remains untouched until the planned evaluation phase.
