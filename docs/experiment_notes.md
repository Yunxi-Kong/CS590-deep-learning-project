# Experiment Notes

## Main Question

The project studies whether neural sequence models can improve next-response correctness prediction over strong non-sequential baselines.

## Model Ladder

The experiments use a staged comparison:

1. Sanity baselines based on global and question-level correctness.
2. MLP baseline using aggregate history features and target question information.
3. GRU sequence model using ordered response histories.
4. GRU ablations for sequence length and question metadata.

## Evaluation Metrics

The main metrics are:

- AUC for ranking quality.
- BCE for probabilistic prediction under binary cross-entropy.
- Accuracy and F1 for thresholded classification.
- Brier score and Expected Calibration Error for probability quality.

## Main Findings

The MLP baseline is competitive at smaller scale. On the 2k-student split, it slightly outperforms the GRU. After scaling to 5k and 10k students, the GRU becomes stronger. This suggests that ordered response histories are useful, but the sequence model needs enough data to make effective use of the additional modeling capacity.

The sequence length ablation supports this interpretation: GRU AUC improves from sequence length 20 to 50 and again to 100 on the 10k-student split.

The metadata ablation shows that removing part and tag metadata has little effect. The main signal appears to come from question identity and response-history dynamics rather than coarse metadata alone.

## Future Work

The project focuses on KT1 response sequences. A natural extension would be to add lightweight behavior summaries from richer EDNet logs, such as recent non-response actions before each answer. That extension was left out of the main experiments to keep the project focused on a reproducible response-sequence prediction pipeline.
