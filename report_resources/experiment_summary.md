# Experiment Summary

This project predicts whether a student will answer the next EDNet question correctly from prior response history. The data pipeline builds student-level response sequences, computes correctness labels by aligning responses with question metadata, and creates train, validation, and test splits by student.

The experiments compare a strong aggregate-feature MLP against a GRU sequence model. The MLP uses summary information such as cumulative correctness, recent correctness, history length, elapsed-time summaries, and target question features. The GRU consumes ordered response histories and is designed to test whether sequence order adds useful information beyond aggregate history features.

The first stage used a 2k-student subset. The MLP slightly outperformed the GRU, showing that aggregate student-history and question features are strong predictors.

The second stage scaled the experiment to 5k and 10k students. At these larger scales, the GRU outperformed the MLP. This changed the interpretation from a simple model comparison to a scale-dependent result: the sequence model appears to benefit more from additional student data.

The third stage added ablations and robustness checks. The sequence length ablation showed that GRU performance improved from sequence length 20 to 50 and from 50 to 100. The metadata ablation showed that removing part and tag metadata had very little effect, suggesting that the main GRU signal comes from question identity and response-history dynamics.

Calibration and paired bootstrap analyses were added to evaluate probability quality and result stability. On the 10k test set, the GRU had better AUC, BCE, Brier score, and Expected Calibration Error than the MLP. Bootstrap confidence intervals for paired model differences supported the stability of the GRU improvement.
