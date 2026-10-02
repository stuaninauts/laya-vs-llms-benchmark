# Paired significance tests

Cascade threshold 0.7. Accuracy: exact McNemar + paired bootstrap (10,000 resamples), Holm-adjusted across the 10 comparisons. Fraud: paired bootstrap (2,000 resamples) of the AUC and ECE differences.

## Routing accuracy

| A | B | n | A | B | Δ (pp) | 95% CI (pp) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| Cascade ModernBERT → luna | gpt-5.6-luna | 1800 | 82.2% | 79.2% | +3.0 | [+1.6, +4.4] | 0.000213 | yes |
| Cascade ModernBERT → luna | ModernBERT fine-tuned | 1800 | 82.2% | 78.8% | +3.4 | [+2.0, +4.8] | 4.66e-05 | yes |
| gpt-5.6-luna | ModernBERT fine-tuned | 1800 | 79.2% | 78.8% | +0.4 | [-1.7, +2.4] | 1 | no |
| gpt-5.6-luna | gpt-5.4-mini | 1800 | 79.2% | 77.1% | +2.1 | [+0.8, +3.3] | 0.0172 | yes |
| ModernBERT fine-tuned | Laya fine-tuned | 1800 | 78.8% | 76.8% | +2.0 | [+0.4, +3.7] | 0.146 | no |
| ModernBERT fine-tuned | TF-IDF + LR | 1800 | 78.8% | 77.2% | +1.6 | [-0.3, +3.3] | 0.52 | no |
| TF-IDF + LR | gpt-5.4-mini | 1800 | 77.2% | 77.1% | +0.1 | [-2.0, +2.3] | 1 | no |
| gpt-5.6-terra | gpt-5.6-luna | 900 | 80.6% | 79.6% | +1.0 | [-0.6, +2.6] | 1 | no |
| gpt-5.6-terra | TF-IDF + LR | 900 | 80.6% | 77.8% | +2.8 | [+0.1, +5.6] | 0.312 | no |
| Cascade ModernBERT → terra | Cascade ModernBERT → luna | 900 | 82.0% | 82.1% | -0.1 | [-1.1, +0.9] | 1 | no |

## Fraud: AUC and calibration error (ECE, lower is better)

| A | B | AUC A | AUC B | Δ AUC, 95% CI | ECE A | ECE B | Δ ECE, 95% CI |
|---|---|---|---|---|---|---|---|
| ModernBERT fine-tuned | gpt-5.4-mini | 0.906 | 0.868 | +0.038 [+0.018, +0.058] | 0.046 | 0.111 | -0.065 [-0.083, -0.045] |
| ModernBERT fine-tuned | Laya fine-tuned | 0.906 | 0.889 | +0.017 [+0.002, +0.033] | 0.046 | 0.049 | -0.004 [-0.015, +0.009] |
| Laya fine-tuned | gpt-5.4-mini | 0.889 | 0.868 | +0.021 [+0.002, +0.040] | 0.049 | 0.111 | -0.061 [-0.079, -0.041] |
| TF-IDF + LR | gpt-5.4-mini | 0.865 | 0.868 | -0.002 [-0.025, +0.019] | 0.052 | 0.111 | -0.058 [-0.078, -0.040] |
| ModernBERT fine-tuned | gpt-5.6-luna | 0.906 | 0.850 | +0.056 [+0.036, +0.077] | 0.046 | 0.164 | -0.119 [-0.134, -0.097] |
