# Paper 2: comparison of the earlier run and the fresh chat-based run

Inspected on 22 October 2026, America/Chicago. The earlier submission, its reports, and its figures are unchanged. This inspection does not replace either notebook's scientific results.

## Completion and scope

The fresh notebook completed all six training runs and all three 61-point paths in 3,914 seconds (65.2 minutes). All eight code cells executed; the saved notebook contains no error outputs. The execution status is complete. Its per-model `converged` labels mean that the notebook's stopping rule fired; they do not establish convergence to a minimizer or successful reproduction of the figure.

The input notebook, executed notebook, histories, path CSVs, endpoint measurements and figures are retained in this separate rerun directory. The six model checkpoints remain on the training server. No earlier checkpoint was an input to these runs.

## Test accuracy in the figures as originally generated

All values are percentages. These columns use different batch-normalisation conventions, so they are not a fully matched comparison.

| Model | Earlier SB | Fresh SB | Earlier LB | Fresh LB |
| --- | ---: | ---: | ---: | ---: |
| F1 | 98.61 | 98.53 | 98.20 | 98.17 |
| C1 | 71.00 | 70.51 | 66.78 | 65.52 |
| C2 | 86.07 | 85.62 | 81.33 | 77.18 |

Small-batch test accuracy exceeds large-batch test accuracy in every pair. This is consistent with the earlier observation, but accuracy alone does not validate the sharpness interpretation.

## Why the new figure looks different

The earlier notebook recomputed batch-normalisation (BN) statistics from training images at each parameter-path point, with dropout disabled. The fresh GPT-based notebook instead interpolates saved means linearly and saved variances geometrically. It also extrapolates those statistics outside the interval between the trained models. The trainable weights follow the same affine-path definition in both implementations, but the full model evaluation is different.

Positive extrapolated variances and exact preservation of the two endpoint states do not ensure that the intervening statistics describe the activations of the intervening model. The fresh path has extreme finite losses at alpha = 2: approximately 2.69e9 for F1, 9.64e5 for C1 and 8.79e29 for C2. Automatic axis scaling compresses the remaining blue curves near zero. Those near-zero-looking lines do not mean the measured losses are near zero.

An evaluation-only control tested alpha = 0, 0.5, 1 and 2 for each new model pair. First it reproduced the saved path values. It then kept the trainable weights fixed and recomputed only BN statistics, using all training images in batches of 512 and disabling dropout, matching the earlier notebook's procedure. Evaluation covered the full training and test datasets. No optimizer steps were performed. Parameter hashes were identical before and after each recalibration, and all six checkpoint file hashes were unchanged at the end.

| Model | Fresh train loss at alpha = 2 | Same weights with training-only BN recalibration |
| --- | ---: | ---: |
| F1 | 2.69e9 | 4.096 |
| C1 | 9.64e5 | 6.910 |
| C2 | 8.79e29 | 9.294 |

The effect is also present between the endpoints. For C2 at alpha = 0.5, training loss changes from 268.043 to 0.07314 and test accuracy from 23.77% to 82.82%, with identical trainable weights. This isolates BN handling as a major source of the changed path. The extreme losses are not evidence that the trained endpoint models themselves diverged.

The control changes the evaluation convention, including the endpoint BN buffers. It does not prove that training-only recalibration is the authors' convention or that it must always be used. Only four alpha values were evaluated under that convention; a complete alternative 61-point figure has not been generated or substituted. Any adoption of this convention must be stated explicitly, rather than presented as a cosmetic plotting change.

## Matched evaluation of the endpoints

The fresh column below uses the evaluation-only BN control, so both versions now use training-only BN recalibration. These are diagnostic measurements, not silently revised entries in the original endpoint file.

| Model | Earlier SB | Fresh SB, matched evaluation | Earlier LB | Fresh LB, matched evaluation |
| --- | ---: | ---: | ---: | ---: |
| F1 | 98.61 | 98.52 | 98.20 | 98.20 |
| C1 | 71.00 | 71.24 | 66.78 | 66.68 |
| C2 | 86.07 | 85.98 | 81.33 | 79.65 |

Five of the six test accuracies are within 0.24 percentage points of the earlier figure. C2 large-batch remains 1.68 points lower. Thus the trained models are broadly similar, while the as-generated figures are not comparable without accounting for their different BN rules.

## Training differences that remain

| Model | Earlier epochs | Fresh epochs |
| --- | ---: | ---: |
| F1 SB | 100 | 114 |
| F1 LB | 100 | 100 |
| C1 SB | 165 | 159 |
| C1 LB | 300 | 111 |
| C2 SB | 165 | 102 |
| C2 LB | 180 | 100 |

Both versions use Adam with learning rate 0.001, seed 42, the same nominal batch sizes and the same datasets. They differ in minibatch order, treatment of incomplete batches, stopping rules, and C1 padding. Equal integer seeds do not make different random-number sequences or code paths identical.

The earlier stopping rule evaluated every five epochs, required 30 epochs without sufficient improvement, and also required at least 99% training accuracy and training loss below 0.1. The fresh rule evaluates every epoch, uses 20-epoch patience, and has no accuracy/loss gate. Both have a 100-epoch minimum and a 300-epoch cap. Both retain the final model rather than restoring the minimum-loss model.

The fresh C1 LB run stops at epoch 111 with stored-BN training accuracy 94.27%. Recalibration raises that same model's training accuracy to 99.258%, showing that its low stored-BN accuracy is not solely a failure to fit the examples. Fresh C2 LB stops at epoch 100, although its lowest recorded training loss occurred at epoch 59. Its last 20 epochs fluctuate rather than show a smooth flat trace. The label `converged` therefore needs qualification. Longer training is one plausible contributor to the earlier result, but these runs change several factors at once and do not establish the cause of the remaining accuracy difference.

## Decision

Keep the earlier submission intact. Preserve the fresh run and the diagnostic separately. Before replacing the figure, choose and document a defensible BN evaluation convention, then recompute the paths from the existing saved weights. This inspection does not justify immediately retraining all six models, changing the seed to chase the paper, or clipping the large losses to make the original plot look better.

Evidence: `results/endpoint_metrics.json`, `results/*/history.json`, `results/*/result.json`, `results/analysis/*/path.csv`, `status.json`, and `diagnostics/comparison_20261022/batchnorm_control.json`. The diagnostic script is retained with that JSON. Earlier comparison values come from `../../results_large_batch/seed42_v1/endpoint_metrics.json` and the earlier per-model summaries.
