# Paper 2: BN correction using the completed chat-based models

## Completed checks

The corrected analysis finished in 306.93 seconds (5.12 minutes), evaluating 61 points for each of F1, C1 and C2. All 183 points are finite. The 12 previously measured diagnostic points match the new complete curves. Independent recalibrated endpoint checks passed, as did all trainable-parameter and original-artifact hash checks. The executed notebook contains the final table and figure and has no error outputs.

## Change

The corrected analysis uses the earlier notebook's training-only batch-normalisation recalibration at each point on the parameter path. It replaces the GPT notebook's linear mean/geometric variance interpolation for this separate analysis version. It does not change the model architectures, learned weights, optimizer settings, training seeds, data order or stopping decisions.

The correction follows a controlled diagnostic: at identical trainable weights, replacing only the BN statistics removed the extreme losses that dominated the new plot. For C2 at alpha = 2, the measured training loss changed from 8.79e29 to 9.294. No clipping, axis truncation, seed search or extra training was used to make the figure resemble the paper.

## Evaluation convention

For each of the 61 alpha values from -1 to 2, trainable parameters follow the original affine path. BN running statistics are reset and recomputed from all training images in batches of 512, with sample-weighted accumulation. Dropout is disabled. Gradients are disabled, and there are no optimizer updates. Trainable parameter hashes are checked before and after every recalibration. Full-data training and test measurements use the original FP32 evaluation code and batches of 256.

This convention also recalibrates the alpha = 0 and alpha = 1 endpoints. Their trainable parameters are exactly those of the saved small-batch and large-batch models, but their BN buffers are newly estimated. Endpoint measurements are therefore explicitly recalibrated measurements, not the unchanged checkpoint's original evaluation. Independent recalibrated endpoint checks are performed for each model pair. This is a disclosed evaluation convention, not a claim that the paper specifies this exact BN treatment.

## Preserved and new files

- The original training input, executed notebook, six checkpoints, endpoint metrics and geometric-BN figure remain unchanged.
- `on_large_batch_training_chat_seed42_v2_bn_recalibrated.ipynb` is the separate corrected input.
- `on_large_batch_training_chat_seed42_v2_bn_recalibrated.executed.ipynb` retains the original executed training cell and the newly executed corrected analysis. The correction runner skips the training cell; it does not claim a second training execution.
- `results_bn_recalibrated/` holds the separate path CSVs, manifests, endpoint JSON and PNG/PDF figure.
- `bn_recalibration_status.json` records correction status, elapsed time and hashes of the preserved artifacts.

The training definitions and experiment plan in the corrected notebook are checked against the archived training input. The original training-source hash identifies the code that produced the saved models. A separate hash of the corrected notebook's code identifies the new analysis. The original training notebook remains alongside the corrected copy as its provenance source.

The fresh run's stopping rule is unchanged. Its `converged` field denotes its patience rule, not a mathematical convergence guarantee. Remaining differences from the earlier run, including C2 large-batch's lower accuracy, are retained. The final submission now uses this corrected notebook and these measurements in the report. The earlier notebook and results remain preserved as historical records.
