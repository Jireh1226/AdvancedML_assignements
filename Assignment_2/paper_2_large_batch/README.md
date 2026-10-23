# Paper 2: On Large-Batch Training for Deep Learning

Figure 3: F1 on MNIST and C1/C2 on CIFAR-10, with one paired seed (42).

## Submitted results

- [Report PDF](on_large_batch_training_for_deep_learning.pdf).
- [Final executed notebook](reruns/chat_seed42_v2_20261022/on_large_batch_training_chat_seed42_v2_bn_recalibrated.executed.ipynb).
- [Final figure](reruns/chat_seed42_v2_20261022/results_bn_recalibrated/figure3_chat_seed42_v2_bn_recalibrated.pdf).
- [Measured paths and endpoint results](reruns/chat_seed42_v2_20261022/results_bn_recalibrated/).
- [BN diagnostic and correction](reruns/chat_seed42_v2_20261022/bn_correction.md).
- [Original paper](76_on_large_batch_training_for_de.pdf), Figure 3 on page 6.
- [ChatGPT/Astra conversation](https://chatgpt.com/share/6adab3d8-c2fc-83ea-8a39-08fb4b3fff4c).
- [Final artifact record](finalization_record.json).

The report uses the fresh chat-derived six-model run. Its final analysis recalibrates batch-normalisation statistics using training images at every path point, including both endpoints. Trainable weights are unchanged. All 183 measured points are finite; no clipping, smoothing or additional training was used for the correction. Final SB/LB test accuracies are F1 98.52/98.20%, C1 71.24/66.68%, and C2 85.98/79.65%.

Open the executed notebook without rerunning cells to view the saved outputs. It combines the actual executed training cell with the corrected analysis; the analysis runner did not train the six models again. Large checkpoints and datasets are not published.

## Fresh training

The recorded environment was Python 3.10.12, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.2.6 and pandas 2.2.3, with Matplotlib and Jupyter. For a CUDA 12.4-compatible Linux environment:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.2.6 pandas==2.2.3 matplotlib jupyter
mkdir fresh_seed42
cp reruns/chat_seed42_v2_20261022/on_large_batch_training_chat_seed42_v2.ipynb fresh_seed42/
cp reruns/chat_seed42_v2_20261022/on_large_batch_training_chat_seed42_v2_bn_recalibrated.ipynb fresh_seed42/
cd fresh_seed42
export PAPER2_RUN_DIR="$PWD"
jupyter notebook on_large_batch_training_chat_seed42_v2_bn_recalibrated.ipynb
```

Run all cells in order. Keep both input notebooks together: the corrected notebook verifies its training definitions against the original input and hashes its analysis source separately. MNIST and CIFAR-10 download automatically when absent. `ASSIGNMENT2_DATA_DIR` can point to an existing shared data directory.

Training is enabled in the corrected input. A fresh directory avoids trying to reuse published manifests without their excluded checkpoints. The six-model run plus original analysis took about 65 minutes on the recorded GPU; the corrected analysis took another 5 minutes. Runtime depends on hardware and contention. Genuine batches of 6,000 MNIST or 5,000 CIFAR-10 images need substantial GPU memory; smaller accumulated batches change BN behaviour and are not equivalent.

The notebook saves epoch checkpoints. Resume only with the matching source, configuration and checkpoints. Its patience-based stopping rule is a practical rule, not a convergence guarantee. The final analysis uses all training images for BN recalibration with batch size 512 and evaluates in FP32 with batch size 256. This is a disclosed convention, not a claim that the authors used it.

## Regenerate report panels without training

From this paper folder, with NumPy and Matplotlib installed:

```sh
python report_assets/render_panels.py
```

This reads the included corrected path CSVs and writes the F1, C1 and C2 report panels plus their source hashes. No data download or model checkpoint is required.

## Earlier results and source history

The root-level [earlier notebook](on_large_batch_training_for_deep_learning.ipynb), `results_large_batch/seed42_v1/`, and its F1 BN diagnostic are retained as historical results. They no longer supply the report's reproduction panels. Within the rerun folder, `results/` and the original executed notebook retain the uncorrected geometric-BN analysis. Use `results_bn_recalibrated/` for the submitted curves.

The earlier IDE-built run preceded the shared chat. The fresh run uses the chat-generated training code with documented corrections, but its execution and BN diagnosis used Codex/IDE assistance. The authors' code had also already been inspected before this rerun, so the fresh attempt does not restore the assignment's original no-source-access sequence.

`phase4_record.json` identifies the earlier local Phase 3 snapshot `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533` and the static comparison with [author commit 8819876](https://github.com/keskarnitish/large-batch-training/tree/8819876245c0ce390eab1c0de1ba3be787e15058). That identity belongs to the earlier notebook, not the new one. The comparison found differences in C1 stride, C2 dropout, BN evaluation, stopping and path density; the author code was not executed or copied. The final report and `finalization_record.json` distinguish these runs explicitly.
