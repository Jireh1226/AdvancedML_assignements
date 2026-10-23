# Paper 2: On Large-Batch Training for Deep Learning

Reproduction of Figure 3 for F1 on MNIST and C1/C2 on CIFAR-10, using one paired seed (42).

- [Report PDF](on_large_batch_training_for_deep_learning.pdf).
- [Executed notebook](on_large_batch_training_for_deep_learning.ipynb).
- [Original paper](76_on_large_batch_training_for_de.pdf) (Figure 3 is on page 6).
- [ChatGPT/Astra design discussion](https://chatgpt.com/share/6adab3d8-c2fc-83ea-8a39-08fb4b3fff4c).
- [Measured paths and training records](results_large_batch/seed42_v1/).
- [Separate F1 batch-normalisation diagnostic](results_large_batch/audit_f1_bn/).
- [Decisions, failures and corrections](../reproduction_changes.md).
- [Phase 3 identity and Phase 4 source record](phase4_record.json).

## View the completed results

Open the notebook without executing cells, or view the [combined figure](results_large_batch/seed42_v1/figure3_f1_c1_c2.pdf). The three interpolation JSON files contain all 61 measured points per network. `endpoint_metrics.json` and each run's `config.json`, `history.json` and `summary.json` contain the settings and results. Full-size report panels are available for [F1](report_assets/F1_panel.pdf), [C1](report_assets/C1_panel.pdf) and [C2](report_assets/C2_panel.pdf).

Datasets and large `checkpoint.pt` files are excluded from Git. The interpolation cells check checkpoint hashes even when saved curves exist; running all cells against the supplied result directory therefore requires the original checkpoints. Inspecting saved outputs does not require them.

## Environment and fresh training

The recorded CUDA runs used Python 3.10.12, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 and NumPy 2.2.6, with Matplotlib and IPython/Jupyter. For a CUDA 12.4-compatible Linux environment:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.2.6 matplotlib jupyter
jupyter notebook
```

Launch Jupyter from this paper folder, or set `ASSIGNMENT2_DIR` to this folder or the assignment root. Creating `../data/` allows shared downloads. The notebook downloads MNIST and CIFAR-10 through torchvision when absent; the train/test sets contain 60,000/10,000 and 50,000/10,000 images respectively.

In a working copy of the notebook, set `RESULTS_DIR` to a fresh directory, for example `PROJECT_DIR / "results_large_batch" / "seed42_fresh_v1"`, then run all cells in order. This trains six models and evaluates their interpolation paths. Allow several hours and sufficient GPU memory for genuine batches of 6,000 MNIST or 5,000 CIFAR-10 images. CPU execution is supported but substantially slower. Gradient accumulation over smaller batches is not equivalent because it changes BN statistics.

The training loop saves checkpoints and skips completed runs. Resuming an interrupted run requires its matching checkpoint and recorded framework/device environment. Do not mix checkpoints and interpolation JSON from different runs. The report documents the optimiser, stopping rule and train-only BN recalibration; these are unchanged in the supplied notebook.

## Regenerate report panels without training

From this folder, with Matplotlib installed:

```sh
python report_assets/render_panels.py
```

This reads the included interpolation JSON and rewrites only the three report panels and `panel_sources.json`. It performs no training or model evaluation, and applies no smoothing. The separate BN diagnostic is the preserved audit script: rerunning it requires CUDA, trained checkpoints, and the sibling-paper notebooks/figures checked by its protection list. None is needed to view its saved results.

## Workflow and source comparison

The executed notebook and six runs preceded the shared chat on 22 October 2026. The chat reviewed a later notebook proposal that was never executed. The measured reproduction used Codex/IDE assistance and does not meet the assignment's chat-interface requirement.

The submitted notebook is byte-identical to the local Phase 3 snapshot `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533`, committed at `2026-10-22T19:37:27-05:00`. Its SHA-256 is `fb8ce76ef7d4de783eb223b8506a5a8edb110f989bcae3db2488e9fba5e8b7c9`. The original snapshot is retained locally; the later repository export is not presented as a pre-inspection commit.

Phase 4 is a static comparison with the [authors' released code at commit 8819876](https://github.com/keskarnitish/large-batch-training/tree/8819876245c0ce390eab1c0de1ba3be787e15058), interpreted using historical Keras 1.2.2 source. It identifies differences in C1 stride, C2 dropout, BN evaluation, stopping and path density. The authors' code was neither run nor copied into this notebook. Its exact historical environment remains unknown, and the effects of those differences have not been isolated experimentally.

`phase4_record.json` records source URLs and Git blob identities, the notebook's saved identity, and a timestamped recheck. The change log places the original inspection after the snapshot, but no exact initial inspection timestamp was preserved; the recheck time does not substitute for it. The older BN diagnostic's `unchanged_files` hashes describe that earlier audit, not the later Phase 3 snapshot.
