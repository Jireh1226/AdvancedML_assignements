# Paper 1: Understanding Deep Learning Requires Rethinking Generalization

## Files

- `figure1_reproduction.ipynb`: executed PyTorch notebook for all three panels of Figure 1.
- `understanding_deep_learning_requires_rethinking_generalization.pdf`: report.
- `results_figure1_reviewed/full/`: original 64-epoch runs, their histories, checkpoints, summaries, and primary figures.
- `results_figure1_reviewed/convergence/`: selected AlexNet/MLP continuations at the disclosed learning-rate floor.
- `results_figure1_reviewed/audit/review_audit.json`: summaries of rejected smaller-batch trials and previous checks.
- `report_diagnostics.py` and `report_assets/`: read-only analysis of saved runs, the full three-threshold timing grid, source hashes, and the supplementary figure.
- `../reproduction_changes.md`: decisions, unsuccessful trials, corrections, and source-access disclosures.

## Environment and data

The recorded training environment was Python 3.10.12, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, and NumPy 2.2.6. Matplotlib is needed for plots; Jupyter is needed to run the notebook. Use an isolated Python environment. For a compatible NVIDIA/CUDA setup, the recorded package versions can be installed with:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.2.6 matplotlib jupyter
```

Open Jupyter from this paper directory. Alternatively set `ASSIGNMENT2_DIR` to this directory or the assignment root. The notebook finds the shared `../data/cifar-10-batches-py/` dataset. If CIFAR-10 is absent, set `ALLOW_DOWNLOAD = True` in the settings cell to download it through torchvision. Downloads are disabled by default. Keep `SMOKE_TEST = False` for the reported full-data experiment.

## Inspect or replot the saved results

The submitted notebook already contains its outputs. To recompute its tables and plots without training, edit only these settings in a working copy:

```python
LOAD_DATA = False
RUN_FIGURE1A = False
RUN_SWEEP = False
```

Run the cells in order. Saved histories and summaries must remain in both `full/` and `convergence/`. The notebook also checks the original parent checkpoint hashes when selecting continuations, so this route requires the parent `checkpoint.pt` files. It does not require the CIFAR-10 download or CUDA to inspect saved results.

The supplementary diagnostic requires only Python, Matplotlib, histories, summaries, and the existing audit JSON; no datasets, PyTorch, or checkpoints are needed:

```sh
python report_diagnostics.py
```

This verifies all 33 label-sweep runs, recalculates first fitting times at 99%, 99.9%, and 100% accuracy, and writes `report_assets/threshold_sensitivity.{json,csv,pdf,png}`. The main figure keeps the original zero-error criterion. The relaxed thresholds are sensitivity checks, not recovered author settings.

## Train a fresh reproduction

Work in a separate copy of the notebook. Retain the reported seed, architecture, data, optimiser, and continuation settings. Use a new `BATCH_NAME`, such as `seed42_fresh_v1`, to avoid mixing new and saved runs. Set:

```python
LOAD_DATA = True
RUN_FIGURE1A = True
RUN_SWEEP = True
SMOKE_TEST = False
```

Run all cells in order. The five Inception conditions and three-model corruption sweep share endpoints, giving 36 unique configurations. All start with 64 epochs. AlexNet/MLP configurations with residual errors then continue at `0.01 * 0.95**64` until zero training error or the 1,024-epoch cap. This continuation changes the authors' exponential schedule and must remain disclosed. The run can take hours; do not treat a finished schedule as evidence that every configuration fitted.

Completed runs reload rather than retrain. Interruptions resume from saved epoch boundaries. Exact continuation checks require the recorded framework versions and device; do not disable those checks to resume a checkpoint in a different environment. A fresh run on another environment can differ even with seed 42.

The diagnostic script defaults to the preserved `seed42_full_v1` experiment. A fresh batch does not silently replace the submitted results.

## Reproduction record

The submission repository is https://github.com/Jireh1226/AdvancedML_assignements, under `Assignment_2/paper_1_generalization/`.

Phase 3 was saved locally as commit `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533` before the author-source comparison. `phase3_record.json` records its timestamp and Paper 1 code hashes. The publication snapshot was exported after that comparison.

Phase 4 statically compared the released demonstration at `pluskid/fitting-random-labels` commit `972e71c94442fc99baeb5baebee8c95e5767963b`. That repository explicitly does not reproduce the original Figure 1 experiments. Its differences are recorded in the report and history. No author code was copied, no demo was run, and no training implementation or measured result was changed.

The submission includes the notebook, report, run instructions, history, and diagnostic source. Saved histories and summaries allow the standalone diagnostic to run without retraining. Large checkpoints and raw datasets are excluded from Git; notebook cells that verify parent checkpoint hashes require a local copy of those checkpoints or a fresh training run. The notebook's existing outputs can be viewed without executing those cells. No external checkpoint download location is claimed.

The report discloses that execution, continuation decisions, and diagnostic trials used IDE assistance outside the shared chat. This record must not be represented as a wholly chat-only reproduction.
