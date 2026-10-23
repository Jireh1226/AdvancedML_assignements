# Assignment 2

Each paper has its own folder containing its report, executed notebook, original paper, saved results, and run instructions.

## Paper 1: Understanding Deep Learning Requires Rethinking Generalization

- [Report PDF](paper_1_generalization/understanding_deep_learning_requires_rethinking_generalization.pdf).
- [Executed Jupyter notebook](paper_1_generalization/figure1_reproduction.ipynb).
- [Run instructions](paper_1_generalization/README.md).
- [Original paper](paper_1_generalization/259_understanding_deep_learning_re.pdf).
- [Decisions, failed attempts, and corrections](reproduction_changes.md).
- [Saved primary figures and run records](paper_1_generalization/results_figure1_reviewed/).
- [Supplementary threshold diagnostic](paper_1_generalization/report_assets/).

The supplementary threshold diagnostic can run using the included histories without downloading CIFAR-10 or training.

## Paper 2: On Large-Batch Training for Deep Learning

- [Report PDF](paper_2_large_batch/on_large_batch_training_for_deep_learning.pdf).
- [Executed Jupyter notebook](paper_2_large_batch/on_large_batch_training_for_deep_learning.ipynb).
- [Run instructions](paper_2_large_batch/README.md).
- [Original paper](paper_2_large_batch/76_on_large_batch_training_for_de.pdf).
- [Saved Figure 3 paths and run records](paper_2_large_batch/results_large_batch/seed42_v1/).
- [Separate F1 batch-normalisation diagnostic](paper_2_large_batch/results_large_batch/audit_f1_bn/).
- [Phase 3 identity and Phase 4 source record](paper_2_large_batch/phase4_record.json).

Datasets and large checkpoints are excluded from Git. The notebooks retain their executed outputs; fresh training uses each paper's documented data and environment. Paper 2's report panels can be regenerated from the included measurements without training.

The reports identify the saved Phase 3 snapshots and author-source versions used for later comparisons. No author implementation is copied into this repository. The [shared history](reproduction_changes.md) records decisions, failed attempts and corrections.
