# Assignment 2

Each paper has a separate report, executed notebook, original paper, saved measurements and run instructions. Start with the links below; earlier attempts are labelled in each paper's README.

## Paper 1: Understanding Deep Learning Requires Rethinking Generalization

- [Report PDF](paper_1_generalization/understanding_deep_learning_requires_rethinking_generalization.pdf).
- [Executed notebook](paper_1_generalization/figure1_reproduction.ipynb).
- [Run instructions](paper_1_generalization/README.md) and [original paper](paper_1_generalization/259_understanding_deep_learning_re.pdf).
- [Saved figures and histories](paper_1_generalization/results_figure1_reviewed/).
- [Supplementary threshold diagnostic](paper_1_generalization/report_assets/).

## Paper 2: On Large-Batch Training for Deep Learning

- [Report PDF](paper_2_large_batch/on_large_batch_training_for_deep_learning.pdf).
- [Final executed notebook](paper_2_large_batch/reruns/chat_seed42_v2_20261022/on_large_batch_training_chat_seed42_v2_bn_recalibrated.executed.ipynb).
- [Run instructions](paper_2_large_batch/README.md) and [original paper](paper_2_large_batch/76_on_large_batch_training_for_de.pdf).
- [Final corrected Figure 3 and measurements](paper_2_large_batch/reruns/chat_seed42_v2_20261022/results_bn_recalibrated/).
- [BN correction](paper_2_large_batch/reruns/chat_seed42_v2_20261022/bn_correction.md) and [final artifact record](paper_2_large_batch/finalization_record.json).

The submitted report uses the fresh chat-derived six-model run with training-only BN recalibration. Earlier notebooks and uncorrected plots remain available as historical records, not the selected result.

## Paper 3: beta-VAE

- [Report PDF](paper_3_beta_vae/beta_vae_learning_basic_visual_concepts_report.pdf).
- [Executed notebook](paper_3_beta_vae/beta_vae_learning_basic_visual_concepts.ipynb).
- [Run instructions](paper_3_beta_vae/README.md) and [original paper](paper_3_beta_vae/291_beta_vae_learning_basic_visual.pdf).
- [Completed Figure 7A/B and measurements](paper_3_beta_vae/results_beta_vae/seed42_v1/).
- [Final artifact record](paper_3_beta_vae/finalization_record.json) and [Phase 4 source check](paper_3_beta_vae/phase4_record.json).

The completed earlier pair supplies all of Paper 3's results. The report identifies its IDE-assisted origin and the later chat's role; no later run is mixed into the comparison.

Datasets, large checkpoints and LaTeX sources are excluded from Git. Reports are supplied as PDFs. Saved figures can be redrawn without training using the commands in each paper's README. Fresh training requires the documented data and environment. The [change log](reproduction_changes.md) records decisions, failed attempts and corrections, with the final selection recorded at the end.
