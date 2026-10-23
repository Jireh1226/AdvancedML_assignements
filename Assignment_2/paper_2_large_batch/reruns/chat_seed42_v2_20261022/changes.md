# Separate GPT-based Paper 2 attempt

## Source and scope

This version starts from the earlier full training notebook pasted from ChatGPT. The newest supplied attachment only reads and plots existing results, so it cannot train new models. The user explicitly approved using the earlier full GPT training version instead. Both pasted sources are preserved alongside `provenance.json`.

The new notebook, checkpoints, histories and figures use this version's own directory. No old trained model or interpolation file is an input. The completed submission remains unchanged; results from this attempt must be reviewed before any replacement.

Corrections and execution use Codex/IDE assistance. This is not represented as an exclusively chat-interface workflow, and the earlier author-code inspection is not erased. No author implementation is consulted or copied during this revision.

## Changes before execution

- Use one seed, 42, for three SB/LB pairs instead of five seeds. This reduces the requested run to six models and does not estimate variation across seeds.
- Remove the unassigned sharpness searches and five-trial summary. Retain measured endpoint values and the three assigned Figure 3 panels.
- Replace the fixed implementation hash with a hash of the actual notebook code cells. The runner checks the complete input notebook against its recorded file hash before execution.
- Reuse the earlier notebook's exact-input caching approach: transform each image once, then batch-index the cached tensors. Retain GPT's DataLoader sampling generator, shuffle order, dropped remainder and saved RNG states. Validate matching batches and exact short-run/resume results before starting.
- Use FP32 evaluation and 61 path points rather than FP64 and 121 points. These are disclosed compute choices, not corrections to paper-specified values. Original float64 interpolation arithmetic is retained before loading into the FP32 model; endpoint checks use FP32-appropriate tolerances.
- Label any 300-epoch budget-exhausted run and its figure as provisional instead of preventing all analysis or calling it converged. The training stopping rule is unchanged.
- Match the paper's blue loss/red accuracy colours, save PNG/PDF plots and endpoint JSON, and keep intermediate progress in a separate execution log.

## GPT choices retained

- F1/C1/C2 definitions, including asymmetric SAME padding for C1. No old C1 checkpoint is loaded into the new architecture.
- Adam learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, no weight decay; genuine small batches of 256 and large batches of 6,000/5,000.
- Identical initial weights within each SB/LB pair, verified by hashes; Xavier initialisation and hidden-layer BN/dropout as in the supplied GPT implementation.
- Full-training evaluation every epoch; minimum 100 epochs, 20-epoch patience, improvement threshold 1e-4 and maximum 300. Unlike the earlier IDE attempt, there is no extra accuracy/loss threshold. Meeting this operational rule is not a mathematical convergence proof.
- Linear interpolation of stored BN means and geometric interpolation of stored variances. No train-only BN recalibration is substituted. Positive variances and preserved endpoints do not establish that these statistics match activations at intermediate weights; this remains a declared assumption.
- Drop incomplete training batches. This was GPT's explicit choice, not a documented requirement of the paper. Evaluation covers all examples.

## Preflight findings

The existing GPU environment lacks pandas. Install the pinned dependency into this run's isolated `.dependencies` directory rather than modifying the shared environment. All software checks and the disposable C2 batch-memory pilot are separate from the six scientific runs.

The first loader check matched batch contents but failed its generator-state assertion. The test used `zip` without fully exhausting the second iterator, so only one sampler performed its final exhaustion step. The check now exhausts both loaders before comparing their states. This corrects the validation harness; it does not change the notebook's sampler or scientific settings.

The corrected preflight passed all architecture/parameter-count checks, cached batch content/order/remainder/RNG comparisons, exact two-epoch training and interrupted/resumed state comparisons, path endpoint identities and positive extrapolated BN variances. Real MNIST/CIFAR-10 sizes and samplewise preprocessing checks passed. A disposable C2 step with 5,000 real CIFAR-10 images completed in 2.11 seconds with 39.56 GiB peak allocated memory on the NVIDIA RTX 6000 Ada GPU. These pilot timings are not full-run duration estimates, and pilot weights are discarded. `validation.json` records the checks. The scientific `results/` directory did not yet exist at the end of validation.

## Launch

The detached six-model run started on GPU 0 at 2026-10-22 21:57:31 America/Chicago (2026-10-23T02:57:31Z), runner PID 2751852. The launch checked that GPU 0 had no existing compute process; GPU 1's other user's process was left alone. Initial monitoring confirmed real F1 small-batch training and checkpoint updates. The saved input notebook is separate from the accumulating `.executed.ipynb`, and progress is in `run.log`, `status.json` and the new per-model histories. Completion is not yet claimed.

No existing report, notebook, figure, checkpoint or repository file was replaced. The pre-existing report, executed notebook and plot files passed their post-save hash checks. A broader local hash pass stalled during a read and was cancelled; it is not represented as a completed full-checkpoint audit.
