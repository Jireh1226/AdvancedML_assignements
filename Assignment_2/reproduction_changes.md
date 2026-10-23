# Paper 1: decisions, failures, corrections, and report review

This log records Paper 1's reproduction and report revisions. The Phase 3 source snapshot was committed locally as `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533` before the new author-implementation comparison. Exporting the saved snapshot for publication did not change the recorded experiment.

## Completion of the existing Zhang experiment

The original 36-run batch subsequently finished all scheduled 64-epoch runs. Each uses all 50,000 training and 10,000 test examples, with 25,024 SGD updates; none is a smoke test. Its code, architectures, seeds, corruptions, and training schedule were left unchanged. The finished notebook validates without errors. Removed only intermediate epoch-log outputs from its two training cells for delivery, retaining final plots, final diagnostics, and an untouched raw executed notebook on Jarvis.

**Important nonconvergence result:** completed execution does not mean every network reached zero training error. Eighteen runs did; eighteen did not. Seven AlexNet settings ended with one training mistake each (p = 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0). All eleven MLP settings retained errors: 15, 43, 102, 157, 244, 293, 387, 451, 506, 538, and 559 for p = 0.0 through 1.0, respectively. These runs remain marked with x symbols in the test-error panel. Their zero-error convergence times are not fabricated. Because the clean MLP itself did not reach zero errors, its normalized convergence-time curve is unavailable. Thus Figure 1(b) is an incomplete reproduction, despite all requested configurations having run.

| Architecture | Clean-label test accuracy | Fully resampled-label test accuracy |
| --- | ---: | ---: |
| Inception | 86.52% | 9.97% |
| AlexNet | 76.52% | 9.94% |
| MLP 1x512 | 50.86% | 9.80% |

The Figure 1(a) curves were inspected against the supplied paper: all five Inception conditions reach near-zero loss, and the random-label condition takes longest. The corruption sweep shows test error approaching 90%, but the nonzero MLP training errors mean those points cannot be described as exactly equal to generalization gap. The AlexNet convergence-time pattern also differs from the paper.

The already-approved 64-epoch budget gives 64 * ceil(50000/128) = 25,024 steps, close to Figure 1(a)'s displayed 25,000-step range. This does not establish that the same budget is sufficient for every model in Figure 1(b/c). Appendix A specifies learning rates and decay but does not establish 64 epochs as a universal convergence budget, and the lecture convergence bounds do not provide such a guarantee either. We did not relax the zero-error definition, select different seeds, or alter the preserved first-paper protocol after seeing the outcomes. The particular cause of its residual errors was not isolated by this run.


## Figure 1(b/c): convergence repair after review

The user correctly identified that finishing 36 scheduled runs was not equivalent to reproducing all three model curves. The earlier completion statement referred to execution, not successful fitting; the original nonconvergence record above is retained. This follow-up resolves the missing curves with real additional training, not a relaxed accuracy threshold or interpolated measurements.

### Diagnosis and unsuccessful attempt

- Inspected the supplied Zhang PDF again: Appendix A specifies SGD momentum 0.9, initial learning rates 0.1/0.01 and decay 0.95 per epoch, but does not specify a universal 64-epoch convergence budget. Figure 1 states that the corruption-sweep models fit perfectly. Table 1 separately reports 99.34% training accuracy for the randomly labelled MLP 1x512, so its rounded table results must not be substituted for the figure's strict zero-error criterion.
- Re-evaluated the actual saved checkpoints and inspected misclassified examples. Their labels/inputs were fixed, and the sampled misclassified examples had nonzero hidden activations; the diagnosis did not show an entirely dead network. No data were dropped, relabelled, or altered.
- Continued the fully randomized MLP from epoch 64 through epoch 128 with the original 0.95 decay in a separate trial. Errors fell from 559 to 337 but remained nonzero; the epoch-128 learning rate was approximately 1.48217e-5. Merely doubling epochs did not solve the problem. Those trial weights were not substituted into the final sweep.
- Geometric-schedule calculation: after the original 64 epochs, the next learning rate is 0.01 * 0.95^64 = 0.000375241392. The sum of all remaining per-epoch learning rates under continued exponential decay is only 0.000375241392 / (1 - 0.95) = 0.00750482785, compared with 0.192495172 during the first 64 epochs. Thus arbitrarily many extra epochs do not restore a substantial update budget. This is a sum of learning rates, not a bound on actual parameter displacement, because gradients and momentum change.
- As discussed in Lecture_4 (PDF pp. 28–37), convergence calculations require objective-specific constants/assumptions unavailable for these networks. The notes do not justify calculating a guaranteed epoch count. The finite geometric-step budget motivated testing a nonvanishing tail; the precise choice below is explicitly ours, not the paper's or lecturer's prescription.

### Adopted continuation and checks

- Preserve all 36 original 64-epoch experiments, their checkpoints, and Figure 1(a) byte-for-byte. Continue only the 18 runs whose final training error was nonzero, in `results_figure1_reviewed/convergence/`. Each new run records the parent checkpoint's SHA256 and retains the original history, optimizer/momentum state, RNG state, fixed corruption labels, batch size 128, architecture, and preprocessing.
- Hold the learning rate at the saved next-epoch value, 0.000375241392, after epoch 64. Stop at the first later full-training evaluation with exactly zero mistakes, subject to a predeclared 1,024-epoch safety cap. Apply the same rule at every corruption probability; runs already fitted at epoch 64 need no continuation. **This learning-rate floor departs from the paper's uninterrupted 0.95 decay. The resulting complete curves are a documented modified-protocol reproduction, not an exact recovery of the authors' experiment.** No settings or checkpoints were selected by held-out test performance or visual resemblance to the paper.
- Test the clean and fully randomized MLP endpoints first. They reached zero errors at total epochs 393 and 458. Reuse those exact completed trials in the sweep rather than retraining or selecting a more attractive seed. No unsuccessful changed protocol replaced a successful original result.
- Cache deterministic standardized label-sweep inputs on the GPU to avoid repeating CPU preprocessing. A full resumed epoch for both MLP and AlexNet produced bitwise-identical weights, exactly matching training/test metrics and CPU/loader RNG states compared with the original loader. This acceleration changes neither examples nor update order. GPU 0 was used; another user's GPU 1 process was left untouched.
- Exact interrupted-versus-uninterrupted continuation passed, including model weights and every minibatch loss. Completed-result reuse does not train, and changed continuation settings are rejected. Checkpoints are atomic and saved at ten-epoch intervals and on stopping; after interruption, at most nine epochs are replayed from the last recorded state.
- Correct panel (b)'s plotting gate: a measured first zero-error epoch remains valid even if the epoch-64 checkpoint later had one error. The prior plot incorrectly hid those known AlexNet first-fit times. All final selected checkpoints now also have zero training error, so panel (c) uses actual fitted-model test errors and requires no missing-point interpolation.

### Final continued epochs

| Label corruption p | AlexNet total epochs | MLP 1x512 total epochs |
| --- | ---: | ---: |
| 0.0 | 64 | 393 |
| 0.1 | 64 | 335 |
| 0.2 | 65 | 293 |
| 0.3 | 64 | 353 |
| 0.4 | 66 | 420 |
| 0.5 | 94 | 332 |
| 0.6 | 66 | 333 |
| 0.7 | 65 | 381 |
| 0.8 | 65 | 549 |
| 0.9 | 64 | 920 |
| 1.0 | 83 | 458 |

All 36 selected experiments now have zero final training errors, with all 33 label-sweep convergence times measured. Each panel (b/c) contains three model curves. None of the continuations hit the 1,024-epoch cap. The MLP slowdown curve need not be monotonic and is not forced to match the paper: its particularly late clean-label fit and the modified schedule affect the normalization. Test accuracy remains a held-out measurement, not a training selection criterion.

| Architecture | Clean-label test accuracy | Fully resampled-label test accuracy |
| --- | ---: | ---: |
| inception | 86.52% | 9.97% |
| alexnet | 76.52% | 9.93% |
| mlp512 | 50.72% | 9.90% |

The regenerated notebook validates and contains completed outputs without intermediate epoch logs. The original notebook/plots and diagnostic logs remain under Jarvis `.execution/figure1_convergence/`; they are not extra deliverable notebooks. Original experiment files and Figure 1(a) were verified unchanged by SHA256.


## Figure 1 timing audit: rejected batch-size replacements

### Diagnosis and controlled trials

The discrepancy is scientific, not a missing line: the existing AlexNet slowdown exceeds Inception's, while the MLP slowdown is below both. Getting all three models to fit with a learning-rate floor did not reproduce the paper's relative optimization dynamics. The current figure is not an exact reproduction.

Rechecked the supplied paper and implementation: input preprocessing, layer widths and parameter counts, ReLU activations, cross-entropy on logits, SGD momentum 0.9, initial learning rates, fixed label resampling, and absence of dropout/weight decay remain as recorded. No definite architecture or label-corruption error was identified. Batch size and initialization are not supplied by the paper; batch size 128 was our assumption. No online answers or author implementations were consulted.

The controlled change was batch size only, restoring the original exponential learning-rate schedule. Batch 128 gives ceil(50000/128) = 391 updates per epoch; batch 32 gives 1,563, about four times as many updates before each 0.95 decay. The independent-gradient approximation in Lecture_6 also predicts about four times the gradient variance. These competing effects motivate an ablation, not a guarantee of faster convergence or a claim that the authors used batch 32. Initial weights, labels, seed 42, input order, architecture, and initial learning rates were matched.

Before training, the acceptance rule required zero training errors with the original decay, a minimum 64-epoch final evaluation, and a 128-epoch safety cap. Clean and fully randomized endpoints screened candidates; a final replacement additionally required every corruption level to pass. Test accuracy and the desired published curve ordering were not acceptance criteria.

| Trial | Clean endpoint | Fully randomized endpoint | Decision |
| --- | --- | --- | --- |
| MLP, batch 32 | First zero errors at epoch 72 | 91 errors after 128 epochs | Rejected as a full-series replacement |
| MLP, batch 16 | Four errors after 128 epochs | Stopped after clean failure; last durable checkpoint at epoch 10 | Rejected; no further batch reduction |
| AlexNet, batch 32 | First zero errors at epoch 22 | First zero errors at epoch 62 | Endpoint screen passed, but full sweep failed at p=0.9 |

The batch-16 random-label run is a partial diagnostic, not a completed result. Its checkpoint remains archived on Jarvis. The full AlexNet sweep was necessary: successful endpoints did not guarantee that intermediate corruption settings would fit.

### Full AlexNet diagnostic sweep

All eleven settings used batch 32, matched initial weights, and learning rate 0.01 * 0.95^(epoch-1), without a floor. The following are diagnostic results, **not selected replacements**:

| Corruption p | First zero-error epoch | Final epoch | Final training errors |
| --- | ---: | ---: | ---: |
| 0.0 | 22 | 64 | 0 |
| 0.1 | 31 | 64 | 0 |
| 0.2 | 46 | 64 | 0 |
| 0.3 | 39 | 64 | 0 |
| 0.4 | 48 | 64 | 0 |
| 0.5 | 51 | 64 | 0 |
| 0.6 | 58 | 64 | 0 |
| 0.7 | 58 | 64 | 0 |
| 0.8 | 60 | 64 | 0 |
| 0.9 | Not reached | 128 | 2 |
| 1.0 | 62 | 64 | 0 |

At p=0.9, two training errors remained after 128 epochs. Its convergence time is unobserved/right-censored; it must not be assigned a fabricated finite timing point. Although the endpoint slowdown 62/22 = 2.818 is closer to the published ordering than the prior AlexNet result, the candidate failed the declared full-series rule. **The entire replacement was rejected.** We did not retain only favorable new points, splice the old p=0.9 result into a different protocol, or lower the threshold to make the replacement pass.

The executed Figure 1 notebook and all its selected runs/plots are unchanged. The original learning-rate-floor continuation remains explicitly disclosed; it fits all 36 selected configurations but leaves the timing mismatch unresolved. Failed diagnostic runs are retained separately on Jarvis under `results_figure1_reviewed/paper_schedule/`, with control logs in `.execution/figure1_schedule/`. Their summaries and the independent checks are copied into `results_figure1_reviewed/audit/review_audit.json`.

### Convergence definition and remaining limitation

For the retained selected trajectories, threshold sensitivity is:

| Model | At least 99% accuracy: clean / random epochs (ratio) | At least 99.9%: clean / random epochs (ratio) | Exactly 100%: clean / random epochs (ratio) |
| --- | --- | --- | --- |
| inception | 12 / 38 (3.167x) | 13 / 38 (2.923x) | 14 / 46 (3.286x) |
| alexnet | 12 / 31 (2.583x) | 14 / 34 (2.429x) | 15 / 83 (5.533x) |
| mlp512 | 25 / 68 (2.720x) | 37 / 201 (5.432x) | 393 / 458 (1.165x) |

The final few mistakes strongly affect exact-zero timing, particularly MLP's clean denominator. The paper does not specify an operational convergence-time threshold, and the assignment asks us to define it. Figure 1's caption says training error is zero, whereas Table 1 separately reports 99.34% for random-label MLP 1x512. That ambiguity is not permission to call 99% perfect fitting. The main figure keeps the original exact-zero criterion; no threshold or seed was chosen after inspection just to move the blue curve. Unknown hyperparameters are possible causes, not established explanations.

All eleven diagnostic AlexNet checkpoints were independently reloaded and their train/test metrics reproduced exactly. Their exponential schedules and first-fit timestamps were checked. Exact interrupted-versus-uninterrupted training, cached reuse, and rejection of incompatible learning-rate settings passed before the sweep. All original 145 artifacts, previous continuation checkpoints, and the previously delivered notebooks/plots were hash-verified unchanged. All three notebooks validate, contain executed cells, and contain no error outputs. No unfinished diagnostic output was added to a submission notebook.

## Paper 1 report: critical review and external-review comparison

Expanded the reproduction section to explain the distinction between completing a baseline, executing all scheduled runs, and actually satisfying the fitting criterion. It now explains why taking 64 epochs from panel (a) was not a convergence guarantee for every architecture, how the decaying learning rate motivated the documented continuation, and how the old final-checkpoint plotting gate could hide an earlier valid first-fit time. These are explanations of the recorded workflow and corrections, not new experiments or newly established causes of every mismatch.

Drafted an independent critical review from the supplied paper and the user's concern about sample size, label corruption, and bias--variance before reading external reviews. The proposed control compares smaller clean datasets with label-corruption sweeps and repeated seeds. The report distinguishes reducing sample size from destroying the image--label relationship; it does not claim their effects are equivalent or that this extra experiment was run. The review also discusses the controlled comparison's strengths, limits of the theoretical interpretation, the practical effect of regularisation, and reproducibility of convergence timings. The two-page problem/methodology summary was preserved.

On 22 October 2026, the live OpenReview forum required browser verification and direct API requests returned HTTP 403. The comparison therefore uses archived original OpenReview records at https://github.com/xcfcode/openreview_spider/blob/b232dbcbbee6a7a39415c2ea046058e9dcd82e1e/res/Sy8gdB9xx.json rather than claiming successful live access. The archive's paper title and forum ID match the supplied paper. Its invitation fields identify the three official reviews (rJUCC0pQg, B1nh-uNVl, HJwljOv4x), which are distinguished from Thomas Dietterich's later public comment (S1zRtYaBG) and the authors' response (HJJR_fQUg). The report links the records and discloses the archive route. The smaller-clean-dataset control was not explicitly requested in those three official reviews.

External-code exposure disclosure: an archived public comment, BkwBqzZ8e, contained a short illustrative nearest-neighbour/random-forest code example. It was encountered while reading the review discussion, after the reproduction runs. It was not used to modify the notebook, settings, or results; no neural-network implementation was consulted in this review comparison. This exposure is disclosed in the report alongside the earlier supplied-chat README exposure and Codex/IDE assistance. This was not a Phase 4 comparison against the authors' implementation, and no such comparison or commit is claimed.

Only the Paper 1 report source, compiled PDF, and this history were updated. No training was started; notebooks, saved results, figures, and the other papers were left unchanged. The exact chat-model version and submission GitHub URL remain pending rather than being invented.

## Paper 1 report: phase-based interaction narrative

Reorganised the reproduction section around the assignment's Phases 1--4 at the user's request. Removed the remote-machine name and hardware narrative from the report. The section now begins with the user's own summary of the paper, follows the requests for a baseline and then the full figure, explains the reviewed code issues and corrections, and relates the final results to the paper's empirical claim. The necessary experimental settings and convergence definition remain, along with the modified-schedule disclosure and unresolved timing mismatch.

Retrieved the user's shared conversation at https://chatgpt.com/share/6ada9094-f644-83e9-86ec-d563eff6b4b3 on 22 October 2026. The browser text tool exposed only its title, but a direct request returned the shared page and its conversation data. The per-message assistant metadata identifies the chat model as `gpt-6-astra-wm`; the report now gives this identifier alongside the user's name, Astra. This resolves the earlier missing model identifier without inferring a version from unrelated page configuration.

The shared conversation confirms that the baseline-only notebook was explicitly requested as the first step. The model later supplied the expanded experiments, and the user acknowledged having inspected the old version. Corrected the report's earlier overly broad attribution of missing conditions: distinguish notebook-version/execution confusion from generated-code weaknesses. The chat also warned that 64 epochs was not a convergence guarantee. The later failure to check fitting across all runs is therefore described as an execution/review failure, rather than falsely attributing that guarantee to the chat model.

Documented the actual revision request: separate plotting from result registration, isolate experiments, and retain complete resume state. Preserved the earlier README-access disclosure and distinguished the chat's small CPU checks from subsequent full training and diagnostics with Codex/IDE assistance. Phase 4 is explicitly marked as not performed; no author-code comparison or commit is invented. The results paragraph states that clean-versus-random-label test performance supports the main empirical contrast, while the exact slowdown ordering remains unmatched. The GitHub submission URL remains pending.

This is a report-only revision. The problem/methodology summary, independent critical review, notebooks, runs, and figures remain unchanged.

## Paper 1 report: plotting-colour mismatch

Added the user's colour-selection observation to Phase 2's review of the LLM output. Checked the executed notebook's `plot_figure1a`: its plotting call does not specify colours and uses Matplotlib's default cycle. The saved panel (a) shows Gaussian in purple rather than the paper's black, random labels in orange rather than red, and random pixels in red rather than magenta. The report identifies this as a presentation mismatch that makes comparison harder without changing the measured values. No recolouring or retraining was performed; the original and reproduced figures, notebook, and results were preserved. Removed one redundant sentence about notebook-version confusion to keep the report concise.

## Paper 1 report: concise account of the reading process

Replaced the detailed Phase 1 retelling with one paragraph describing the user's process: summarise the paper, check alignment with the LLM, request the relevant section when explanations differ, and return to the paper to verify corrections before building the notebook in stages. Moved the explanation of why labels remain fixed across epochs into the experimental-settings paragraph so the assignment's question remains answered. Other report sections, notebooks, plots, and numerical results were preserved.

At the user's request, removed the notebook-version narrative from Phase 2 and shortened its opening transition. The section now starts with the substantive experiment-management weakness and its correction. This editorial change leaves the recorded experiments and technical findings unchanged.

Removed the sentence explaining the assignment's source-access restriction back to the reader. Retained the factual disclosure of the LLM's README access, reported lack of implementation-file access, and framework-documentation consultation.

Added the user's stated motivation for requesting seed 42: considering it a common default and hoping for results closer to the paper. Distinguished this initial expectation from evidence: the paper does not report its seed, and no advantage for matching the paper has been established for 42. Did not assert that it is the most common seed. The single-seed experiment and all saved results remain unchanged.

Removed manual bold emphasis from the Paper 1 report's condition names, learning-rate note, figure labels, and link/disclosure labels. Kept the normal section headings. This formatting-only edit preserves the wording, figures, and experimental results.

Rewrote Phase 2 to focus on unnecessary operating instructions, result-isolation and checkpoint weaknesses, incomplete convergence after 64 epochs, the incorrect first-fit plotting gate, and the colour mismatch. Kept classification errors distinct from code-execution failures and retained the detailed training diagnosis in Phase 3. Moved the factual source-access disclosure to the code/interaction record instead of deleting it. No implementation, measurements, or figures changed.

## Paper 1 report: prose and sentence-flow review

Consulted the user's Wikipedia page, https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing, as a descriptive editing reference rather than an authorship test. Also consulted Harvard College Writing Center's guidance on transitions (https://writingcenter.fas.harvard.edu/transitions) and focused paragraphs (https://writingcenter.fas.harvard.edu/clone-anatomy-body-paragraph), and Monash's scientific-report guidance (https://www.monash.edu/student-academic-success/excel-at-writing/annotated-assessment-samples/science/science-lab-report). The assignment's required phases and the user's first-person voice take precedence over generic lab-report structure or passive-voice conventions.

Edited repetitive introductions, vague links between observations, and unnecessary summaries. Connected training failures to the diagnostic evidence and correction; separated experimental choices from their justification; made review claims explicit before discussing their supporting evidence. Corrected the grammar of the input-condition descriptions without reintroducing bold emphasis. Retained uncertainty where the diagnostics do not establish a cause and retained the seed-choice caveat. The independent critique's arguments were preserved while its wording was revised after the review comparison.

Preserved the equations, numerical results, figure assets, phase structure, model identifier, and disclosures of chat/IDE assistance and external-source access. The pending GitHub entry now states that its URL has not been supplied instead of instructing the reader when to publish it. No experiments or notebooks changed. No claim is made that stylistic editing establishes authorship or guarantees an AI-detector outcome.

Removed the ambiguous commentary about describing a completed schedule as a successful reproduction. Phase 2 now directly states that 18 of 36 runs had nonzero training error at the epoch-64 final checkpoint, without implying that every such run lacked an earlier first-fit time.

Replaced the critique's vague bias--variance connection with the specific question of how limited data and label corruption affect memorisation and test performance. A low-training/high-test-error pattern alone does not identify bias or variance. Kept the proposed smaller-clean-dataset control and the distinction between reducing sample size and corrupting supervision; no new experimental claim was added.

Standardised the report's narrative voice: retained singular first person for the user's prompts, decisions, and critical assessment, and replaced collective first person with neutral descriptions of methods and results. Removed all instances of “we,” “our,” and related collective pronouns, including in captions and the chat-process paragraph. Experimental settings, numerical results, source attribution, and assistance disclosures were unchanged.

Removed the review-comparison preamble about drafting order and browser access, and shortened the assessment heading. The comparison now begins with the reviewers' substantive comments. Retained the archived-record citation and bibliography entry so the source remains identifiable; the independent-review chronology and retrieval route remain documented in this history.

## Paper 1: corrections after the external report audit

Checked the pasted audit against the supplied assignment, original paper, executed notebook, saved histories, and the preserved shared-chat page. The following changes are report and documentation changes, with no retraining or replacement of the primary experimental figures.

- Added Jireh Katebe to the title using the name already present in the course proposal, plus the course and draft date. No submission-repository URL or Git commit was found in the local Assignment 2 directory. The report marks that omission explicitly rather than substituting the original authors' repository. Repository creation/publication and Phase 4 are not claimed complete.
- Made the label-corruption sampling rule and actual-change probability explicit. Distinguished training cross-entropy from the classification-error plot, and noted that corrupted training labels and clean test labels follow different distributions. Added the finite-sample expressivity result and the conditional minimum-norm argument for zero-initialised linear SGD, without claiming that this solves nonlinear-network generalisation.
- Clarified Astra as the model used in ChatGPT and preserved the shared-page model identifier `gpt-6-astra-wm`; did not invent an externally verified product-version name. Grouped the conversation, notebook, and pending GitHub information at the start of reproduction.
- Added a stated/assumed/changed settings table and the three verified parameter counts. Expanded the rejected batch-32 AlexNet diagnostic without adopting its more favourable endpoint or replacing any original point.
- Enlarged the primary reproduced plots on a dedicated page. Kept their saved colours and 100-update averaging unchanged. Added the three-threshold timing table and a separately labelled 99%-accuracy diagnostic curve derived from all 33 selected label-sweep histories.
- Added `report_diagnostics.py`, which validates configurations, epoch counts, steps, first-fit timestamps, original/continuation prefix agreement, and agreement with the existing audit's endpoint timings. It writes the 99 threshold measurements, source-file SHA-256 hashes, one-error intervals, and supplementary plot to `report_assets/`, outside the preserved results directory. It does not import the training notebook, load a model, or train.
- Corrected overstatements in the external audit: the 99% MLP curve is not monotone (2.76 at p=0.9, then 2.72 at p=1), and its ordering against AlexNet varies across the grid. AlexNet p=0.4 first fits at epoch 46; it is not continuously at one error from 40 to 64. The clean MLP's one-error plateau spans 175--392, and AlexNet p=0.5 has one error throughout 42--93. Counts alone do not identify whether the same example is wrong at every evaluation.
- The threshold experiment demonstrates sensitivity of the selected trajectories, not the authors' true stopping rule or the complete cause of cross-implementation differences. The report preserves the original zero-error criterion and discloses that some diagnostic times occur after the learning-rate change. It does not declare 99% a recovered paper setting.
- Expanded source-access disclosure using the decoded shared-page records: both `kobybibas/training_with_random_labels` and `google-research/heldout-influence-estimation` appear as `search_result` entries with snippets. The saved web-tool messages are redacted, so a separate page opening cannot be confirmed or ruled out. No new implementation pages were opened for this verification.
- Specified which work occurred outside the shared chat: GPU execution, constant-rate continuation, batch-size trials, and subsequent diagnostics. Retained the fact that this only partly meets the assignment's chat-interface rule. No retrospective chat entry is treated as a record of the original decision.
- Strengthened the smaller-clean-dataset proposal with a count-matched hypothesis and an explicit warning that chance-correct random labels are not equivalent to useful supervision. Added the alternative unresampled-label count control. Framed Table 1 versus Figure 1 as a conditional reporting inconsistency, since their checkpoint identity is unverified.
- Compared the independent assessment with official reviews through agreement, interpretation differences, and additional points. Included Dietterich's early-stopping argument as a later public comment, distinct from the official reviews and from changing the measured fit threshold. The earlier independent-review chronology remains in this log, not as defensive report prose.
- Added a paper-specific README with environment, data, saved-result inspection, fresh training, diagnostic-generation, report-build, and submission-status instructions. The executed notebook and all original results remain unchanged.

The report structure is the blueprint for the remaining papers: concise mathematical understanding; required reproduction links and model identification; phase-specific failures and corrections; stated/assumed/changed settings; legible original/reproduced figures and numerical evidence; uncertainty and access disclosures; independent assessment followed by substantive review comparison. Scientific details and diagnostics must be specific to each paper.

## Paper 1 Phase 4: saved snapshot and static implementation comparison

The local Phase 3 snapshot was committed as `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533` at 2026-10-22 19:37:27 America/Chicago. New author-source inspection began after that commit, at 2026-10-23 00:37:36 UTC (19:37:36 local). The original commit is retained in the local Git reference `refs/local/assignment2-phase3`. The Paper 1 notebook and diagnostic script in the publication match that snapshot byte-for-byte; `phase3_record.json` records their hashes. The paper-only export commit was created later and is not represented as preceding the comparison.

Compared the author-maintained [demonstration repository at commit 972e71c94442fc99baeb5baebee8c95e5767963b](https://github.com/pluskid/fitting-random-labels/tree/972e71c94442fc99baeb5baebee8c95e5767963b). This was a static reading of five files, not execution or a copied implementation:

| Source | Verified difference or agreement |
| --- | --- |
| [README.md, lines 9–11](https://github.com/pluskid/fitting-random-labels/blob/972e71c94442fc99baeb5baebee8c95e5767963b/README.md#L9) | The author describes the repository as a separately written demonstration, not the original Google-owned implementation or an exact Inception/AlexNet reproduction. Its defaults cannot be retroactively attributed to Figure 1. |
| [cifar10_data.py, lines 27–37](https://github.com/pluskid/fitting-random-labels/blob/972e71c94442fc99baeb5baebee8c95e5767963b/cifar10_data.py#L27) | Labels are uniformly resampled once with a Bernoulli-style mask, agreeing with the fixed-label interpretation. The demo resets NumPy seed to 12345; the original paper's seed remains unknown. |
| [train.py, lines 18–51](https://github.com/pluskid/fitting-random-labels/blob/972e71c94442fc99baeb5baebee8c95e5767963b/train.py#L18) | Uses 32×32 images with fixed channel statistics and passes the corruption probability to both train and validation datasets. The reproduction uses a 28×28 crop, per-image standardisation, and clean test labels for its label sweep. |
| [model_mlp.py, lines 12–25](https://github.com/pluskid/fitting-random-labels/blob/972e71c94442fc99baeb5baebee8c95e5767963b/model_mlp.py#L12) | Demo MLP uses bias-free layers and Gaussian initialisation; the reproduction has biases and Xavier-uniform weights. The 32×32 demo MLP has 1,577,984 weights versus 1,209,866 parameters in the paper-matched 28×28 reconstruction. |
| [cmd_args.py, lines 11–18](https://github.com/pluskid/fitting-random-labels/blob/972e71c94442fc99baeb5baebee8c95e5767963b/cmd_args.py#L11) and [train.py, lines 184–188](https://github.com/pluskid/fitting-random-labels/blob/972e71c94442fc99baeb5baebee8c95e5767963b/train.py#L184) | Defaults are batch 128, 300 epochs, momentum 0.9, weight decay 1e-4, and learning-rate drops by ten at epochs 150 and 225 within the default budget. This differs from the paper schedule and the explicitly modified continuation. |
| [train.py, lines 76–106](https://github.com/pluskid/fitting-random-labels/blob/972e71c94442fc99baeb5baebee8c95e5767963b/train.py#L76) | Cross-entropy and momentum SGD agree in kind. The loop reports full-training-set metrics per epoch by default, but contains no Figure 1 convergence-time sweep; it does not recover the original stopping rule. |

The complete README was read, including its illustrative linear/kernel example. No linked third-party implementation was followed. No author code was copied into the submission, and no demo training was run. No notebook, training implementation, checkpoint, selected trajectory, primary plot, or diagnostic threshold was changed as a result. The comparison changes the report's explanation of source availability and limits, not the experimental evidence. It was performed with IDE assistance and remains outside the original shared chat.

## Paper 1 initial publication

The first upload to `https://github.com/Jireh1226/AdvancedML_assignements.git` added Paper 1 before the final assignment submission, earlier than the publication day specified in the guidelines. Datasets, large checkpoints, and the original mixed-paper Phase 3 commit were excluded from that upload.

Validation for this revision includes recalculating all 99 timing measurements from 33 saved runs, checking those endpoints against the previous audit, validating notebook cell syntax and the absence of saved error outputs, comparing original artifact hashes, and compiling the report twice. The report's summary remains limited to its first two pages. No full notebook training was rerun during report revision or publication preparation.

## Paper 1 second review: stale PDF correction

The second review correctly identified that the distributed PDF still contained the earlier pending-Phase-4 paragraph, even though the adjacent local source contained the completed comparison. Both the assignment-folder PDF and repository PDF had SHA-256 `c83318692c834f4616b63f7eb7d6c03f440e135ac5a2de5c8496a8a53fce2524`. The previous statement that the final report had been compiled and synchronized was inaccurate for the distributed artifact.

Rebuilt the exact current source twice in a fresh output directory, without editing or publishing the LaTeX source. Source SHA-256: `4dac7773a4eb4359e96a10f63ba169db55fc449d6b6084255794441096328fc0`. Corrected PDF SHA-256: `fb8ac17c1f626da849ec691e35b091885aa97a7a8c8269ab82013cd0c12a3bfd`. The report remains ten pages.

Validation now checks the PDF itself rather than inferring its contents from the source: page 7 contains the completed Phase 4 section and commits `8b28d5b` and `972e71c`; page 8 contains the updated interaction record; page 10 contains the third bibliography entry for the released demonstration. The outdated claims that the comparison remains pending and that no Phase 3 commit exists are absent. The rendered Phase 4 page was also inspected. The build has no overfull boxes or unresolved references; the shell-escape-disabled notice concerns an unused image-conversion capability.

Synchronized the local Paper 1 README with the repository's PDF-only version by removing the LaTeX-source description and report-build section. The repository's Assignment 2 README already has no report-source link, so it requires no change.

No notebook, checkpoint, original plot, diagnostic measurement, or other paper report was changed. The first public export was created after author-source inspection; tagging that export would not make it the original pre-inspection Phase 3 commit. The original mixed-paper commit remains separately identified in the provenance record. A visibility change would restrict future access, not undo earlier publication.

On 22 October 2026, the repository was changed to private at the user's request and GitHub's API confirmed private visibility. This limits further public access; the earlier-publication disclosure is retained.


## Paper 2: On Large-Batch Training for Deep Learning

### Paper-specified settings

- F1: MNIST, five 512-unit batch-normalized ReLU hidden layers and ten outputs.
- C1: CIFAR-10, two 64-filter 5x5 stride-2 convolution/max-pool stages, dense 384/192, batch normalization and 0.5 retention dropout in hidden dense layers.
- C2: CIFAR-10, VGG-style convolution groups 64x2, 128x2, 256x3, 512x3, 512x3, pooling between groups, a 512-unit hidden dense layer and ten outputs. The Appendix B.4 reference to C3 is interpreted as a naming typo within the C2/C4 section.
- Adam; small batch 256; large batch 10% of the training set: 6,000 for MNIST and 5,000 for CIFAR-10.
- Mean cross entropy and accuracy along theta(alpha) = (1-alpha)*theta_SB + alpha*theta_LB, alpha in [-1, 2]. Alpha 0 is SB and alpha 1 is LB.

### Explicit implementation choices

- One prespecified seed, 42, and matched initial weights for each SB/LB pair. The paper reports five trials and selects a pair for Figure 3; this reproduction does not estimate between-seed variability or search seeds for an attractive plot.
- Decode pixels to float32 in [0,1]; retain MNIST 28x28 and CIFAR-10 32x32. No centering, per-image standardization, augmentation, or weight decay.
- Xavier-uniform weights, zero biases, BN affine scale one/shift zero, epsilon 1e-5 and momentum 0.1. BN is applied to hidden layers, not output logits. C1 convolution/pooling padding and pooling strides require an explicit choice because the paper does not fully specify them.
- Adam learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, no learning-rate schedule. Evaluate every five epochs; stop after at least 100 epochs when full-training cross entropy has not improved by 0.0001 for 30 epochs, provided training accuracy is at least 99% and cross entropy is below 0.1. A 300-epoch safety cap is an explicit departure from the paper's unbounded convergence budget; capped runs must not be called converged automatically.
- C1 uses convolution padding 2 and 3x3 max pooling with stride 2/padding 1; these unspecified padding choices yield a final 64x2x2 feature map. No change is made simply to match published accuracy.
- The interpolation grid contains 61 evenly spaced alpha values, including exact alpha 0 and 1. BN recalibration uses the whole training split in fixed 512-example batches, with batch-size-weighted running-statistic updates. This is an explicitly chosen calibration convention, not a claim about the authors' unreported handling.
- Interpolate trainable parameters, not negative/extrapolated running variances. Recompute BN statistics from training data only at each interpolation point; disable dropout for loss/accuracy evaluation.

## Completed large-batch results

All six models completed their specified training-only plateau rule: F1 SB/LB at 100/100 epochs, C1 at 165/300, and C2 at 165/180. C1 LB met the plateau rule on the final allowed epoch; it is not an experiment with an unbounded budget. Full training and test splits were evaluated at all 61 interpolation points for each network. The executed notebook passes nbformat validation, matches the validated code, and contains no error outputs or intermediate training logs.

Figure 3 endpoint values below use the same train-only BN recalibration as the plotted curves. They therefore differ slightly from the raw running-BN metrics saved at the end of training.

| Network | SB test accuracy | LB test accuracy | SB minus LB |
| --- | ---: | ---: | ---: |
| F1 | 98.61% | 98.20% | 0.41 percentage points |
| C1 | 71.00% | 66.78% | 4.22 percentage points |
| C2 | 86.07% | 81.33% | 4.74 percentage points |

The completed curves were visually compared with the supplied Figure 3. They show the intended qualitative pattern: a broad low-training-loss region around the small-batch endpoint and greater sensitivity around/beyond the large-batch endpoint. All three SB endpoints have better test accuracy than their LB counterpart. This is not an exact numerical reproduction: the paper's Table 2 averages over five trials and reports F1 98.03%/97.81%, C1 80.04%/77.26%, and C2 89.24%/87.26% for SB/LB. In particular, our C1 and C2 accuracies are lower, and extrapolation heights/transition locations differ. One seed, unspecified optimizer/initialization/padding settings, finite stopping, and our explicit BN calibration convention limit comparison. These differences were not hidden or corrected by selecting a favorable seed, test epoch, or plot scale.

## Paper 2 F1 accuracy rebound: controlled BN comparison

The user requested leaving Papers 1 and 3 untouched and investigating the upward red tail in Paper 2. Red represents accuracy, not cross-entropy. Alpha is a weight-path coordinate, not training time: weights are `(1-alpha)*SB + alpha*LB`; alpha beyond 1 extrapolates past the large-batch model. Classification accuracy is not required to decrease monotonically along this path.

Used the two existing F1 checkpoints without retraining. Kept all trainable parameters identical at each of the 61 alpha values, changing only the batch-normalization running statistics. Evaluated all 60,000 training and 10,000 test examples. The original training-data recalibration reproduced all 61 saved train/test points exactly.

| BN treatment | Test accuracy at alpha 1.3 | Test accuracy at alpha 2 | Interpretation |
| --- | ---: | ---: | --- |
| Recalibration from training images, current notebook | 2.20% | 27.91% | Reproduces the observed rebound |
| Fixed small-batch endpoint statistics | 10.10% | 11.12% | Largely collapses to predicting digit 3; not a faithful replacement |
| Fixed large-batch endpoint statistics | 3.26% | 18.04% | Rebound remains without recalibration |
| Affine interpolation/extrapolation of stored means and variances | Invalid | Invalid | Negative variances; no accuracy assigned |

Thus BN handling materially changes the curve but recalibration alone does not explain the rebound. With fixed LB statistics, accuracy still recovers while test cross-entropy rises from 66.83 at alpha 1.3 to 56,913.09 at alpha 2: improved accuracy need not mean improved probabilistic predictions, because confidently wrong examples can dominate cross-entropy. This control does not reproduce the paper either.

The classwise predictions confirm a real change, not a plotting error. Under the current recalibration protocol, at alpha 2 the test accuracies for digits 0, 3, 4, and 9 are approximately 69.8%, 81.4%, 55.8%, and 56.0%, respectively; several other digits remain almost entirely wrong. That partial recovery produces the aggregate upward red tail. The network is still a poor classifier, not a recovered high-accuracy endpoint.

The proposed direct BN-buffer interpolation cannot cover the full displayed range for these checkpoints: every sampled alpha from 1.05 through 2 contains negative running variances; at alpha 1.3 all 2,560 BN variance entries are negative. These points were rejected, not clamped, silently converted into standard deviations, or plotted as finite results. Fixed-SB and fixed-LB statistics are diagnostic controls only: each preserves its own full endpoint model but substitutes the wrong BN buffers at the opposite endpoint, so neither is automatically a valid replacement for the complete two-endpoint curve.

No primary notebook, checkpoint, or submitted figure was changed. In particular, Papers 1 and 3 remain byte-identical. All three notebook hashes, the protected figures, and all eight classifier/VAE checkpoint hashes passed before/after checks. The exact reason our extrapolated F1 behavior differs from the authors' selected pair remains unresolved; the experiment does not establish that initialization, stopping, or BN convention alone is responsible. No visual-match-driven replacement was selected.

The evaluation loop took about 20 seconds on GPU 0, excluding environment/data setup and file transfer. Diagnostic artifacts are in `results_large_batch/audit_f1_bn/`: `curves.json`, `audit_metadata.json`, `bn_comparison.png`, and `diagnostic.py`. These are separate from the submitted Figure 3 results.

## Paper 2 report: chat review, completed evidence, and Phase 4 comparison

The user requested a complete Paper 2 report using Paper 1's structure. The revision combines the mathematical summary, phase-specific reproduction account, original/reproduced Figure 3 panels, numerical comparisons, diagnostic evidence, an independent critical assessment, and comparison with public reviews. Author and course details follow Paper 1. No experiments were started, no training code was changed, and nothing was published.

### Shared chat and assumptions

- Shared conversation: https://chatgpt.com/share/6adab3d8-c2fc-83ea-8a39-08fb4b3fff4c. The fetched page's conversation default and assistant-message metadata identify `gpt-6-astra-wm` (Astra). The public chat was checked directly; it was not inferred from Paper 1's model.
- Compared the 34-cell chat proposal and the user's pasted design-choice response with the actual executed notebook. The shared conversation records the initial understanding check, request for paper-only code with assumptions explained, complaint about excessive instructions, subsequent design questions, and acknowledgment that no full runs were performed there.
- The completed six seed-42 runs predate this chat proposal. The report explicitly separates them. It does not imply the proposed five-seed, asymmetric-padding, geometric-BN-variance implementation produced the saved outputs.
- The chat's log-space variance interpolation is motivated by positivity, exact endpoint buffers, and convenience, but has not been evaluated on the existing checkpoints. It is distinct from the earlier affine-variance diagnostic that produced negative extrapolated variances. No new geometric-interpolation experiment is claimed.
- Five trials follow the paper; the existing single-seed study is a reduced reproduction. Other differences include the plateau rule, dropping incomplete batches, FP64 evaluation, grid density, out-of-scope sharpness searches, and a literal implementation hash without runtime validation. These are described as assumptions or weaknesses, not automatically labelled scientific bugs.
- Proposed removals and provenance fixes remain proposals for the unexecuted chat notebook. The executed notebook already excludes the extra sharpness suite. Retaining an old result is not evidence that every old convention is uniquely correct.

### Verified local evidence

- All six summaries, training histories, paired initial-state hashes, interpolation measurements, and the prior BN diagnostic were checked. Reported endpoints remain F1 98.61/98.20%, C1 71.00/66.78%, and C2 86.07/81.33% after recalibration. Raw running-BN endpoint values are listed separately.
- C1 SB raw test accuracy peaked at 71.67% at epoch 30 and ended at 69.56%; C1 LB peaked at 68.33% at epoch 30 and ended at 65.67%. These observations do not prove stopping explains the mismatch. No checkpoint was selected by test accuracy.
- All six runs meet their recorded operational plateau-and-fitting rule. C1 LB also reaches the 300-epoch cap, and both facts are reported. The rule is not an assertion of mathematical convergence.
- The calibration description now explicitly states that batch-weighted running-statistic updates with all BN layers active are not exact layerwise population moments or pooled activation variance. Dropout stays off and only training images calibrate the statistics.
- The prior nine-point calibration-batch comparison and 61-point F1 control are reused as existing evidence, not presented as new experiments. The fixed-LB rebound and invalid affine extrapolation remain visible; their cause is not overclaimed.
- `report_assets/render_panels.py` renders all 61 stored points into three report-only PDF panels with readable labels. `panel_sources.json` identifies the exact source JSON hashes. No points are changed, selected away, fitted, smoothed, or newly evaluated. The original combined Figure 3 files remain unchanged.

### Saved snapshot and released implementation

Before source access, the executed Paper 2 notebook was verified byte-identical to `Assignment_2/paper_2_large_batch/on_large_batch_training_for_deep_learning.ipynb` in local commit `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533`, saved at 2026-10-22T19:37:27-05:00 and retained as `refs/local/assignment2-phase3`. Notebook SHA-256: `fb8ce76ef7d4de783eb223b8506a5a8edb110f989bcae3db2488e9fba5e8b7c9` (corrected during the later submission audit; the earlier entry mistakenly reused the older BN-audit hash). The preserved mixed-paper commit was not pushed. No new commit was made for this report task.

Static inspection used https://github.com/keskarnitish/large-batch-training/tree/8819876245c0ce390eab1c0de1ba3be787e15058 on 22 October 2026, after the saved snapshot. Files read: `README.md`, `network_zoo.py`, `plot_parametric_plot.py`, `PyTorch/README.md`, `PyTorch/plot_parametric_pytorch.py`, and `PyTorch/vgg.py`.

- The Keras example covers C1--C4, not F1. It agrees on [0,1] scaling, Adam, SB256/LB5000, and restoring the same initial weights while resetting the optimizer. It uses a fixed 20 epochs and 25 interpolation points, rather than the paper's unbounded loss-based stopping.
- The released shallow convolutions omit `subsample`, whose historical Keras default is (1,1). This differs from Appendix B.3's stride two and the reproduction's stride two; the resulting dense input sizes differ. The released settings are not silently substituted for the paper.
- Released C2 adds convolutional dropout at 0.3 and 0.4 and dropout before its dense layer. These are absent from the paper-based reconstruction. Their effect on the reproduction's accuracy has not been isolated.
- `BatchNormalization(mode=2)` in the released Keras model uses each input batch's statistics in training and testing. Its plotting script evaluates in batches of 5,000, including test batches. This differs from the reproduction's train-only recalibration/frozen evaluation. The distinction is supported by the official historical Keras 1.2.2 `keras/layers/normalization.py` source; `keras/layers/convolutional.py` confirms the stride default. That release is a reference for interpreting the API, not a verified version of the authors' environment.
- The PyTorch README describes a preliminary VGG11-based example, not an exact F1/C1/C2 implementation. Its defaults are not used as recovered paper settings.
- No released code was executed or copied into the notebook. No training/evaluation setting, checkpoint, or original curve was changed because of this comparison. The effect of these discrepancies remains unmeasured.

### Critical review and external discussion

The independent assessment was written before opening the review archive. It concerns empirical strength, causal controls for batch size versus update budgets and BN, one-dimensional geometry, parameterisation dependence, missing implementation details, and the scope of the overfitting explanation. The report presents the assessment before its comparison and does not include process-defence prose.

The original OpenReview page required browser verification and its API could not be fetched through the browser tool. The discussion was read from the archived primary records at https://github.com/xcfcode/openreview_spider/blob/b232dbcbbee6a7a39415c2ea046058e9dcd82e1e/res/H1oyRlYgg.json. Official reviews `S1Eq3CUEl`, `SJVFyZfNe`, and `HkYQOlGEl` are distinguished from public comments. The comparison notes agreement on empirical value, the authors' lack of a second-order optimality guarantee, withdrawal of loss-scale-invariance wording, and unsuccessful reported noise-injection trials. Loss-unit scaling and network reparameterisation are explicitly distinguished. The review comparison does not claim that these two objections are identical.

Only the local Paper 2 report, its report-rendering assets, and this history are updated. The report retains a concise two-page summary and discloses that IDE-generated/executed results do not fully satisfy the chat-interface requirement. Report PDF and LaTeX remain local; this task does not authorize publication of Paper 2 or the upload of LaTeX sources.

The final save follows the user's PDF-only scope: the adjacent pre-existing LaTeX source is left unchanged, and does not contain this report revision. The revised build source is retained only in the temporary report-build directory. Auto-review declined overwriting the assignment-folder LaTeX source; the save was narrowed to the PDF, its report-only rendering assets, and this history. No bypass was attempted. Six endpoint values, epoch counts, paired initialisation hashes, operational stopping conditions, and source-measurement hashes were checked. The final ten-page PDF compiled without layout/reference warnings; its first two pages contain the summary. The title page and comparison figure were inspected visually, and all protected original notebook/result hashes remain unchanged.

## Paper 2 prose review and PDF refresh

The user requested another writing review, particularly the sentence about the stopping numbers remaining practical assumptions. That sentence was removed because the settings table already distinguishes the paper's specifications from implementation choices. The actual stopping thresholds and their limits remain in Phase 3.

Reviewed sentence connections and paragraph focus using Harvard's [Transitions](https://writingcenter.fas.harvard.edu/transitions) and [Anatomy of a Body Paragraph](https://writingcenter.fas.harvard.edu/clone-anatomy-body-paragraph). Used [Wikipedia's Signs of AI Writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) as an editorial checklist, not an authorship detector. Reduced repeated qualifications, generic concluding statements and unnecessary procedural assurances. Phase 2 now focuses more directly on the generated notebook's scope, BN proposal, padding, stopping rules and unchecked implementation hash. The diagnostics distinguish observed effects from unresolved causes without repeatedly restating that distinction. The critique and review comparison have clearer links between claims and evidence.

The revised text retains the unexecuted-chat versus executed-notebook distinction, the chat-interface disclosure, the single-seed limitation, the untested geometric interpolation, and the timing and scope of author-code access. Equations, table data, figure inclusions, bibliography and unique numerical values were checked against the previous source and are unchanged. No collective pronouns or manual bold emphasis were added.

Compiled twice and checked the resulting PDF's text and rendered Phase 2, figure-comparison and critical-review pages. The report remains ten pages, with the summary confined to its first two pages; the final build has no layout or unresolved-reference warnings. All 45 protected files, including the executed notebook, saved results, checkpoints, report assets, adjacent earlier LaTeX source, Paper 1 PDF and Paper 3 notebook, remain unchanged. The Paper 2 notebook still matches the saved Phase 3 commit. No training, checkpoint evaluation, code changes, commits or publication occurred.

Only the assignment-folder Paper 2 PDF and this history are saved. The latest build source is `/private/tmp/paper2-style-review/on_large_batch_training_for_deep_learning.tex`; the adjacent assignment-folder LaTeX remains the earlier draft under the existing PDF-only save scope.

## Paper 2 review: lecture connections and Adam momentum

Added two lecture-grounded points to the critical review at the user's request. Lecture 4, pages 19–27, supplies the Lipschitz-gradient condition; Lecture 6.1, pages 9–11, connects the Hessian to directional curvature. The report explains that a gradient-Lipschitz bound limits the Hessian's operator norm where the loss is twice differentiable. A one-dimensional path cannot determine that bound over a neighbourhood. The discussion is conditional on smooth regions of the ReLU loss and makes no claim that smoothness alone establishes generalisation or that a Lipschitz constant was measured.

Lecture 6.1, pages 5–7 and 23–24, supports the distinction between momentum and Adam's adaptive scaling. Momentum can maintain movement through a small-gradient region without guaranteeing escape from a sharp minimum. The zero-gradient, zero-first-moment example follows directly from Adam's update. The algorithm was also checked against Kingma and Ba's original [Adam paper](https://arxiv.org/pdf/1412.6980), Algorithm 1. The supplied Keskar paper confirms that both batch regimes already use Adam and reports similar large-batch findings with other optimisers. The review therefore proposes a controlled momentum-coefficient comparison, not adding an allegedly missing momentum mechanism. That comparison is a proposed experiment and was not run.

Updated the lecture bibliography entry with the relevant pages. The summary and reproduction sections, tables and figures are unchanged. Recompiled twice, checked the PDF text and rendered review page, and verified that the report remains ten pages with a two-page summary and no build warnings. All 45 protected files remain unchanged. Only the report PDF and this change log are saved; there are no new numerical experiments, notebook edits, commits or uploads. The current build source is `/private/tmp/paper2-lecture-review/on_large_batch_training_for_deep_learning.tex`; the adjacent earlier LaTeX source remains unchanged under the PDF-only save scope.

## Paper 2 review: initialisation sensitivity

Added the requested initialisation discussion to Section 3.1. It acknowledges the paper's five random starting points, its brief statement about other initialisation strategies, and Figure 5's improvements when large-batch training starts from later small-batch checkpoints. It distinguishes the lower mean large-batch accuracy across Table 2's six networks from an unsupported claim that every individual trial was worse. The proposed extension compares named initialisation schemes across paired seeds, separating random-start sensitivity from the benefit of preliminary small-batch training. Removed the earlier short warm-start sentence to avoid repeating the same point.

Only the opening of the critical assessment changes; the summary, reproduction and remaining review are unchanged. The rebuilt PDF remains ten pages with a two-page summary and no build warnings. Checked its extracted text and the rendered opening review page. No experiments, notebook edits or publication were performed. The current build source is `/private/tmp/paper2-initialisation-review/on_large_batch_training_for_deep_learning.tex`; only the assignment PDF and this history are saved.

## Papers 1 and 2: author-repository findings linked to results and review

The user requested clearer discussion of the authors' released repositories and what those implementations suggest for the reproduction. Both reports already contained Phase 4 comparisons. This revision rechecked the same pinned sources and integrated their implications into the results and critical assessments. This is a further inspection, not a new claim that author-code access first occurred now. Existing chronology and chat-interface disclosures remain intact.

### Rechecked sources

- Paper 1: [pluskid/fitting-random-labels](https://github.com/pluskid/fitting-random-labels/tree/972e71c94442fc99baeb5baebee8c95e5767963b), commit `972e71c94442fc99baeb5baebee8c95e5767963b`. Read `README.md`, `cifar10_data.py`, `cmd_args.py`, `model_mlp.py` and `train.py`.
- Paper 2: [keskarnitish/large-batch-training](https://github.com/keskarnitish/large-batch-training/tree/8819876245c0ce390eab1c0de1ba3be787e15058), commit `8819876245c0ce390eab1c0de1ba3be787e15058`. Read `README.md`, `plot_parametric_plot.py`, `network_zoo.py` and `PyTorch/README.md`.
- Historical API checks: official Keras tag `1.2.2`, `keras/layers/normalization.py` and the `Convolution2D` constructor in `keras/layers/convolutional.py`. This interprets the released API; it does not establish the authors' exact installed version.
- Browser requests for the pinned repository pages failed; the GitHub connector successfully returned the files at the pinned refs. No repository code was executed or copied into either reproduction notebook.

### Report changes

Paper 1 now explains that the author repository is an independently written demonstration, whose README explicitly excludes exact reproduction of the original Inception/AlexNet experiments. Its printed random-label MLP example still reports 99.97% training accuracy after 300 epochs. The report treats this as a separate example of near-complete fitting, not evidence explaining the saved timing ratios. The Phase 4 section explains why copying the demo's corrupted validation targets would change the meaning of panel (c). It proposes a controlled Gaussian-versus-uniform initialisation comparison at matched variance, keeping other settings fixed across seeds. The critical review identifies original run configurations and histories as the evidence needed to resolve the timing definition. The demo's stated purpose and recommendations are distinguished from inferred experimental motivations.

Paper 2 now links the result mismatches to the specific BN, C1-stride and C2-dropout differences already established from source. It explicitly notes that the repository has no F1 implementation from which to recover that panel's evaluation convention. The proposed first control keeps C1/C2 checkpoint weights fixed and evaluates using input-batch BN statistics, 5,000-image batches and dropout disabled, with historical normalisation details checked. Architecture and training-dropout changes require separate paired training runs. The review explains that evaluation-time batch statistics make predictions and path measurements depend on the evaluation batch's composition. These are documented discrepancies and proposed tests, not demonstrated causes or completed corrections.

### Validation and scope

Both final PDFs were compiled twice, their extracted text was checked, and the Phase 4 pages and Paper 2 reference page were inspected visually. Both remain ten pages, with their original two-page summaries. Equations, table data, measured figures and bibliography entries are unchanged; Paper 2 references use a compact 10-point font to avoid a nearly empty final page. Removed a duplicated no-new-training sentence in Paper 1 to avoid an isolated line on its own page. There are no layout or unresolved-reference warnings. All 253 protected source, notebook, figure, history and measurement files remain unchanged. No training, checkpoint evaluation, commit, push or repository-visibility change occurred.

Only the two assignment-folder report PDFs and this history are saved. The adjacent LaTeX sources are left unchanged under the PDF-only save scope. Current build sources are `/private/tmp/papers12-repository-review/paper1/understanding_deep_learning_requires_rethinking_generalization.tex` and `/private/tmp/papers12-repository-review/paper2/on_large_batch_training_for_deep_learning.tex`.

PDF SHA-256 values: Paper 1 `30ebb9b963d0fc8fa0e17967e3a4eb0d16a9a893c1a5f9bda016fcc9a9068459`; Paper 2 `3e6684063da30c1e2d82f8819d4e88b4b32aa209aaedafbcffb652266e87170a`.

## Paper 2: saved checkpoint and calibration audit

- Checked the supplied Appendix B against the layers, actual SB/LB batch sizes, Adam objective, dropout/BN treatment, and matched initializations. Padding, initialization, stopping, and BN calibration remain disclosed choices where the paper is incomplete; this does not establish that these choices match the authors'.
- Reloaded all six checkpoints and recomputed interpolation at alpha = 0, 1, and 1.2 for all three architectures. All saved training/test losses and accuracies matched exactly. At alpha 0 and 1, trainable parameters exactly matched the endpoint checkpoint tensors. All six runs met the recorded training-only plateau rule.
- Recalibrating BN with training batches of 256 rather than 512 changed test accuracy by at most 0.02 percentage points and cross-entropy by at most 0.00959 across those nine points. This specific choice cannot explain C1's large accuracy gap; the test does not exclude every possible BN convention.
- Numeric curve shapes, extrapolation losses, and CIFAR test accuracies differ from the paper. Our SB/LB accuracies are F1 98.61%/98.20%, C1 71.00%/66.78%, and C2 86.07%/81.33%; the paper reports five-trial table averages of 98.03%/97.81%, 80.04%/77.26%, and 89.24%/87.26%, respectively. Broad SB behavior and greater sensitivity near/beyond LB are qualitatively present, not exact matches.
- The cause of the remaining performance gaps is unresolved. A single seed and unreported implementation/training choices are limitations, not proven causes. No checkpoint or plotted value was changed merely to hide the mismatch.

## Papers 1 and 2: submission audit and repository update

The user authorised pushing the missing submission files and addressing the verifiable audit findings. Paper 2 now has a README, executed notebook, report PDF, original paper, saved measurements, report panels, diagnostic and Phase 4 record in the repository export. The assignment index links both reports and run instructions. Paper 1's latest report PDF includes the previously completed author-repository discussion. Datasets, checkpoints and LaTeX sources are excluded.

### Report corrections

- State the actual chronology: the six Codex/IDE-assisted runs preceded the later shared chat on 22 October. The chat proposed a different notebook that was never executed. This does not meet the chat-interface requirement; changing the report cannot retroactively repair that workflow.
- Add the paper's sharpness equation with its search matrix, perturbation box and pseudoinverse. The simpler equation suggested by the audit omitted the subspace matrix, so the supplied paper's equations (3)--(4) were checked directly.
- Shorten repeated seed, stopping and workflow qualifications while preserving settings and limitations. The summary remains two pages.
- Identify the Appendix B.4 C3/C2 naming error using the section heading, Table 1 and B.3. The review still distinguishes implementation uncertainty from measured results.
- Correct the audit's suggested OpenReview comparison: Reviewer 3 explicitly questioned minima along one-dimensional slices in note `ByV60KkQg`. The report acknowledges this overlap rather than claiming that no reviewer raised it.
- Do not adopt the claim that fewer processors than batch examples is inherently inconsistent with ideal parallel efficiency. That public comment questions a modelling assumption; it does not establish the claimed contradiction. The later Figure 5 epsilon comment is not needed to support the independently checked B.4 error.

### Source and notebook provenance

The executed Paper 2 notebook is byte-identical to the original local snapshot `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533`. Its actual SHA-256 is `fb8ce76ef7d4de783eb223b8506a5a8edb110f989bcae3db2488e9fba5e8b7c9`. The older `68a06d56...` value was incorrectly reused in the Phase 3 history entry and is corrected there. Historical BN-audit metadata remains unchanged; its hashes describe an earlier audit, not the later snapshot.

`phase4_record.json` lists the six author files at commit `8819876245c0ce390eab1c0de1ba3be787e15058`, the two historical Keras 1.2.2 files, their Git blob identities and source URLs. A new static recheck began at `2026-10-23T02:05:27Z`; it confirms the stride, dropout, BN, 20-epoch and 25-point findings. No author code was executed. The exact initial inspection timestamp could not be recovered and is explicitly null. The historical change log records inspection after the snapshot, but the new recheck timestamp is not presented as proof of the initial access time.

The mixed-paper local Phase 3 commit is not exported or relabelled as a public pre-inspection commit. The repository remains private until the submission-day visibility change; this does not undo its earlier public period. New measurements, retraining and changes to the executed notebooks are outside this update.


### Validation of the submitted artifacts

The Paper 2 PDF was rebuilt twice and inspected, including its equation, comparison figure, Phase 4 text and references. An initial eleven-page build was reduced to ten pages by shortening repeated prose; the summary remains two pages. No layout or unresolved-reference warnings remain. The original panels remain vector crops for zooming, and the README links full-size original and reproduced figures.

The executed notebook passed schema and syntax validation, has six executed code cells and no saved errors, and matches the original Phase 3 snapshot byte-for-byte. Checked all 183 interpolation rows, six endpoint measurements and epoch counts, report-panel input hashes, README relative links and the excluded-file list. Regenerating the three report panels from the exported JSON passed without datasets or checkpoints. All 321 protected original files remain unchanged. A pronoun check initially matched hidden text from the cropped original paper; checking the authored report source avoided attributing the paper's prose to this report.

Final report SHA-256 values: Paper 1 `30ebb9b963d0fc8fa0e17967e3a4eb0d16a9a893c1a5f9bda016fcc9a9068459`; Paper 2 `5fea3ef08e3e8871286b73ad2a64a5da3483ce07582ab58d42111e07a48a6c96`. The adjacent earlier LaTeX sources are unchanged under the PDF-only save scope.
