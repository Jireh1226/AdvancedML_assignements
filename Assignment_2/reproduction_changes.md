# Paper 1: decisions, failures, corrections, and report review

This publication log contains only the Paper 1 portions of the local Assignment 2 history. Papers 2 and 3 and their histories remain local. The Phase 3 source snapshot was committed locally as `8b28d5b4c9f86b092aead6e1bec1ee5a5a43e533` before the new author-implementation comparison. Exporting only Paper 1 for publication does not change the recorded experiment.

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
