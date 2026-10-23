# Later chat-derived VAE run

[Minimal executed notebook](beta_vae_learning_basic_visual_concepts_chat_seed42_v2_minimal.executed.ipynb) · [Figure 7A/B](results/figure7_AB.pdf) · [Comparison record](comparison_to_submitted/comparison.json)

Both models completed 100 epochs. The notebook has one title and six code cells, matching the submitted notebook's minimal layout. Code, execution counts and final outputs are unchanged from the completed run; only explanatory Markdown was removed. `presentation_record.json` records the source and export hashes.

This is a separate later result. The submitted notebook, report and ZIP still use the earlier completed pair.

| Final measurement, nats/image | Earlier β=4 | New β=4 | Earlier β=1 | New β=1 |
|---|---:|---:|---:|---:|
| Posterior-mean reconstruction | 45.145 | 42.427 | 33.456 | 31.186 |
| Mean KL | 11.900 | 12.190 | 17.606 | 19.093 |

Both figures show more selective factors for β=4 and more mixing for β=1. The new β=1 model has six low-variance coordinates rather than five. Coordinate identities and signs differ, and the reference oval uses scale index 2 rather than 3. The results are qualitatively similar, not identical. Posterior-mean reconstruction is not the expected reconstruction term in the ELBO; one seed does not establish statistical equivalence.

Open the notebook without rerunning cells to inspect its output. For fresh training, copy it into an empty folder and use the environment described in the [paper README](../../README.md). It downloads dSprites when absent; `ASSIGNMENT2_DATA` can point to a directory containing the archive. Checkpoints are excluded from Git. Run in a fresh directory to avoid overwriting these saved measurements.

The [shared chat](https://chatgpt.com/share/6adadb56-35dc-83e9-9c94-7e1077a90301) supplied the notebook. The Adagrad accumulator correction and execution used IDE assistance. The training source, seed, optimiser settings and checkpoints were not changed for this presentation export. The separate reconstruction/KL evaluation performed no optimiser updates and verified unchanged checkpoint hashes.
