# Paper 3: beta-VAE

Figure 7A/B: beta=4 and beta=1 on all 737,280 dSprites images, seed 42.

- [Report PDF](beta_vae_learning_basic_visual_concepts_report.pdf).
- [Submitted executed notebook](beta_vae_learning_basic_visual_concepts.ipynb).
- [Completed figure](results_beta_vae/seed42_v1/figure7ab.pdf).
- [Metrics, histories and analysis arrays](results_beta_vae/seed42_v1/).
- [Original paper](291_beta_vae_learning_basic_visual.pdf), Figure 7 on page 10.
- [Later ChatGPT/Astra conversation](https://chatgpt.com/share/6adadb56-35dc-83e9-9c94-7e1077a90301).
- [Final artifact record](finalization_record.json) and [Phase 4 source check](phase4_record.json).
- [Later completed chat-derived run](reruns/chat_seed42_v2_20261022/README.md), with a minimal notebook and comparison to the submitted results. It does not replace the report's earlier run or the submission ZIP.

## Which run is submitted?

The completed earlier IDE-assisted pair supplies all notebook outputs, figures and report measurements. Both models trained for 100 epochs with Adagrad 0.01, accumulator 0.1, batch size 256 and matched seed-42 initialisation. No later run is mixed into this submission.

The report discloses that the submitted run preceded the shared chat and therefore does not fully meet the assignment's chat-interface requirement. The earlier notebook is unchanged from the saved local Phase 3 snapshot. No training code from an author or third-party implementation was copied.

## View and regenerate the figure

Open the notebook without executing cells, or view the saved PDF/PNG. The analysis arrays contain the factor-response curves, posterior variances and traversal images. To redraw the figure without PyTorch, data or checkpoints:

```sh
python -m pip install numpy matplotlib
python report_assets/render_figure.py
```

This uses the plotting cell from the submitted notebook and the saved analysis arrays. It does not run training or recompute model outputs. The report's figure is a copy of the original completed figure, not a newly selected run.

## Data and fresh training

The original environment was Python 3.10.12, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 and NumPy 2.2.6, with Matplotlib and Jupyter. In a CUDA 12.4-compatible Linux environment:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.2.6 matplotlib jupyter
```

From this paper folder, prepare the official dataset archive and its provenance file. The notebook reads the archive directly; it does not download it automatically.

```sh
mkdir -p ../data/dsprites
curl -L --fail https://raw.githubusercontent.com/google-deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz -o ../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
cp dataset_source.json ../data/dsprites/source.json
python -c 'import hashlib,pathlib; source=pathlib.Path("../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"); assert hashlib.sha256(source.read_bytes()).hexdigest()=="857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2"'
jupyter notebook beta_vae_learning_basic_visual_concepts.ipynb
```

In a working copy, change `RESULTS_DIR` to a new directory, for example `PROJECT_DIR / "results_beta_vae" / "seed42_fresh"`, then run all cells. Launch Jupyter from this paper folder, or set `ASSIGNMENT2_DIR` to it. The notebook locates the shared `../data/` directory. The dataset is about 3 GB uncompressed and is cached as uint8 on the selected device, in addition to model and training memory.

The recorded pair took about 18.6 minutes of training, excluding data preparation and figure analysis. Different hardware, data movement and competing jobs can substantially change that time. Checkpoints are saved after each epoch. An interrupted run requires its matching environment and checkpoint to resume.

Large checkpoints and data are excluded from Git. Running all cells against the published completed directory still requires the missing checkpoints, because analysis validates their hashes. Use a fresh results directory for new training or the standalone plotting command for saved results.

## Result limits

Both models have five informative and five approximately unused coordinates, but the beta=4 response maps and traversals show greater factor selectivity. Its final posterior-mean reconstruction loss is 45.145 nats/image versus 33.456 for beta=1; KL is 11.900 versus 17.606. Posterior-mean decoding is not the expected reconstruction term in the ELBO. Training still improved near epoch 100, and one seed does not establish robustness.

Phase 4 checked the author publication page and linked official dataset. No author Figure 7 training implementation was identified in those resources; unknown optimiser and stopping settings remain unresolved. The separate records distinguish this limited search from a claim that no implementation exists anywhere.
