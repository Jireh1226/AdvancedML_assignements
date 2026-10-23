import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    paper_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Redraw Figure 7A/B from saved analysis arrays.")
    parser.add_argument("--output-dir", type=Path, default=paper_dir / "report_assets" / "regenerated")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = paper_dir / "results_beta_vae" / "seed42_v1"
    analyses = {}
    for beta in (4.0, 1.0):
        with np.load(results_dir / f"beta{beta:g}_seed42" / "analysis.npz", allow_pickle=False) as saved:
            analyses[beta] = {key: saved[key] for key in saved.files}
    notebook = json.loads((paper_dir / "beta_vae_learning_basic_visual_concepts.ipynb").read_text())
    plot_cells = ["".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "figure = plt.figure(figsize=(25, 10))" in "".join(cell["source"])]
    if len(plot_cells) != 1 or "\nplt.show()" not in plot_cells[0]:
        raise ValueError("The submitted notebook's plotting cell could not be identified.")
    namespace = {"plt": plt, "np": np, "vae_analyses": analyses, "BETAS": (4.0, 1.0),
                 "LATENT_SIZE": 10, "TRAVERSAL_VALUES": np.linspace(3, -3, 5, dtype=np.float32),
                 "RESULTS_DIR": arguments.output_dir}
    exec(compile(plot_cells[0].split("\nplt.show()", 1)[0], "submitted_notebook_plot", "exec"), namespace)
    plt.close("all")
    print(f"Saved figure7ab.pdf and figure7ab.png in {arguments.output_dir}")


if __name__ == "__main__":
    main()
