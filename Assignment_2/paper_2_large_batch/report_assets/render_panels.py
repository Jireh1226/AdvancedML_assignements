import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


paper_dir = Path(__file__).resolve().parents[1]
results_dir = paper_dir / "reruns" / "chat_seed42_v2_20261022" / "results_bn_recalibrated" / "analysis"
output_dir = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
provenance = {}

for network in ("F1", "C1", "C2"):
    candidates = list(results_dir.glob(f"path_{network}_seed42_*/path.csv"))
    assert len(candidates) == 1
    source = candidates[0]
    payload = source.read_bytes()
    rows = [{"alpha": float(row["alpha"]),
             "train": {"loss": float(row["train_loss"]), "accuracy": float(row["train_accuracy"]) / 100},
             "test": {"loss": float(row["test_loss"]), "accuracy": float(row["test_accuracy"]) / 100}}
            for row in csv.DictReader(payload.decode().splitlines())]
    alphas = [row["alpha"] for row in rows]
    assert len(rows) == 61 and alphas[0] == -1 and alphas[-1] == 2
    assert 0 in alphas and 1 in alphas
    figure, loss_axis = plt.subplots(figsize=(4.3, 3.0), layout="constrained")
    accuracy_axis = loss_axis.twinx()
    for split, style in (("train", "-"), ("test", "--")):
        loss_axis.plot(alphas, [row[split]["loss"] for row in rows],
                       color="blue", linestyle=style, label=split.capitalize())
        accuracy_axis.plot(alphas, [100 * row[split]["accuracy"] for row in rows],
                           color="red", linestyle=style)
    loss_axis.set(title=network, xlabel=r"$\alpha$", ylabel="Cross-entropy",
                  xlim=(-1, 2), ylim=(0, None))
    accuracy_axis.set(ylabel="Accuracy (%)", ylim=(0, 100))
    loss_axis.yaxis.label.set_color("blue")
    accuracy_axis.yaxis.label.set_color("red")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(loc="upper left", fontsize=9)
    for endpoint in (0, 1):
        loss_axis.axvline(endpoint, color="gray", linewidth=0.6, alpha=0.5)
    destination = output_dir / f"{network}_panel.pdf"
    figure.savefig(destination, metadata={"Title": f"{network}: saved seed-42 path", "Author": "Jireh Katebe"})
    plt.close(figure)
    provenance[network] = {"source": str(source.relative_to(paper_dir)),
                           "source_sha256": hashlib.sha256(payload).hexdigest(),
                           "points": len(rows), "smoothing": False,
                           "output": destination.name}

(output_dir / "panel_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
