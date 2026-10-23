import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARCHITECTURES = ("inception", "alexnet", "mlp512")
THRESHOLDS = (500, 50, 0)


def read_json(file_path):
    return json.loads(file_path.read_text())


def source_record(file_path, results_root):
    return {
        "path": file_path.relative_to(results_root).as_posix(),
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
    }


def select_history(results_root, architecture, probability):
    run_id = f"seed42_full_v1_{architecture}_labels_p{probability:g}_seed42"
    parent = results_root / "full" / run_id
    selected = parent
    parent_summary = read_json(parent / "summary.json")
    continuation = results_root / "convergence" / run_id
    if (continuation / "summary.json").is_file():
        continued_summary = read_json(continuation / "summary.json")
        if continued_summary["train"]["errors"] <= parent_summary["train"]["errors"]:
            selected = continuation
    summary_path = selected / "summary.json"
    history_path = selected / "history.json"
    summary = read_json(summary_path)
    history = read_json(history_path)
    epochs = history["epochs"]
    config = summary["config"]
    expected = {
        "architecture": architecture,
        "condition": "labels",
        "p": probability,
        "seed": 42,
        "batch_size": 128,
        "train_count": 50000,
        "smoke_test": False,
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError(f"Unexpected configuration: {run_id}")
    if summary["status"] != "completed" or not summary["final_fit"]:
        raise ValueError(f"Selected run is unfinished: {run_id}")
    if len(epochs) != summary["epochs_completed"]:
        raise ValueError(f"History length differs from summary: {run_id}")
    if epochs[-1]["train"] != summary["train"] or epochs[-1]["step"] != summary["steps"]:
        raise ValueError(f"Final history differs from summary: {run_id}")
    for expected_epoch, record in enumerate(epochs, start=1):
        if record["epoch"] != expected_epoch or record["step"] != expected_epoch * 391:
            raise ValueError(f"Nonconsecutive evaluation history: {run_id}")
        if record["train"]["count"] != 50000:
            raise ValueError(f"Unexpected evaluation sample size: {run_id}")
    first_zero = next(record for record in epochs if record["train"]["errors"] == 0)
    if first_zero["step"] != summary["first_fit_step"]:
        raise ValueError(f"First-fit timestamp differs from history: {run_id}")
    if selected != parent:
        parent_epochs = read_json(parent / "history.json")["epochs"]
        if epochs[:len(parent_epochs)] != parent_epochs:
            raise ValueError(f"Continuation changed its parent history: {run_id}")
    sources = [source_record(file_path, results_root) for file_path in (
        summary_path, history_path, parent / "summary.json"
    )]
    return run_id, epochs, sources


def one_error_intervals(epochs):
    intervals = []
    for record in epochs:
        if record["train"]["errors"] != 1:
            continue
        if intervals and intervals[-1]["end_epoch"] + 1 == record["epoch"]:
            intervals[-1]["end_epoch"] = record["epoch"]
            intervals[-1]["end_lr"] = record["lr"]
        else:
            intervals.append({
                "start_epoch": record["epoch"], "end_epoch": record["epoch"],
                "start_lr": record["lr"], "end_lr": record["lr"],
            })
    return intervals


def main():
    parser = argparse.ArgumentParser(description="Recompute timing diagnostics without training.")
    parser.add_argument("--results-root", type=Path,
                        default=Path(__file__).resolve().parent / "results_figure1_reviewed")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "report_assets")
    arguments = parser.parse_args()
    results_root = arguments.results_root.resolve()
    output_dir = arguments.output_dir.resolve()
    if output_dir == results_root or results_root in output_dir.parents:
        raise ValueError("Write diagnostics outside the preserved experimental results.")
    rows, runs, sources = [], [], {}
    for architecture in ARCHITECTURES:
        baseline_steps = {}
        for probability_index in range(11):
            probability = probability_index / 10
            run_id, epochs, records = select_history(results_root, architecture, probability)
            for record in records:
                sources[record["path"]] = record
            runs.append({"run_id": run_id, "one_error_intervals": one_error_intervals(epochs)})
            for allowed_errors in THRESHOLDS:
                first_fit = next(record for record in epochs
                                 if record["train"]["errors"] <= allowed_errors)
                if probability_index == 0:
                    baseline_steps[allowed_errors] = first_fit["step"]
                rows.append({
                    "architecture": architecture,
                    "p": probability,
                    "allowed_training_errors": allowed_errors,
                    "first_epoch": first_fit["epoch"],
                    "first_step": first_fit["step"],
                    "relative_steps": first_fit["step"] / baseline_steps[allowed_errors],
                    "history_path": records[1]["path"],
                })
    if len(rows) != 99:
        raise ValueError("The diagnostic must contain all 33 runs at all three thresholds.")
    audit_path = results_root / "audit" / "review_audit.json"
    existing_audit = read_json(audit_path)
    sources[audit_path.relative_to(results_root).as_posix()] = source_record(audit_path, results_root)
    for recorded in existing_audit["timing_sensitivity"]:
        endpoints = [row for row in rows
                     if row["architecture"] == recorded["architecture"]
                     and row["allowed_training_errors"] == recorded["allowed_training_errors"]
                     and row["p"] in (0, 1)]
        if (endpoints[0]["first_epoch"] != recorded["clean_epoch"]
                or endpoints[1]["first_epoch"] != recorded["random_epoch"]
                or abs(endpoints[1]["relative_steps"] - recorded["ratio"]) > 1e-12):
            raise ValueError("Recomputed endpoint timing differs from the existing audit.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "First epoch-end fit at three error thresholds on unchanged selected runs.",
        "training_performed": False,
        "primary_threshold_errors": 0,
        "diagnostic_threshold_errors": [500, 50],
        "sources": sorted(sources.values(), key=lambda record: record["path"]),
        "timings": rows,
        "runs": runs,
    }
    (output_dir / "threshold_sensitivity.json").write_text(json.dumps(output, indent=2) + "\n")
    with (output_dir / "threshold_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    colors = {"inception": "black", "alexnet": "tab:green", "mlp512": "tab:blue"}
    names = {"inception": "Inception", "alexnet": "AlexNet", "mlp512": "MLP 1x512"}
    for architecture in ARCHITECTURES:
        selected_rows = [row for row in rows if row["architecture"] == architecture
                         and row["allowed_training_errors"] == 500]
        axis.plot([row["p"] for row in selected_rows],
                  [row["relative_steps"] for row in selected_rows], "o-",
                  label=names[architecture], color=colors[architecture], linewidth=1.6)
    axis.set(xlabel="Label resampling probability p", ylabel="Steps / clean baseline steps",
             title="Diagnostic: first epoch-end training accuracy at least 99%",
             xlim=(0, 1), ylim=(0.9, 3.4))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "threshold_sensitivity.pdf")
    figure.savefig(output_dir / "threshold_sensitivity.png", dpi=180)
    plt.close(figure)
    print(f"Verified 33 saved runs and 99 threshold measurements; outputs: {output_dir}")


if __name__ == "__main__":
    main()
