import hashlib
import json
import os
import sys
import time
from pathlib import Path


run_dir = Path(os.environ["PAPER2_RUN_DIR"])
output_dir = run_dir / "diagnostics" / "comparison_20261022"
output_dir.mkdir(parents=True, exist_ok=False)
notebook_path = run_dir / "on_large_batch_training_chat_seed42_v2.ipynb"
notebook = json.loads(notebook_path.read_text())
provenance = json.loads((run_dir / "provenance.json").read_text())
assert hashlib.sha256(notebook_path.read_bytes()).hexdigest() == provenance["notebook_sha256"]
checkpoint_paths = sorted((run_dir / "results").glob("*/last.pt"))
assert len(checkpoint_paths) == 6


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


before_hashes = {str(path.relative_to(run_dir)): file_hash(path) for path in checkpoint_paths}
for cell_index in (1, 2, 3, 4, 5):
    exec(compile("".join(notebook["cells"][cell_index]["source"]), str(notebook_path), "exec"), globals())

DOWNLOAD_DATA = False


def parameter_hash(model):
    return state_hash(dict(model.named_parameters()))


def variance_summary(model):
    variances = torch.cat([
        module.running_var.detach().flatten()
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ])
    return {
        "minimum": float(variances.min()),
        "maximum": float(variances.max()),
        "median": float(variances.median()),
        "finite": bool(torch.isfinite(variances).all()),
    }


@torch.no_grad()
def recalibrate(model, training_data):
    model.eval()
    normalizations = [
        module for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    momenta = [module.momentum for module in normalizations]
    for module in normalizations:
        module.reset_running_stats()
        module.train()
    count = 0
    try:
        for start in range(0, len(training_data), 512):
            images = training_data.images[start:start + 512]
            momentum = len(images) / (count + len(images))
            for module in normalizations:
                module.momentum = momentum
            model(images)
            count += len(images)
    finally:
        for module, momentum in zip(normalizations, momenta):
            module.momentum = momentum
        model.eval()


started = time.perf_counter()
records = []
for network in ("F1", "C1", "C2"):
    saved_states = {}
    for regime in ("SB", "LB"):
        matches = list((run_dir / "results").glob(f"{network}_{regime}_seed42_*/last.pt"))
        assert len(matches) == 1
        saved = load_checkpoint(matches[0])
        saved_states[regime] = saved["model"]
        del saved
    train_set, test_set, _ = get_data(network, CFG)
    train_loader = make_loader(train_set, CFG.eval_batch, False, 4042)
    test_loader = make_loader(test_set, CFG.eval_batch, False, 5042)
    model = make_model(network, CFG).to(DEVICE).eval()
    path_files = list((run_dir / "results/analysis").glob(f"path_{network}_seed42_*/path.csv"))
    assert len(path_files) == 1
    original_path = pd.read_csv(path_files[0])
    for alpha in (0.0, 0.5, 1.0, 2.0):
        model.load_state_dict(interpolated_state(saved_states["SB"], saved_states["LB"], alpha))
        unchanged_parameters = parameter_hash(model)
        baseline_variances = variance_summary(model)
        baseline = {"train": evaluate(model, train_loader), "test": evaluate(model, test_loader)}
        original_row = original_path.loc[original_path.alpha == alpha].iloc[0]
        for split in ("train", "test"):
            assert np.isclose(baseline[split]["loss"], original_row[f"{split}_loss"], rtol=1e-6, atol=1e-7)
            assert np.isclose(baseline[split]["accuracy"], original_row[f"{split}_accuracy"], rtol=0, atol=1e-8)
        recalibrate(model, train_set)
        assert parameter_hash(model) == unchanged_parameters
        recalibrated = {"train": evaluate(model, train_loader), "test": evaluate(model, test_loader)}
        record = {
            "network": network,
            "alpha": alpha,
            "trainable_weights_unchanged": True,
            "saved_path_reproduced": True,
            "original_log_variance_path": baseline,
            "training_only_recalibration": recalibrated,
            "original_variances": baseline_variances,
            "recalibrated_variances": variance_summary(model),
        }
        records.append(record)
        json_write(output_dir / "partial.json", records)
        print(json.dumps(record), flush=True)
    del model, saved_states
    torch.cuda.empty_cache()

after_hashes = {str(path.relative_to(run_dir)): file_hash(path) for path in checkpoint_paths}
assert before_hashes == after_hashes
json_write(output_dir / "batchnorm_control.json", {
    "training_performed": False,
    "purpose": "Inspect the completed attempt; not a replacement figure or parameter selection.",
    "protocol": "Same trained weights at each alpha; only stored BN statistics change. Full training-data recalibration, dropout disabled, batch size 512, matching the earlier notebook's procedure.",
    "caveat": "Recalibration also changes endpoint BN buffers, so it is a different evaluation convention, not a proof that it matches the authors' protocol.",
    "elapsed_seconds": time.perf_counter() - started,
    "checkpoint_hashes_unchanged": before_hashes,
    "records": records,
})
print("INSPECTION COMPLETE: all six checkpoints unchanged; no training performed.", flush=True)
