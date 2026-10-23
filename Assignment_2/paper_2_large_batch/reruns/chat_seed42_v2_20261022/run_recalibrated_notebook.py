import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


run_dir = Path(__file__).resolve().parent
source = run_dir / "on_large_batch_training_chat_seed42_v2_bn_recalibrated.prepared.ipynb"
destination = run_dir / "on_large_batch_training_chat_seed42_v2_bn_recalibrated.executed.ipynb"
notebook = nbformat.read(source, as_version=4)
training_notebook = nbformat.read(run_dir / "on_large_batch_training_chat_seed42_v2.ipynb", as_version=4)
for cell_index in (1, 2, 3, 4, 7):
    assert notebook.cells[cell_index].source == training_notebook.cells[cell_index].source
checkpoints = sorted((run_dir / "results").glob("*/last.pt"))
assert len(checkpoints) == 6
original_artifacts = checkpoints + [
    run_dir / "on_large_batch_training_chat_seed42_v2.ipynb",
    run_dir / "on_large_batch_training_chat_seed42_v2.executed.ipynb",
    run_dir / "results/figure3_chat_seed42_v2.png",
    run_dir / "results/endpoint_metrics.json",
]


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


original_hashes = {str(path.relative_to(run_dir)): file_hash(path) for path in original_artifacts}
started = time.perf_counter()
started_utc = datetime.now(timezone.utc).isoformat()


def save_status(state, **details):
    payload = {"state": state, "started_utc": started_utc,
               "updated_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter() - started,
               "training_performed": False, **details}
    (run_dir / "bn_recalibration_status.json").write_text(json.dumps(payload, indent=2))


class LoggingClient(NotebookClient):
    def process_message(self, message, cell, cell_index):
        output = super().process_message(message, cell, cell_index)
        if message["msg_type"] == "stream":
            print(message["content"]["text"], end="", flush=True)
        return output


environment = dict(os.environ)
environment["PAPER2_RUN_DIR"] = str(run_dir)
environment["PYTHONPATH"] = str(run_dir / ".dependencies") + os.pathsep + environment.get("PYTHONPATH", "")
environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
environment["MPLCONFIGDIR"] = str(run_dir / ".cache/matplotlib")
client = LoggingClient(notebook, kernel_name="assignment2-inception", timeout=None,
                       resources={"metadata": {"path": str(run_dir)}})
save_status("running")
try:
    with client.setup_kernel(cwd=str(run_dir), env=environment):
        execution_count = 0
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if cell_index == 7:
                print("Preserving the already-executed training cell; no training call is made.", flush=True)
                execution_count = max(execution_count, cell.execution_count or 0)
                continue
            execution_count += 1
            client.execute_cell(cell, cell_index, execution_count=execution_count)
            nbformat.write(notebook, destination)
    after_hashes = {str(path.relative_to(run_dir)): file_hash(path) for path in original_artifacts}
    assert after_hashes == original_hashes
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = ([output for output in cell.outputs if output.output_type != "stream"]
                            if cell.id == "gpt-v2-figure" else [])
    notebook.metadata["reproduction"]["original_artifacts_sha256"] = original_hashes
    nbformat.write(notebook, destination)
    save_status("complete", executed_notebook=destination.name, original_artifacts_sha256=original_hashes)
    print("COMPLETE: 183 recalibrated path points; all original checkpoints and outputs unchanged.", flush=True)
except BaseException as error:
    traceback.print_exc()
    nbformat.write(notebook, destination)
    save_status("failed", error=f"{type(error).__name__}: {error}")
    raise
