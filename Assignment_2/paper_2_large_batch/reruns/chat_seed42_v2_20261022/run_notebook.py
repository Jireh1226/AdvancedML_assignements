import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

run_dir = Path(__file__).resolve().parent
source = run_dir / 'on_large_batch_training_chat_seed42_v2.ipynb'
destination = run_dir / 'on_large_batch_training_chat_seed42_v2.executed.ipynb'
provenance = json.loads((run_dir / 'provenance.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance['notebook_sha256']
assert json.loads((run_dir / 'validation.json').read_text())['passed']
notebook = nbformat.read(source, as_version=4)
started_at = datetime.now(timezone.utc).isoformat()
started_counter = time.perf_counter()


def write_status(state, **details):
    payload = {'state': state, 'pid': os.getpid(), 'started_utc': started_at, 'updated_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - started_counter, 'notebook': source.name, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'seed': 42, 'training_runs': 6, **details}
    temporary = run_dir / 'status.tmp.json'
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(run_dir / 'status.json')


def save_notebook():
    temporary = destination.with_suffix('.tmp.ipynb')
    nbformat.write(notebook, temporary)
    temporary.replace(destination)


class StreamingNotebookClient(NotebookClient):
    def process_message(self, message, cell, cell_index):
        output = super().process_message(message, cell, cell_index)
        if message['msg_type'] == 'stream':
            print(message['content']['text'], end='', flush=True)
        elif message['msg_type'] == 'error':
            print(f"{message['content']['ename']}: {message['content']['evalue']}", flush=True)
        return output


def cell_started(cell, cell_index):
    print(f'Executing {cell.id}', flush=True)
    write_status('running', cell_id=cell.id)


def cell_finished(cell, cell_index, execute_reply):
    save_notebook()


environment = dict(os.environ)
environment['PAPER2_RUN_DIR'] = str(run_dir)
environment['MPLCONFIGDIR'] = str(run_dir / '.cache/matplotlib')
environment['PYTHONPATH'] = str(run_dir / '.dependencies') + os.pathsep + environment.get('PYTHONPATH', '')
environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
client = StreamingNotebookClient(notebook, kernel_name='assignment2-inception', timeout=None, resources={'metadata': {'path': str(run_dir)}}, on_cell_execute=cell_started, on_cell_executed=cell_finished)
write_status('starting')
try:
    client.execute(cwd=str(run_dir), env=environment)
except BaseException as error:
    traceback.print_exc()
    save_notebook()
    write_status('failed', error=f'{type(error).__name__}: {error}')
    sys.exit(1)
else:
    for cell in notebook.cells:
        if cell.cell_type == 'code' and cell.id != 'gpt-v2-figure':
            cell.outputs = []
        elif cell.cell_type == 'code':
            cell.outputs = [output for output in cell.outputs if output.output_type != 'stream']
    save_notebook()
    write_status('complete', executed_notebook=destination.name)
    print('COMPLETE: six fresh models and all three measured paths saved.', flush=True)
