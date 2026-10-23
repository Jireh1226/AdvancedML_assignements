import ast
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
run_dir = Path(__file__).resolve().parent
os.environ['PAPER2_RUN_DIR'] = str(run_dir)
os.environ['MPLCONFIGDIR'] = str(run_dir / '.cache/matplotlib')

import nbformat
import torch
from torch.utils.data import TensorDataset

notebook_path = run_dir / 'on_large_batch_training_chat_seed42_v2.ipynb'
provenance = json.loads((run_dir / 'provenance.json').read_text())
assert hashlib.sha256(notebook_path.read_bytes()).hexdigest() == provenance['notebook_sha256']
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
namespace = {}
for cell in notebook.cells:
    if cell.cell_type == 'code':
        ast.parse(cell.source)
        if cell.id in {'gpt-v2-setup', 'gpt-v2-data', 'gpt-v2-models', 'gpt-v2-training', 'gpt-v2-paths'}:
            exec(compile(cell.source, cell.id, 'exec'), namespace)
assert namespace['IMPLEMENTATION_SHA256'] == provenance['code_cells_sha256']
assert namespace['SEEDS'] == (42,)
assert namespace['CFG'].path_points == 61
assert namespace['DEVICE'].type == 'cuda'
checks = {}
checks['architecture'] = namespace['architecture_checks']().to_dict(orient='records')
expected_counts = {'F1': 1462794, 'C1': 283274, 'C2': 14991946}
for row in checks['architecture']:
    assert row['parameters'] == expected_counts[row['network']]

namespace['seed_all'](177)
images = torch.rand(40, 1, 28, 28)
labels = torch.arange(40) % 10
reference_dataset = TensorDataset(images, labels)
cached_dataset = namespace['CachedTensorDataset'](images.cuda(), labels.cuda())
for shuffled in (False, True):
    for drop_last in (False, True):
        reference_loader = namespace['make_loader'](reference_dataset, 7, shuffled, 52, drop_last)
        cached_loader = namespace['make_loader'](cached_dataset, 7, shuffled, 52, drop_last)
        assert len(reference_loader) == len(cached_loader)
        reference_batches = list(reference_loader)
        cached_batches = list(cached_loader)
        for reference_batch, cached_batch in zip(reference_batches, cached_batches):
            assert torch.equal(reference_batch[0], cached_batch[0].cpu())
            assert torch.equal(reference_batch[1], cached_batch[1].cpu())
        assert torch.equal(reference_loader.generator.get_state(), cached_loader.generator.get_state())
checks['cached_loader_matches_batch_values_order_remainder_and_rng'] = True

config = namespace['replace'](namespace['CFG'], small_batch=8, eval_batch=8)
synthetic_info = {'dataset': 'synthetic_software_test_only', 'hashes': {}, 'train_n': 40, 'test_n': 40}
with tempfile.TemporaryDirectory(prefix='validation-', dir=run_dir) as temporary:
    scratch = Path(temporary)
    full = namespace['train_trial']('F1', 'SB', 42, reference_dataset, reference_dataset, synthetic_info, cfg=config, max_epochs=2, root=scratch / 'reference')
    cached = namespace['train_trial']('F1', 'SB', 42, cached_dataset, cached_dataset, synthetic_info, cfg=config, max_epochs=2, root=scratch / 'cached')
    namespace['train_trial']('F1', 'SB', 42, cached_dataset, cached_dataset, synthetic_info, cfg=config, max_epochs=1, root=scratch / 'resumed')
    resumed = namespace['train_trial']('F1', 'SB', 42, cached_dataset, cached_dataset, synthetic_info, cfg=config, max_epochs=2, root=scratch / 'resumed')
    saved = [namespace['load_checkpoint'](folder / 'last.pt') for folder in (full, cached, resumed)]
    for alternate in saved[1:]:
        assert saved[0]['initial_state_sha256'] == alternate['initial_state_sha256']
        assert namespace['state_hash'](saved[0]['model']) == namespace['state_hash'](alternate['model'])
        assert saved[0]['global_step'] == alternate['global_step']
        for left, right in zip(saved[0]['history'], alternate['history']):
            assert {key: value for key, value in left.items() if key != 'seconds'} == {key: value for key, value in right.items() if key != 'seconds'}
    checks['cached_training_and_exact_resume_match_reference'] = True
    left_state = saved[0]['model']
    right_state = {name: value.clone() for name, value in left_state.items()}
    for name, value in right_state.items():
        if name.endswith('running_var'):
            value.mul_(1.1)
        elif value.is_floating_point():
            value.add_(0.01)
    for alpha, state in ((0., left_state), (1., right_state)):
        assert namespace['state_hash'](namespace['interpolated_state'](left_state, right_state, alpha)) == namespace['state_hash'](state)
    for alpha in (-1., 0.5, 2.):
        state = namespace['interpolated_state'](left_state, right_state, alpha)
        assert all(torch.isfinite(value).all() for value in state.values())
        assert all((value > 0).all() for name, value in state.items() if name.endswith('running_var'))
    checks['path_endpoints_and_positive_extrapolated_variances'] = True

del saved, left_state, right_state, reference_dataset, cached_dataset
torch.cuda.empty_cache()
checks['real_datasets'] = []
for network in ('F1', 'C1'):
    training, testing, info = namespace['get_data'](network)
    checks['real_datasets'].append(info)
    assert len(training) == (60000 if network == 'F1' else 50000)
    assert len(testing) == 10000
checks['dataset_cache_transform_checks'] = True

namespace['seed_all'](42)
training, _, _ = namespace['get_data']('C2')
model = namespace['make_model']('C2').cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False, fused=False)
loader = namespace['make_loader'](training, 5000, True, 1042, True)
batch_images, batch_labels = next(iter(loader))
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
optimizer.zero_grad(set_to_none=True)
loss = namespace['F'].cross_entropy(model(batch_images), batch_labels)
assert torch.isfinite(loss)
loss.backward()
optimizer.step()
torch.cuda.synchronize()
checks['c2_full_batch_pilot'] = {'examples': len(batch_labels), 'seconds': time.perf_counter() - started, 'loss': loss.item(), 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3}
checks['training_results_created'] = False
assert not (run_dir / 'results').exists()
checks['passed'] = True
(run_dir / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2), flush=True)
