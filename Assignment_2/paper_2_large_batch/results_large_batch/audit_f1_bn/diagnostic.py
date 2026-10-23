import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


namespace = {}
notebook_path = Path(__file__).resolve().parents[2] / 'on_large_batch_training_for_deep_learning.ipynb'
notebook = json.loads(notebook_path.read_text())
os.environ.setdefault('ASSIGNMENT2_DIR', str(notebook_path.parent))
for cell in notebook['cells']:
    if cell.get('id') in {'large_batch-common', 'large_batch-model', 'large_batch-training'}:
        exec(''.join(cell['source']), namespace)
project = namespace['PROJECT_DIR']
assignment = project.parent if project.name == 'paper_2_large_batch' else project
paper1 = assignment / 'paper_1_generalization' if (assignment / 'paper_1_generalization').is_dir() else assignment
paper3 = assignment / 'paper_3_beta_vae' if (assignment / 'paper_3_beta_vae').is_dir() else assignment
destination = project / 'results_large_batch/audit_f1_bn'
destination.mkdir(exist_ok=True)
protected = [paper1 / 'figure1_reproduction.ipynb', paper3 / 'beta_vae_learning_basic_visual_concepts.ipynb',
             project / 'on_large_batch_training_for_deep_learning.ipynb',
             paper1 / 'results_figure1_reviewed/full/figure1a.png',
             paper1 / 'results_figure1_reviewed/full/figure1bc.png',
             paper3 / 'results_beta_vae/seed42_v1/figure7ab.png',
             project / 'results_large_batch/seed42_v1/figure3_f1_c1_c2.png']
protected.extend((project / 'results_large_batch/seed42_v1').glob('*/checkpoint.pt'))
protected.extend((paper3 / 'results_beta_vae/seed42_v1').glob('*/checkpoint.pt'))
hashes = {str(filename.relative_to(assignment)): hashlib.sha256(filename.read_bytes()).hexdigest() for filename in protected}
assert namespace['DEVICE'].type == 'cuda'
training, testing = namespace['classification_data']('F1')
states = [torch.load(project / f'results_large_batch/seed42_v1/F1_{regime}_seed42/checkpoint.pt',
                     map_location=namespace['DEVICE'], weights_only=True)['model'] for regime in ('SB', 'LB')]
baseline = json.loads((project / 'results_large_batch/seed42_v1/F1_interpolation.json').read_text())['rows']
baseline_by_alpha = {row['alpha']: row for row in baseline}
model = namespace['F1']().to(namespace['DEVICE']).eval()
curves = {}
selected_alphas = (1.3, 1.5, 2.0)
started = time.perf_counter()


@torch.no_grad()
def test_confusion():
    images, labels = testing
    matrix = torch.zeros((10, 10), dtype=torch.long, device=namespace['DEVICE'])
    for start in range(0, len(images), namespace['EVALUATION_BATCH']):
        stop = start + namespace['EVALUATION_BATCH']
        prediction = model(images[start:stop]).argmax(1)
        matrix += torch.bincount(labels[start:stop] * 10 + prediction, minlength=100).reshape(10, 10)
    matrix = matrix.cpu()
    return {'matrix': matrix.tolist(), 'prediction_counts': matrix.sum(0).tolist(),
            'per_class_accuracy': (matrix.diagonal().double() / matrix.sum(1)).tolist()}


for mode in ('recalibrated', 'fixed_sb', 'fixed_lb', 'affine_buffers'):
    rows = []
    for alpha in namespace['ALPHAS']:
        alpha = float(alpha)
        with torch.no_grad():
            model.load_state_dict(states[0])
            for name, parameter in model.named_parameters():
                parameter.copy_((1 - alpha) * states[0][name] + alpha * states[1][name])
            for name, buffer in model.named_buffers():
                if mode == 'fixed_lb':
                    buffer.copy_(states[1][name])
                elif mode == 'affine_buffers' and name.endswith(('running_mean', 'running_var')):
                    buffer.copy_((1 - alpha) * states[0][name] + alpha * states[1][name])
            variances = torch.cat([buffer.flatten() for name, buffer in model.named_buffers() if name.endswith('running_var')])
            invalid_count = int((variances < 0).sum().item())
            minimum_variance = float(variances.min().item())
        if invalid_count:
            rows.append({'alpha': alpha, 'valid': False, 'negative_variance_count': invalid_count,
                         'minimum_variance': minimum_variance})
            continue
        if mode == 'recalibrated':
            namespace['calibrate_batchnorm'](model, training[0])
        row = {'alpha': alpha, 'valid': True, 'train': namespace['evaluate_classifier'](model, training),
               'test': namespace['evaluate_classifier'](model, testing)}
        assert np.isfinite(row['train']['loss']) and np.isfinite(row['test']['loss'])
        if mode == 'recalibrated':
            assert row['train'] == baseline_by_alpha[alpha]['train']
            assert row['test'] == baseline_by_alpha[alpha]['test']
        if alpha in selected_alphas:
            row['test_confusion'] = test_confusion()
        rows.append(row)
    curves[mode] = rows
    namespace['atomic_json'](destination / 'curves.json', curves)
    print(mode, 'valid points', sum(row['valid'] for row in rows), flush=True)
    for alpha in selected_alphas:
        row = next(row for row in rows if row['alpha'] == alpha)
        print(json.dumps({key: row.get(key) for key in ('alpha', 'valid', 'train', 'test', 'negative_variance_count', 'minimum_variance')}), flush=True)

titles = {'recalibrated': 'Training-data recalibration (current)', 'fixed_sb': 'Fixed SB statistics (control)',
          'fixed_lb': 'Fixed LB statistics (control)', 'affine_buffers': 'Affine statistics (gaps = invalid variance)'}
figure, axes = plt.subplots(1, 4, figsize=(20, 4.8))
for axis, (mode, rows) in zip(axes, curves.items()):
    accuracy_axis = axis.twinx()
    for split, style in (('train', '-'), ('test', '--')):
        axis.plot([row['alpha'] for row in rows], [row[split]['loss'] if row['valid'] else np.nan for row in rows],
                  color='blue', linestyle=style, label=split.capitalize())
        accuracy_axis.plot([row['alpha'] for row in rows], [100 * row[split]['accuracy'] if row['valid'] else np.nan for row in rows],
                           color='red', linestyle=style)
    axis.set(title=titles[mode], xlabel='alpha', ylabel='Cross entropy', xlim=(-1, 2), ylim=(0, None))
    accuracy_axis.set(ylabel='Accuracy (%)', ylim=(0, 100))
    axis.yaxis.label.set_color('blue')
    accuracy_axis.yaxis.label.set_color('red')
    axis.axvline(0, color='gray', linewidth=0.6)
    axis.axvline(1, color='gray', linewidth=0.6)
    axis.legend(loc='upper left')
    axis.grid(alpha=0.2)
figure.suptitle('F1 batch-normalization controls: identical trained weights; no retraining or result selection')
figure.tight_layout()
figure.savefig(destination / 'bn_comparison.png', dpi=160)
plt.close(figure)
for relative, expected in hashes.items():
    assert hashlib.sha256((assignment / relative).read_bytes()).hexdigest() == expected, relative
metadata = {'state': 'complete', 'training_performed': False, 'elapsed_seconds': time.perf_counter() - started,
            'all_61_baseline_points_reproduced_exactly': True, 'unchanged_files': hashes,
            'protocol': 'identical affine trainable weights; change BN running statistics only; dropout disabled',
            'fixed_buffer_caveat': 'Each fixed-buffer control preserves only its own endpoint full model, not both endpoints.',
            'affine_buffer_caveat': 'Negative extrapolated variances are invalid and omitted; no clamping or invented points.',
            'selection_changed': False}
namespace['atomic_json'](destination / 'audit_metadata.json', metadata)
print(json.dumps({key: metadata[key] for key in ('state', 'elapsed_seconds', 'selection_changed')}), flush=True)
