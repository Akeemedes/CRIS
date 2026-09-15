"""Auditable selection from retained weights within an accepted-update budget."""
import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / 'runs/lower_branch_sweep'
ARMS = ('mse', 'lambda_0p01', 'lambda_0p1', 'lambda_1')
BUDGET = 10000


def candidates(name, budget=BUDGET):
    run = RUNS / name
    replay=BASE/'runs/budget_10k'/name
    if (replay/'result.json').exists() and (replay/'final_checkpoint.bin').exists():
        result=json.loads((replay/'result.json').read_text())
        if int(result['accepted_updates'])!=budget:
            raise ValueError(f'Recovery run stopped before the requested budget: {replay}')
        run=replay
    with (run/'optimizer_trace.csv').open() as stream:
        trace = list(csv.DictReader(stream))
    eligible = [r for r in trace if int(r['accepted_updates']) <= budget]
    result = []
    for path in sorted(run.glob('milestone_*.bin')):
        threshold = 10.**(-2-int(path.stem.split('_')[1]))
        row = next(r for r in trace if r['status']=='accepted' and float(r['validation_mse_normalized']) <= threshold)
        if int(row['accepted_updates']) > budget:
            continue
        result.append(dict(path=str(path.relative_to(BASE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            accepted=int(row['accepted_updates']), proposals=int(row['iteration']),
            hours=float(row['cumulative_phase_seconds'])/3600,
            validation_mse_normalized=float(row['validation_mse_normalized'])))
    # The final artifact contains the selected best weights, not necessarily the last iterate.
    # It is eligible only when the entire completed run is inside the analysis budget.
    if int(trace[-1]['accepted_updates']) <= budget:
        path = run/'final_checkpoint.bin'
        best = min(trace, key=lambda r:float(r['validation_mse_normalized']))
        result.append(dict(path=str(path.relative_to(BASE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            accepted=int(best['accepted_updates']), proposals=int(best['iteration']),
            hours=float(best['cumulative_phase_seconds'])/3600,
            validation_mse_normalized=float(best['validation_mse_normalized'])))
    if not result:
        raise ValueError(f'No eligible retained checkpoint: {name}')
    best_by_budget=min(eligible,key=lambda r:float(r['validation_mse_normalized']))
    return result, dict(run=str(run), trace_sha256=hashlib.sha256((run/'optimizer_trace.csv').read_bytes()).hexdigest(),
        budget_accepted=budget, budget_hours=float(eligible[-1]['cumulative_phase_seconds'])/3600,
        actual_best_by_budget_accepted=int(best_by_budget['accepted_updates']),
        actual_best_by_budget_mse_physical=20.25*float(best_by_budget['validation_mse_normalized']),
        best_by_budget_weights_available=int(trace[-1]['accepted_updates'])<=budget,
        full_run_hours=float(trace[-1]['cumulative_phase_seconds'])/3600)


def selection():
    selected = {}
    for name in ARMS:
        retained, cost = candidates(name)
        selected[name] = dict(**min(retained,key=lambda r:r['validation_mse_normalized']), **cost)
    return selected


if __name__ == '__main__':
    chosen = selection()
    (BASE/'analysis/checkpoint_selection.json').write_text(json.dumps(chosen,indent=2)+'\n')
    print(json.dumps(chosen,indent=2))
