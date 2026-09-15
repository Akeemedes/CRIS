"""Run paired SRDM/CRIS shock refinements and compare with independent references."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT/'training/twophasetransport/analysis/regular_v3/packages/residual_lambda_0p01_final.pt'
PUBLISHED_MESHES = {1: [100,200,400,800,1600,3200],
                    10: [2000,4000,8000,10000,16000],
                    100: [8000,16000,20000,32000],
                    1000: [20000,40000,50000,80000]}
LENGTH, T_END, X_INITIAL, LEFT, RIGHT = 1000., 200., 250., .25, .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def scalar_flow(u):
    s = min(1., max(0., u))
    return s*s/(s*s+.4*(1-s)**2)


def shock_speed():
    return (scalar_flow(LEFT)-scalar_flow(RIGHT))/(LEFT-RIGHT)


def cell_averages(n, t):
    h = LENGTH/n
    occupied = np.clip((X_INITIAL+shock_speed()*t-np.arange(n)*h)/h, 0., 1.)
    return RIGHT+(LEFT-RIGHT)*occupied


def check_entropy():
    # f'' has the sign of p(s)=2.8*s^3-4.2*s^2+.4; p'<0 on (0,1).
    # Hence p(LEFT)>0 establishes strict convexity between the two states.
    if not (0 < RIGHT < LEFT < 1 and 2.8*LEFT**3-4.2*LEFT**2+.4 > 0):
        raise ValueError('The analytic single-shock solution requires a convex flux on these states')
    if not (0 < X_INITIAL < X_INITIAL+shock_speed()*T_END < LENGTH):
        raise ValueError('The shock must stay inside the domain')


def norms(error):
    return {'L1':float(np.mean(abs(error))), 'L2':float(np.sqrt(np.mean(error**2))),
            'Linf':float(np.max(abs(error)))}


def read_state(path):
    with path.open('rb') as stream:
        count = np.fromfile(stream, dtype='<u8', count=1)
        values = np.fromfile(stream, dtype='<f8')
    if len(count) != 1 or int(count[0]) != len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f'Invalid saturation file: {path}')
    return values


def config(n, beta, method):
    h, dt = LENGTH/n, beta*LENGTH/n
    return f'''# Single-shock accuracy study; fixed dt/h within each cohort.
NX {n}
NY 1
NZ 1
GEOMETRY UNIFORM
DX {h:.17g}
DY 1
DZ 1
DISPERSION HOMOGENEOUS
DISPX 0
DISPY 0
DISPZ 0
BOUNDARY_FIELD HOMOGENEOUS
BC_VALUE {LEFT} {RIGHT} 0 0 0 0
DIRICHLET_BC 1 1 0 0 0 0
INIT_FIELD HETEROGENEOUS
INIT_VALUE 0
INIT_FILE ../initial.txt
INTR_VEL_FIELD HOMOGENEOUS
INTR_VEL_VALS 1 0 0
BNDR_VEL_FIELD HOMOGENEOUS
BNDR_VEL_VALS -1 1 0 0 0 0
MU_W 1
MU_NW 0.4
N_W 2
N_NW 2
NG 0
LAMBDA 1
DT_INIT {dt:.17g}
DT_MIN {dt:.17g}
DT_MAX {dt:.17g}
DT_INCR 1.5
DT_DECR 0.5
T_END {T_END:.17g}
REPORT_FREQ {T_END/2:.17g}
VERBOSITY 0
DUTOL 1e-10
RTOL 1e-10
LINTOL 1e-2
LINTOL_MIN 1e-6
LINMAX 7
PRESSURE_LINMAX 100
MAX_STEPS_NEWTON {100 if beta == 1 else 1000}
REFRESH_TRANSPORT_PRECONDITIONER 1
LINEAR_SOLVE_POLICY CONTINUE_ON_BUDGET
NEWTONUPDATER {'LINESEARCH' if method == 'SRDM' else 'STANDARD'}
MODE {method}
{('CRIS_MODEL ../../../inputs/model.pt' if method == 'CRIS' else '')}
'''


def run_native(args, n, beta, method, identity, env):
    folder = args.output/f'beta_{beta:g}'/f'N{n:05d}'/method
    folder.mkdir(parents=True, exist_ok=True)
    text = config(n, beta, method)
    run_identity = dict(identity, config_sha256=hashlib.sha256(text.encode()).hexdigest())
    stamp = folder/'execution.json'
    if stamp.exists():
        old = json.loads(stamp.read_text())
        if any(old.get(key) != value for key,value in run_identity.items()):
            raise RuntimeError(f'Existing run inputs differ: {folder}. Choose a new output directory.')
        if old['returncode'] != 0 or old['timed_out']:
            raise RuntimeError(f'Previous run was incomplete: {folder}. Retain its outputs and use a new directory.')
        if not (folder/'output/solver_report.csv').is_file():
            raise RuntimeError(f'Completed run is missing its solver report: {folder}')
        return old
    if any(folder.iterdir()):
        raise RuntimeError(f'Unrecorded files already exist: {folder}. Choose a new output directory.')
    initial = folder.parent/'initial.txt'
    if not initial.exists():
        np.savetxt(initial, cell_averages(n, 0), fmt='%.17g')
    elif not np.array_equal(np.loadtxt(initial), cell_averages(n, 0)):
        raise RuntimeError(f'Initial condition has changed: {initial}')
    (folder/'sim.txt').write_text(text, encoding='utf-8')
    start = time.perf_counter()
    with (folder/'run.log').open('w', encoding='utf-8') as log:
        try:
            completed = subprocess.run([str(args.solver), '.'], cwd=folder, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            returncode, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timed_out = None, True
    result = dict(run_identity, seconds=time.perf_counter()-start,
                  returncode=returncode, timed_out=timed_out)
    stamp.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if returncode != 0:
        raise RuntimeError(f'Incomplete {method}, beta={beta:g}, N={n}: see {folder}/run.log')
    return result


def discrete_reference(n, steps, beta):
    """Independent causal elimination is used as an accuracy reference only."""
    u, t, maximum = cell_averages(n, 0), 0., 0.
    profiles = {}
    xtol, rtol = ((5e-324, 8.881784197001252e-16) if beta >= 1000 else (5e-15, 1e-14))
    for dt in steps:
        coefficient = dt/(LENGTH/n)
        old = u.copy()
        incoming = scalar_flow(LEFT)
        for i in range(n):
            rhs = old[i]+coefficient*incoming
            def residual(s):
                return s+coefficient*scalar_flow(s)-rhs
            u[i] = brentq(residual, 0., 1., xtol=xtol, rtol=rtol)
            maximum = max(maximum, abs(residual(u[i])))
            incoming = scalar_flow(u[i])
        t += dt
        for target in (T_END/2, T_END):
            if abs(t-target) < 1e-7:
                profiles[target] = u.copy()
    if maximum >= 1e-12:
        raise RuntimeError(f'Independent reference residual {maximum:g} exceeds 1e-12')
    if set(profiles) != {T_END/2, T_END}:
        raise RuntimeError('The prescribed report times were not reached')
    return profiles, maximum


def collect(args, n, beta, execution):
    folder = args.output/f'beta_{beta:g}'/f'N{n:05d}'
    expected_dt = beta*LENGTH/n
    expected_steps = round(T_END/expected_dt)
    steps, work = {}, []
    for method in ('SRDM','CRIS'):
        report = read_rows(folder/method/'output/solver_report.csv')
        if len(report) != expected_steps or any(row['converged'] != '1' for row in report):
            raise RuntimeError('A rejected or missing step prevents a common-time-grid accuracy comparison')
        steps[method] = [float(row['DT']) for row in report]
        if not np.allclose(steps[method], expected_dt, rtol=0, atol=1e-10):
            raise RuntimeError('The solver did not follow the prescribed fixed time grid')
        if abs(sum(steps[method])-T_END) >= 1e-7:
            raise RuntimeError('The prescribed final horizon was not reached')
        counts = {'N_J':sum(int(r['NLNSTEPS']) for r in report),
                  'N_K':sum(int(r['LINSTEPS']) for r in report),
                  'N_R':sum(int(r['NFEVAL']) for r in report)}
        work.append(dict(beta=beta, N=n, method=method, accepted_steps=len(report),
                         rejected_steps=0, **counts, W=sum(counts.values()),
                         seconds=execution[method]['seconds']))
    if steps['SRDM'] != steps['CRIS']:
        raise RuntimeError('Paired time grids differ')
    reference, maximum = discrete_reference(n, steps['SRDM'], beta)
    rows, fields = [], {}
    for t in (T_END/2, T_END):
        analytic, discrete = cell_averages(n, t), reference[t]
        disc_norm = norms(discrete-analytic)
        for method in ('SRDM','CRIS'):
            u = read_state(folder/method/f'output/saturation_{int(t)}.bin')
            if len(u) != n:
                raise ValueError('Unexpected solution length')
            total, difference = norms(u-analytic), norms(u-discrete)
            row = dict(beta=beta, N=n, h=LENGTH/n, dt=expected_dt, time=t, method=method,
                       saturation_min=float(min(u)), saturation_max=float(max(u)))
            for norm in disc_norm:
                row.update({f'total_{norm}':total[norm], f'discretization_{norm}':disc_norm[norm],
                            f'vs_discrete_{norm}':difference[norm],
                            f'vs_discrete_over_discretization_{norm}':difference[norm]/disc_norm[norm]})
            rows.append(row)
            fields[f'{method}_{int(t)}'] = u
        fields[f'analytic_{int(t)}'], fields[f'discrete_{int(t)}'] = analytic, discrete
    np.savez_compressed(folder/'reference_and_profiles.npz', x=(np.arange(n)+.5)*LENGTH/n, **fields)
    (folder/'reference_checks.json').write_text(json.dumps({'maximum_original_residual':maximum}, indent=2)+'\n')
    write_rows(folder/'errors.csv', rows)
    write_rows(folder/'work.csv', work)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, required=True, help='Built tp_transport executable')
    parser.add_argument('--model', type=Path, default=MODEL, help='TorchScript local inverse')
    parser.add_argument('--output', type=Path, required=True, help='New directory for generated runs')
    parser.add_argument('--runtime-dir', type=Path, action='append', default=[],
                        help='Runtime library directory; repeat for LibTorch, MKL and OpenMP')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--beta', type=float, nargs='+', default=[1,10,100,1000],
                        help='dt/h at unit advective velocity')
    parser.add_argument('--meshes', type=int, nargs='+', help='Override meshes for one beta only')
    parser.add_argument('--timeout', type=float, default=900, help='Seconds per native solver run')
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('threads and timeout must be positive')
    if args.meshes is not None and len(args.beta) != 1:
        parser.error('--meshes requires exactly one --beta')
    if any(not math.isfinite(b) or b <= 0 for b in args.beta):
        parser.error('beta must be finite and positive')
    jobs = []
    for beta in args.beta:
        meshes = args.meshes if args.meshes is not None else PUBLISHED_MESHES.get(beta)
        if not meshes:
            parser.error('Supply --meshes for beta values outside the published cohorts')
        for n in sorted(set(meshes)):
            half_steps = T_END*n/(2*beta*LENGTH)
            if n <= 0 or n % 4 or abs(half_steps-round(half_steps)) > 1e-9 or half_steps < 1:
                parser.error('Meshes must align the initial shock and both report times: N/4 and N/(10*beta) must be integers')
            jobs.append((beta,n))
    args.solver, args.model, args.output = (p.resolve() for p in (args.solver,args.model,args.output))
    if not args.solver.is_file() or not args.model.is_file():
        parser.error('The solver executable and model must exist')
    for path in args.runtime_dir:
        if not path.is_dir():
            parser.error(f'Runtime directory does not exist: {path}')
    check_entropy()
    env = os.environ.copy()
    runtime = os.pathsep.join(str(p.resolve()) for p in args.runtime_dir)
    if runtime:
        for variable in (['PATH'] if os.name == 'nt' else ['PATH','LD_LIBRARY_PATH']):
            env[variable] = runtime+os.pathsep+env.get(variable,'')
    for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[variable] = str(args.threads)
    identity = dict(executable_sha256=sha(args.solver), checkpoint_sha256=sha(args.model),
                    thread_count=args.threads, runner_sha256=sha(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'manifest.json'
    manifest = dict(identity, platform=platform.platform(), solver=str(args.solver), model=str(args.model),
                    shock_speed=shock_speed(), domain=[0,LENGTH], final_time=T_END,
                    reference='Exact analytic cell averages and independently solved discrete balances')
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(k) != v for k,v in identity.items()):
            raise RuntimeError('Existing campaign identity differs; choose a new output directory')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8')
    staged_model = args.output/'inputs/model.pt'
    staged_model.parent.mkdir(exist_ok=True)
    if staged_model.exists() and sha(staged_model) != identity['checkpoint_sha256']:
        raise RuntimeError('The staged model changed; choose a new output directory')
    if not staged_model.exists():
        shutil.copyfile(args.model, staged_model)
    for beta,n in jobs:
        execution = {}
        for method in ('SRDM','CRIS'):
            print(f'Running beta={beta:g}, N={n}, {method}', flush=True)
            execution[method] = run_native(args,n,beta,method,identity,env)
        rows = collect(args,n,beta,execution)
        final = next(r for r in rows if r['method']=='CRIS' and r['time']==T_END)
        print(json.dumps(dict(beta=beta,N=n,E_inv=final['vs_discrete_L2'],
                              E_disc=final['discretization_L2'],E_tot=final['total_L2'],
                              ratio=final['vs_discrete_over_discretization_L2'])), flush=True)
        for name in ('errors.csv','work.csv'):
            combined = []
            for path in sorted(args.output.glob(f'beta_*/N*/{name}')):
                combined.extend(read_rows(path))
            write_rows(args.output/name,combined)
    if sha(args.model) != identity['checkpoint_sha256'] or sha(staged_model) != identity['checkpoint_sha256']:
        raise RuntimeError('The checkpoint changed during the campaign')
    print(f'Completed {len(jobs)} requested mesh pairs. Results: {args.output}/errors.csv')


if __name__ == '__main__':
    main()
