"""Prepare and run seven paired transport benchmarks, compare solutions and plot results."""
import argparse
import json
import os
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
from benchmark_inputs import ROOT, BASE, KEYS, sources, sha, read, write

DEFAULT=BASE/'paired_benchmarks'
CASES=('1D','2DSPE10_Layer50','2DSPE10_Layer75','2DSPE10_Layer84',
       'Norne_5SpotBase','EDFM','3DSPE10_5SpotBase')
PAIR_DIFFERENCES={'MODE','NEWTONUPDATER'}


def config(path):
    result={}
    for raw in path.read_text().splitlines():
        fields=raw.split('#',1)[0].split(maxsplit=1)
        if len(fields)!=2:continue
        key,value=fields;key=key.upper()
        if key in result:raise ValueError(f'Duplicate setting {key}: {path}')
        result[key]=value
    return result


def validate_controls(c, method):
    """Reject physics/globalization drift before an executable is started."""
    fixed={'MODE':method,'NEWTONUPDATER':'STANDARD' if method=='CRIS' else 'LINESEARCH',
           'REFRESH_TRANSPORT_PRECONDITIONER':'1','LINEAR_SOLVE_POLICY':'CONTINUE_ON_BUDGET',
           'DISPERSION':'HOMOGENEOUS'}
    for key,value in fixed.items():
        if c.get(key)!=value:raise ValueError(f'{method}: require {key} {value}')
    numeric={'N_W':2.,'N_NW':2.,'NG':0.,'DISPX':0.,'DISPY':0.,'DISPZ':0.,
             'PRESSURE_LINMAX':100.,'LINTOL':1e-2,'LINTOL_MIN':1e-6,
             'DUTOL':1e-6,'RTOL':1e-6}
    for key,value in numeric.items():
        if float(c.get(key,'nan'))!=value:raise ValueError(f'{method}: require {key} {value}')
    if not math.isclose(float(c['MU_NW'])/float(c['MU_W']),.4):
        raise ValueError('Regular model requires mobility ratio 0.4')
    for key in ('LINMAX','MAX_STEPS_NEWTON'):
        if int(c[key])<1:raise ValueError(f'{key} must be positive')
    for key in ('DT_INIT','DT_MIN','DT_MAX','T_END','REPORT_FREQ'):
        if not math.isfinite(float(c[key])) or float(c[key])<=0:
            raise ValueError(f'{key} must be finite and positive')
    # The native parser reads one whitespace-delimited path and prepends the
    # case folder. Absolute paths, even quoted, are not its input-file syntax.
    for key in (KEYS | {'CRIS_MODEL'}) & c.keys():
        if Path(c[key]).is_absolute() or any(char.isspace() for char in c[key]):
            raise ValueError(f'{key} must be a whitespace-free, case-relative path')


def input_copy(output, source, digest):
    """Short, content-addressed path shared by paired cases."""
    return output/'inputs'/(digest[:16]+'.txt')


def copy_inputs(copies):
    import shutil
    for destination,(source,digest) in copies.items():
        if destination.exists():
            if sha(destination)!=digest:raise ValueError(f'Staged input collision/change: {destination}')
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        if sha(destination)!=digest:raise ValueError(f'Input copy verification failed: {destination}')


def stage(args):
    model=BASE/'packages/residual_lambda_0p01_final.pt'
    jobs=[];pending=[];copies={}
    for case,source in sources():
        c=config(source)
        assert float(c['MU_NW'])/float(c['MU_W'])==.4
        assert float(c['N_W'])==float(c['N_NW'])==2 and float(c['NG'])==0
        inputs={}
        for key in KEYS & c.keys():
            path=(source.parent/c[key]).resolve();digest=sha(path);inputs[str(path)]=digest
            destination=input_copy(args.output,path,digest);copies[destination]=(path,digest)
            inputs[str(destination)]=digest;c[key]=str(destination).replace('\\','/')
        for method in ('CRIS','SRDM'):
            folder=args.output/case/method
            controls={**c,'MODE':method,'NEWTONUPDATER':'STANDARD' if method=='CRIS' else 'LINESEARCH',
                      'CRIS_MODEL':os.path.relpath(model,folder).replace('\\','/'),'REFRESH_TRANSPORT_PRECONDITIONER':'1',
                      'LINMAX':str(args.linmax),'PRESSURE_LINMAX':'100','LINTOL':'1e-2',
                      'LINTOL_MIN':'1e-6','LINEAR_SOLVE_POLICY':'CONTINUE_ON_BUDGET'}
            for key in KEYS & c.keys():
                controls[key]=os.path.relpath(c[key],folder).replace('\\','/')
            validate_controls(controls,method)
            text='\n'.join(f'{k} {v}' for k,v in controls.items())+'\n'
            target=folder/'sim.txt'
            if target.exists() and target.read_text()!=text:raise ValueError(f'Existing configuration differs: {target}; choose another output explicitly')
            pending.append((target,text))
            jobs.append(dict(case=case,method=method,model='lambda_0p01' if method=='CRIS' else 'SRDM',
                folder=str(folder),source=str(source),source_sha256=sha(source),config_sha256=None,
                controls=controls,inputs=inputs))
    # Validate the entire proposed campaign before touching an existing stage.
    import hashlib
    for job,(target,text) in zip(jobs,pending):
        payload=text.replace('\n',os.linesep).encode()
        job['config_sha256']=sha(target) if target.exists() else hashlib.sha256(payload).hexdigest()
    manifest=dict(jobs=jobs,executable=str(args.solver),executable_sha256=sha(args.solver),
                  model=str(model),model_sha256=sha(model),thread_count=args.threads,
                  policy='fresh ILU0; CRIS plain Newton; SRDM line search')
    target=args.output/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:raise ValueError('Staged manifest differs; existing evidence preserved')
    copy_inputs(copies)
    for path,text in pending:
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    if not target.exists():target.write_text(json.dumps(manifest,indent=2))
    check(args)
    print(f'Staged {len(jobs)} jobs. No solver launched.')


def checked_manifest(output):
    m=json.loads((output/'manifest.json').read_text())
    pairs={(j['case'],j['method']) for j in m['jobs']}
    expected={(case,method) for case in CASES for method in ('CRIS','SRDM')}
    if len(m['jobs'])!=14 or pairs!=expected:raise ValueError('Require seven distinct CRIS/SRDM pairs')
    if int(m['thread_count'])<1:raise ValueError('Invalid thread count')
    for key in ('executable','model'):
        if sha(Path(m[key]))!=m[key+'_sha256']:raise ValueError(f'{key} changed after staging')
    for job in m['jobs']:
        folder=Path(job['folder'])
        if folder.resolve()!=output.resolve()/job['case']/job['method']:
            raise ValueError('Job folder is outside its declared campaign location; restage after moving a checkout')
        if sha(folder/'sim.txt')!=job['config_sha256']:raise ValueError('Staged configuration changed')
        if config(folder/'sim.txt')!=job['controls']:raise ValueError('Manifest controls differ from staged file')
        validate_controls(job['controls'],job['method'])
        for key in KEYS & job['controls'].keys():
            if str((folder/job['controls'][key]).resolve()) not in job['inputs']:
                raise ValueError(f'Unhashed input: {key}')
        if (folder/job['controls']['CRIS_MODEL']).resolve()!=Path(m['model']).resolve():
            raise ValueError('Job model differs from campaign model')
        for path,digest in job['inputs'].items():
            if sha(Path(path))!=digest:raise ValueError(f'Input changed: {path}')
    for case in CASES:
        controls=[j['controls'] for j in m['jobs'] if j['case']==case]
        if {k:v for k,v in controls[0].items() if k not in PAIR_DIFFERENCES}!={k:v for k,v in controls[1].items() if k not in PAIR_DIFFERENCES}:
            raise ValueError(f'Paired controls differ: {case}')
    return m


def check(args):
    """Read-only stage and result audit; never executes a solver."""
    m=checked_manifest(args.output);states=[]
    print(f'Verified model/executable/config/input hashes; {m["thread_count"]} worker(s).')
    for job in m['jobs']:
        folder=Path(job['folder']);r=folder/'result.json'
        state='staged'
        if r.exists():state='completed' if verified_result(job,m)['completed'] else 'incomplete'
        elif (folder/'stdout.log').exists() or (folder/'output').exists():state='interrupted/unrecorded'
        states.append(state)
        c=job['controls']
        print(f'{job["case"]:22} {job["method"]:4} {state:22} '
              f'dt={c["DT_INIT"]}..{c["DT_MAX"]} end={c["T_END"]} LINMAX={c["LINMAX"]}')
    return states


def output_hashes(folder):
    """Freeze every file consumed by numerical post-processing plus stdout."""
    files=[folder/'stdout.log',*sorted((folder/'output').glob('*.bin')),
           folder/'output/solver_report.csv',folder/'output/well_report.csv']
    return {p.relative_to(folder).as_posix():sha(p) for p in files if p.is_file()}


def verified_result(job, manifest):
    folder=Path(job['folder']);result=json.loads((folder/'result.json').read_text())
    for key,expected in (('config_sha256',job['config_sha256']),
                         ('executable_sha256',manifest['executable_sha256']),
                         ('model_sha256',manifest['model_sha256'])):
        if result.get(key)!=expected:raise ValueError(f'Result provenance mismatch: {folder}/{key}')
    if not result.get('output_sha256'):raise ValueError(f'Missing output hashes: {folder}')
    if output_hashes(folder)!=result['output_sha256']:raise ValueError(f'Run outputs changed: {folder}')
    return result


def archive_startup_failures(output, manifest, archive_name='startup_launch_failure.zip'):
    """Explicitly retry only audited input-open failures with no solver work.

    The caller holds run.lock. Failed records are hash-verified in an archive
    before removing generated files; configurations and successful runs remain.
    Numerical failures are retained, not retried. Interrupted/unrecorded
    attempts still require review before the queue can proceed past them.
    """
    import zipfile
    generated=[];configs=[];folders=[]
    for job in manifest['jobs']:
        folder=Path(job['folder']).resolve()
        if not (folder/'result.json').exists():continue
        result=verified_result(job,manifest)
        if result['completed']:continue
        if 'ASSERTION FAILED: Cannot open file:' not in (folder/'stdout.log').read_text(errors='replace'):
            continue  # preserve any genuine numerical or other startup failure
        zero=('accepted_time','accepted_steps','failed_attempts','newton','krylov','residual_evaluations')
        if any(result.get(key)!=0 for key in zero) or result.get('timed_out'):
            raise ValueError(f'Not a zero-work startup failure: {folder}')
        if list((folder/'output').glob('*.bin')):
            raise ValueError(f'Unexpected simulation fields: {folder}')
        files=[folder/'result.json',folder/'stdout.log',*sorted((folder/'output').glob('*'))]
        for path in files:
            if not path.is_file() or path.is_symlink() or output.resolve() not in path.resolve().parents:
                raise ValueError(f'Unsafe startup recovery target: {path}')
        generated.extend(files);configs.append(folder/'sim.txt');folders.append(folder)
    if not generated:return
    archive=output/archive_name
    records=[output/'manifest.json',*configs,*generated]
    digests={p:sha(p) for p in records}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for path in records:bundle.write(path,path.relative_to(output).as_posix())
    import hashlib
    with zipfile.ZipFile(archive) as bundle:
        for path,digest in digests.items():
            if hashlib.sha256(bundle.read(path.relative_to(output).as_posix())).hexdigest()!=digest:
                raise ValueError(f'Startup archive verification failed: {path}')
    for path in generated:
        if sha(path)!=digests[path]:raise ValueError(f'Startup file changed during recovery: {path}')
        path.unlink()
    for folder in folders:
        if (folder/'output').exists():(folder/'output').rmdir()
    print(f'Archived {len(folders)} zero-work startup failures to {archive}; retrying with unchanged configurations.',flush=True)


def localize_inputs(args):
    """Shorten paths only for unrun/zero-work jobs, preserving completed evidence."""
    import copy
    m=checked_manifest(args.output);updated=copy.deepcopy(m);copies={};changes=[]
    lock=args.output/'run.lock'
    with lock.open('x') as stream:json.dump(dict(pid=os.getpid(),host=platform.node(),action='localize-inputs'),stream)
    try:
        for job in updated['jobs']:
            folder=Path(job['folder']);result_path=folder/'result.json'
            if result_path.exists():
                result=verified_result(job,m)
                if result['completed']:continue
                zero=('accepted_time','accepted_steps','failed_attempts','newton','krylov','residual_evaluations')
                if any(result.get(k)!=0 for k in zero) or result.get('timed_out') or 'ASSERTION FAILED: Cannot open file:' not in (folder/'stdout.log').read_text(errors='replace'):
                    raise ValueError(f'Cannot restage a numerical/unknown failure: {folder}')
            elif (folder/'stdout.log').exists() or (folder/'output').exists():
                raise ValueError(f'Cannot restage an unrecorded attempt: {folder}')
            controls=dict(job['controls'])
            for key in KEYS & controls.keys():
                source=(folder/controls[key]).resolve();digest=job['inputs'][str(source)]
                destination=input_copy(args.output,source,digest)
                copies[destination]=(source,digest);job['inputs'][str(destination)]=digest
                controls[key]=os.path.relpath(destination,folder).replace('\\','/')
            if controls==job['controls']:continue
            validate_controls(controls,job['method']);job['controls']=controls
            changes.append((job,folder/'sim.txt','\n'.join(f'{k} {v}' for k,v in controls.items())+'\n'))
        if not changes:
            print('No input-path changes needed.');return
        copy_inputs(copies)
        # This archive retains the preceding manifest/configs and only removes
        # proven zero-work failed products, never successful simulation fields.
        archive_startup_failures(args.output,m,'startup_native_open_failure.zip')
        for job,path,text in changes:
            path.write_text(text);job['config_sha256']=sha(path)
        (args.output/'manifest.json').write_text(json.dumps(updated,indent=2))
        checked_manifest(args.output)
        print(f'Localized inputs for {len(changes)} unrun jobs; completed configurations/results unchanged.',flush=True)
    finally:lock.unlink()


def native_input_preflight(job):
    """Use the Windows C runtime, matching narrow native file opening semantics."""
    if os.name!='nt':return
    import ctypes
    crt=ctypes.CDLL('ucrtbase')
    crt.fopen.argtypes=[ctypes.c_char_p,ctypes.c_char_p];crt.fopen.restype=ctypes.c_void_p
    crt.fclose.argtypes=[ctypes.c_void_p]
    previous=Path.cwd()
    try:
        os.chdir(job['folder'])
        for key in KEYS & job['controls'].keys():
            path='./'+job['controls'][key]
            stream=crt.fopen(path.encode('mbcs'),b'rb')
            if not stream:raise ValueError(f'Native input preflight failed: {job["case"]}/{job["method"]} {key}: {path}')
            crt.fclose(stream)
    finally:os.chdir(previous)


def run(args):
    m=checked_manifest(args.output)
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]=str(m['thread_count'])
    env.pop('TPT_SAVE_FIRST_ATTEMPT',None)
    if args.dll_dir:env['PATH']=str(args.dll_dir)+os.pathsep+env['PATH']
    if args.dll_dir and not args.dll_dir.is_dir():raise ValueError('Runtime directory does not exist')
    # An atomic campaign lock prevents two launchers writing the same reports.
    lock=args.output/'run.lock'
    with lock.open('x') as stream:json.dump(dict(pid=os.getpid(),host=platform.node()),stream)
    power=None
    try:
        if os.name=='nt':
            import ctypes
            power=ctypes.windll.kernel32.SetThreadExecutionState
            if not power(0x80000001):print('Warning: could not inhibit idle sleep',flush=True)
        if getattr(args,'retry_startup_failures',False):archive_startup_failures(args.output,m)
        # Validate every queued job before spending time on the first solve.
        for job in m['jobs'][:getattr(args,'max_jobs',None)]:
            if not (Path(job['folder'])/'result.json').exists():native_input_preflight(job)
        run_jobs(args,m,env)
    finally:
        if power is not None:power(0x80000000)
        lock.unlink()


def run_jobs(args,m,env):
    work=[]
    for job in m['jobs'][:getattr(args,'max_jobs',None)]:
        folder=Path(job['folder']);result_path=folder/'result.json'
        if result_path.exists():
            prior=verified_result(job,m)
            if prior['completed']:
                work.append(prior);write(args.output/'work.csv',work);continue
            if getattr(args,'retry_startup_failures',False):
                print(f'Retaining incomplete result without rerunning: {folder}',flush=True)
                work.append(prior);write(args.output/'work.csv',work);continue
            raise ValueError(f'{folder} has an incomplete attempt. Inspect it and choose a fresh output; it will not be overwritten.')
        if (folder/'stdout.log').exists() or (folder/'output').exists():raise ValueError(f'{folder} has an interrupted attempt without a final record; preserve it before retrying')
        native_input_preflight(job)
        write_status(args.output,m,work,active=f'{job["case"]}/{job["method"]}')
        start=time.perf_counter();code=None;timeout=False
        with (folder/'stdout.log').open('w') as log:
            # The native reader concatenates this argument with relative input
            # paths without normalizing them. Use the case as cwd to avoid long
            # absolute-plus-parent paths exceeding Windows stream path limits.
            try:code=subprocess.run([m['executable'],'.'],cwd=folder,env=env,stdout=log,stderr=log,timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:timeout=True
        wall_seconds=time.perf_counter()-start
        report=read(folder/'output/solver_report.csv');text=(folder/'stdout.log').read_text(errors='replace')
        reached=sum(float(r['DT']) for r in report if r['converged']=='1')
        target=float(job['controls']['T_END'])
        completed=code==0 and 'Simulation Completed Successfully' in text and abs(reached-target)<=max(1e-5,target*1e-8)
        result=dict(case=job['case'],model=job['model'],method=job['method'],completed=completed,
            returncode=code,timed_out=timeout,wall_seconds=wall_seconds,accepted_time=reached,
            accepted_steps=sum(r['converged']=='1' for r in report),
            failed_attempts=sum(r['converged']!='1' for r in report),
            model_sha256=m['model_sha256'],output_sha256=output_hashes(folder),
            host=platform.node(),platform=platform.platform(),python=sys.version,
            thread_count=m['thread_count'],runtime_dir=str(args.dll_dir) if args.dll_dir else None,
            config_sha256=job['config_sha256'],executable_sha256=m['executable_sha256'],
            **{name:sum(int(r.get(key,0)) for r in report) for name,key in [('newton','NLNSTEPS'),('krylov','LINSTEPS'),('residual_evaluations','NFEVAL'),('linear_tolerance_misses','LINEAR_TOL_MISSES'),('budget_continuations','BUDGET_CONTINUATIONS'),('linear_breakdowns','LINEAR_BREAKDOWNS')]})
        result_path.write_text(json.dumps(result,indent=2));work.append(result);write(args.output/'work.csv',work)
        print(result,flush=True)
    write_status(args.output,m,work)


def write_status(output,m,work,active=None):
    # Include completed jobs later in queue order, not only the traversed
    # prefix. This matters when earlier startup failures are being retried.
    records={(r['case'],r['method']):r for r in work}
    for job in m['jobs']:
        key=(job['case'],job['method']);path=Path(job['folder'])/'result.json'
        if key not in records and path.exists():records[key]=json.loads(path.read_text())
    successful={key for key,r in records.items() if r['completed']}
    complete_cases=[case for case in CASES if all((case,method) in successful for method in ('CRIS','SRDM'))]
    (output/'status.json').write_text(json.dumps(dict(active=active,completed=len(successful),
        finished_attempts=len(records),failed=sum(not r['completed'] for r in records.values()),total=len(m['jobs']),
        completed_cases=complete_cases,total_cases=len(CASES),
        all_successful=len(successful)==len(m['jobs']))))


def publication_runs(args, m):
    """Require a completed result for every member of the paired campaign."""
    selected={}
    for job in m['jobs']:
        result=verified_result(job,m)
        key=(job['case'],job['method'])
        if result['completed']:
            selected[key]=dict(folder=Path(job['folder']),result=result,provenance='corrected')
            continue
        raise ValueError(f'Paired run incomplete: {key}')
    return selected


def references(args):
    m=checked_manifest(args.output)
    selected=publication_runs(args,m)
    import numpy as np
    import reference_regular_benchmarks as reference
    import summarize_regular_benchmarks as summarize
    # Reuse the existing independent-reference implementations, restricted to CRIS.
    reference.DEST=args.output
    cris=[j for j in m['jobs'] if j['method']=='CRIS']
    for job in cris:
        case=job['case'];folder=Path(job['folder'])
        if not json.loads((folder/'result.json').read_text())['completed']:raise ValueError(f'CRIS incomplete: {case}')
        if case=='1D':
            # The existing scalar recurrence supports the canonical 1D case.
            summarize.DEST=args.output;summarize.main(jobs=[job])
        else:
            saved_argv=sys.argv
            try:
                sys.argv=['reference_regular_benchmarks','--case',case,'--model','lambda_0p01'];reference.main()
            finally:sys.argv=saved_argv
            for method in ('CRIS','SRDM'):
                run_folder=selected[(case,method)]['folder'];wells={}
                p,*_=reference.graph(run_folder,producer_completions=wells)
                values=[{'Time':0.,**{name:0. for name in wells}}]
                for path in sorted((run_folder/'output').glob('saturation_*.bin'),key=lambda p:int(p.stem.split('_')[-1])):
                    u=summarize.state(path)
                    values.append({'Time':int(path.stem.split('_')[-1]),**{name:float(np.dot(w['rates'],reference.flow(u[w['cells']]))/sum(w['rates'])) for name,w in wells.items()}})
                write(args.output/case/(method.lower()+'_physical_watercut.csv'),values)
    products=[args.output/'accuracy.csv']
    for job in cris:
        if job['case']=='1D':continue
        products.append(args.output/job['case']/'reference_accuracy.csv')
        products.extend(args.output/job['case']/(method.lower()+'_physical_watercut.csv') for method in ('CRIS','SRDM'))
    (args.output/'reference_products.json').write_text(json.dumps({
        'simulation_manifest_sha256':sha(args.output/'manifest.json'),
        'files':{p.relative_to(args.output).as_posix():sha(p) for p in products}},indent=2))
    print('Independent CRIS references and physical producer water cuts updated.')


def publish(args):
    m=checked_manifest(args.output)
    selected=publication_runs(args,m)
    products=json.loads((args.output/'reference_products.json').read_text())
    if products['simulation_manifest_sha256']!=sha(args.output/'manifest.json'):raise ValueError('References belong to another campaign')
    for name,digest in products['files'].items():
        if sha(args.output/name)!=digest:raise ValueError(f'Reference product changed: {name}')
    import importlib.util
    path=ROOT/'Figures/fig4_transport/build_fig4_transport.py'
    spec=importlib.util.spec_from_file_location('regular_figure',path);figure=importlib.util.module_from_spec(spec);spec.loader.exec_module(figure)
    cases=json.loads((ROOT/'Figures/fig4_transport/data/publication_sources.json').read_text())['cases']
    for case in cases:
        slug=case['slug']
        for method,key in [('CRIS','cris'),('SRDM','srdm')]:
            record=selected[(slug,method)];folder=record['folder'];r=record['result']
            if not r['completed']:raise ValueError(f'Cannot publish a full-run comparison: {slug}/{method} incomplete')
            case[key]=str(folder/'output');case[key+'_seconds']=r['wall_seconds']
            case[key+'_provenance']=record['provenance']
            if slug!='1D':case[key+'_watercut']=str(args.output/slug/(method.lower()+'_physical_watercut.csv'))
        accuracy=args.output/('accuracy.csv' if slug=='1D' else slug+'/reference_accuracy.csv')
        if not accuracy.exists():raise FileNotFoundError('Run the references action first: '+str(accuracy))
        case['accuracy']=str(accuracy)
        if 'image' in case:case['image']=str(case['image'])
    # Repository-relative paths keep saved-evidence redraw independent of the host.
    for case in cases:
        for key in ('cris','srdm','accuracy','image','cris_watercut','srdm_watercut'):
            if key in case:case[key]=Path(case[key]).resolve().relative_to(ROOT).as_posix()
    manifest=ROOT/'Figures/fig4_transport/data/publication_sources.json'
    manifest.write_text(json.dumps(dict(all_completed=True,all_corrected_completed=all(v['provenance']=='corrected' for v in selected.values()),
        scope='Completed paired transport runs.',
        cases=cases,simulation_manifest_sha256=sha(args.output/'manifest.json')),indent=2))
    subprocess.run([sys.executable,str(path),'--sources',str(manifest)],check=True,cwd=ROOT)
    print('Transport comparison figure updated.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['stage','check','run','localize-inputs','references','publish'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--solver',type=Path,default=ROOT/('build/transport/Release/tp_transport.exe' if os.name=='nt' else 'build/transport/tp_transport'))
    parser.add_argument('--dll-dir',type=Path,help='Runtime DLL directory when not already on PATH')
    parser.add_argument('--linmax',type=int,default=7)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--timeout',type=float,default=86400,help='Per-process seconds; incomplete results are retained')
    parser.add_argument('--max-jobs',type=int,help='Run at most this many campaign entries; completed jobs are reused')
    parser.add_argument('--retry-startup-failures',action='store_true',help='Archive and retry verified zero-work input-open failures; never retries numerical failures')
    args=parser.parse_args();args.output=args.output.resolve();args.solver=args.solver.resolve()
    if args.linmax<1 or args.threads<1 or args.timeout<=0:parser.error('Budgets must be positive')
    if args.max_jobs is not None and args.max_jobs<1:parser.error('--max-jobs must be positive')
    globals()[args.action.replace('-','_')](args)


if __name__=='__main__':main()
