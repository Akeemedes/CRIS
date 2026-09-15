"""Numerical summaries only: independent 1D and archived physical-SRDM comparisons."""
import csv
import json
from pathlib import Path
import numpy as np
from benchmark_inputs import ROOT, FIG, DEST, read, write


def state(path):
    with path.open("rb") as f:
        n = int(np.fromfile(f, dtype="<u8", count=1)[0])
        x = np.fromfile(f, dtype="<f8")
    if len(x)!=n or not np.isfinite(x).all(): raise ValueError(f"Invalid state {path}")
    return x


def measures(ref, pred):
    e=pred-ref; a=abs(e)
    return dict(rmse=float(np.sqrt(np.mean(e*e))),mae=float(a.mean()),q99=float(np.quantile(a,.99)),
                max_error=float(a.max()),minimum=float(pred.min()),maximum=float(pred.max()),
                oob=int(np.count_nonzero((pred<0)|(pred>1))))


def main(jobs=None):
    if jobs is None:jobs=json.loads((DEST/"manifest.json").read_text())["jobs"]
    completed={(r["case"],r["model"]) for r in read(DEST/"work.csv") if r["completed"]=="True"}
    results=[]; cache={}
    for job in jobs:
        key=(job["case"],job["model"])
        if key not in completed: continue
        out=Path(job["folder"])/"output"
        steps=tuple(float(r["DT"]) for r in read(out/"solver_report.csv") if r["converged"]=="1")
        if job["case"]=="1D":
            if steps not in cache:
                # Resolve the exact native staged input, not the source template's
                # differently rooted path. Older records keep the source fallback.
                staged=Path(job['folder'])/'sim.txt'
                source=staged if staged.exists() else Path(job['source'])
                initial=Path(next(line.split(maxsplit=1)[1] for line in source.read_text().splitlines() if line.startswith("INIT_FILE ")))
                u=np.loadtxt(initial if initial.is_absolute() else source.parent/initial); t=0.; profiles={}
                max_resid=0.
                for dt in steps:
                    old=u.copy(); incoming=1.
                    for i in range(len(u)):
                        lo,hi=0.,1.
                        for _ in range(56):
                            s=(lo+hi)/2;f=s*s/(s*s+.4*(1-s)**2)
                            if s-old[i]+dt*(f-incoming)>0:hi=s
                            else:lo=s
                        u[i]=(lo+hi)/2; f=u[i]*u[i]/(u[i]*u[i]+.4*(1-u[i])**2)
                        max_resid=max(max_resid,abs(u[i]-old[i]+dt*(f-incoming)));incoming=f
                    t+=dt
                    if abs(t-round(t))<.01:profiles[int(round(t))]=u.copy()
                cache[steps]=(profiles,max_resid)
            profiles,max_resid=cache[steps]
            for t,ref in profiles.items():
                path=out/f"saturation_{t}.bin"
                if path.exists():results.append(dict(case=job["case"],model=job["model"],time=t,
                    reference="independent_implicit_same_time_grid",reference_residual_inf=max_resid,
                    same_time_grid=True,**measures(ref,state(path))))
        else:
            reference=FIG/"benchmark_spe10_srdm/cases"/job["case"]/"SRDM/output"
            reports=read(reference/"solver_report.csv")
            if not reports: raise ValueError(f"Missing physical SRDM report {reference}")
            refsteps=tuple(float(r.get("DT",r.get("DT_TRIAL"))) for r in reports if r["converged"]=="1")
            # Canonical index confirms complete matching-physics SRDM runs.
            for path in out.glob("saturation_*.bin"):
                rp=reference/path.name
                if not rp.exists():continue
                t=int(path.stem.split("_")[-1])
                results.append(dict(case=job["case"],model=job["model"],time=t,
                    reference="archived_physical_SRDM",same_time_grid=steps==refsteps,
                    **measures(state(rp),state(path))))
    write(DEST/"accuracy.csv",results)
    final=[]
    for key in sorted(completed):
        rows=[r for r in results if (r["case"],r["model"])==key]
        if rows:final.append(max(rows,key=lambda r:r["time"]))
    write(DEST/"final_accuracy.csv",final)
    print(json.dumps(final,indent=2),flush=True)


if __name__=="__main__":main()
