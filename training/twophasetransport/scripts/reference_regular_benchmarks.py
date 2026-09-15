"""Independent fixed-pressure DAG references; no plots and no solver changes."""
import argparse
import hashlib
import json
import re
import time
import numpy as np
from benchmark_inputs import DEST, read, write
from summarize_regular_benchmarks import state, measures
from pathlib import Path


def config(path):
    return {p[0].upper():p[1] for line in path.read_text().splitlines()
            if len(p:=line.split(maxsplit=1))==2 and not line.startswith("#")}


def graph(folder, expected_nnw=2., producer_completions=None):
    c=config(folder/"sim.txt")
    assert float(c["MU_NW"])/float(c["MU_W"])==.4
    assert float(c["N_W"])==2 and float(c["N_NW"])==expected_nnw and float(c["NG"])==0
    assert c["INIT_FIELD"]=="HOMOGENEOUS" and float(c["INIT_VALUE"])==0
    assert c["DISPERSION"]=="HOMOGENEOUS" and all(float(c[k])==0 for k in ("DISPX","DISPY","DISPZ"))
    load=lambda key,typ=float:np.loadtxt(folder/c[key],dtype=typ)
    p=state(folder/"output/pressure_0.bin"); n=len(p)
    rg=float(c.get("RHO",62.4))/144.
    if c.get("MESH")=="UNSTRUCTURED":
        a,b=load("CL_FILE",int).reshape(-1,2).T
        trans=load("TRANS_FILE");vol=load("VOL_FILE");depth=load("DEPTH_FILE")
        mapping=None
        if "GLOBAL_IDX_FILE" in c:
            global_ids=load("GLOBAL_IDX_FILE",int)
            mapping={int(g):i for i,g in enumerate(global_ids)}
        K=None
    else:
        nx,ny,nz=(int(c[k]) for k in ("NX","NY","NZ"))
        dx,dy,dz=(float(c[k]) for k in ("DX","DY","DZ"))
        dims=(dx,dy,dz);vol=np.full(n,dx*dy*dz)
        depth=(np.arange(n)//(nx*ny)+.5)*dz
        K=[load(k+"_FILE") if k+"_FILE" in c else np.full(n,float(c.get(k,1))) for k in ("KX","KY","KZ")]
        grid=np.arange(n).reshape(nz,ny,nx); aa=[];bb=[];tt=[]
        for dim,axis in enumerate((2,1,0)):
            left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
            ai=grid[tuple(left)].ravel();bi=grid[tuple(right)].ravel()
            area=dx*dy*dz/dims[dim]
            tr=area/(.5*dims[dim]/K[dim][ai]+.5*dims[dim]/K[dim][bi])
            aa.append(ai);bb.append(bi);tt.append(tr)
        a=np.concatenate(aa);b=np.concatenate(bb);trans=np.concatenate(tt);mapping=None
    q=.001127*trans*(p[a]-p[b]+rg*(depth[b]-depth[a]))
    up=np.where(q>0,a,b);down=np.where(q>0,b,a);mask=q!=0
    up,down,q=up[mask],down[mask],abs(q[mask])
    potential=p-rg*depth
    if np.any(potential[up]<=potential[down]):raise ValueError("Flow graph not strictly potential-ordered")
    injection=np.zeros(n);production=np.zeros(n)
    def cell(i,j,k):
        global_id=i+int(c["NX"])*(j+int(c["NY"])*k)
        return mapping[global_id] if mapping is not None else global_id
    for block in re.split(r"(?m)^WELL ",(folder/c["WELLS_FILE"]).read_text())[1:]:
        schedules=re.findall(r"([\d.eE+\-]+) ON BHP ([\d.eE+\-]+)",block)
        if len(schedules)!=1 or float(schedules[0][0])!=0:raise ValueError("Nonconstant well schedule")
        if re.search(r"(?m)^CELLS\s*$",block):
            cells=np.fromstring(re.search(r"(?ms)^CELLS\s*\n(.*?)^END",block)[1],sep=" ",dtype=int)
            datum=int(re.search(r"DATUMCELL\s+(\d+)",block)[1])
        else:
            i,j=map(int,re.search(r"(?m)^IJ\s+(\d+)\s+(\d+)",block).groups())
            ranges=re.findall(r"(?m)^K\s+(\d+)\s+(\d+)",block)
            if len(ranges)!=1:raise ValueError("Multiple well segments require explicit parser extension")
            k0,k1=map(int,ranges[0]);cells=np.array([cell(i,j,k) for k in range(k0,k1+1)])
            datum=cell(i,j,int(re.search(r"DATUMK\s+(\d+)",block)[1]))
        match=re.search(r"(?ms)^WI\s*\n(.*?)^END",block)
        if match:
            wi=np.fromstring(match[1],sep=" ")
        else:
            if K is None:raise ValueError("Missing unstructured WI")
            rw=float(re.search(r"RW\s+([\d.eE+\-]+)",block)[1]);skin=float(re.search(r"SKIN\s+([\d.eE+\-]+)",block)[1])
            wi=2*np.pi*np.sqrt(K[0][cells]*K[1][cells])*dz/(np.log(.14*np.sqrt(dx*dx+dy*dy)/rw)+skin)
        if len(wi)!=len(cells):raise ValueError("WI length mismatch")
        rates=.001127*wi*(float(schedules[0][1])+rg*(depth[cells]-depth[datum])-p[cells])
        if producer_completions is not None and np.any(rates<0):
            name=block.splitlines()[0].strip()
            entry=producer_completions.setdefault(name,dict(cells=[],rates=[]))
            entry['cells'].extend(cells[rates<0].tolist());entry['rates'].extend((-rates[rates<0]).tolist())
        np.add.at(injection,cells,np.maximum(rates,0));np.add.at(production,cells,np.maximum(-rates,0))
    outgoing=np.bincount(up,weights=q,minlength=n)+production
    incoming=np.bincount(down,weights=q,minlength=n)+injection
    balance=float(max(abs(incoming-outgoing)))
    edgeorder=np.argsort(up);starts=np.r_[0,np.cumsum(np.bincount(up,minlength=n))]
    level=np.zeros(n,dtype=int)
    for node in np.argsort(-potential):
        np.maximum.at(level,down[edgeorder[starts[node]:starts[node+1]]],level[node]+1)
    def groups(labels):
        order=np.argsort(labels);count=np.bincount(labels,minlength=int(level.max())+1)
        return np.split(order,np.cumsum(count)[:-1])
    return p,vol,up,down,q,outgoing,injection,groups(level),groups(level[up]),balance


def flow(u):
    x=np.clip(u,0,1);return x*x/(x*x+.4*(1-x)**2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--case",required=True)
    parser.add_argument('--model',help='Restrict reference evaluation to one model')
    parser.add_argument("--allow-partial",action="store_true");args=parser.parse_args()
    jobs=[j for j in json.loads((DEST/"manifest.json").read_text())["jobs"] if j["case"]==args.case and (args.model is None or j['model']==args.model)]
    if not jobs:raise ValueError("Unknown case")
    start=time.perf_counter();base=Path(jobs[0]["folder"])
    p,vol,up,down,q,outgoing,inj,groups,edgegroups,balance=graph(base)
    print(dict(case=args.case,cells=len(p),levels=len(groups),pressure_balance_inf=balance),flush=True)
    cache={};rows=[];checks=[]
    for job in jobs:
        folder=Path(job["folder"]);out=folder/"output"
        completed={(r["case"],r["model"]) for r in read(DEST/"work.csv") if r["completed"]=="True"}
        if (args.case,job["model"]) not in completed:
            if args.allow_partial:continue
            raise ValueError("Native model run not complete")
        if not np.array_equal(p,state(out/"pressure_0.bin")):raise ValueError("Pressure fields differ")
        reports=read(out/"solver_report.csv");steps=tuple(float(r["DT"]) for r in reports if r["converged"]=="1")
        if abs(sum(steps)-float(job["controls"]["T_END"]))>max(1e-5,sum(steps)*1e-6):raise ValueError("Incomplete run")
        if steps not in cache:
            signature=hashlib.sha256(repr(steps).encode()+p.tobytes()+vol.tobytes()+q.tobytes()+outgoing.tobytes()+inj.tobytes()).hexdigest()
            cachepath=DEST/args.case/("reference_"+signature[:16]+".npz")
            if cachepath.exists():
                with np.load(cachepath) as stored:
                    profiles={int(k[1:]):stored[k] for k in stored.files if k.startswith("t")}
                    worst=float(stored["residual_inf"])
                cache[steps]=(profiles,worst)
        if steps not in cache:
            u=np.zeros(len(p));profiles={};t=0.;worst=0.
            for dt in steps:
                old=u.copy();accum=inj.copy();beta=dt*outgoing/vol
                for cells,edges in zip(groups,edgegroups):
                    rhs=old[cells]+dt*accum[cells]/vol[cells];lo=np.zeros(len(cells));hi=np.ones(len(cells))
                    for _ in range(56):
                        mid=(lo+hi)/2;r=mid+beta[cells]*flow(mid)-rhs
                        lo=np.where(r<0,mid,lo);hi=np.where(r>=0,mid,hi)
                    u[cells]=(lo+hi)/2
                    u[cells]=np.where(rhs<0,rhs,np.where(rhs>1+beta[cells],rhs-beta[cells],u[cells]))
                    np.add.at(accum,down[edges],q[edges]*flow(u[up[edges]]))
                residual=u-old+dt*(outgoing*flow(u)-accum)/vol
                worst=max(worst,float(max(abs(residual))));t+=dt
                checks.append(dict(case=args.case,schedule_model=job["model"],time=t,residual_inf=float(max(abs(residual)))))
                rt=int(round(t))
                if (out/f"saturation_{rt}.bin").exists():profiles[rt]=u.copy()
            if worst>1e-6:raise ValueError(f"Reference residual exceeds tolerance: {worst}")
            cache[steps]=(profiles,worst)
            np.savez_compressed(cachepath,residual_inf=worst,**{f"t{k}":v for k,v in profiles.items()})
        profiles,worst=cache[steps]
        for t,ref in profiles.items():
            path=out/f"saturation_{t}.bin"
            if path.exists():
                pred=state(path);row=dict(case=args.case,model=job["model"],time=t,reference="independent_DAG_same_time_grid",
                    reference_residual_inf=worst,**measures(ref,pred))
                row["volume_weighted_rmse"]=float(np.sqrt(np.sum(vol*(pred-ref)**2)/sum(vol)))
                rows.append(row)
        write(DEST/args.case/"reference_accuracy.csv",rows)
        print(job["model"],rows[-1] if rows else "No matching reports",flush=True)
    if checks:write(DEST/args.case/"reference_checks.csv",checks)
    (DEST/args.case/"reference_manifest.json").write_text(json.dumps(dict(cells=len(p),levels=len(groups),
        pressure_balance_inf=balance,unique_schedules=len(cache),seconds=time.perf_counter()-start),indent=2))


if __name__=="__main__":main()
