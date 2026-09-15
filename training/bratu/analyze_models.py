"""Compare learned and exact CRIS initialization basins."""
import csv
import hashlib
import json
import time
import os
from pathlib import Path
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
_cache = Path(__file__).resolve().parent / "analysis" / ".mplconfig"
_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_cache)
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy import sparse
from scipy.sparse.linalg import spsolve
from checkpoint_selection import selection

BASE = Path(__file__).resolve().parent
RUNS = BASE / 'runs/lower_branch_sweep'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

OUT = BASE / "analysis"
COLORS = {"exact": "#555555", "mse": "#1f4e79", "lambda_0p01": "#008c95", "lambda_0p1":"#C77D16", "lambda_1":"#A24E77"}
LABELS = {"exact": "Exact CRIS", "mse": "λ = 0", "lambda_0p01": "λ = 0.01", "lambda_0p1":"λ = 0.1", "lambda_1":"λ = 1"}
MODELS = {}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def learned(name, a, beta, derivative=False):
    shape = np.shape(a); a = np.asarray(a).ravel(); beta = np.broadcast_to(beta, shape).ravel()
    x = np.column_stack([(a+2)/4-1, 2*np.log1p(beta/1e-6)/np.log1p(1e6)-1])
    dx = np.column_stack([np.full(a.size, .25), np.zeros(a.size)]) if derivative else None
    w = MODELS[name]; offset = 0; width = 2
    for _ in range(4):
        matrix = w[offset:offset+20*width].reshape(20, width); offset += 20*width
        bias = w[offset:offset+20]; offset += 20
        x = np.tanh(x @ matrix.T + bias)
        if derivative: dx = (dx @ matrix.T) * (1-x*x)
        width = 20
    u = 4.5 * (x @ w[offset:offset+20] + w[offset+20] + 1)-2
    du = 4.5 * (dx @ w[offset:offset+20]) if derivative else None
    return u.reshape(shape), None if du is None else du.reshape(shape)


def inverse(name, a, beta, derivative=False):
    if name != "exact": return learned(name, a, beta, derivative)
    q = np.asarray(beta)*np.exp(a)
    valid = q <= 1/np.e
    z = np.where(valid, -q, 0)
    w = lambertw(z, 0).real
    u = np.where(valid, a-w, np.nan)
    return u, np.where(valid, 1/(1+w), np.nan) if derivative else None


def two_cell(name, scheme):
    grid = np.linspace(0, 3, 91)
    x,y = np.meshgrid(grid, grid, indexing="xy")
    start = np.column_stack([x.ravel(), y.ravel()]); u = start.copy()
    lower = 1.4161790490383697
    initial_valid = np.all(.1*np.exp(.65+.25*start[:, ::-1]) <= 1/np.e, axis=1)
    active = np.ones(len(u), bool); converged = np.zeros(len(u), bool)
    steps = np.zeros(len(u), int); failure = np.full(len(u), "iteration_limit", dtype=object)
    started = time.perf_counter()
    for iteration in range(81):
        ids = np.flatnonzero(active)
        if not len(ids): break
        state = u[ids]; a = .65 + .25*state[:, ::-1]
        phi, slope = inverse(name, a, .1, scheme == "newton")
        g = state-phi
        valid = np.all(np.isfinite(g), axis=1) & np.all(np.abs(state)<100, axis=1)
        failure[ids[~valid]] = "undefined_or_diverged"; active[ids[~valid]] = False
        done = valid & (np.max(np.abs(g), axis=1) <= 1e-10)
        converged[ids[done]] = True; active[ids[done]] = False; failure[ids[done]] = "map_converged"
        move = valid & ~done
        if iteration == 80: break
        if scheme == "picard":
            u[ids[move]] = phi[move]
        else:
            p = .25*slope[move]; residual = g[move]; det = 1-p[:,0]*p[:,1]
            delta = np.column_stack([(-residual[:,0]-p[:,0]*residual[:,1])/det,
                                     (-p[:,1]*residual[:,0]-residual[:,1])/det])
            u[ids[move]] += delta
        steps[ids[move]] += 1
    r = .65+.25*u[:, ::-1]-u+.1*np.exp(np.clip(u,-700,700))
    residual = np.max(np.abs(r),axis=1); error = np.max(np.abs(u-lower),axis=1)
    captured = converged & (error <= 1e-3) & (residual <= 1e-4)
    rows = [dict(method=name,scheme=scheme,u1_initial=s[0],u2_initial=s[1],initial_exact_domain=bool(v),
                 map_converged=bool(c),lower_capture=bool(ok),nonlinear_updates=int(n),
                 lower_max_error=float(e),physical_residual_inf=float(rn),status=f)
            for s,v,c,ok,n,e,rn,f in zip(start,initial_valid,converged,captured,steps,error,residual,failure)]
    summary = dict(method=name,scheme=scheme,starts=len(u),initially_admissible=int(initial_valid.sum()),
                   captured=int(captured.sum()),captured_inside=int((captured&initial_valid).sum()),
                   captured_outside=int((captured&~initial_valid).sum()),
                   median_updates_captured=float(np.median(steps[captured])) if captured.any() else None,
                   max_physical_residual_captured=float(residual[captured].max()) if captured.any() else None,
                   max_lower_error_captured=float(error[captured].max()) if captured.any() else None,
                   analysis_wall_seconds=time.perf_counter()-started)
    print(summary,flush=True)
    return rows,summary


def setup3d():
    # Unit cube, lambda=2 and sin(pi*x) boundary data on z=0.
    # The initialization sweep uses 3,249 interior unknowns.
    x,y,z = np.linspace(0,1,21),np.linspace(0,1,21),np.linspace(0,1,11)
    nx,ny,nz = len(x)-2,len(y)-2,len(z)-2
    def adjacency(n): return sparse.diags([np.ones(n-1),np.ones(n-1)],[-1,1],shape=(n,n),format="csr")
    B = (400*sparse.kron(sparse.eye(nz),sparse.kron(sparse.eye(ny),adjacency(nx)))
         +400*sparse.kron(sparse.eye(nz),sparse.kron(adjacency(ny),sparse.eye(nx)))
         +100*sparse.kron(adjacency(nz),sparse.eye(nx*ny))).tocsr()
    boundary = np.zeros((nz,ny,nx)); boundary[0,:,:] = 100*np.sin(np.pi*x[1:-1])[None,:]
    zz,yy,xx = np.meshgrid(z[1:-1],y[1:-1],x[1:-1],indexing="ij")
    shape = (np.sin(np.pi*xx)*np.sin(np.pi*yy)*np.sin(np.pi*zz)).ravel()
    return B/1800,boundary.ravel()/1800,2/1800,shape


def solve3d(method, amplitude, A,b,beta,shape, *, scheme="newton", max_updates=80):
    if scheme not in ("newton", "newton_linesearch", "jacobi"):
        raise ValueError("Unknown update scheme")
    u = amplitude*shape.copy(); identity=sparse.eye(len(u),format="csr")
    start=time.perf_counter(); nfeval=0; updates=0; max_outside=0; max_training_range=0
    def evaluate(state, jacobian=False):
        nonlocal nfeval,max_outside,max_training_range
        nfeval+=1; a=A@state+b
        max_outside=max(max_outside,int(np.count_nonzero(beta*np.exp(np.clip(a,-700,700))>1/np.e)))
        max_training_range=max(max_training_range,int(np.count_nonzero((a < -2)|(a > 6))))
        if method.startswith("srdm"):
            reaction=beta*np.exp(state); g=state-a-reaction
            J=identity-A-sparse.diags(reaction) if jacobian else None
        else:
            phi,dphi=inverse(method,a,beta,jacobian); g=state-phi
            J=identity-sparse.diags(dphi)@A if jacobian else None
        return g,J
    initial_a=A@u+b
    initial_outside=int(np.count_nonzero(beta*np.exp(initial_a)>1/np.e))
    status="iteration_limit"
    for iteration in range(max_updates+1):
        if time.perf_counter()-start>45: status="time_limit";break
        if not np.all(np.isfinite(u)) or np.max(np.abs(u))>100: status="diverged";break
        g,J=evaluate(u,scheme != "jacobi")
        if not np.all(np.isfinite(g)): status="undefined_inverse";break
        if np.max(np.abs(g))<=1e-10: status="map_converged";break
        if iteration==max_updates:break
        delta=-g if scheme == "jacobi" else spsolve(J,-g)
        if not np.all(np.isfinite(delta)):status="linear_failure";break
        if method=="srdm_linesearch" or scheme=="newton_linesearch":
            merit=float(g@g); step=1.; accepted=False
            for _ in range(20):
                trial=u+step*delta
                if np.all(np.isfinite(trial)) and np.max(np.abs(trial))<100:
                    gt,_=evaluate(trial)
                    if float(gt@gt)<=(1-1e-4*step)*merit: u=trial;accepted=True;break
                step*=.5
            if not accepted:status="line_search_failed";break
        else:u+=delta
        updates+=1
    residual=u-A@u-b-beta*np.exp(np.clip(u,-700,700))
    return u,dict(method=method,scheme=scheme,initial_amplitude=amplitude,status=status,nonlinear_updates=updates,
                  residual_evaluations=nfeval,wall_seconds=time.perf_counter()-start,
                  initial_outside_exact_cells=initial_outside,max_outside_exact_cells=max_outside,
                  max_outside_training_a_cells=max_training_range,
                  scaled_physical_residual_inf=float(np.max(np.abs(residual))),
                  pde_residual_inf=float(1800*np.max(np.abs(residual))))


def figures(rows,three):
    plt.rcParams.update({"font.family":"Arial","font.size":10,"axes.titlesize":11,"pdf.fonttype":42,"savefig.dpi":300})
    for scheme in ("picard","newton"):
        fig,axes=plt.subplots(2,3,figsize=(7.2,4.9),sharex=True,sharey=True,layout="constrained")
        for col,name in enumerate(COLORS):
            values=[r for r in rows if r["method"]==name and r["scheme"]==scheme]
            capture=np.array([r["lower_capture"] for r in values]).reshape(91,91)
            steps=np.array([r["nonlinear_updates"] for r in values],float).reshape(91,91)
            axes[0,col].imshow(capture,origin="lower",extent=(0,3,0,3),vmin=0,vmax=1,cmap="Greys",interpolation="nearest")
            im=axes[1,col].imshow(np.where(capture,steps,np.nan),origin="lower",extent=(0,3,0,3),cmap="viridis",vmin=0,vmax=80)
            axes[0,col].set_title(LABELS[name]); axes[1,col].set_xlabel(r"$u_1^{(0)}$")
            for ax in axes[:,col]:
                boundary=(np.log(10)-1-.65)/.25
                ax.axvline(boundary,color="#c7352e",lw=.7,ls="--");ax.axhline(boundary,color="#c7352e",lw=.7,ls="--")
        axes[0,0].set_ylabel(r"$u_2^{(0)}$");axes[1,0].set_ylabel(r"$u_2^{(0)}$")
        fig.colorbar(im,ax=axes[1,:],shrink=.8,label="Nonlinear updates")
        fig.savefig(OUT/f"two_cell_{scheme}_basins.png");fig.savefig(OUT/f"two_cell_{scheme}_basins.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout="constrained")
    for name,color in COLORS.items():
        vals=[r for r in three if r["method"]==name]
        amps=[r["initial_amplitude"] for r in vals]
        axes[0].plot(amps,[r["nonlinear_updates"] if r["lower_capture"] else np.nan for r in vals],"o-",color=color,label=LABELS[name])
        axes[1].semilogy(amps,[r["lower_rmse"] if r["lower_capture"] else np.nan for r in vals],"o-",color=color)
        for r in vals:
            if not r["lower_capture"]:axes[0].plot(r["initial_amplitude"],.03,"x",color=color,transform=axes[0].get_xaxis_transform())
    axes[0].set_ylabel("Nonlinear updates");axes[1].set_ylabel("Lower-solution RMSE")
    for ax in axes:ax.set_xlabel("Initial amplitude");ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(OUT/"three_dimensional_initialization.png");fig.savefig(OUT/"three_dimensional_initialization.pdf");plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    selected=selection()
    (OUT/'checkpoint_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    for name,record in selected.items():
        MODELS[name]=np.fromfile(BASE/record['path'],dtype="<f8")
        if len(MODELS[name])!=1341:raise ValueError("Unexpected checkpoint architecture")
    # Check the derivative used in CRIS Newton before running any global solve.
    for name in MODELS:
        a=np.array([.2,.9,1.2,2.,6.]); beta=.1; eps=1e-6
        _,d=learned(name,a,beta,True);plus,_=learned(name,a+eps,beta);minus,_=learned(name,a-eps,beta)
        if not np.allclose(d,(plus-minus)/(2*eps),rtol=1e-5,atol=1e-7):raise ValueError("Input derivative check failed")
    allrows=[]; summaries=[]
    for scheme in ("picard","newton"):
        for name in COLORS:
            rows,summary=two_cell(name,scheme);allrows+=rows;summaries.append(summary)
    write_csv(OUT/"two_cell_basins.csv",allrows);write_csv(OUT/"two_cell_summary.csv",summaries)
    A,b,beta,shape=setup3d(); reference,refinfo=solve3d("srdm_linesearch",0,A,b,beta,shape)
    if refinfo["status"]!="map_converged" or refinfo["pde_residual_inf"]>1e-7:raise ValueError("3D lower reference did not converge")
    np.save(OUT/"three_dimensional_reference.npy",reference)
    three=[]
    for amplitude in (0,1,2,4,6,8,10,12):
        for method in ("srdm","srdm_linesearch",*COLORS):
            u,row=solve3d(method,amplitude,A,b,beta,shape)
            row["lower_rmse"]=float(np.sqrt(np.mean((u-reference)**2)))
            row["lower_capture"]=bool(row["status"]=="map_converged" and row["lower_rmse"]<=1e-3 and row["scaled_physical_residual_inf"]<=1e-4)
            three.append(row); print(row,flush=True)
            write_csv(OUT/"three_dimensional_initialization.csv",three)
    (OUT/"manifest.json").write_text(json.dumps({
        "models":{name:record['sha256'] for name,record in selected.items()},
        "checkpoint_selection":selected,
        "two_cell":"Figure 3: a0=.65, coupling=.25, beta=.1; 91x91 starts on [0,3]^2; separate Picard and unrestricted Newton",
        "three_dimensional":"21x21x11 nodes, 3249 interior unknowns; unit cube, lambda=2, z=0 sin(pi*x) boundary; amplitude*sin(pi*x)*sin(pi*y)*sin(pi*z) initial conditions",
        "linear_solver":"Sparse direct solve with a new factorization at every Newton update",
        "termination":"Own map infinity residual<=1e-10; at most 80 updates. 3D time cap45s per case.",
        "capture_gate":"Map convergence AND lower-solution error<=1e-3 (two-cell max; 3D RMSE) AND scaled physical residual<=1e-4",
        "timing":"Python timings for the branch and initialization study; native C++ timings are reported separately for transport",
        "scope":"The 3D initialization sweep uses a coarser mesh than the 101x101x51-node solution plot",
    },indent=2)+"\n")


if __name__=="__main__":
    main()
