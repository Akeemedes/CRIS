"""Restored residual landscapes with explicit Newton trajectories.

Mechanism diagnostics: Bratu uses exact selected-branch CRIS; transport uses
the current regular lambda=.01 network and the original smooth algebraic flux
extension. Neither panel is a field-benchmark performance comparison.
"""
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import lambertw
from publication_style import enforce_print_fonts

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#332288','#882255']


def newton(start, fun, line_search):
    u=np.array(start,dtype=float);path=[u.copy()];status='iteration_limit'
    for _ in range(80):
        r,j=fun(u)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):status='undefined';break
        if np.linalg.norm(r)<1e-10:status='converged';break
        try:step=np.linalg.solve(j,-r)
        except np.linalg.LinAlgError:status='singular';break
        alpha=1.
        if line_search:
            for backtrack in range(35):
                trial=u+alpha*step;rr,_=fun(trial)
                if np.all(np.isfinite(rr)) and np.dot(rr,rr)<=(1-1e-4*alpha)*np.dot(r,r):break
                alpha*=.5
            else:status='line_search_failed';break
        u=u+alpha*step;path.append(u.copy())
        if not np.all(np.isfinite(u)) or max(abs(u))>1e6:status='diverged';break
    return np.array(path),status


def bratu_functions():
    def physical(u):
        with np.errstate(over='ignore',invalid='ignore'):
            return .65+.25*u[::-1]-u+.1*np.exp(u),np.diag(-1+.1*np.exp(u))+np.array([[0,.25],[.25,0]])
    def cris(u):
        a=.65+.25*u[::-1];z=-.1*np.exp(a)
        if np.any(z < -1/np.e):return np.full(2,np.nan),np.full((2,2),np.nan)
        w=lambertw(z).real
        return u-a+w,np.array([[1,-.25/(1+w[0])],[-.25/(1+w[1]),1]])
    return physical,cris


def transport_functions():
    package=ROOT/'training/twophasetransport/runs/regular_balanced_beta1e6_v3_62k_log1p_coldstart_residual_softcap_1_lambda_0p01/final_checkpoint.bin'
    weights=np.fromfile(package,dtype='<f8')
    assert weights.size==1361
    widths=(3,20,20,20,20,1);layers=[];offset=0
    for ni,no in zip(widths,widths[1:]):
        layers.append((weights[offset:offset+ni*no].reshape(no,ni),weights[offset+ni*no:offset+ni*no+no]));offset+=ni*no+no
    def net(a):
        x=np.array([2*a-1,2*np.log1p(200)/np.log1p(1e6)-1,-1.]);der=np.array([2.,0.,0.])
        for k,(w,b) in enumerate(layers):
            x=w@x+b;der=w@der
            if k<len(layers)-1:x=np.tanh(x);der*=1-x*x
        return .5*(x[0]+1),.5*der[0]
    def flux(u):
        d=u*u+.4*(1-u)**2
        return u*u/d,.8*u*(1-u)/d**2
    def physical(u):
        f,d=flux(u)
        return u+200*np.array([f[0]-1,f[1]-f[0]]),np.array([[1+200*d[0],0],[-200*d[0],1+200*d[1]]])
    def cris(u):
        f,d=flux(u);v0,_=net(1.);v1,d1=net(f[0])
        return u-np.array([v0,v1]),np.array([[1,0],[-d1*d[0],1]])
    return physical,cris,package


def draw_pair(axes, kind, colorscale=True):
    if kind=='bratu':
        funs=bratu_functions();limits=(0,3);starts=[(.1,.1),(1.2,1.8),(1.8,1.2),(2.2,2.2),(2.3,1.8)]
        model=None
    else:
        p,c,model=transport_functions();funs=(p,c);limits=(0,1.5)
        starts=[(.05,.95),(.2,.8),(.8,1.2),(1.2,.8),(1.2,1.2)]
    x=np.linspace(*limits,301 if kind=='transport' else 151);xx,yy=np.meshgrid(x,x)
    paths=[];images=[]
    for ax,fun,title in zip(axes,funs,['SRDM + line search','Exact CRIS' if kind=='bratu' else r'Learned CRIS: $\lambda=0.01$']):
        values=np.array([np.linalg.norm(fun(np.array([a,b]))[0]) for a,b in zip(xx.ravel(),yy.ravel())]).reshape(xx.shape)
        base=np.linalg.norm(fun(np.array(starts[0]))[0]);z=np.log10(np.maximum(values/base,1e-8))
        cmap='Greys' if kind=='bratu' else LinearSegmentedColormap.from_list('residual_landscape',plt.get_cmap('Greys')(np.linspace(.02,.9,256)))
        im=ax.contourf(xx,yy,z,levels=np.linspace(-4,2,61 if kind=='transport' else 31),cmap=cmap,extend='both',**({'antialiased':False} if kind=='transport' else {}))
        if kind=='transport':im.set_rasterized(True)  # Avoid PDF seam artifacts between filled levels.
        ax.contour(xx,yy,z,levels=np.arange(-4,2.01,.25) if kind=='transport' else np.arange(-4,3),colors='#555555' if kind=='transport' else '#999999',linewidths=.65 if kind=='transport' else .45,negative_linestyles='solid' if kind=='transport' else 'dashed')
        images.append(im)
        for k,start in enumerate(starts):
            path,status=newton(start,fun,line_search=title.startswith('SRDM'))
            paths.append(dict(method=title,start=start,status=status,trajectory=path.tolist()))
            ax.plot(path[:,0],path[:,1],color=COLORS[k],lw=1.25,zorder=4)
            ax.plot(*path[0],'o',color=COLORS[k],mec='white',mew=.7,ms=5,zorder=6)
            if status=='converged':
                if kind=='transport':
                    admissible=bool(np.all((path[-1]>=0)&(path[-1]<=1)))
                    paths[-1]['terminal_class']='physical' if admissible else 'nonphysical'
                    ax.plot(*path[-1],marker='*' if admissible else 'D',color='#009E73' if admissible else '#D55E00',mec='black',mew=.6,ms=12 if admissible else 7,zorder=7)
                else:ax.plot(*path[-1],'*',color=COLORS[k],mec='black',mew=.4,ms=10,zorder=7)
            for a,b in zip(path[:-1],path[1:]):
                if np.linalg.norm(b-a)>.025 and np.all((a>=limits[0])&(a<=limits[1])) and np.all((b>=limits[0])&(b<=limits[1])):
                    ax.annotate('',xy=a+.70*(b-a),xytext=a+.40*(b-a),arrowprops=dict(arrowstyle='-|>',color=COLORS[k],lw=1.2,mutation_scale=10),zorder=5)
        if kind=='bratu' and not title.startswith('SRDM'):
            bound=(np.log(10)-1-.65)/.25
            ax.axvline(bound,color='#555555',ls='--',lw=.8);ax.axhline(bound,color='#555555',ls='--',lw=.8)
        ax.set(xlim=limits,ylim=limits,xlabel='$u_1$',ylabel='$u_2$',title=title,aspect='equal')
        ax.set_xticks([0,1,2,3] if kind=='bratu' else [0,.5,1,1.5])
        ax.set_yticks([0,1,2,3] if kind=='bratu' else [0,.5,1,1.5])
    axes[1].set_ylabel('')
    return images,dict(kind=kind,normalization='Each formulation norm divided by its norm at the first common start',
        model_sha256=None if model is None else hashlib.sha256(model.read_bytes()).hexdigest(),trajectories=paths)


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Two-cell residual landscapes and solver trajectories.')
    parser.add_argument('--only',choices=['bratu','transport'])
    args=parser.parse_args()
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    for kind,folder in [('bratu','fig3_bratu'),('transport','figS2_transport_geometry')]:
        if args.only and args.only!=kind:continue
        compact=kind=='transport'
        fig=plt.figure(figsize=(183/25.4,(95 if compact else 112)/25.4))
        axes=[fig.add_axes([.09+.395*k,.14,.33,.70] if compact else [.09+.47*k,.28,.37,.59]) for k in range(2)]
        images,evidence=draw_pair(axes,kind)
        cb=fig.colorbar(images[0],cax=fig.add_axes([.86,.22,.016,.53] if compact else [.25,.15,.5,.025]),orientation='vertical' if compact else 'horizontal',ticks=[-4,-2,0,2])
        cb.set_label('Log normalized residual')
        handles=[Line2D([],[],marker='o',color='none',mfc='#555555',label='Initial state')]
        if kind=='transport':
            handles += [Line2D([],[],marker='*',ls='none',mfc='#009E73',mec='black',ms=12,label='Physical root'),Line2D([],[],marker='D',ls='none',mfc='#D55E00',mec='black',ms=7,label='Nonphysical root')]
        else:handles += [Line2D([],[],marker='*',color='none',mfc='#555555',ms=10,label='Converged root')]
        fig.legend(handles=handles,loc='upper center',ncol=len(handles),frameon=False,columnspacing=1.2,handletextpad=.4)
        enforce_print_fonts(fig)
        out=ROOT/'Figures'/folder/'two_cell_residual_contours'
        for ext in ('pdf','png','svg'):fig.savefig(out.with_suffix('.'+ext),dpi=600 if compact and ext!='png' else 220)
        out.with_suffix('.json').write_text(json.dumps(evidence,indent=2))
        plt.close(fig)


if __name__=='__main__':main()
