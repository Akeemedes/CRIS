"""Plot Bratu branch selection and transport training-to-simulation accuracy.

The figures combine the supplied root, solver and training results. Each export
includes a manifest identifying its numerical inputs and model checkpoints.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from publication_style import enforce_print_fonts, MIN_FONT_PT
from scipy.special import lambertw

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'Figures'
BASE=ROOT/'training/twophasetransport/analysis/regular_v3'
sys.path.insert(0,str(ROOT/'training/twophasetransport/scripts'))

COLORS=['#245781','#008C95','#C77D16','#A24E77']
LABELS=[r'$\lambda=0$',r'$\lambda=0.01$',r'$\lambda=0.1$',r'$\lambda=1$']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,
    'axes.titlesize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
SOURCES=set()
def rows(path):
    SOURCES.add(path)
    with path.open() as f:return list(csv.DictReader(f))
def label(ax,letter,title):
    ax.set_title(title,loc='left',pad=9)
    ax.text(-.12,1.065,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
def compact_label(ax,letter,title):
    """Keep panel letter and title on one baseline within narrow columns."""
    ax.set_title('',loc='center')
    ax.set_title(r'$\mathbf{'+letter+r'}$  '+title,loc='left',pad=7,fontsize=12)
def save(fig,folder,name):
    enforce_print_fonts(fig)
    SOURCES.add(FIG/'publication_style.py')
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'{name}.{ext}')
    (folder/f'{name}_manifest.json').write_text(json.dumps(dict(
        size_mm=(fig.get_size_inches()*25.4).tolist(),font_min_pt=MIN_FONT_PT,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)},
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    plt.close(fig);SOURCES.clear()

def bratu():
    base=ROOT/'training/bratu/analysis'
    basins=rows(base/'two_cell_basins.csv')
    three=rows(base/'three_dimensional_initialization.csv')
    evidence=base/'manifest.json';SOURCES.add(evidence)
    for name,digest in json.loads(evidence.read_text())['models'].items():
        checkpoint=ROOT/'training/bratu'/json.loads(evidence.read_text())['checkpoint_selection'][name]['path']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        SOURCES.add(checkpoint)
    fig=plt.figure(figsize=(183/25.4,120/25.4))
    # Allocate actual square panels, rather than wide slots which equal-aspect
    # axes silently shrink. Four basin maps share one comparison row.
    columns=(17,59,103,147)
    def panel(column,bottom):
        return fig.add_axes([columns[column]/183,bottom/120,33/183,33/120])
    ax=panel(0,75);a=np.linspace(0,np.log(10)-1,400,endpoint=False)
    ax.plot(a,a-lambertw(-.1*np.exp(a),0).real,color=COLORS[0],label='Selected branch')
    ax.plot(a,a-lambertw(-.1*np.exp(a),-1).real,'--',color='#999999',label='Upper branch')
    ax.plot(np.log(10)-1,np.log(10),'o',color='black',ms=4)
    ax.set(xlabel=r'$a$',ylabel=r'$u$',ylim=(0,5.5),xticks=[0,.6,1.2],yticks=[0,2,4])
    ax.text(.03,4.9,'Upper',fontsize=11,color='#777777')
    ax.text(.05,1.75,'Selected',fontsize=11,color=COLORS[0])
    compact_label(ax,'a','Local roots')
    ax=fig.add_axes([59/183,78/120,26/183,26/120]);path=FIG/'fig3_bratu/data/bratu3d_regenerated/bratu3d_lam2_dx001_dz002.npz';SOURCES.add(path)
    reference=np.load(path);iz=int(np.argmin(abs(reference['z']-.5)))
    im=ax.imshow(reference['U'][:,:,iz].T,origin='lower',extent=(0,1,0,1),cmap='cividis')
    ax.set(xlabel='$x$',xticks=[0,1],yticks=[0,1]);ax.set_ylabel('$y$',labelpad=0)
    compact_label(ax,'b','3D slice')
    ax.set_title(ax.get_title(loc='left'),loc='left',y=30/26)
    cb=fig.colorbar(im,cax=fig.add_axes([87/183,78/120,2/183,26/120]),ticks=[0,.2])
    cb.ax.set_title('$u$',pad=3)
    from two_cell_contours import draw_pair
    contour_axes=[panel(k,75) for k in (2,3)]
    _,contour_evidence=draw_pair(contour_axes,'bratu')
    for axis,letter,title in zip(contour_axes,'cd',('SRDM + LS','Exact CRIS')):
        axis.set(xticks=[0,1.5,3],yticks=[0,3])
        compact_label(axis,letter,title)
    contour_axes[0].yaxis.set_label_coords(-.12,.5)
    contour_axes[1].tick_params(labelleft=False)
    (FIG/'fig3_bratu/main_bratu_contours.json').write_text(json.dumps(contour_evidence,indent=2))
    SOURCES.add(FIG/'two_cell_contours.py')
    # Evaluate SRDM on the same initial-state lattice and with the same full-step
    # Newton policy as the CRIS basin data.
    grid=np.linspace(0,3,91);x,y=np.meshgrid(grid,grid);start=np.c_[x.ravel(),y.ravel()];u=start.copy()
    active=np.ones(len(u),bool);converged=np.zeros(len(u),bool)
    for iteration in range(81):
        ids=np.flatnonzero(active);v=u[ids]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            exp=.1*np.exp(v);r=.65+.25*v[:,::-1]-v+exp
            valid=np.all(np.isfinite(r),axis=1)&np.all(abs(v)<100,axis=1)
            done=valid&(np.max(abs(r),axis=1)<=1e-10)
            converged[ids[done]]=True;active[ids[done|~valid]]=False
            move=valid&~done
            if iteration==80:break
            d=exp[move]-1;rr=r[move];det=d[:,0]*d[:,1]-.25**2
            u[ids[move]]+=np.c_[(-d[:,1]*rr[:,0]+.25*rr[:,1])/det,(.25*rr[:,0]-d[:,0]*rr[:,1])/det]
    captured=converged&(np.max(abs(u-1.4161790490383697),axis=1)<=1e-3)
    srdm=np.where(captured,1,np.where(converged,2,0))
    palette=ListedColormap(['#E1E1E1',COLORS[0],'#D58C48','#FFFFFF'])
    boundary=(np.log(10)-1-.65)/.25
    summary=[]
    masks={}
    for k,method in enumerate(['srdm','exact','mse','lambda_1']):
        ax=panel(k,23)
        if method=='srdm':codes=srdm
        else:
            rr=[r for r in basins if r['scheme']=='newton' and r['method']==method]
            codes=np.array([1 if r['lower_capture']=='True' else (3 if r['status']=='undefined_or_diverged' and method=='exact' else 0) for r in rr])
        ax.imshow(codes.reshape(91,91),origin='lower',extent=(0,3,0,3),cmap=palette,norm=BoundaryNorm(np.arange(-.5,4),4),interpolation='nearest')
        ax.axhline(boundary,ls='--',color='#555555',lw=.8);ax.axvline(boundary,ls='--',color='#555555',lw=.8)
        ax.set(xlabel=r'$u_1^{(0)}$',xticks=[0,1.5,3],yticks=[0,1.5,3])
        if k==0:
            ax.set_ylabel(r'$u_2^{(0)}$');ax.yaxis.set_label_coords(-.30,.5)
        else:ax.tick_params(labelleft=False)
        compact_label(ax,chr(101+k),['SRDM','Exact CRIS',r'CRIS: $\lambda=0$',r'CRIS: $\lambda=1$'][k])
        summary.append(dict(method=method,lower_capture=int(sum(codes==1)),starts=len(codes)))
        masks[method]=codes.reshape(91,91)
    from matplotlib.ticker import FormatStrFormatter
    for axis in fig.axes:
        axis.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    fig.legend(handles=[Patch(color=COLORS[0],label='Lower root'),Patch(color='#D58C48',label='Other root'),Patch(color='#E1E1E1',label='Not captured'),Patch(facecolor='white',edgecolor='#888888',label='Undefined')],loc='lower center',bbox_to_anchor=(.51,0),ncol=4,frameon=False,handlelength=1.2,columnspacing=1)
    out=FIG/'fig3_bratu';(out/'main_bratu_counts.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'main_bratu_basins.npz',grid=grid,**masks)
    save(fig,out,'main_bratu')

def training():
    history=rows(BASE/'validation_residual_history.csv');transfer=rows(BASE/'transport_accuracy_summary.csv')
    current=rows(BASE/'regular_campaign_summary.csv')
    expected={r['checkpoint_sha256'] for r in current if r['dataset']=='validation'}
    assert {r['checkpoint_sha256'] for r in transfer if r['case'].endswith('_final')}==expected
    arms=list(dict.fromkeys(r['arm'] for r in history))
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,115/25.4),layout='constrained')
    root_ax,resid_ax,transfer_ax,work_ax,cost_ax,reuse_ax=axes.ravel()
    for k,arm in enumerate(arms):
        rr=sorted([r for r in history if r['arm']==arm and float(r['hours'])>0],key=lambda r:float(r['hours']))
        for ax,key in zip((root_ax,resid_ax),['prediction_rmse','residual_rmse']):
            ax.loglog([float(r['hours']) for r in rr],[float(r[key]) for r in rr],'-o',color=COLORS[k],ms=3,lw=1,label=LABELS[k])
    for ax in (root_ax,resid_ax):ax.set_xlabel('Time (h)');ax.grid(alpha=.15)
    root_ax.set_ylabel('Root RMSE');resid_ax.set_ylabel('Residual RMSE')
    for k,arm in enumerate(['MSE','0.01','0.1','1']):
        rr=sorted([r for r in transfer if r['arm']==arm],key=lambda r:float(r['local_validation_rmse']))
        transfer_ax.loglog([float(r['local_validation_rmse']) for r in rr],[float(r['rmse']) for r in rr],'-o',color=COLORS[k],ms=4,lw=1)
        work_ax.semilogx([float(r['rmse']) for r in rr],[float(r['nonlinear_updates']) for r in rr],'-o',color=COLORS[k],ms=4,lw=1)
    exact=next(r for r in transfer if r['case']=='exact_cris')
    work_ax.axhline(float(exact['nonlinear_updates']),ls='--',color='#555555',lw=1,label='Exact CRIS')
    work_ax.text(.98,.9,'Exact CRIS',ha='right',va='top',transform=work_ax.transAxes,fontsize=11,color='#555555')
    transfer_ax.set(xlabel='Root RMSE',ylabel='Field RMSE')
    work_ax.set(xlabel='Field RMSE',ylabel='Newton updates')
    from build_publication_completion import amortization
    amortization((cost_ax,reuse_ax),compact=True)
    for ax,letter,title in zip(axes.ravel(),'abcdef',['Root accuracy','Residual accuracy','Error transfer','Nonlinear work','Cost recovery','Reuse sensitivity']):compact_label(ax,letter,title)
    from matplotlib.ticker import LogLocator,NullFormatter
    for ax in axes.ravel():
        for axis in (ax.xaxis,ax.yaxis):
            if axis.get_scale()=='log':
                axis.set_major_locator(LogLocator(base=10,numticks=4));axis.set_minor_formatter(NullFormatter())
    SOURCES.update((FIG/'build_publication_completion.py',BASE/'training_cost_summary.csv',FIG/'fig4_transport/data/current_model_cost_table.csv'))
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    save(fig,FIG/'fig5_local_inverse','main_training')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='+',choices=['bratu','training'],default=['bratu','training'])
    for name in parser.parse_args().only:globals()[name]()
