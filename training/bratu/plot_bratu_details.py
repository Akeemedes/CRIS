"""Plot Bratu local accuracy, residual histories and initialization studies."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import lambertw
from checkpoint_selection import BASE, ARMS, BUDGET, candidates, selection
import analyze_models as bm
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'Figures'))
from build_main_results import save as export_figure, SOURCES, COLORS, LABELS, label, compact_label
import matplotlib.pyplot as plt
OUT=BASE/'analysis'
FIG=ROOT/'Figures/fig3_bratu'


def save(fig,folder,name):
    SOURCES.update((Path(__file__).resolve(),BASE/'checkpoint_selection.py',OUT/'checkpoint_selection.json'))
    export_figure(fig,folder,name)


def map_strip(height, map_height, xlabel, ylabel, side_scale=False):
    """Four full-resolution maps at fixed print typography, with shared labels."""
    fig=plt.figure(figsize=(183/25.4,height/25.4))
    bottom=16 if side_scale else 33
    positions=(18,54.5,91,127.5) if side_scale else (21,62,104,145)
    axes=np.array([fig.add_axes([x/183,bottom/height,(31 if side_scale else 32)/183,map_height/height])
                   for x in positions])
    fig.text(.024,(bottom+map_height/2)/height,ylabel,rotation=90,ha='center',va='center',fontsize=12)
    fig.supxlabel(xlabel,x=.48 if side_scale else .54,y=(1.5 if side_scale else 19)/height,fontsize=12)
    color_axis=fig.add_axes([164/183,bottom/height,2/183,map_height/height] if side_scale else [.23,12.5/height,.58,2.5/height])
    return fig,axes,color_axis


def data(name):
    path=BASE/'datasets/lower_branch'/f'{name}.bin'
    SOURCES.add(path)
    with path.open('rb') as f:
        if f.read(8)!=b'BRATUD01':raise ValueError(path)
        n=int.from_bytes(f.read(8),'little')
    arr=np.fromfile(path,dtype='<f8',offset=16).reshape(-1,4)
    assert len(arr)==n and np.isfinite(arr).all()
    return arr


def metrics(pred,arr):
    err=pred-arr[:,2];r=arr[:,0]-pred+arr[:,1]*np.exp(pred)
    return dict(count=len(arr),prediction_rmse=float(np.sqrt(np.mean(err**2))),
        prediction_mse=float(np.mean(err**2)),absolute_error_q99=float(np.quantile(abs(err),.99)),
        maximum_error=float(np.max(abs(err))),residual_rmse=float(np.sqrt(np.mean(r**2))),
        residual_max=float(np.max(abs(r))),capped_residual_rmse=float(np.sqrt(np.mean(np.tanh(r/4.5)**2))))


def load(name,record):
    path=BASE/record['path'];SOURCES.add(path)
    assert bm.digest(path)==record['sha256']
    weights=np.fromfile(path,dtype='<f8')
    assert len(weights)==1341 and np.isfinite(weights).all()
    bm.MODELS[name]=weights


def histories(selected):
    valid=data('validation');history=[];sources=[]
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,170/25.4),layout='constrained')
    for k,name in enumerate(ARMS):
        retained,cost=candidates(name);sources.append(cost)
        path=Path(cost['run'])/'optimizer_trace.csv';SOURCES.add(path)
        trace=list(csv.DictReader(path.open()))
        trace=[r for r in trace if int(r['accepted_updates'])<=BUDGET]
        h=np.array([float(r['cumulative_phase_seconds'])/3600 for r in trace])
        n=np.array([int(r['accepted_updates']) for r in trace])
        ids=np.unique(np.r_[np.arange(min(25,len(trace))),np.arange(0,len(trace),25),len(trace)-1])
        for row,x in enumerate((h,n)):
            for key,style in [('train_mse_normalized','--'),('validation_mse_normalized','-')]:
                axes[row,0].loglog(x[ids],4.5*np.sqrt([float(trace[i][key]) for i in ids]),style,color=COLORS[k])
        for rec in retained:
            load(name,rec);pred,_=bm.learned(name,valid[:,0],valid[:,1]);values=metrics(pred,valid)
            assert np.isclose(values['prediction_mse']/20.25,rec['validation_mse_normalized'],rtol=2e-6,atol=1e-18), (name,rec)
            history.append(dict(arm=LABELS[k],model=name,stage=k,checkpoint=Path(rec['path']).name,
                checkpoint_sha256=rec['sha256'],hours=rec['hours'],accepted=rec['accepted'],
                proposals=rec['proposals'],selected_final=rec['sha256']==selected[name]['sha256'],**values))
        rr=sorted([r for r in history if r['model']==name],key=lambda r:r['accepted'])
        for row,key in enumerate(('hours','accepted')):
            axes[row,1].loglog([r[key] for r in rr],[r['residual_rmse'] for r in rr],'-o',ms=3,color=COLORS[k],label=LABELS[k])
    for i,ax in enumerate(axes.flat):
        label(ax,chr(97+i),'Root accuracy' if i%2==0 else 'Validation residual')
        ax.set_xlabel('Optimization time (h)' if i<2 else 'Accepted updates')
        ax.set_ylabel('Root RMSE' if i%2==0 else 'Raw residual RMSE');ax.grid(alpha=.15)
    axes[0,1].legend(frameon=False)
    save(fig,FIG,'supp_bratu_training')
    bm.write_csv(OUT/'validation_residual_history.csv',history)
    (OUT/'validation_residual_history_manifest.json').write_text(json.dumps(dict(sources=sources,
        budget=BUDGET,sampling='Retained eligible checkpoints only; no residual interpolation claim; selected model at actual saving time. Budget and full cost separately recorded.',
        residual='a-u+beta*exp(u), on all fixed validation rows; capped tanh(R/4.5).'),indent=2)+'\n')


def local(selected):
    sets={name:data(name) for name in ('validation','test','test_near_fold')}
    summary=[];bins=[]
    edges=np.r_[0.,np.logspace(-10,0,21)]
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,120/25.4),layout='constrained')
    for k,name in enumerate(ARMS):
        load(name,selected[name])
        for dataset,arr in sets.items():
            pred,_=bm.learned(name,arr[:,0],arr[:,1]);summary.append(dict(model=name,dataset=dataset,**metrics(pred,arr)))
        # Pool only independent matched and closer-fold tests for conditional bins.
        arr=np.vstack([sets['test'],sets['test_near_fold']]);pred,_=bm.learned(name,arr[:,0],arr[:,1])
        deficit=1-np.e*arr[:,1]*np.exp(arr[:,0])
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(deficit>=lo)&((deficit<hi) if hi<1 else (deficit<=hi))
            if mask.any():bins.append(dict(model=name,deficit_lower=lo,deficit_upper=hi,deficit_median=float(np.median(deficit[mask])),**metrics(pred[mask],arr[mask])))
        rr=[r for r in bins if r['model']==name]
        for j,key in enumerate(('prediction_rmse','residual_rmse')):
            axes[0,j].loglog([r['deficit_median'] for r in rr],[r[key] for r in rr],'-o',ms=3,color=COLORS[k],label=LABELS[k])
        for j,key in enumerate(('prediction_rmse','absolute_error_q99')):
            vals=[next(r[key] for r in summary if r['model']==name and r['dataset']==d) for d in sets]
            axes[1,j].semilogy(np.arange(3)+(k-1.5)*.10,vals,'o',color=COLORS[k])
    for j,ax in enumerate(axes[0]):
        ax.set_xlabel(r'Fold deficit $1-e\beta e^a$');ax.set_ylabel('Root RMSE' if j==0 else 'Raw residual RMSE')
        ax.axvline(1e-8,ls=':',color='#777777',lw=1)
    for j,ax in enumerate(axes[1]):
        ax.set_xticks(range(3),['Validation','Matched\ntest','Closer-fold\ntest']);ax.set_ylabel('Root RMSE' if j==0 else '99th-percentile error')
    for i,ax in enumerate(axes.flat):label(ax,chr(97+i),['Near-fold accuracy','Near-fold residual','Distributional accuracy','Error tails'][i]);ax.grid(alpha=.15)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False,handlelength=1.2,columnspacing=1.2)
    save(fig,FIG,'supp_bratu_local_error')
    bm.write_csv(OUT/'local_accuracy.csv',summary);bm.write_csv(OUT/'fold_error_bins.csv',bins)
    # Deterministic input-domain slices, equal colour limits for all weights.
    aa=np.linspace(-1,6,161);dd=np.logspace(-10,0,181);a,d=np.meshgrid(aa,dd)
    beta=np.exp(-a-1)*(1-d);root=a-lambertw(-beta*np.exp(a),0).real
    from matplotlib.colors import LogNorm
    fig,axes,color_axis=map_strip(90,43,'$a$','Fold deficit')
    for k,(name,ax) in enumerate(zip(ARMS,axes.flat)):
        load(name,selected[name]);pred,_=bm.learned(name,a,beta)
        im=ax.pcolormesh(a,d,np.maximum(abs(pred-root),1e-9),norm=LogNorm(1e-9,1e-3),cmap='magma',shading='auto',rasterized=True)
        ax.set(yscale='log',xticks=[-1,3,6],yticks=[1e-10,1e-6,1e-2,1])
        if k:ax.tick_params(labelleft=False)
        ax.axhline(1e-8,ls=':',color='white',lw=1);compact_label(ax,chr(97+k),LABELS[k])
    fig.colorbar(im,cax=color_axis,orientation='horizontal',label='Absolute root error',extend='both',ticks=[1e-9,1e-7,1e-5,1e-3])
    save(fig,FIG,'supp_bratu_error_domain')


def global_figures():
    path=OUT/'two_cell_basins.csv';SOURCES.add(path);rows=list(csv.DictReader(path.open()))
    fig,axes,color_axis=map_strip(60,31,r'$u_1^{(0)}$',r'$u_2^{(0)}$',side_scale=True)
    paired=[];exact=[r for r in rows if r['scheme']=='newton' and r['method']=='exact']
    for k,(name,ax) in enumerate(zip(ARMS,axes.flat)):
        rr=[r for r in rows if r['scheme']=='newton' and r['method']==name]
        captured=np.array([r['lower_capture']=='True' for r in rr]);steps=np.array([int(r['nonlinear_updates']) for r in rr])
        im=ax.imshow(np.where(captured,steps,np.nan).reshape(91,91),origin='lower',extent=(0,3,0,3),vmin=0,vmax=20,cmap='viridis',interpolation='nearest')
        ax.set_facecolor('#dedede');boundary=(np.log(10)-1-.65)/.25
        ax.axhline(boundary,color='white',ls='--',lw=1);ax.axvline(boundary,color='white',ls='--',lw=1)
        ax.set_xticks([0,1.5,3],['0','1.5','3']);ax.set_yticks([0,1.5,3],['0','1.5','3'])
        compact_label(ax,chr(97+k),LABELS[k])
        if k:ax.tick_params(labelleft=False)
        common=np.array([r['lower_capture']=='True' for r in exact])&captured
        en=np.array([int(r['nonlinear_updates']) for r in exact])[common];ln=steps[common]
        paired.append(dict(model=name,common_captures=int(common.sum()),exact_updates=int(en.sum()),learned_updates=int(ln.sum()),
            learned_fewer=int((ln<en).sum()),equal=int((ln==en).sum()),learned_more=int((ln>en).sum())))
    fig.colorbar(im,cax=color_axis,orientation='vertical',label='Newton updates',extend='max',ticks=[0,10,20])
    save(fig,FIG,'supp_bratu_basins');bm.write_csv(OUT/'paired_newton_work.csv',paired)
    path=OUT/'three_dimensional_initialization.csv';SOURCES.add(path);three=list(csv.DictReader(path.open()))
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,95/25.4),layout='constrained')
    names=('srdm','srdm_linesearch','exact',*ARMS)
    capture_grid=np.full((len(names),8),np.nan)
    for k,name in enumerate(names):
        rr=[r for r in three if r['method']==name];ok=[r for r in rr if r['lower_capture']=='True']
        color=('#888888','#333333','#111111',*COLORS)[k]
        title={'srdm':'SRDM','srdm_linesearch':'SRDM + LS','exact':'Exact CRIS'}.get(name,bm.LABELS.get(name,name))
        for j,r in enumerate(rr):
            if r['lower_capture']=='True':capture_grid[k,j]=int(r['nonlinear_updates'])
        if name in ARMS:
            axes[1].semilogy([float(r['initial_amplitude']) for r in ok],[float(r['lower_rmse']) for r in ok],'o-',color=color,ms=3,label=title)
    from matplotlib.colors import ListedColormap
    axes[0].imshow(np.isfinite(capture_grid),cmap=ListedColormap(['#dedede','#d2e6ef']),vmin=0,vmax=1,aspect='auto')
    for i,j in zip(*np.where(np.isfinite(capture_grid))):axes[0].text(j,i,str(int(capture_grid[i,j])),ha='center',va='center')
    axes[0].set_xticks(range(8),[0,1,2,4,6,8,10,12]);axes[0].set_yticks(range(7),['SRDM','SRDM + LS','Exact CRIS',*LABELS])
    for ax in axes:ax.set_xlabel('Initial amplitude')
    axes[1].grid(alpha=.15);axes[1].set_ylabel('Lower-solution RMSE');axes[1].legend(frameon=False,fontsize=11)
    label(axes[0],'a','Capture / Newton updates');label(axes[1],'b','Learned-solution error')
    save(fig,FIG,'supp_bratu_3d')


def matched_accuracy():
    result=[]
    for name in ARMS:
        record=next(r for r in candidates(name)[0] if Path(r['path']).name=='milestone_9.bin')
        load(name,record)
        _,summary=bm.two_cell(name,'newton')
        summary.update(accepted=record['accepted'],physical_validation_mse=20.25*record['validation_mse_normalized'])
        result.append(summary)
    bm.write_csv(OUT/'matched_accuracy_basins.csv',result)


if __name__=='__main__':
    selected=selection()
    histories(selected);local(selected);global_figures();matched_accuracy()
    print('Bratu supplementary figures exported.',flush=True)
