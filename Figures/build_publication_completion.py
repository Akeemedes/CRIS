"""Plot transport completion, solver sensitivity, amortization and endpoint errors."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from build_main_results import ROOT, FIG, BASE, COLORS, LABELS, SOURCES, rows, save, label, compact_label

IMP=ROOT/'training/twophasetransport/analysis/imp_v3/checkpoint_assessment'


def record(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


def completion_work(report, target_time):
    """Count every reported attempt; only accepted attempts advance time."""
    counts=np.array([[int(r[key]) for key in ('NLNSTEPS','LINSTEPS','NFEVAL')] for r in report])
    dt=np.array([float(r['DT']) for r in report])
    accepted=np.array([int(r['converged'])==1 for r in report])
    if not len(report) or not np.isfinite(target_time) or target_time<=0 or np.any(counts<0) or not np.all(np.isfinite(dt)) or np.any(dt<=0):
        raise ValueError('Invalid attempt ledger')
    work=np.cumsum(counts.sum(axis=1));time=np.cumsum(np.where(accepted,dt,0.))
    if time[-1]>target_time*(1+1e-10):raise ValueError('Accepted time exceeds target')
    return work,100*time/target_time,dict(recorded_work=int(work[-1]),accepted_time=float(time[-1]),
        progress_percent=float(100*time[-1]/target_time),reported_attempts=len(report),accepted_steps=int(accepted.sum()))


def degenerate():
    """Use one validation-selected residual model on both geometries."""
    continuation=record(IMP/'continuation_comparison.json')
    water=rows(IMP/'watercut_curves.csv')
    fig,grid=plt.subplots(2,3,figsize=(183/25.4,122/25.4),gridspec_kw={'width_ratios':[1,1,1.12]})
    fig.subplots_adjust(left=.085,right=.975,bottom=.195,top=.925,wspace=.65,hspace=.83)
    axes=grid.T  # each row is one geometry; columns are water cut, error, work
    ledger=[]
    for j,case in enumerate(('EDFM','Norne')):
        final=next(r for r in continuation if r['case']==case)
        assert final['completed'] and final['matched_time_grid'] and final['matched_pressure']
        checkpoint=IMP/'residual_continued_best.bin';SOURCES.add(checkpoint)
        import hashlib
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==final['checkpoint_sha256']
        tend=final['accepted_time']
        for k,model in enumerate(('mse_125k','residual_continued_best')):
            w=[r for r in water if r['case']==case and r['model']==model and r['producer']=='FIELD_TOTAL']
            w.sort(key=lambda r:float(r['time']))
            t=np.array([float(r['time']) for r in w])/tend
            if k==0:axes[0,j].plot(t,[float(r['reference']) for r in w],color='#333333',lw=1.8,label='Exact reference')
            axes[0,j].plot(t,[float(r['prediction']) for r in w],color=COLORS[k],ls=('--','-.')[k],lw=1.7,label=LABELS[k])
            if k==0:
                e=[r for r in rows(IMP/f'{case.lower()}_reference_errors.csv') if r['model']==model]
            else:e=rows(IMP/case/model/'reference_errors.csv')
            e.sort(key=lambda r:float(r['time']))
            axes[1,j].plot([float(r['time'])/tend for r in e],[float(r['rmse']) for r in e],'-o',color=COLORS[k],ms=3)
            report=rows(IMP/case/model/'output/solver_report.csv')
            assert sum(r['converged']!='1' for r in report)==0
            vals=[sum(int(r[key]) for r in report) for key in ('NLNSTEPS','LINSTEPS','NFEVAL')]
            work,progress,metrics=completion_work(report,tend)
            assert np.isclose(progress[-1],100)
            axes[2,j].step(np.r_[0,work],np.r_[0,progress],where='post',color=COLORS[k],ls=('--','-.')[k],lw=1.7)
            axes[2,j].plot(work[-1],progress[-1],'o',color=COLORS[k],ms=5,mec='white',mew=.6,zorder=5)
            ledger.append(dict(case=case,model=model,work=dict(zip(('Newton','Krylov','Residual'),vals)),final_rmse=float(e[-1]['rmse']),**metrics))
        probe_folder=IMP/case/('srdm_lin100' if case=='EDFM' else 'srdm_lin100_tol1e-06')
        probe=record(probe_folder/'result.json')
        vals=[probe[key] for key in ('NLNSTEPS','LINSTEPS','NFEVAL')]
        work,progress,metrics=completion_work(rows(probe_folder/'output/solver_report.csv'),tend)
        assert metrics['recorded_work']==sum(vals) and np.isclose(metrics['accepted_time'],probe['accepted_time'])
        assert not probe['completed'] and progress[-1]<100
        axes[2,j].step(np.r_[0,work],np.r_[0,progress],where='post',color='#666666',ls=':',lw=1.8)
        axes[2,j].plot(work[-1],progress[-1],marker='X',color='#666666',ms=6,mec='white',mew=.5,zorder=6)
        axes[2,j].annotate(f'Stopped\n{progress[-1]:.3g}%',xy=(work[-1],progress[-1]),xytext=(.99,.20),textcoords='axes fraction',ha='right',va='bottom',fontsize=11,color='#444444',arrowprops=dict(arrowstyle='-',lw=.8,color='#666666',shrinkA=3,shrinkB=5))
        ledger.append(dict(case=case,model='srdm_partial',work=dict(zip(('Newton','Krylov','Residual'),vals)),completed_fraction=probe['accepted_time']/tend,seconds=probe['seconds'],**metrics))
        for i in range(3):
            ax=axes[i,j];compact_label(ax,chr(97+3*j+i),case if i==0 else ('Field accuracy' if i==1 else 'Progress'))
            ax.grid(axis='x' if i==2 else 'y',alpha=.15)
            if i<2:ax.set(xlabel=r'$t$',xlim=(0,1),xticks=[0,.5,1])
        axes[0,j].set(ylim=(0,1),ylabel='Water cut',yticks=[0,.5,1])
        axes[1,j].set(yscale='log',ylabel='Field RMSE',ylim=(.002,.05))
        axes[1,j].set_yticks([.003,.01,.03])
        from matplotlib.ticker import FuncFormatter,NullFormatter,NullLocator
        axes[1,j].yaxis.set_major_formatter(FuncFormatter(lambda v,p:format(v,'.3g')))
        axes[1,j].yaxis.set_minor_formatter(NullFormatter())
        axes[2,j].set(xscale='log',ylim=(-5,118),yticks=[0,50,100],ylabel='Progress (%)',xlabel='Cumulative work',xlim=(30,2e6),xticks=[1e2,1e4,1e6])
        axes[2,j].xaxis.set_minor_locator(NullLocator())
        axes[2,j].axhline(100,color='#dddddd',lw=.8,zorder=0)
        axes[2,j].text(.98,.99,'Complete',transform=axes[2,j].transAxes,ha='right',va='top',color='#555555',fontsize=11)
    handles,texts=axes[0,0].get_legend_handles_labels()
    fig.legend(handles+[Line2D([],[],color='#666666',ls=':',marker='X',ms=5)],texts+['SRDM + LS'],loc='lower center',bbox_to_anchor=(.51,.012),ncol=4,frameon=False,handlelength=1.7,columnspacing=1)
    SOURCES.add(Path(__file__).resolve());save(fig,FIG/'figC1_impossible','main_degenerate')
    (FIG/'figC1_impossible/main_degenerate_ledger.json').write_text(json.dumps(ledger,indent=2))


def solver_sensitivity():
    path=FIG/'figS3_conditioning/data/frozen_linear/policy_edfm_full_budget_results.json'
    data=record(path)
    entries=data.get('results',data.get('runs',[]))
    if not entries:raise ValueError(f'No runs in {path}: {list(data)}')
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,125/25.4),layout='constrained')
    for method,col,title in [('srdm_linesearch','#555555','SRDM + LS'),('learned_cris',COLORS[1],'Learned CRIS')]:
        selected=[r for r in entries if r['method']==method and r['floor']==1e-6]
        selected.sort(key=lambda r:r['cap'])
        for j,(key,name) in enumerate((('nonlinear_steps','Newton updates'),('linear_iterations','Krylov iterations'),('residual_evaluations','Residual evaluations'))):
            ax=axes.flat[j]
            ax.plot([r['cap'] for r in selected],[r['common_horizon_work'][key] for r in selected],'-o',color=col,label=title)
            ax.set(yscale='log',xlabel='LINMAX',ylabel=name);label(ax,chr(97+j),['Nonlinear work','Linear work','Residual work'][j]);ax.grid(alpha=.15)
        ax=axes[1,1]
        ax.plot([r['cap'] for r in selected],[r['transport_work']['accepted_time']/data['final_time'] for r in selected],'-o',color=col)
    axes[1,1].set(xlabel='LINMAX',ylabel='Completed time fraction',ylim=(0,1.05));label(axes[1,1],'d','Full-run progress')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    SOURCES.add(Path(__file__).resolve());save(fig,FIG/'figS3_conditioning','supp_corrected_solver')


def amortization(axes=None,compact=False):
    costs=rows(BASE/'training_cost_summary.csv')
    standalone=axes is None
    if standalone:fig,axes=plt.subplots(1,2,figsize=(183/25.4,100/25.4),layout='constrained')
    else:fig=axes[0].figure
    measured=rows(FIG/'fig4_transport/data/current_model_cost_table.csv')
    measured=[r for r in measured if r['cris_provenance']==r['srdm_provenance']=='paired_benchmark']
    train=float(next(r for r in costs if '0.01' in r['model'])['instrumented_hours'])*3600
    sweep=sum(float(r['instrumented_hours'])*3600 for r in costs)
    savings=np.geomspace(1,1e4,150)
    axes[0].loglog(savings,train/savings,color='#333333',lw=1)
    ledger=[]
    for r in measured:
        delta=float(r['srdm_seconds'])-float(r['cris_seconds'])
        if delta<=0:raise ValueError('Nonpositive measured savings')
        axes[0].loglog(delta,train/delta,'o',color='#333333',ms=4)
        ledger.append(dict(case=r['case'],saved_seconds=delta,final_rmse=float(r['final_rmse']),
            selected_optimization_seconds=train,sweep_optimization_seconds=sweep,
            selected_break_even=int(np.ceil(train/delta)),sweep_break_even=int(np.ceil(sweep/delta))))
    import csv
    with (FIG/'fig5_local_inverse/amortization.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    axes[0].set(xlabel='Time saved per reuse (s)',ylabel='Optimization break-even reuses',xticks=[1,100,10000])
    label(axes[0],'a' if standalone else 'e','Optimization-cost recovery')
    reuse=np.geomspace(1,1e5,200)
    for case,name,ls in [('1D','1D','-'),('Norne_5SpotBase','Norne','--'),('EDFM','EDFM',':')]:
        r=next(r for r in measured if r['case']==case)
        ratio=reuse*float(r['srdm_seconds'])/(train+reuse*float(r['cris_seconds']))
        axes[1].loglog(reuse,ratio,color='#333333',ls=ls,label=name)
    axes[1].set(yscale='log',xlabel='Simulation reuses',ylabel='Total-cost speedup')
    axes[1].axhline(1,color='#777777',ls=':',lw=.8)
    label(axes[1],'b' if standalone else 'f','Reuse sensitivity')
    if compact:
        axes[0].set(xlabel='Time saved (s)',ylabel='Break-even reuses')
        axes[1].set(xlabel='Reuses',ylabel='Total speedup',ylim=(.01,1e3))
        axes[1].legend(frameon=False,loc='upper left',handlelength=.9,handletextpad=.35,labelspacing=0,borderaxespad=0)
        # Embedded panels use the parent's single-baseline labels.
        for ax in axes:
            for artist in list(ax.texts):
                if artist.get_text() in ('e','f'):artist.remove()
    else:axes[1].legend(title='Runtime speedup',frameon=False,loc='upper left')
    for ax in axes:ax.grid(alpha=.15)
    if standalone:
        fig.suptitle(r'Measured reuse: selected $\lambda=0.01$ model',fontsize=12)
        SOURCES.add(Path(__file__).resolve());save(fig,FIG/'fig5_local_inverse','supp_amortization')


def endpoints():
    data=rows(IMP/'front_input_accuracy.csv')
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,125/25.4),layout='constrained')
    for i,partition in enumerate(('sharp_front','near_front')):
        for k,model in enumerate(('mse_125k','residual_continued_best')):
            rr=[r for r in data if r['model']==model and r['partition']==partition]
            assert len(rr)==8
            SOURCES.add(IMP/(model+'.bin'))
            for j,key in enumerate(('rmse','flux_rmse')):
                axes[i,j].loglog([float(r['beta']) for r in rr],[float(r[key]) for r in rr],'-o',color=COLORS[k],ms=4,label=LABELS[k])
        for j in range(2):
            ax=axes[i,j];label(ax,chr(97+2*i+j),('Exact front' if i==0 else 'Front neighborhood')+(': root' if j==0 else ': flux'))
            ax.set(xlabel=r'$\beta$',ylabel='Saturation RMSE' if j==0 else 'Fractional-flow RMSE',xticks=[.1,10,1000]);ax.grid(alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    SOURCES.add(Path(__file__).resolve());save(fig,FIG/'figC1_impossible','supp_degenerate_endpoints')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only',nargs='+',choices=['degenerate','solver_sensitivity','amortization','endpoints'],default=['degenerate','solver_sensitivity','amortization','endpoints'])
    for name in p.parse_args().only:globals()[name]()
