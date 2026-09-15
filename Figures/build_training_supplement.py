"""Plot training histories, validation errors, input distributions and optimizer cost."""
import sys
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import numpy as np
import matplotlib.pyplot as plt
from build_main_results import ROOT,FIG,BASE,COLORS,LABELS,rows,save as save_common,label,SOURCES
from artifact_paths import training_run_path

OUT=FIG/'fig5_local_inverse'

def save(fig,folder,name):
    SOURCES.add(Path(__file__).resolve())
    save_common(fig,folder,name)

def histories(axes=None):
    campaigns=[('Regular transport',BASE,2),('Degenerate transport',ROOT/'training/twophasetransport/analysis/imp_v3',2),('Bratu',ROOT/'training/bratu/analysis',1/4.5)]
    combined=axes is not None
    if not combined:fig,axes=plt.subplots(3,2,figsize=(183/25.4,225/25.4),layout='constrained')
    else:fig=axes[0,0].figure
    for i,(title,base,scale) in enumerate(campaigns):
        history=rows(base/'validation_residual_history.csv');path=base/'validation_residual_history_manifest.json';SOURCES.add(path)
        sources=json.loads(path.read_text())['sources'];offset=0
        for k,source in enumerate(sources):
            if i==1 and k and source.get('arm')!=sources[k-1].get('arm'):offset=0
            directory=training_run_path(ROOT,source['run']);trace=rows(directory/'optimizer_trace.csv')
            if i==2:trace=[r for r in trace if int(r['accepted_updates'])<=10000]
            h=np.cumsum([sum(float(r.get(key) or 0) for key in ['normal_seconds','solve_seconds','candidate_seconds']) for r in trace])/3600+offset
            if i==2:h=np.array([float(r['cumulative_phase_seconds'])/3600 for r in trace])
            indices=np.unique(np.r_[np.arange(min(25,len(trace))),np.arange(0,len(trace),25),len(trace)-1]);indices=indices[h[indices]>0]
            color=COLORS[1 if 'residual_lambda' in source['run'] else 0] if i==1 else COLORS[k]
            for field,style in [('train_mse_normalized','--'),('validation_mse_normalized','-')]:
                axes[i,0].loglog(h[indices],np.sqrt([float(trace[j][field]) for j in indices])/scale,style,color=color,lw=1)
            if i==1:
                if offset>0:
                    for ax in axes[i]:ax.axvline(offset,color='#999999',ls=':',lw=1)
                offset=float(h[-1])
        arms=list(dict.fromkeys(r['arm'] for r in history))
        for k,arm in enumerate(arms):
            rr=[r for r in history if r['arm']==arm and float(r['hours'])>0]
            axes[i,1].loglog([float(r['hours']) for r in rr],[float(r['residual_rmse']) for r in rr],'-o',ms=3,color=COLORS[k],lw=1,label=LABELS[k])
        if not combined:axes[i,1].legend(frameon=False,fontsize=9)
        for j,ax in enumerate(axes[i]):
            ax.set_xlabel('Optimization time (h)');ax.grid(alpha=.15)
            label(ax,chr(97+i*2+j),title if j==0 else 'Validation residual')
        axes[i,0].set_ylabel('Root RMSE');axes[i,1].set_ylabel('Raw residual RMSE')
        limits=[ax.get_xlim() for ax in axes[i]]
        for ax in axes[i]:ax.set_xlim(min(v[0] for v in limits),max(v[1] for v in limits))
    from matplotlib.lines import Line2D
    if not combined:
        fig.legend(handles=[Line2D([],[],color='#555555',ls='--',label='Training'),Line2D([],[],color='#555555',label='Validation')],loc='outside lower center',ncol=2,frameon=False)
        save(fig,OUT,'supp_training_histories')

def distributions():
    data=rows(BASE/'regular_campaign_summary.csv');models=list(dict.fromkeys(r['model'] for r in data))
    datasets=['validation','uniform','balanced_transport','log_beta','target_tail','edge']
    names=['Validation','Uniform','Balanced','Log β','Target tails','Edges']
    fig,axes=plt.subplots(2,3,figsize=(183/25.4,156/25.4),layout='constrained')
    titles=['Root RMSE','99th-percentile\nerror','Maximum root\nerror','Bounds-violation\nfraction','Maximum\novershoot','Maximum\nundershoot']
    for j,ax in enumerate(axes.ravel()):
        for k,model in enumerate(models):
            rr=[next(r for r in data if r['model']==model and r['dataset']==d) for d in datasets]
            keys=['physical_rmse','physical_abs_error_q99','physical_max_abs_error']
            if j<3:values=[float(r[keys[j]]) for r in rr]
            elif j==3:values=[float(r['out_of_bounds_count'])/float(r['count']) for r in rr]
            elif j==4:values=[max(0,float(r['prediction_max'])-1) for r in rr]
            else:values=[max(0,-float(r['prediction_min'])) for r in rr]
            ax.plot(np.maximum(values,1e-8),np.arange(6)+(k-1.5)*.14,'o',ms=3.6,color=COLORS[k],label=LABELS[k])
            for y,v in enumerate(values):
                if v==0:ax.plot(1e-8,y+(k-1.5)*.14,'<',ms=4,color=COLORS[k])
        ax.set(xscale='log',yticks=np.arange(6),yticklabels=names if j%3==0 else [],ylim=(5.6,-.6));ax.grid(axis='x',alpha=.15)
        ticks=[[1e-6,1e-5],[1e-5,1e-4],[1e-5,1e-4],[1e-8,1e-5,1e-2],[1e-8,1e-6,1e-4],[1e-8,1e-6,1e-4]][j]
        ax.set_xticks(ticks)
        from matplotlib.ticker import NullFormatter,LogFormatterSciNotation
        ax.xaxis.set_major_formatter(LogFormatterSciNotation())
        ax.xaxis.set_minor_formatter(NullFormatter())
        label(ax,chr(97+j),titles[j])
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    save(fig,OUT,'supp_regular_distributions')

def efficiency(axes=None):
    costs=rows(BASE/'training_cost_summary.csv');thresholds=rows(BASE/'training_accuracy_thresholds.csv')
    combined=axes is not None
    if not combined:fig,axes=plt.subplots(2,2,figsize=(183/25.4,145/25.4),layout='constrained')
    else:fig=axes[0,0].figure
    for k,c in enumerate(costs):
        rr=[r for r in thresholds if r['model']==c['model']]
        axes[0,0].loglog([float(r['physical_validation_mse_threshold']) for r in rr],[float(r['instrumented_hours_to_first_crossing']) for r in rr],'-o',color=COLORS[k],ms=4,label=LABELS[k])
        axes[1,0].semilogx([float(r['physical_validation_mse_threshold']) for r in rr],[float(r['accepted_updates']) for r in rr],'-o',color=COLORS[k],ms=4)
    bottom=np.zeros(4)
    for key,text,col in [('normal_seconds','Normal equations','#245781'),('candidate_seconds','Candidate evaluation','#008C95'),('solve_seconds','Linear solve','#C77D16')]:
        vals=np.array([float(c[key])/3600 for c in costs])
        if combined:axes[0,1].barh(np.arange(4),vals,left=bottom,label=text,color=col,height=.55)
        else:axes[0,1].bar(np.arange(4),vals,bottom=bottom,label=text,color=col)
        bottom+=vals
    if combined:
        axes[0,1].set(yticks=np.arange(4),yticklabels=LABELS,xlabel='Optimization time (h)')
        axes[0,1].invert_yaxis()
    else:
        axes[0,1].set(xticks=np.arange(4),xticklabels=['0','0.01','0.1','1'],xlabel=r'Residual weight $\lambda$',ylabel='Optimization time (h)')
        fig.legend(*axes[0,1].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
    manifest=BASE/'regular_campaign_manifest.json';SOURCES.add(manifest)
    for k,source in enumerate(json.loads(manifest.read_text())['source_runs'].values()):
        rr=rows(training_run_path(ROOT,source['run_directory'])/'optimizer_trace.csv');ii=np.unique(np.r_[np.arange(1,len(rr),50),len(rr)-1])
        axes[1,1].plot([float(rr[j]['iteration']) for j in ii],[float(rr[j]['accepted_updates'])/float(rr[j]['iteration']) for j in ii],color=COLORS[k],lw=1)
    axes[0,0].set(xlabel='Validation MSE threshold',ylabel='Time to first crossing (h)')
    axes[1,0].set(xlabel='Validation MSE threshold',ylabel='Accepted updates to crossing')
    axes[1,1].set(xlabel='LM proposals',ylabel='Cumulative acceptance fraction',ylim=(0,1))
    axes[1,1].set_xticks([0,10000,20000])
    for ax,letter,title in zip(axes.ravel(),'abcd',['Time to accuracy','Measured cost components','Updates to accuracy','Proposal acceptance']):label(ax,letter,title)
    if not combined:
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
        save(fig,OUT,'supp_training_efficiency')


def combined():
    """Combine training histories and optimizer-efficiency measurements."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
    fig=plt.figure(figsize=(183/25.4,245/25.4),layout='constrained')
    grid=fig.add_gridspec(4,6,height_ratios=[1,1,1,.65])
    history_axes=np.array([[fig.add_subplot(grid[0,2*i:2*i+2]),fig.add_subplot(grid[1,2*i:2*i+2])] for i in range(3)])
    eff_axes=np.array([[fig.add_subplot(grid[2,0:2]),fig.add_subplot(grid[3,1:5])],
                       [fig.add_subplot(grid[2,2:4]),fig.add_subplot(grid[2,4:6])]])
    histories(history_axes);efficiency(eff_axes)
    # Relabel the transposed layout without duplicating panel letters.
    for ax in fig.axes:
        for text in list(ax.texts):text.remove()
        ax.set_title('')
    for i,title in enumerate(('Regular transport','Degenerate transport','Bratu')):
        for j in range(2):
            ax=history_axes[i,j]
            label(ax,chr(97+j*3+i),title if j==0 else '')
            ax.set_xlabel('Time (h)')
            ax.set_ylabel(('Root RMSE' if j==0 else 'Raw residual RMSE') if i==0 else '')
            ax.xaxis.set_major_locator(LogLocator(base=10,numticks=3))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
            ax.yaxis.set_minor_formatter(NullFormatter())
    for ax,letter,title in [(eff_axes[0,0],'g','Time to accuracy'),(eff_axes[1,0],'h','Updates to accuracy'),
                            (eff_axes[1,1],'i','Acceptance'),(eff_axes[0,1],'j','Regular: operation costs')]:
        label(ax,letter,title)
    eff_axes[0,1].texts[-1].set_x(-.028)
    eff_axes[0,0].set(xlabel='MSE target',ylabel='Time (h)')
    eff_axes[1,0].set(xlabel='MSE target',ylabel='Accepted updates')
    eff_axes[1,1].set(xlabel='LM proposals',ylabel='Accepted fraction')
    for ax in (eff_axes[0,0],eff_axes[1,0]):
        ax.xaxis.set_major_locator(LogLocator(base=10,numticks=3));ax.xaxis.set_minor_formatter(NullFormatter())
    eff_axes[1,0].yaxis.set_major_locator(MaxNLocator(4,integer=True))
    eff_axes[1,1].set_xticks([0,10000,20000],['0','10k','20k'])
    fig.legend(handles=[Line2D([],[],color=c,label=l) for c,l in zip(COLORS,LABELS)],
               loc='outside upper center',ncol=4,frameon=False)
    components,component_labels=eff_axes[0,1].get_legend_handles_labels()
    fig.legend(handles=[*components,Line2D([],[],color='#555555',ls='--'),Line2D([],[],color='#555555')],
               labels=[*component_labels,'Training','Validation'],loc='outside lower center',ncol=3,frameon=False)
    save(fig,OUT,'supp_training_histories')

def domain():
    import plot_regular_error_topology as topology
    manifest=BASE/'regular_campaign_manifest.json'
    sources=json.loads(manifest.read_text())['source_runs']
    sys.argv=['plot_regular_error_topology','--output-dir',str(OUT)]
    for k,(name,source) in enumerate(sources.items()):
        sys.argv+=['--model',LABELS[k]+'='+str(training_run_path(ROOT,source['run_directory'])/'final_checkpoint.bin')]
    topology.main()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='+',choices=['combined','histories','distributions','efficiency','domain'],
                        default=['combined','distributions','domain'])
    for name in parser.parse_args().only:globals()[name]()
