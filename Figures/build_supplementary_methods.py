"""Reproduce the transport work, local construction and fixed-point figures."""
import os, sys, json, csv, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DESTINATIONS={'supp_timestep_work':'fig4_transport','supp_transport_embedding':'fig5_local_inverse','supp_local_construction':'fig3_bratu','supp_residual_error_indicator':'fig5_local_inverse','supp_matrix_free_pilot':'fig4_transport'}
def product(name, ext):return HERE/DESTINATIONS[name]/(name+ext)
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'training/twophasetransport/scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from evaluate_checkpoints import load_weights, network
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
C=['#0072B2','#B2182B','#009E73','#CC79A7']
sources={}; outputs=[]; findings={}; page_sizes=[]
def source(p):
    p=ROOT/p if not Path(p).is_absolute() else Path(p)
    sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def records(p):
    with source(p).open() as f:return list(csv.DictReader(f))
def configuration(p):
    return {parts[0].upper():parts[1] for line in source(p).read_text().splitlines()
            if not line.lstrip().startswith('#') and len(parts:=line.split(maxsplit=1))==2}
def save(fig,name):
    fig.canvas.draw()
    from matplotlib.text import Text
    renderer=fig.canvas.get_renderer()
    box=fig.bbox
    for obj in fig.findobj(Text):
        if obj.get_visible() and obj.get_text():
            bb=obj.get_window_extent(renderer)
            # Off-axis tick artists are not drawn by Matplotlib.
            if obj in [t for ax in fig.axes for t in ax.get_xticklabels()+ax.get_yticklabels()]:continue
            assert bb.x0>=-2 and bb.y0>=-2 and bb.x1<=box.width+2 and bb.y1<=box.height+2,(name,obj.get_text(),bb.bounds)
    fig.savefig(product(name,'.pdf'));fig.savefig(product(name,'.png'),dpi=200);fig.savefig(product(name,'.svg'))
    page_sizes.append(np.round(fig.get_size_inches()*25.4,2).tolist())
    outputs.append(name);plt.close(fig)
def grid(h=145):
    fig,ax=plt.subplots(2,2,figsize=(183/25.4,h/25.4))
    fig.subplots_adjust(left=.12,right=.97,bottom=.11,top=.92,wspace=.43,hspace=.65)
    return fig,ax.ravel()
def title(ax,s):ax.set_title(s,loc='left',pad=10)

# Paired EDFM timestep histories, including work spent on rejected attempts.
base=Path('training/twophasetransport/analysis/regular_v3/paired_benchmarks/EDFM')
fig,axs=grid(125);runs={}
fig.subplots_adjust(left=.115,right=.98,bottom=.135,top=.875,wspace=.42,hspace=.9)
for name,col in [('CRIS',C[0]),('SRDM',C[1])]:
    meta=json.loads(source(base/name/'result.json').read_text());assert meta['completed']
    cfg=configuration(base/name/'sim.txt')
    final_time=float(cfg['T_END']);initial_dt=float(cfg['DT_INIT'])
    p=source(base/name/'output/solver_report.csv')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['output_sha256']['output/solver_report.csv']
    rr=records(p);t=0;start=[];end=[]
    good=np.array([int(r['converged'])==1 for r in rr]);dt=np.array([float(r['DT']) for r in rr])
    for d,g in zip(dt,good):start.append(t);t+=d if g else 0;end.append(t)
    assert np.isclose(t,final_time)
    start=np.array(start)/final_time;end=np.array(end)/final_time
    axs[0].plot(start[good],dt[good]/initial_dt,'o-',ms=3,lw=.8,color=col,label=name)
    axs[0].scatter(start[~good],dt[~good]/initial_dt,marker='x',s=20,color='#C98A00')
    waste=[]
    for ax,key in zip(axs[1:3],['NLNSTEPS','LINSTEPS']):
        v=np.array([float(r[key]) for r in rr]);ax.step(np.r_[0,end],np.r_[np.nan,np.cumsum(v)],where='post',color=col,label=name)
        if name=='SRDM':ax.step(np.r_[0,end],np.r_[np.nan,np.cumsum(v*good)],where='post',color=col,ls='--',label='SRDM accepted')
    for key in ['NLNSTEPS','LINSTEPS','NFEVAL']:
        v=np.array([float(r[key]) for r in rr]);waste.append(100*sum(v[~good])/sum(v))
    runs[name]={'accepted':int(sum(good)),'rejected':int(sum(~good)),'discarded_percent':waste,'wall_seconds':meta['wall_seconds']}
axs[0].set(yscale='log',xlabel=r'$\hat t$',ylabel=r'$\Delta t/\Delta t_{\rm init}$')
fig.legend(handles=[Line2D([],[],color=C[0],label='CRIS'),Line2D([],[],color=C[1],label='SRDM'),Line2D([],[],color=C[1],ls='--',label='Accepted only'),Line2D([],[],color='#C98A00',marker='x',ls='none',label='Rejected')],loc='upper center',bbox_to_anchor=(.52,1.01),ncol=4,frameon=False,handlelength=1.5,columnspacing=1)
for ax,label in zip(axs[1:3],['Newton updates','Krylov iterations']):ax.set(yscale='log',xlabel=r'$\hat t$',ylabel=label)
y=np.arange(3);w=runs['SRDM']['discarded_percent']
axs[3].barh(y,100-np.array(w),color=C[1],label='Accepted')
axs[3].barh(y,w,left=100-np.array(w),color='#E69F00',label='Discarded')
axs[3].set(yticks=y,yticklabels=['Newton','Krylov','Residual'],xlim=(0,100),xlabel='SRDM work (%)');axs[3].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.04),ncol=2,fontsize=11,handlelength=1,columnspacing=.9,borderaxespad=0)
axs[3].set_ylim(-.6,3.35)
axs[3].spines['left'].set_bounds(-.4,2.4)
for j,v in enumerate(w):axs[3].text(53,j,f'{v:.1f}%',ha='center',va='center',fontsize=12)
for ax,s in zip(axs,['a  Timestep attempts','b  Total nonlinear work','c  Total linear work','d  Cost of rejected steps']):title(ax,s)
findings['edfm']=runs;save(fig,'supp_timestep_work')

# Assembly of the three inputs to the transport local inverse.
fig=plt.figure(figsize=(183/25.4,49/25.4));ax=fig.add_axes([.025,.03,.95,.94]);ax.axis('off')
for x in [.015,.365,.715]:
    ax.add_patch(FancyBboxPatch((x,.32),.27,.62,boxstyle='round,pad=.008,rounding_size=.025',facecolor='#F3F6F8',edgecolor='#B5C4CE',lw=.8))
for x,head,body in [(.15,'Transport assembly','Fluxes and sources\nStorage and previous state'),(.5,'Three local inputs',r'$a_i=\dfrac{mA_i}{mB_i+\epsilon}$'+'\n'+r'$\beta_i=mB_i\Delta t,\quad u_i^n$'),(.85,'Selected inverse',r'$\hat u_i=\widehat\Phi(a_i,\beta_i,u_i^n)$')]:
    ax.text(x,.81,head,ha='center',va='center',fontsize=11,fontweight='bold')
    if x==.15:body='Fluxes and sources\nStorage\nPrevious state'
    if x==.5:
        ax.text(x,.62,r'$a_i=\dfrac{mA_i}{mB_i+\epsilon}$',ha='center',va='center',fontsize=11)
        ax.text(x,.395,r'$\beta_i=mB_i\Delta t,\quad u_i^n$',ha='center',va='center',fontsize=11)
    else:ax.text(x,.54,body,ha='center',va='center',fontsize=11,linespacing=1.3)
for x in [.325,.675]:ax.annotate('',xy=(x+.032,.62),xytext=(x-.032,.62),arrowprops=dict(arrowstyle='->',lw=1.3,color=C[0]))
ax.text(.5,.12,r'$u_i+\beta_i[f(u_i)-a_i]=u_i^n$',ha='center',fontsize=13)
save(fig,'supp_transport_embedding')
fig,axs=plt.subplots(1,3,figsize=(183/25.4,77/25.4))
fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.76,wspace=.72)
rows=records('Figures/fig3_bratu/data/continuation_validation.csv')
for idx,sch in enumerate(['alpha1_only','admissible_curved']):
    rr=[r for r in rows if r['schedule']==sch];rho=sorted(set(float(r['rho']) for r in rr));x=1-np.array(rho)
    for ax,key,op in [(axs[0],'abs_label_error',max),(axs[2],'min_derivative',min)]:
        vals=[op(float(r[key]) for r in rr if float(r['rho'])==v) for v in rho]
        ax.loglog(x,np.maximum(vals,1e-16),color=C[idx],ls=['-','--'][idx],label=[r'Fixed $\alpha_2$',r'Curved, $\rho<1$'][idx])
rr=[r for r in rows if r['schedule']=='alpha1_only'];vals=[max(float(r['path_disagreement']) for r in rr if float(r['rho'])==v) for v in rho]
axs[1].loglog(x,np.maximum(vals,1e-16),color=C[2])
for ax,s,yl in zip(axs,['a  Label error','b  Path difference','c  Regularity'],['Maximum error','Maximum difference',r'Minimum $|r_u|$']):
    title(ax,s);ax.set(ylabel=yl);ax.set_xticks([1e-3,1e-1,1])
fig.supxlabel(r'Fold deficit $1-\rho$',y=.03,fontsize=12)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.0),ncol=2,frameon=False)
save(fig,'supp_local_construction')

# Compare a derivative-scaled residual indicator with local inverse error.
fig=plt.figure(figsize=(183/25.4,78/25.4))
axs=[fig.add_axes([.12,.35,.22,.52]),fig.add_axes([.43,.35,.22,.52])]
bottom=fig.add_axes([.78,.35,.20,.52]);axs.extend([bottom,bottom])
rng=np.random.default_rng(1042);n=12000
a=rng.uniform(0,1,n);beta=10**rng.uniform(-8,6,n);old=rng.uniform(0,1,n)
rows=np.column_stack([a,beta,old])
models=[Path('training/twophasetransport/runs/regular_balanced_beta1e6_v3_62k_log1p_coldstart_residual_softcap_1_lambda_0p01/final_checkpoint.bin'),Path('training/twophasetransport/analysis/imp_v3/checkpoint_assessment/residual_continued_best.bin')]
for k,(name,q,p) in enumerate(zip(['Regular','Degenerate'],[2.,.2],models)):
    def flux(u):
        u=np.clip(u,0,1);return u*u/(u*u+.4*(1-u)**q)
    def deriv(u):
        u=np.clip(u,1e-15,1-1e-15);v=1-u;return .4*(2*u*v**q+q*u*u*v**(q-1))/(u*u+.4*v**q)**2
    lo=np.zeros(n);hi=np.ones(n)
    for _ in range(60):
        mid=(lo+hi)/2;pos=mid+beta*(flux(mid)-a)-old>0;hi=np.where(pos,mid,hi);lo=np.where(pos,lo,mid)
    truth=(lo+hi)/2;pred=network(load_weights(source(p)),rows,1e6,'log1p')
    err=abs(pred-truth);res=abs(pred+beta*(flux(pred)-a)-old)
    valid=(pred>0)&(pred<1)&(err>1e-11);indicator=res/(1+beta*deriv(pred))
    ax=axs[k];ax.loglog(err[valid],indicator[valid],'.',color=C[k],ms=1.5,alpha=.22,rasterized=True)
    lim=[1e-11,1e-3];ax.plot(lim,lim,'--',color='.3',lw=1);ax.set(xlim=lim,ylim=lim,xlabel='Absolute root error',ylabel=r'$|r(\hat u)/r_u(\hat u)|$',aspect='equal');title(ax,chr(97+k)+'  '+name)
    ax.set_xticks([1e-10,1e-7,1e-4]);ax.set_yticks([1e-10,1e-7,1e-4])
    if k:ax.set_ylabel('')
    ax=axs[k+2];dist=np.minimum(truth,1-truth);edges=np.logspace(-8,np.log10(.5),13)
    for quant,ls in [(50,'-'),(95,'--')]:
        xx=[];yy=[]
        for l,h in zip(edges[:-1],edges[1:]):
            m=valid&(dist>=l)&(dist<h)
            if sum(m)>=15:xx.append(np.sqrt(l*h));yy.append(np.percentile(abs(indicator[m]/err[m]-1),quant))
        ax.loglog(xx,yy,ls,marker='o',ms=3,color=C[k],label=f'{quant}th percentile')
    ax.set(xlabel='Root distance\nto endpoint',ylabel='Relative indicator error',xlim=(1e-8,.5),ylim=(1e-11,10))
    ax.set_xticks([1e-8,1e-5,1e-2]);ax.set_yticks([1e-10,1e-6,1e-2])
    findings[name]={'n':n,'excluded_oob':int(sum((pred<=0)|(pred>=1))),'plotted':int(sum(valid)),'root_rmse':float(np.sqrt(np.mean(err**2)))}
title(bottom,'c  Endpoints')
fig.legend(handles=[Line2D([],[],color=C[0],label='Regular'),Line2D([],[],color=C[1],label='Degenerate'),Line2D([],[],color='.25',ls='-',label='Median'),Line2D([],[],color='.25',ls='--',label='95th percentile')],loc='lower center',bbox_to_anchor=(.54,.005),ncol=4,frameon=False,columnspacing=1,handlelength=1.3)
save(fig,'supp_residual_error_indicator')

# Fixed-point evaluation on the three-dimensional SPE10 problem. CPU and GPU
# loop timings are reported separately from the full-process Newton timing.
pilot=Path('training/twophasetransport/analysis/regular_v3/benchmarks/3DSPE10_5SpotBase/picard')
timings=[json.loads(source(pilot/f).read_text()) for f in ['cpu_summary.json','cuda_summary.json','cuda_float32_check1_tol0.0001_summary.json']]
newton=next(r for r in records(pilot.parent.parent/'summary.csv')
            if r['case']=='3DSPE10_5SpotBase' and r['model']=='lambda_0p01')
newton_accuracy=next(r for r in records(pilot.parent/'reference_accuracy.csv')
                     if r['model']=='lambda_0p01' and float(r['time'])==20000)
newton_steps=records(pilot.parent/'lambda_0p01/output/solver_report.csv')
assert len(newton_steps)==10 and all(int(r['converged'])==1 and float(r['DT'])==2000 for r in newton_steps)
assert sum(int(r['NLNSTEPS']) for r in newton_steps)==int(newton['newton'])
assert newton['completed']=='True' and float(newton['last_accepted_time'])==20000
newton_seconds=float(newton['wall_seconds']); newton_rmse=float(newton_accuracy['rmse'])
fig,axs=plt.subplots(1,2,figsize=(183/25.4,63/25.4))
fig.subplots_adjust(left=.105,right=.98,bottom=.29,top=.85,wspace=.45)
ax=axs[0]
ax.scatter([r['seconds'] for r in timings[:2]],[r['rmse_vs_physical_reference']*1e5 for r in timings[:2]],c=[C[0],C[2]],s=55)
ax.scatter([timings[2]['seconds']],[timings[2]['rmse_vs_physical_reference']*1e5],edgecolors=C[3],facecolors='none',marker='D',s=65,linewidths=1.6)
ax.scatter([newton_seconds],[newton_rmse*1e5],color='.2',marker='s',s=55)
ax.annotate('Newton†',xy=(newton_seconds,newton_rmse*1e5),
            xytext=(newton_seconds,4.5),ha='center',fontsize=11,
            arrowprops=dict(arrowstyle='-',color='.2'))
ax.annotate('CPU 64',xy=(timings[0]['seconds'],timings[0]['rmse_vs_physical_reference']*1e5),xytext=(412,4.5),ha='center',fontsize=11,arrowprops=dict(arrowstyle='-',color=C[0]))
ax.annotate('GPU 64',xy=(timings[1]['seconds'],timings[1]['rmse_vs_physical_reference']*1e5),xytext=(285,6.5),ha='center',fontsize=11,arrowprops=dict(arrowstyle='-',color=C[2]))
ax.annotate('GPU 32*',xy=(timings[2]['seconds'],timings[2]['rmse_vs_physical_reference']*1e5),xytext=(70,9.6),fontsize=11)
ax.set(xlim=(0,470),ylim=(4,10.7),xlabel='Recorded time (s)',ylabel=r'Saturation RMSE ($10^{-5}$)')
title(ax,'a  Accuracy and time')
means=[1000*newton_seconds/int(newton['newton'])]+[1000*r['seconds']/r['iterations'] for r in timings]
bars=axs[1].bar(['Newton†','CPU\n64','GPU\n64','GPU\n32*'],means,color=['.2',C[0],C[2],C[3]],width=.4)
bars[-1].set_hatch('//')
axs[1].set_xlim(-.6,3.6)
axs[1].set_ylabel('Milliseconds per update');title(axs[1],'b  Mean update cost')
findings['picard_mean_ms_per_update']=means[1:]
findings['newton_baseline']={'seconds':newton_seconds,'rmse':newton_rmse,
    'updates':int(newton['newton']),'mean_ms_per_update':means[0],
    'timing_scope':'full process, unlike Picard iteration loop; not a controlled speedup'}
# Cell count and the starred float32 qualification are carried by the caption.
save(fig,'supp_matrix_free_pilot')
sizes=page_sizes
assert all(abs(w-183)<.01 for w,h in sizes)
(HERE/'supplementary_methods_manifest.json').write_text(json.dumps({
    'schema_version':1,
    'description':'Input checksums, computed summaries and export sizes for the supplementary-method figures.',
    'findings_description':'Computed from the source files and model evaluations by build_supplementary_methods.py.',
    'sources':sources,'findings':findings,'outputs':outputs,'page_sizes_mm':sizes,
    'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print(json.dumps(findings,indent=2))
