"""Independent CPU audit of the exported S45 discovery data; no model execution."""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results/stage4_s45_discovery_v2'
OUT = Path(__file__).resolve().parent
REPORT = ROOT / 'חומר כתוב/Stage45_Overleaf'
def read(name): return json.loads((SRC/name).read_text(encoding='utf-8-sig'))
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x,float); mean=x.mean(); ma=np.abs(x).mean()
    sign=(np.sign(x)==np.sign(mean)).mean(); frac=(np.abs(x)>=.1).mean()
    cls='coherent' if abs(mean)>=.1 and sign>=.7 else 'heterogeneous' if ma>=.1 and frac>=.2 else 'below_rule'
    ix=np.random.default_rng(20260926).integers(0,len(x),(20000,len(x)))
    ci=np.quantile(x[ix].mean(1),[.025,.975])
    return dict(mean=float(mean),mean_absolute=float(ma),sign_fraction=float(sign),above_tau_fraction=float(frac),
                classification=cls,interval=ci.tolist(),families=len(x))
def vectors(d):
    m={c:d[[f'M_{c}_order0',f'M_{c}_order1']].to_numpy(float) for c in ['x00','x10','x01','x11']}
    g=m['x00']-m['x10']; b=(m['x00']-m['x10']-m['x01']+m['x11'])/4
    return m,g,b

def main(figures=False):
    OUT.mkdir(exist_ok=True,parents=True)
    a=read('analysis/stage_a/stage_a_summary.json'); b=read('analysis/stage_b/stage_b_summary.json')
    c=read('analysis/stage_c/stage_c_summary.json'); ext=read('analysis/full_discovery/extension_summary.json')
    freeze=read('freeze_manifest.json'); sel=read('frozen_selection.json')
    fa=pd.read_csv(SRC/'analysis/stage_a/core_behavior.csv')
    fb=pd.read_csv(SRC/'analysis/stage_b/core_behavior_stage_b.csv')
    routes=pd.read_csv(SRC/'analysis/stage_c/core_route_effects.csv')
    fe=pd.read_csv(SRC/'analysis/full_discovery/extension_behavior.csv')
    ledger=pd.read_csv(SRC/'ri_membership_ledger.csv')
    checks=[]
    def check(name,actual,expected,tol=1e-10):
        if isinstance(expected,str): ok=actual==expected;err=None
        else: err=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))));ok=err<=tol
        checks.append(dict(check=name,passed=bool(ok),max_error=err))
    # Reproduce four-cell contrasts, gaps, accuracies, fidelity and paired family intervals.
    for phase,df,summaries in [('A',fa,a),('B',fb,b['states'])]:
        full=df[df.state=='full'].sort_values('family'); fullg=vectors(full)[1].mean(1)
        for rec in summaries:
            d=df[df.state==rec['state']].sort_values('family')
            assert len(d)==20 and not d.family.duplicated().any()
            m,g,bv=vectors(d);gm=g.mean(1);bm=bv.mean(1)
            for key,val in [('g_mean',gm.mean()),('b_mean',bm.mean()),('g_order0',g[:,0].mean()),('g_order1',g[:,1].mean()),
                            ('F',gm.mean()/fullg.mean()),('L',np.abs(gm-fullg).mean()/np.abs(fullg).mean())]:
                check(f'{phase}/{rec["state"]}/{key}',val,rec[key])
            check(f'{phase}/{rec["state"]}/stored_b',bm,d.b)
            check(f'{phase}/{rec["state"]}/g_interval',stats(gm)['interval'],rec['g_interval'])
            check(f'{phase}/{rec["state"]}/b_interval',stats(bm)['interval'],rec['b_interval'])
            for cell,arr in m.items():
                sign=1 if cell in ['x00','x11'] else -1
                for order in [0,1]:check(f'{phase}/{rec["state"]}/{cell}/acc{order}',np.mean(sign*arr[:,order]>0),rec['accuracy'][f'{cell}_order{order}'])
    cand=[]
    for r in b['candidates']:
        plus=fb[fb.state==r['state_plus']].sort_values('family');minus=fb[fb.state==r['state_minus']].sort_values('family')
        mp,gp,bp=vectors(plus);mm,gm,bm=vectors(minus);delta=(bp-bm).mean(1);st=stats(delta)
        for k in ['mean','mean_absolute','sign_fraction','interval','classification']:check(f'candidate/{r["candidate"]}/{k}',st[k],r['d_b'][k])
        for cell in mp:
            sign=1 if cell in ['x00','x11'] else -1
            sd=stats(sign*(mp[cell]-mm[cell]).mean(1))
            for k in ['mean','interval','sign_fraction']:check(f'candidate/{r["candidate"]}/{cell}/{k}',sd[k],r['gold_diff_'+cell][k])
        cand.append(dict(candidate=r['candidate'],kind=r['kind'],**st,order0=float((bp-bm)[:,0].mean()),order1=float((bp-bm)[:,1].mean()),
                         plus=r['state_plus'],minus=r['state_minus'],top_accuracy_changes={k:v for k,v in r['delta_accuracy'].items() if k.startswith('top_') and v!=0}))
    # Raw route endpoint arithmetic, complete 8 x 5 x 40 coverage, then matched Gamma.
    assert not routes.duplicated(['state','anchor','pair_id','direction']).any()
    assert len(routes)==1600
    check('route endpoint arithmetic',routes.intact_margin-routes.endpoint_margin,routes.effect)
    grouped={};route_out=[]
    for r in c['routes']:
        d=routes[(routes.state==r['state'])&(routes.anchor==r['anchor'])]
        assert len(d)==40 and set(d.order)=={0,1}
        v=d.pivot(index='family',columns='order',values='effect').sort_index();assert v.shape==(20,2)
        grouped[r['state'],r['anchor']]=v
        st=stats(v.mean(axis=1))
        for k in ['mean','mean_absolute','sign_fraction','interval','classification']:check(f'route/{r["state"]}/{r["anchor"]}/{k}',st[k],r[k])
        route_out.append(dict(state=r['state'],label=r['label'],anchor=r['anchor'],**st,order0=float(v[0].mean()),order1=float(v[1].mean())))
    candidates={r['candidate']:r for r in b['candidates']};matrix=[]
    for r in c['matrix']:
        h,t=r['candidate'],r['structure']; cr=candidates[h]
        plus=grouped[cr['state_plus'],t].mean(axis=1); minus=grouped[cr['state_minus'],t].mean(axis=1)
        st=stats(plus-minus)
        for k in ['mean','mean_absolute','sign_fraction','interval','classification']:check(f'gamma/{h}/{t}/{k}',st[k],r['gamma'][k])
        eligible=stats(plus)['classification']!='below_rule' or stats(minus)['classification']!='below_rule'
        check(f'gamma/{h}/{t}/eligible',int(eligible),int(r['route_eligible']))
        matrix.append(dict(candidate=h,structure=t,eligible=eligible,**st,route_plus=float(plus.mean()),route_minus=float(minus.mean())))
    # File provenance: independently validate canonical freeze digest and available declared hashes.
    body={k:v for k,v in freeze.items() if k!='signature'}
    signature=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    check('freeze signature',signature,freeze['signature'])
    check('pipeline freeze signature',read('pipeline_done.json')['freeze_signature'],signature)
    hash_rows=[]
    for stage,rec in freeze['stages'].items():
        for fn,expected in rec['analysis'].items():
            matches=list(SRC.rglob(fn))
            for path in matches:
                check('hash/'+str(path.relative_to(SRC)),sha(path),expected)
            hash_rows.append(dict(stage=stage,file=fn,present=bool(matches)))
    code_hashes={fn:sha(ROOT/'pilot_v2/s45'/fn)==expected if (ROOT/'pilot_v2/s45'/fn).exists() else None for fn,expected in freeze['code'].items()}
    ip=ROOT/'results/stage4_s45_inputs_v1'
    input_hashes={name:sha(ip/name)==freeze[key] for name,key in [('plan.json','plan_hash'),('pairs.jsonl.gz','pair_hash')]}
    # Core/extension export consistency. Extension CSV contains only the 20-family four-cell rows.
    check('extension exported family count',fe.family.nunique(),20)
    check('extension four-cell row count',len(fe),len(fb))
    merged=fe.merge(fb,on=['state','baseline','family'],suffixes=('_e','_b'),validate='one_to_one')
    for k in [x for x in fb.columns if x.startswith('M_') or x=='b']:check('extension retained core/'+k,merged[k+'_e'],merged[k+'_b'])
    ex=[r for r in ext['states'] if r['families']==89]
    full89=next(r for r in ex if r['label']=='full')['g_mean']
    for r in ex:
        check('extension/'+r['label']+'/'+r['baseline']+'/order averaging',(r['g_order0']+r['g_order1'])/2,r['g_mean'])
        check('extension/'+r['label']+'/'+r['baseline']+'/F',r['g_mean']/full89,r['F'])
    ext_by={(r['label'],r['baseline']):r for r in ex};diff=[]
    for base in ['mean','donor']:
        gb=ext_by['B',base]['g_mean']
        for name,val in [('L9H16',ext_by['B+L9H16',base]['g_mean']-gb),('L1H27',gb-ext_by['B-L1H27',base]['g_mean']),('R(B)',gb-ext_by['B-R',base]['g_mean'])]:
            diff.append(dict(candidate=name,baseline=base,delta_g=val))
    # Descriptive preference for the answer-side mother in the first chain, among the two candidates.
    prim=[]
    for label in ['full','C33','C50','empty']:
        d=fa[fa.label==label];m,_,_=vectors(d);first=[];query=[]
        for cell,arr in m.items():
            sign=1 if cell in ['x00','x01'] else -1
            first.extend((arr*np.array([sign,-sign])>0).ravel().tolist())
            gold=1 if cell in ['x00','x11'] else -1;query.extend((arr*gold>0).ravel().tolist())
        prim.append(dict(label=label,first_chain_candidate_preference=float(np.mean(first)),gold_candidate_accuracy=float(np.mean(query))))
    bid=next(r['state'] for r in a if r['label']=='C50')
    nested={key:stats((grouped[sid,'T1']-grouped[sid,'T3']).mean(axis=1)) for key,sid in [('full','full'),('B',bid)]}
    summary=dict(checks=len(checks),failures=[r for r in checks if not r['passed']],hash_coverage=hash_rows,code_hashes=code_hashes,input_hashes=input_hashes,nested_route_increment=nested,
                 raw_coverage=dict(stage_a_families=int(fa.family.nunique()),stage_a_rows=len(fa),stage_b_rows=len(fb),route_rows=len(routes),
                    extension_csv_families=int(fe.family.nunique()),extension_csv_rows=len(fe),extension_summary_families=89,worker_chunks_present=False),
                 primacy=prim,candidates=cand,routes=route_out,matrix=matrix,extension=ex,extension_differences=diff,
                 counts=dict(B=int(ledger.B.sum()),R_B=int(ledger.R_B.sum()),original59=59,extra_candidates_in_B=int((ledger.B&ledger.candidate).sum()),
                 selected=len(sel['selected']),heldout_mean_states=6,heldout_donor_states=4,heldout_scored_endpoints=5568))
    dump('audit_summary.json',summary);dump('checks.json',checks)
    pd.DataFrame(cand).to_csv(OUT/'recomputed_candidate_effects.csv',index=False)
    pd.DataFrame(matrix).to_csv(OUT/'recomputed_gamma.csv',index=False)
    pd.DataFrame(diff).to_csv(OUT/'extension_contrasts_from_summaries.csv',index=False)
    print(json.dumps({k:summary[k] for k in ['checks','failures','raw_coverage','counts','primacy','extension_differences','input_hashes']},indent=2))
    if figures: make_figures(a,fa,summary)
    assert not summary['failures'],'Reproduction mismatch: inspect checks.json'

def make_figures(a,fa,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    dest=REPORT/'figures';dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=150);plt.close(fig)
    colors={'full':'#174a6e','B':'#da7c30','other':'#79848c'}
    fig,ax=plt.subplots(1,2,figsize=(11.8,3.9),gridspec_kw={'width_ratios':[1,1.25]})
    labels=['full','empty','T1','T2','T3','T4','T5','C33','C50']; dd={r['label']:r for r in a}
    vals=np.array([dd[k]['g_mean'] for k in labels]);ci=np.array([dd[k]['g_interval'] for k in labels])
    ax[0].barh(labels,vals,color=[colors['full'] if k=='full' else colors['B'] if k in ['C33','C50'] else colors['other'] for k in labels])
    ax[0].errorbar(vals,np.arange(len(labels)),xerr=np.array([vals-ci[:,0],ci[:,1]-vals]),fmt='none',ecolor='black',capsize=2,lw=.9)
    ax[0].axvline(0,color='black',lw=.7);ax[0].invert_yaxis();ax[0].set_xlabel('Fact-swap gap g (logits)');ax[0].set_title('(a) Retention: 20 discovery families',loc='left')
    x=np.arange(4);cl=['x00\ngold a','x10\ngold b','x01\ngold b','x11\ngold a']
    for label,col in [('full',colors['full']),('C50',colors['B'])]:
        d=fa[fa.label==label];m,_,_=vectors(d)
        for o,ls,marker in [(0,'-','o'),(1,'--','s')]:
            ax[1].plot(x,[m[k][:,o].mean() for k in ['x00','x10','x01','x11']],color=col,ls=ls,marker=marker,label=f'{label}, order {o}')
    ax[1].axhline(0,color='black',lw=.7);ax[1].set_xticks(x,cl);ax[1].set_ylabel('Fixed-sign margin M = logit(a) - logit(b)');ax[1].set_title('(b) C50: order dominates the queried child',loc='left');ax[1].legend(fontsize=8,ncol=2)
    fig.tight_layout();save(fig,'retention_and_query')
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.0),gridspec_kw={'width_ratios':[1,1.25]})
    order=['L9H16','L1H27','L11H4','L17H5','L23H10','L25H18','L18H19','L8H15','R'];cd={r['candidate']:r for r in s['candidates']}
    vv=np.array([cd[h]['mean'] for h in order]);ii=np.array([cd[h]['interval'] for h in order]);labs=[h if h!='R' else 'RI31 jointly' for h in order]
    ax[0].barh(labs,vv,color=[colors['B'] if h in order[:2] else colors['full'] if h in ['L18H19','L8H15','R'] else colors['other'] for h in order])
    ax[0].errorbar(vv,np.arange(9),xerr=[vv-ii[:,0],ii[:,1]-vv],fmt='none',ecolor='black',capsize=2,lw=.9);ax[0].axvline(.1,color='gray',ls=':',label='Effect threshold 0.1');ax[0].axvline(0,color='black',lw=.7);ax[0].invert_yaxis();ax[0].set_xlabel('Binding contrast change, delta b (logits)');ax[0].set_title('(a) Functional participation, mean background',loc='left');ax[0].legend(fontsize=8,loc='lower right')
    hs=order[:6];tt=['T1','T2','T3','T4','T5'];mm={(r['candidate'],r['structure']):r for r in s['matrix']};z=np.array([[mm[h,t]['mean'] for t in tt] for h in hs])
    im=ax[1].imshow(z,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.035,vcenter=0,vmax=.035),aspect='auto')
    for i,h in enumerate(hs):
        for j,t in enumerate(tt):ax[1].text(j,i,f'{z[i,j]:+.3f}'+(' *' if not mm[h,t]['eligible'] else ''),ha='center',va='center',fontsize=8,color='white' if abs(z[i,j])>.022 else 'black')
    ax[1].set_xticks(range(5),tt);ax[1].set_yticks(range(6),hs);ax[1].set_title('(b) Route modulation Gamma: no retained pair',loc='left');ax[1].set_xlabel('* Route below retention rule in both head states');fig.colorbar(im,ax=ax[1],label='Gamma (logits)',fraction=.046,pad=.035)
    fig.tight_layout();save(fig,'functional_and_routes')
    fig,ax=plt.subplots(1,2,figsize=(11.8,3.8))
    bid=next(r['state'] for r in a if r['label']=='C50');rs={(r['state'],r['anchor']):r for r in s['routes']};xx=np.arange(5);w=.34
    for off,key,col,label in [(-w/2,'full',colors['full'],'Full model'),(w/2,bid,colors['B'],'C50 mean background')]:
        v=np.array([rs[key,t]['mean'] for t in tt]);ci=np.array([rs[key,t]['interval'] for t in tt]);ax[0].bar(xx+off,v,w,color=col,label=label);ax[0].errorbar(xx+off,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='none',ecolor='black',capsize=2,lw=.9)
    ax[0].axhline(0,color='black',lw=.7);ax[0].set_xticks(xx,tt);ax[0].set_ylabel('Route noising effect (logits)');ax[0].set_title('(a) Three routes remain above the rule',loc='left');ax[0].legend(fontsize=8)
    names=['B','B-R','B+L9H16','B-L1H27'];ee={(r['label'],r['baseline']):r for r in s['extension']};xx=np.arange(4)
    for off,base,col in [(-w/2,'mean',colors['B']),(w/2,'donor',colors['full'])]:
        v=np.array([ee[n,base]['g_mean'] for n in names]);ci=np.array([ee[n,base]['g_interval'] for n in names]);ax[1].bar(xx+off,v,w,color=col,label=base);ax[1].errorbar(xx+off,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='none',ecolor='black',capsize=2,lw=.9)
    ax[1].axhline(0,color='black',lw=.7);ax[1].set_xticks(xx,['B','B - RI31','B + L9H16','B - L1H27'],rotation=12);ax[1].set_ylabel('Fact-swap gap g (logits)');ax[1].set_title('(b) Baseline sensitivity: 89 discovery families',loc='left');ax[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'routes_and_baselines')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',action='store_true');args=parser.parse_args();main(args.figures)
