from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,actual,expected):
    ok=bool(np.allclose(actual,expected,atol=1e-10,rtol=1e-10)) if not isinstance(actual,(str,bool,list,set)) else actual==expected
    checks.append({'name':name,'passed':ok})
def ci(a):
    a=np.array(a);ix=np.random.default_rng(20260926).integers(0,len(a),(20000,len(a)))
    return list(np.quantile(a[ix].mean(axis=1),[.025,.975]))
def summary(a):
    a=np.asarray(a)
    return {'mean':float(a.mean()),'interval':ci(a),'positive':int((a>0).sum()),'n':len(a),'mean_absolute':float(abs(a).mean()),'min':float(a.min()),'max':float(a.max()),'above_tau':int((abs(a)>=.1).sum())}

plan=read(ROOT/'results/heldout_plan.json')
discplan=read(ROOT/'results/stage4_s45_inputs_v1/plan.json')
freeze=read(ROOT/'results/stage4_s45_discovery_v2/freeze_manifest.json')
manifest=read(OUT/'pipeline_manifest.json');done=read(OUT/'pipeline_done.json')
fbody={k:v for k,v in freeze.items() if k!='signature'}
sig=hashlib.sha256(json.dumps(fbody,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
check('canonical freeze signature',sig,freeze['signature'])
check('run used discovery freeze',done['freeze_signature'],freeze['signature'])
check('freeze file hash',manifest['freeze'],digest(ROOT/'results/stage4_s45_discovery_v2/freeze_manifest.json'))
check('heldout plan hash',manifest['plan_hash'],digest(ROOT/'results/heldout_plan.json'))
check('discovery plan link',plan['discovery_plan_hash'],freeze['plan_hash'])
check('run completion',done['complete'],True)
check('families disjoint',set(plan['families'])&set(discplan['families']),set())
check('87 families',len(plan['families']),87)
for key in ['policy','structures','C33','C50','candidates','references','original59','control_rosters']:
    check('fixed '+key,plan[key]==discplan[key],True)
for f,h in manifest['code'].items():
    check('frozen code '+f,h,freeze['code'][f])
    check('local code '+f,digest(ROOT/'pilot_v2/s45'/f),h)

saved=read(OUT/'analysis/validation/validation_summary.json')
states={(s['label'],s['baseline']):s for s in saved['states']}
groups=defaultdict(list)
with (OUT/'analysis/validation/validation_behavior.csv').open(encoding='utf-8',newline='') as f:
    rows=list(csv.DictReader(f))
for r in rows:groups[(r['label'],r['baseline'])].append(r)
check('exported mean family rows',len(rows),522)
vectors={};diagnostics={}
for (label,base),rr in groups.items():
    rr=sorted(rr,key=lambda x:int(x['family']))
    check(label+' family coverage',[int(r['family']) for r in rr],sorted(plan['families']))
    m=np.array([[float(r[f'M_{c}_order{o}']) for c in ['x00','x10','x01','x11'] for o in (0,1)] for r in rr]).reshape(87,4,2)
    g_order=m[:,0]-m[:,1];b_order=(m[:,0]-m[:,1]-m[:,2]+m[:,3])/4
    g=g_order.mean(1);b=b_order.mean(1)
    v={'g':g,'b':b,'g_order':g_order,'b_order':b_order,'m':m}
    vectors[label]=v;s=states[label,base]
    vals={'g_mean':g.mean(),'b_mean':b.mean(),'g_interval':ci(g),'b_interval':ci(b),
          'g_order0':g_order[:,0].mean(),'g_order1':g_order[:,1].mean(),
          'b_order0':b_order[:,0].mean(),'b_order1':b_order[:,1].mean(),
          'g_abs_order_difference':abs(g_order[:,0]-g_order[:,1]).mean(),
          'g_abs_family_mean':abs(g).mean(),
          'query_contrast_original_facts':(m[:,0]-m[:,2]).mean(),
          'query_contrast_swapped_facts':(m[:,1]-m[:,3]).mean(),
          'fact_contrast_alt_query':(m[:,2]-m[:,3]).mean()}
    for k,vv in vals.items():check(label+'/'+k,np.array(vv),np.array(s[k]))
    check(label+'/exported b',b,np.array([float(r['b']) for r in rr]))
    gold=np.array([1,-1,-1,1])[None,:,None]
    acc=(m*gold>0).mean()
    for c,k in enumerate(['x00','x10','x01','x11']):
        for o in (0,1):check(f'{label}/accuracy/{k}/{o}',float((m[:,c,o]*gold[0,c,0]>0).mean()),s['accuracy'][f'{k}_order{o}'])
    # Originally queried chain first in order 0; swapping mothers exchanges its name.
    first=np.array([[1,-1],[-1,1],[1,-1],[-1,1]])[None,:,:]
    diagnostics[label]={'candidate_accuracy':float(acc),'first_chain_preference':float((m*first>0).mean()),'b':summary(b),'g':summary(g)}

for (label,base),s in states.items():
    check(f'{label}/{base}/F algebra',s['F'],s['g_mean']/states['full','mean']['g_mean'])
    if base=='mean':
        check(label+'/L',float(abs(vectors[label]['g']-vectors['full']['g']).mean()/abs(vectors['full']['g']).mean()),s['L'])

effects={}
for h,p,mn in [('L9H16','B+L9H16','B'),('L1H27','B','B-L1H27'),('RI31','B','B-R')]:
    db=vectors[p]['b']-vectors[mn]['b'];dg=vectors[p]['g']-vectors[mn]['g']
    bo=vectors[p]['b_order']-vectors[mn]['b_order'];mo=vectors[p]['m']-vectors[mn]['m']
    rec={'delta_b':summary(db),'delta_g_mean':summary(dg),'delta_b_order_means':bo.mean(0).tolist(),
         'delta_b_positive_by_order':(bo>0).sum(0).tolist(),
         'gold_oriented_cell_changes':(mo*np.array([1,-1,-1,1])[None,:,None]).mean((0,2)).tolist(),
         'candidate_accuracy_change':diagnostics[p]['candidate_accuracy']-diagnostics[mn]['candidate_accuracy'],
         'delta_g_donor_summary_only':states[p,'donor']['g_mean']-states[mn,'donor']['g_mean']}
    effects[h]=rec
    if h!='RI31':
        ss=next(x for x in saved['selected'] if x['candidate']==h)['d_b']
        check(h+'/delta b',rec['delta_b']['mean'],ss['mean'])
        check(h+'/delta b interval',np.array(rec['delta_b']['interval']),np.array(ss['interval']))
        check(h+'/all positive',rec['delta_b']['positive']/87,ss['sign_fraction'])

gate=read(OUT/'runs/gate_r0/gate.json')
check('gate passed',gate['passed'],True)
gate_errors={k:max(float(p['errors'].get(k) or 0) for p in gate['pairs']) for k in ['x00_inert','x10_inert','x00_sdpa','x10_sdpa','all_live_equals_full','prefix_invariance_x00_x01','prefix_invariance_x10_x11']}
result={'checks':len(checks),'failures':[x for x in checks if not x['passed']],
        'families':87,'export_rows':len(rows),'export_baselines':sorted({r['baseline'] for r in rows}),
        'gate_max_errors':gate_errors,'states':diagnostics,'effects':effects,
        'coverage_limit':'Mean four-cell family margins exported; donor family margins, raw worker records, top-token identities and mean-bank files absent.',
        'label_logic':'analyze_validation labels functional reproduction before sensitivity and does not inspect donor effects when assigning the final label.'}
(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(OUT/'independent_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps({'checks':len(checks),'failures':result['failures'],'effects':effects,'state_overview':{k:{kk:v[kk] for kk in ['candidate_accuracy','first_chain_preference']} for k,v in diagnostics.items()},'gates':gate_errors},indent=2))
