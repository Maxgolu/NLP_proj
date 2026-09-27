"""Generate report tables from audited exports (no model execution)."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/stage4_s45_discovery_v2'
DEST=ROOT/'חומר כתוב/Stage45_Overleaf/tables'
DEST.mkdir(parents=True,exist_ok=True)
S=json.loads((Path(__file__).parent/'audit_summary.json').read_text())
A=json.loads((SRC/'analysis/stage_a/stage_a_summary.json').read_text())
def f(x):return f'{x:+.3f}'
def ci(r):return '['+', '.join(f(x) for x in r['interval'])+']'
def write(name,spec,header,rows):
    out='\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
    out+='\n'.join(' & '.join(row)+' \\\\' for row in rows)
    out+='\n\\bottomrule\n\\end{tabular}\n'
    (DEST/(name+'.tex')).write_text(out,encoding='utf-8')
a={r['label']:r for r in A}
rows=[]
for h in ['full','empty','T1','T2','T3','T4','T5','C33','C50']:
    r=a[h];acc=sum(v for k,v in r['accuracy'].items() if not k.startswith('top_'))/8
    rows.append([h,f(r['g_mean']),f"{r['F']:.3f}",f"{r['L']:.3f}",f(r['b_mean']),f"{100*acc:.1f}\\%",'pass' if r['passes'] else 'fail'])
write('retention','lrrrrrl','Mask & $g$ & $F$ & $L$ & $b$ & Candidate acc. & Guards',rows)
rows=[]
for r in sorted(S['candidates'],key=lambda r:(r['kind']!='candidate',-r['mean'])):
    h=r['candidate'];h='RI31 jointly' if h=='R' else h+(' (ref.)' if r['kind']=='reference' else '')
    rows.append([h,f(r['mean']),ci(r),f"{round(20*r['sign_fraction'])}/20",f(r['order0']),f(r['order1']),r['classification'].replace('_',' ')])
write('candidates','lrlrrrl','Head/group & $\Delta b$ & 95\% interval & Sign & Order 0 & Order 1 & Class',rows)
rs={(r['state'],r['anchor']):r for r in S['routes']};bid=a['C50']['state'];rows=[]
for t in ['T1','T2','T3','T4','T5']:
    r=rs[bid,t];full=rs['full',t]
    rows.append([t,f(full['mean']),f(r['mean']),ci(r),f"{100*r['mean']/full['mean']:.1f}\\%",f(r['order0']),f(r['order1']),r['classification'].replace('_',' ')])
write('routes','lrrlrrrl','Route & Full & $B$ & 95\% interval in $B$ & Ratio & Order 0 & Order 1 & Class',rows)
ex={(r['label'],r['baseline']):r for r in S['extension']};rows=[]
for n,bas in [('full','mean'),('empty','mean')]+[(n,base) for n in ['B','B-R','B+L9H16','B-L1H27'] for base in ['mean','donor']]:
    r=ex[n,bas];c='['+', '.join(f(x) for x in r['g_interval'])+']'
    rows.append([n.replace('B-R',r'$B-R(B)$'),bas,f(r['g_mean']),c,f"{r['F']:.3f}",f(r['g_order0']),f(r['g_order1'])])
write('extension','llrlrrr','Configuration & Baseline & $g$ & 95\% interval & $F$ & Order 0 & Order 1',rows)
rows=[]
for h in ['L9H16','L1H27','R(B)']:
    rr={r['baseline']:r for r in S['extension_differences'] if r['candidate']==h}
    rows.append([h,f(rr['mean']['delta_g']),f(rr['donor']['delta_g'])])
write('sensitivity','lrr','Contribution & Mean $\Delta g$ & Donor $\Delta g$',rows)
df=pd.read_csv(SRC/'analysis/stage_a/core_behavior.csv');rows=[]
for lab in ['full','C50']:
    d=df[df.label==lab]
    for o in [0,1]:rows.append([lab,str(o)]+[f(d[f'M_{c}_order{o}'].mean()) for c in ['x00','x10','x01','x11']])
write('cells','llrrrr','Mask & Order & $M_{00}$ & $M_{10}$ & $M_{01}$ & $M_{11}$',rows)
print('Wrote six tables from audited data.')
