"""Pure planning/statistics helpers; no GPU dependency."""
import collections
import json
import random
from pathlib import Path
import numpy as np


def atomic_json(path,obj):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=1,allow_nan=False),encoding='utf-8');temp.replace(path)


def groups(devices, minimum=2):
    if not 2<=len(devices)<=6 or minimum<1 or minimum>len(devices) or len(set(devices))!=len(devices):
        raise ValueError('Use 2--6 allocated GPUs, with unique IDs and a valid replica size.')
    n=len(devices)//minimum; size,extra=divmod(len(devices),n);out=[];start=0
    for i in range(n):
        end=start+size+(i<extra);out.append(devices[start:end]);start=end
    return out


def make_pairs(rows,prompt_ids):
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate data IDs')
    selected=[by_id[i] for i in prompt_ids]
    paired=collections.defaultdict(dict)
    for r in selected:
        if r['split']!='discovery':raise ValueError('Validation data in stage-1 selection')
        if r['variant'] in ('base','corrupted'):
            key=(r['family'],r['option_order'])
            if r['variant'] in paired[key]:raise ValueError('Duplicate pair cell')
            paired[key][r['variant']]=r
    result=[]
    for (family,order),v in sorted(paired.items()):
        if set(v)!={'base','corrupted'}:raise ValueError('Incomplete twin pair')
        a,b=v['base'],v['corrupted']
        if set(a['candidates'])!=set(b['candidates']) or a['gold']==b['gold']:
            raise ValueError('Twins must exchange the same two candidate answers')
        if a['demo_ids']!=b['demo_ids']:raise ValueError('Different ICL demonstrations in twins')
        result.append(dict(id=a['id'],family=family,order=order,clean=a,corr=b))
    counts=collections.Counter(p['family'] for p in result)
    if not counts or set(counts.values())!={2}:raise ValueError('Need both orders in every family')
    return result


def plan_work(pairs,replicas,exact_families=20,seed=20260914):
    fams=sorted({p['family'] for p in pairs})
    if not 1<=exact_families<=len(fams):raise ValueError('Invalid exact subset size')
    exact=set(random.Random(seed).sample(fams,exact_families))
    # Families remain intact. Balance heavy all-head exact tasks first, then
    # light attribution-only tasks. Prompt character length estimates relative cost.
    costs={f:sum(len(p['clean']['prompt']) for p in pairs if p['family']==f)*(1024 if f in exact else 4) for f in fams}
    loads=[0]*replicas;assignment={}
    for f in sorted(fams,key=lambda f:(-costs[f],f)):
        i=min(range(replicas),key=lambda i:(loads[i],i));assignment[f]=i;loads[i]+=costs[f]
    return [dict(p,worker=assignment[p['family']],exact_all=p['family'] in exact) for p in pairs]


def rankdata(a):
    a=np.asarray(a);order=np.argsort(a,kind='stable');rank=np.empty(len(a),float);i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        rank[order[i:j]]=(i+j-1)/2;i=j
    return rank


def spearman(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or not(np.isfinite(a).all() and np.isfinite(b).all()):return None
    x,y=rankdata(a),rankdata(b)
    if not x.std() or not y.std():return None
    return float(np.corrcoef(x,y)[0,1])
