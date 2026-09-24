"""Family-balanced role means. Statistics contain discovery families only."""
import collections
from pathlib import Path
import numpy as np
from stage3_common import json_read,json_write,digest,npz_write
from s42_plan import key_levels,POLICY


def accumulate(sums,counts,keys,values,vocabulary):
    # Every event contributes once at each fallback level. Names are not labels.
    for j,key in enumerate(keys):
        if key is None:continue
        for k in key_levels(key):
            i=vocabulary[k];sums[:,i]+=values[:,j];counts[i]+=1


def reduce_bank(inputs,request,runs,out):
    from s42_run import mean_identity
    plan=json_read(inputs/'plan.json');req=json_read(request);families=plan['families'];heads=req['heads']
    records={}
    for rank,run in enumerate(runs):
        ident=mean_identity(inputs,request,rank,len(runs))
        if json_read(run/'manifest.json')['identity']!=ident or not json_read(run/'done.json')['complete']:raise ValueError('Mean worker identity/completion')
        for f in (run/'families').glob('*.npz'):
            mark=json_read(f.with_suffix('.ok'))
            if mark['identity']!=ident or mark['sha256']!=digest(f):raise ValueError('Mean family checksum')
            with np.load(f,allow_pickle=False) as z:
                fam=int(z['family'])
                if fam in records or fam not in families[rank::len(runs)]:raise ValueError('Mean family coverage')
                counts=z['counts'];sums=z['sums']
                if sums.shape[:2]!=(len(heads),len(plan['mean_keys'])) or counts.shape!=(len(plan['mean_keys']),):raise ValueError('Mean shape')
                if not np.isfinite(sums).all() or np.any(counts<0):raise ValueError('Nonfinite mean')
                records[fam]=(sums/np.maximum(counts,1)[None,:,None],counts>0)
    if set(records)!=set(families):raise ValueError('Missing mean families')
    # Global sums are sums of FAMILY means, not token/event sums.
    fm=np.stack([records[f][0] for f in families]).astype(np.float32)
    support=np.stack([records[f][1] for f in families])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        bank=MeanBank(out,inputs);meta=json_read(out/'manifest.json')
        if meta['request_hash']!=digest(request) or bank.heads!=heads or bank.families!=families:raise ValueError('Mean bank resume identity')
        if not np.array_equal(bank.fm,fm) or not np.array_equal(bank.support,support):raise ValueError('Changed source family means')
        return
    npz_write(out/'bank.npz',family_means=fm,support=support,families=np.array(families),heads=np.array(heads))
    json_write(out/'manifest.json',dict(plan_hash=digest(inputs/'plan.json'),request_hash=digest(request),
        sha256=digest(out/'bank.npz'),keys=plan['mean_keys'],policy=POLICY,source_manifests=[digest(r/'manifest.json') for r in runs]))


class MeanBank:
    def __init__(self,path,inputs):
        meta=json_read(path/'manifest.json')
        if meta['policy']!=POLICY or meta['plan_hash']!=digest(inputs/'plan.json') or meta['sha256']!=digest(path/'bank.npz'):raise ValueError('Mean bank identity')
        with np.load(path/'bank.npz',allow_pickle=False) as z:
            self.fm=z['family_means'];self.support=z['support'];self.families=z['families'].tolist();self.heads=z['heads'].tolist()
        self.keys={s:i for i,s in enumerate(meta['keys'])};self.total=self.fm.astype(np.float64).sum(0);self.count=self.support.sum(0)
        if len(set(self.families))!=len(self.families) or not np.isfinite(self.fm).all():raise ValueError('Invalid mean bank')

    def vectors(self,head,family,keys):
        hi=self.heads.index(head);fi=self.families.index(family);result=[];info=collections.Counter()
        for key in keys:
            for level,label in enumerate(key_levels(key)):
                k=self.keys[label];n=int(self.count[k])-int(self.support[fi,k])
                if n>=POLICY['mean_min_families']:
                    value=(self.total[hi,k]-self.fm[fi,hi,k])/n
                    result.append(value);info[str(level)]+=1;break
            else:raise ValueError('Too few families even for component fallback')
        return np.asarray(result,dtype=np.float32),dict(info)
