"""All-head, family-balanced role means with two evaluation modes.

Discovery: leave-one-family-out (the evaluated family's own mean is subtracted).
Validation: the fixed full discovery bank, with no subtraction or addition.
Statistics come from discovery x00/x10 cells only (both orders); changed-query and
held-out inputs never enter the bank. Keys are syntax/slot/bucket roles only.
"""
import collections
from pathlib import Path
import numpy as np
from stage3_common import digest,json_read,json_write
from s42_plan import key_levels
from s45_plan import POLICY

def accumulate(sums,counts,keys,values,vocab):
    """values: (heads, positions, DH) float64; one contribution per event at every fallback level."""
    for j,key in enumerate(keys):
        if key is None:continue
        for k in key_levels(key):
            i=vocab[k];sums[:,i]+=values[:,j];counts[i]+=1

def reduce_bank(inputs,runs,out,heads=1024):
    """Sum FAMILY means (equal family weight). Idempotent: an existing bank must match."""
    plan=json_read(inputs/'plan.json');families=plan['families'];K=len(plan['mean_keys']);records={}
    for run in runs:
        ident=json_read(run/'manifest.json')['identity']
        if not json_read(run/'done.json')['complete']:raise ValueError('Mean worker incomplete: '+str(run))
        for f in sorted((run/'families').glob('*.npz')):
            mark=json_read(f.with_suffix('.ok'))
            if mark['identity']!=ident or mark['sha256']!=digest(f):raise ValueError('Mean family checksum: '+str(f))
            with np.load(f,allow_pickle=False) as z:
                fam=int(z['family']);sums=z['sums'];counts=z['counts']
                if fam in records or fam not in families:raise ValueError('Mean family coverage')
                if sums.shape!=(heads,K,z['sums'].shape[-1]) or counts.shape!=(K,) or not np.isfinite(sums).all():raise ValueError('Mean shape/finite')
                records[fam]=((sums/np.maximum(counts,1)[None,:,None]).astype(np.float32),counts>0)
    if set(records)!=set(families):raise ValueError('Missing mean families')
    out.mkdir(parents=True,exist_ok=True);(out/'families').mkdir(exist_ok=True)
    support=np.stack([records[f][1] for f in families]);total=np.zeros(records[families[0]][0].shape,dtype=np.float64)
    for f in families:total+=records[f][0]
    if (out/'mean_bank_manifest.json').exists():
        old=json_read(out/'mean_bank_manifest.json');bank=MeanBank(out,inputs,'full')
        if not np.array_equal(bank.support,support) or not np.allclose(bank.total,total):raise ValueError('Changed source family means on resume')
        return old
    for f in families:np.save(out/'families'/f'{f:03d}.npy',records[f][0])
    np.save(out/'total.npy',total);np.save(out/'support.npy',support)
    manifest=dict(policy=POLICY,plan_hash=digest(inputs/'plan.json'),keys=plan['mean_keys'],families=families,heads=heads,
        files={'total.npy':digest(out/'total.npy'),'support.npy':digest(out/'support.npy'),**{f'families/{f:03d}.npy':digest(out/'families'/f'{f:03d}.npy') for f in families}},
        source_manifests=[digest(r/'manifest.json') for r in runs],
        statistics='discovery x00/x10 cells, both orders, all 1024 heads; equal family weight',
        modes=dict(lofo='discovery: subtract the evaluated family',full='held-out: fixed bank, no subtraction or addition'))
    json_write(out/'mean_bank_manifest.json',manifest);return manifest

class MeanBank:
    def __init__(self,path,inputs,mode):
        if mode not in ['lofo','full']:raise ValueError('Mean bank mode')
        path=Path(path);meta=json_read(path/'mean_bank_manifest.json');self.path=path;self.mode=mode;self.meta=meta
        plan=json_read(inputs/'plan.json')
        ref=plan.get('discovery_plan_hash',None) or digest(inputs/'plan.json')
        if meta['policy']!=POLICY or meta['plan_hash']!=ref:raise ValueError('Mean bank identity')
        if meta['keys']!=plan['mean_keys']:raise ValueError('Mean bank vocabulary')
        for name in ['total.npy','support.npy']:
            if digest(path/name)!=meta['files'][name]:raise ValueError('Mean bank file hash: '+name)
        self.total=np.load(path/'total.npy');self.support=np.load(path/'support.npy');self.families=meta['families']
        self.keys={s:i for i,s in enumerate(meta['keys'])};self.count=self.support.sum(0);self._fam={}
        if not np.isfinite(self.total).all() or self.total.shape[0]!=meta['heads']:raise ValueError('Invalid mean bank')
    def family_means(self,family):
        if family not in self._fam:
            name=f'families/{family:03d}.npy'
            if digest(self.path/name)!=self.meta['files'][name]:raise ValueError('Mean bank family hash')
            self._fam={family:np.load(self.path/name)}
        return self._fam[family]
    def identity(self):return dict(manifest_hash=digest(self.path/'mean_bank_manifest.json'),mode=self.mode)
    def cell_vectors(self,family,keys):
        """Replacement vectors for every keyed position: (positions, heads, DH) float32, and fallback counts."""
        held=family in self.families
        if self.mode=='lofo' and not held:raise ValueError('LOFO evaluation of a family outside the bank')
        if self.mode=='full' and held:raise ValueError('Full-bank evaluation must not touch a discovery family')
        fi=self.families.index(family) if held else None;fm=self.family_means(family) if held else None
        out=[];info=collections.Counter();positions=[]
        for j,key in enumerate(keys):
            if key is None:continue
            for level,label in enumerate(key_levels(key)):
                if label not in self.keys:continue
                k=self.keys[label];n=int(self.count[k])-(int(self.support[fi,k]) if held else 0)
                if n>=POLICY['mean_min_families']:
                    v=self.total[:,k,:]-(fm[:,k,:] if held else 0.)
                    out.append((v/n).astype(np.float32));info[str(level)]+=1;positions.append(j);break
            else:raise ValueError('Too few families even for component fallback')
        return positions,np.stack(out),dict(info)
