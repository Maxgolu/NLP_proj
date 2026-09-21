"""Mandatory head diagnostics; all returned projections are pre-attention-norm diagnostics."""
import collections
import re
import numpy as np
from stage3_common import POLICY, SEED, annotate, normalized_ri


def checked_capture(e, ids, heads, rows, context_inputs=False):
    c=e.capture(ids,heads,rows,context_inputs)
    err=max(x['reconstruction_error'] for x in c['heads'].values())
    if not np.isfinite(err) or err>POLICY['reconstruction_tolerance']:
        raise ValueError(f'Attention/value reconstruction failed: {err}')
    return c


def position_roles(row, ann, j):
    a,b=ann['offsets'][j]; roles=[]
    for f in ann['facts']:
        for role in ['source','target']:
            pos=f[role+'_positions']
            if j in pos:
                roles.append(dict(fact=f['index'],query=f['query'],role=role,
                                  first=j==pos[0],last=j==pos[-1]))
    text=row['prompt'][a:b].strip()
    fact_ids=[]
    for f in ann['facts']:
        original=row['facts'][f['index']];line=original.get('line','')
        start=row['prompt'].rfind(line,0,max(original['head_span'][1],original['tail_span'][1])+len(line)) if line else -1
        if start>=0 and a<start+len(line) and b>start:fact_ids.append(f['index'])
    return dict(j=j,token=ann['ids'][j],text=text,entities=roles,fact_indices=fact_ids,
                question=a>=row['prompt'].rfind('Question:'),
                punctuation=bool(text and not any(c.isalnum() for c in text)),
                final=j==len(ann['ids'])-1)


def causal_pair(e, pair, heads, plan, ref, pair_index):
    spec=e.encode(pair); n=spec['n']; rows=list(range(n))
    if spec['shared_prefix']!=pair['shared_prefix']: raise ValueError('Answer prefix changed')
    base=e.readout(spec['clean'],spec['g'],spec['d'])
    corrupt=e.readout(spec['corr'],spec['g'],spec['d'])
    if np.max(np.abs(np.array([base[0],corrupt[0]])-ref['baselines'][pair_index]))>POLICY['probe_tolerance']:
        raise ValueError('Saved baseline drift: '+pair['id'])
    clean=checked_capture(e,spec['clean'][:,:n],heads,rows)
    corr=checked_capture(e,spec['corr'][:,:n],heads,rows)
    first=next(j for j in rows if spec['clean'][0,j]!=spec['corr'][0,j])
    ann=annotate(e.tok,pair['clean']); test=set(ann['test_positions'])
    positions=[j for j in range(first,n) if j in test] if pair['exact_all'] else []
    F=[]; av=[]; profile=[]; reverse=[]; reverse_heads=[]; f_reused=[]; max_drift=0.
    for h in heads:
        ix=plan['heads'].index(h); donor=corr['heads'][h]['z']
        if np.isfinite(ref['scopeF_delta'][pair_index,ix]):
            f=ref['scopeF_readout'][pair_index,ix].copy(); f_reused.append(True)
        else:
            f=e.patched(spec,h,[n-1],donor[n-1:n]);f_reused.append(False)
        F.append(f)
        # Joint and self reconstructions are checked on EVERY measured pair/head.
        measurements=np.array([e.patched(spec,h,[n-1],v) for v in e.mix_vectors(clean,corr,h,n-1)])
        drift=float(max(np.max(np.abs(measurements[0]-base)),np.max(np.abs(measurements[3]-f))))
        max_drift=max(max_drift,drift)
        if not np.isfinite(measurements).all() or drift>POLICY['probe_tolerance']:
            raise ValueError(f'AV joint/self readout mismatch {pair["id"]}, head {h}: {drift}')
        av.append(measurements)
        profile.append([f if j==n-1 else e.patched(spec,h,[j],donor[j:j+1]) for j in positions])
        if h in plan['reverse_heads']:
            reverse_heads.append(h);z=clean['heads'][h]['z']
            reverse.append([e.patched(spec,h,rows,z,True),e.patched(spec,h,[n-1],z[n-1:n],True)])
    return dict(pair_id=pair['id'],family=pair['family'],order=pair['order'],heads=heads,
                baseline=base,corrupt_baseline=corrupt,n=n,first_diff=first,
                positions=positions,scopeF_readout=F,scopeF_reused=f_reused,av_readout=av,
                position_readout=np.array(profile).reshape(len(heads),len(positions),3),
                reverse_heads=reverse_heads,reverse_readout=np.array(reverse).reshape(-1,2,3),
                av_max_drift=max_drift),[position_roles(pair['clean'],ann,j) for j in positions]


def matched_ri(e, row, capture, events):
    """Rescore the saved event and controls verbatim; verify raw replication first."""
    ids=row['token_ids']; byhead=collections.defaultdict(list); results=[]
    for ev in events:
        if ev['scores']: byhead[ev['layer']*32+ev['head']].append(ev)
    embed=e.model.get_input_embeddings().weight
    for h,evs in byhead.items():
        positions=sorted({x['j'] for x in evs}); cache={}
        for start in range(0,len(positions),8):
            chunk=positions[start:start+8]
            raw=embed[e.t.tensor([ids[j] for j in chunk],device=embed.device)]
            ctx=capture['value_inputs'][h//e.H][chunk]
            # Full-vocabulary fp32 softmax, then the same unique-visible-token normalization.
            prob=[]
            for x in [raw,ctx]:
                logits=e.projected_logits(h,x,True)
                prob.append(e.t.from_numpy(logits).softmax(-1).numpy())
            for k,j in enumerate(chunk):
                visible=sorted(set(ids[:j+1]));cache[j]=(visible,*[normalized_ri(p[k,visible]) for p in prob])
        for ev in evs:
            visible,raw,ctx=cache[ev['j']]; lookup={v:i for i,v in enumerate(visible)}
            if raw is None: raise ValueError('Saved scored RI event became undefined')
            for anchor,s in ev['scores'].items():
                checked=[(s['target_id'],s['target'])]+[(x['token_id'],x['score']) for kind in ['names','words'] for x in s[kind]]
                errors=[]
                for token,saved in checked:
                    actual=float(raw[lookup[token]]);errors.append(abs(actual-saved))
                    if not np.isclose(actual,saved,atol=POLICY['ri_atol'],rtol=POLICY['ri_rtol']):
                        raise ValueError(f'Raw RI replication failed: {ev["id"]}, {h}, {ev["event_id"]}, {anchor}: {actual} vs {saved}')
                scores={}
                for label,values in [('raw',raw),('contextual',ctx)]:
                    if values is None: scores[label]=None;continue
                    target=float(values[lookup[s['target_id']]])
                    scores[label]=dict(target=target)
                    for kind in ['names','words']:
                        v=[float(values[lookup[x['token_id']]]) for x in s[kind]]
                        scores[label][kind]=v
                        scores[label][kind+'_gap']=target-float(np.mean(v)) if v else None
                results.append(dict(head=h,id=ev['id'],event_id=ev['event_id'],family=ev['family'],order=ev['order'],
                                    variant=ev['variant'],fact=ev['fact'],query_fact=ev['query_fact'],j=ev['j'],
                                    position=ev['position'],anchor=anchor,scores=scores,
                                    raw_replication_error=max(errors),common_valid=ctx is not None))
    return results


def anatomy(e,row,heads,capture,ann):
    """All test rows; full attention retained separately. Ranks use a declared local population."""
    ids=ann['ids'];rows=ann['test_positions']; records=[]
    anchors=sorted({ids[p] for f in ann['facts'] for role in ['source','target'] for p in [f[role+'_positions'][0],f[role+'_positions'][-1]]})
    population=sorted(set(anchors)|{ids[j] for j in rows});lookup={x:i for i,x in enumerate(population)}
    embed=e.model.get_input_embeddings().weight
    for h in heads:
        cache=capture['heads'][h]; ix=[cache['rows'].index(j) for j in rows]
        out=e.projected_logits(h,cache['z'][ix],token_ids=population)
        raw=e.projected_logits(h,embed[e.t.tensor([ids[j] for j in rows],device=embed.device)],True,population)
        for k,j in enumerate(rows):
            a=cache['pattern'][ix[k]].numpy();arg=int(a.argmax());top=np.sort(a[:j+1])[-2:]
            ratio=float(top[-1]/max(top[-2],1e-30)) if len(top)>1 else None
            facts=[]; visible_tokens={ids[j]}
            for f in ann['facts']:
                entry=dict(fact=f['index'],query=f['query'])
                for role in ['source','target']:
                    pos=f[role+'_positions']; entry[role+'_mass']=float(a[pos].sum())
                    entry[role+'_first']=float(a[pos[0]]);entry[role+'_last']=float(a[pos[-1]])
                    entry[role+'_visible']=pos[-1]<=j
                    if pos[-1]<=j:visible_tokens.update([ids[pos[0]],ids[pos[-1]]])
                entry['qk_pass']=bool(f['target_positions'][-1]<=j and f['source_positions'][-1]<=j and
                                      arg==f['source_positions'][-1] and ratio is not None and ratio>POLICY['dominance'])
                facts.append(entry)
            vt=sorted(visible_tokens); col=[lookup[x] for x in vt]
            projections={}
            for label,v in [('output',out[k,col]),('raw',raw[k,col])]:
                projections[label]=dict(logits=v.tolist(),rank_min=[int(1+sum(v>x)) for x in v],
                                        rank_max=[int(sum(v>=x)) for x in v])
            records.append(dict(head=h,id=row['id'],family=row['family'],order=row['order'],variant=row['variant'],
                                **position_roles(row,ann,j),argmax=arg,offset=j-arg,self_attention=float(a[j]),
                                previous_attention=float(a[j-1]) if j else None,dominance=ratio,
                                argmax_text=e.tok.decode([ids[arg]]),facts=facts,projection_tokens=vt,projections=projections))
    return records


def token_pool(tok,prompts):
    # First/last test-name token IDs, with equal weight per distinct token.
    names=set()
    for r in prompts:
        a=annotate(tok,r)
        for f in a['facts']:
            for role in ['source','target']:
                p=f[role+'_positions'];names.update([a['ids'][p[0]],a['ids'][p[-1]]])
    candidates=[]
    for i in range(len(tok)):
        if i in names or i in tok.all_special_ids:continue
        s=tok.decode([i])
        if re.fullmatch(r' [a-z]{3,12}',s) and tok.encode(s,add_special_tokens=False)==[i]:candidates.append(i)
    if len(candidates)<POLICY['weight_nonname_tokens']:raise ValueError('Insufficient ordinary-word reference pool')
    sample=np.random.default_rng(SEED).choice(candidates,POLICY['weight_nonname_tokens'],replace=False)
    return dict(name_tokens=sorted(names),nonname_tokens=sorted(map(int,sample)),seed=SEED)


def copying_weights(e,heads,pool):
    ids=pool['name_tokens']+pool['nonname_tokens'];split=len(pool['name_tokens'])
    emb=e.model.get_input_embeddings().weight;stats=[];matrices=[]
    for h in heads:
        chunks=[]
        for j in range(0,len(ids),32):
            x=emb[e.t.tensor(ids[j:j+32],device=emb.device)]
            chunks.append(e.projected_logits(h,x,True,ids))
        M=np.concatenate(chunks);matrices.append(M)
        for label,sl in [('name',slice(0,split)),('nonname',slice(split,None))]:
            m=M[sl,sl];diag=m.diagonal();n=len(m)
            stats.append(dict(head=h,population=label,tokens=n,self_minus_others=float(np.mean(diag-(m.sum(1)-diag)/(n-1))),
                              mean_self_rank_min=float(np.mean(1+(m>diag[:,None]).sum(1))),
                              mean_self_rank_max=float(np.mean((m>=diag[:,None]).sum(1))),
                              self_top_fraction=float(np.mean(diag==m.max(1)))))
    return dict(heads=heads,tokens=ids,name_count=split,logits=np.array(matrices)),stats


def synthetic(e,heads,pool):
    rng=np.random.default_rng(SEED+1);tokens=pool['nonname_tokens'];out=[]
    for kind in ['repeated_sequence','key_value_retrieval']:
        for trial in range(POLICY['synthetic_trials']):
            if kind=='repeated_sequence':
                seq=list(map(int,rng.choice(tokens,POLICY['repeat_length'],replace=False)))
                ids=seq+seq; L=len(seq);rows=list(range(L,2*L-1));source=list(range(1,L));gold=seq[1:]
                candidates=seq
            else:
                chosen=list(map(int,rng.choice(tokens,2*POLICY['retrieval_records'],replace=False)))
                keys=chosen[::2];values=chosen[1::2];target=trial%len(keys)
                ids=e.tok.encode('Copy the value paired with the requested key.\n',add_special_tokens=False)
                positions=[]
                for key,value in zip(keys,values):
                    ids+=[key]+e.tok.encode(':',add_special_tokens=False);positions.append(len(ids));ids+=[value]+e.tok.encode('\n',add_special_tokens=False)
                ids+=e.tok.encode('Key:',add_special_tokens=False)+[keys[target]]+e.tok.encode('\nValue:',add_special_tokens=False)
                rows=[len(ids)-1];source=[positions[target]];gold=[values[target]];candidates=values
            tensor=e.t.tensor([ids],device=e.device);cap=checked_capture(e,tensor,heads,rows)
            # Model predictions at every probe row, without attention-capture changes.
            with e.t.no_grad(): logits=e.model(tensor,use_cache=False).logits[0,rows].float().cpu()
            correct=(logits.argmax(-1).numpy()==gold)
            for h in heads:
                c=cap['heads'][h];projection=e.projected_logits(h,c['z'],token_ids=candidates)
                for k,j in enumerate(rows):
                    a=c['pattern'][k].numpy();v=projection[k];g=candidates.index(gold[k])
                    out.append(dict(head=h,kind=kind,trial=trial,j=j,source=source[k],gold_token=gold[k],
                                    model_correct=bool(correct[k]),model_gold_minus_best_other=float(logits[k,gold[k]]-logits[k,[x for x in candidates if x!=gold[k]]].max()),
                                    source_attention=float(a[source[k]]),source_argmax=bool(a.argmax()==source[k]),
                                    previous_attention=float(a[j-1]),self_attention=float(a[j]),
                                    output_gold_minus_others=float(v[g]-np.delete(v,g).mean()),
                                    output_gold_rank_min=int(1+sum(v>v[g])),input_ids=ids))
    return out
