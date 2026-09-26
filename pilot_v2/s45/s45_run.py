"""S4.5 worker: gated, restartable, checksummed measurements. CPU subcommands import no torch."""
import argparse
import collections
import os
from pathlib import Path
import signal
import numpy as np
from stage3_common import digest,json_read,json_write,read_lines,run_lock,npz_write
from stage4_run import check,error,CODE_FILES as S41_FILES
from s45_plan import load,assigned,record_keys,state,ids,hid,hname,STRUCTURES,C33,CANDIDATES,attachment,CELLS,POLICY

ROOT=Path(__file__).resolve().parent
CODE_FILES=sorted(set(S41_FILES+['s42_plan.py','s42_engine.py','s42_means.py','s43_engine.py','s45_plan.py','s45_means.py','s45_engine.py',
    's45_run.py','s45_analyze.py','s45_freeze.py','s45_pipeline.py','test_s45.py','test_s43.py','test_s42.py']))

def code_identity():return {n:digest(ROOT/n) for n in CODE_FILES if (ROOT/n).exists()}
def identity(inputs,schedule,shard,shards,bank=None,mode=None):
    return dict(plan_hash=digest(inputs/'plan.json'),schedule_hash=digest(schedule),code=code_identity(),shard=shard,shards=shards,
                bank_hash=digest(bank/'mean_bank_manifest.json') if bank else None,bank_mode=mode)
def mean_identity(inputs,shard,shards):return dict(plan_hash=digest(inputs/'plan.json'),code=code_identity(),shard=shard,shards=shards,kind='means')
def key(r):return (r['pair_id'],r['job_id'],r['cell'])
def reuse_key(r):
    if r['kind']=='behavior':return ('behavior',r['pair_id'],r['state'],r['baseline'],r['cell'],r['bank_hash'])
    return (r['kind'],r['pair_id'],r['state'],r['key'],r['direction'],r['bank_hash'])

def chunks(out,ident):
    rows=[]
    for path in sorted((out/'chunks').glob('*.json')):
        marker=path.with_suffix('.ok')
        if not marker.exists():continue
        mark=json_read(marker)
        if mark['identity']!=ident or mark['sha256']!=digest(path):raise ValueError('Checkpoint identity/hash: '+str(path))
        rows.extend(json_read(path))
    return rows

def write_chunk(out,ident,number,rows):
    path=out/'chunks'/f'{number:07d}.json';json_write(path,rows);json_write(path.with_suffix('.ok'),dict(identity=ident,sha256=digest(path)))

def verify(out,inputs,schedule,require_done=True):
    ident=json_read(out/'manifest.json')['identity'];plan,sc,pairs=load(inputs,schedule)
    if ident['plan_hash']!=digest(inputs/'plan.json') or ident['schedule_hash']!=digest(schedule):raise ValueError('Run/input mismatch')
    needs_bank=any(j['kind']!='behavior' or j['baseline']=='mean' for j in sc['jobs']) and any(sc['states'][j['state']]['live'] is not None for j in sc['jobs'])
    if needs_bank and ident['bank_hash'] is None:raise ValueError('Mean bank required')
    if ident['bank_mode'] is not None and ident['bank_mode']!=('full' if sc['population']=='heldout' else 'lofo'):raise ValueError('Mean bank mode/population')
    rows=chunks(out,ident);keys=[key(r) for r in rows];want=record_keys(sc,pairs,ident['shard'],ident['shards'])
    if len(keys)!=len(set(keys)) or set(keys)!=want:raise ValueError(f'Incomplete/duplicate coverage {len(set(keys))}/{len(want)}')
    by={p['id']:p for p in pairs};jobs={j['id']:j for j in sc['jobs']}
    for r in rows:
        p=by[r['pair_id']];j=jobs[r['job_id']]
        if r['family']!=p['family'] or r['order']!=p['order'] or r['state']!=j['state'] or r['split']!=p['split']:raise ValueError('Record labels')
        if r['kind']=='behavior':
            for v in ['margin','logit_a','logit_b']:
                if not np.isfinite(r[v]):raise ValueError('Nonfinite readout')
            check(abs(r['margin']-(r['logit_a']-r['logit_b'])),1e-4,'stored margin')
            if r['gold_side']!=('a' if r['cell'] in ['x00','x11'] else 'b'):raise ValueError('Cell gold side')
        else:
            if not np.isfinite(r['effect']):raise ValueError('Nonfinite effect')
            v=r['intact_margin']-r['endpoint_margin'] if r['direction']=='noise' else r['endpoint_margin']-r['intact_margin']
            check(abs(v-r['effect']),1e-4,'stored effect')
    if require_done and not json_read(out/'done.json')['complete']:raise ValueError('Missing completion')
    if not json_read(out/'gate.json')['passed']:raise ValueError('Missing gate')
    return rows

def prior_cache(inputs,prior):
    found={};sources=[]
    for item in prior:
        schedule=Path(item['schedule'])
        for folder in item['runs']:
            run=Path(folder);rows=verify(run,inputs,schedule);sources.append(digest(run/'manifest.json'))
            for r in rows:
                k=reuse_key(r)
                if k in found:
                    a,b=found[k],r;check(abs(a.get('margin',a.get('effect'))-b.get('margin',b.get('effect'))),.001,'duplicate cached record')
                found.setdefault(k,r)
    return found,sources

# ---------------------------------------------------------------- model side ----
def start_model():
    import unittest
    from test_s42 import TinyJoint
    from test_stage4 import TinyRoutes
    from test_s43 import TinyS43
    from test_s45 import TinyBackground
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [TinyRoutes,TinyJoint,TinyS43,TinyBackground])
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.skipped:raise ValueError('Required tiny model gate failed/skipped')
    from s45_engine import S45Engine
    if os.environ.get('S45_TINY_MODEL'):
        # Test-only: a random 32x32-head OLMo2 so the whole pipeline can be exercised on CPU. Never a measurement.
        import torch
        from transformers import Olmo2Config,Olmo2ForCausalLM
        torch.manual_seed(int(os.environ['S45_TINY_MODEL']));cfg=Olmo2Config(vocab_size=128,hidden_size=256,intermediate_size=64,num_hidden_layers=32,
            num_attention_heads=32,num_key_value_heads=32,max_position_embeddings=512,attention_dropout=0.);cfg._attn_implementation='sdpa'
        return S45Engine(torch,None,Olmo2ForCausalLM(cfg).eval())
    from stage1_scan import load_model
    t,tok,model,lock=load_model();return S45Engine(t,tok,model)

def spec_checked(e,p,cell):
    sp=e.spec(p,cell);c=p['cells'][cell]
    ids=sp['ids'][0].tolist()
    if ids[:sp['n']]!=c['token_ids'] or ids[sp['n']:]!=c['shared_prefix'] or sp['positions'][-1]!=sp['n']-1:raise ValueError('Frozen token IDs/prefix changed: '+c['id'])
    return sp

def baseline_check(plan,p,cell,out,label):
    """Saved-baseline replication within 0.05, or the explicit frozen compatibility exception (S4.3 family-034 policy)."""
    if not p['saved_baselines']:return None
    saved=p['saved_baselines'][0 if cell=='x00' else 1];err=abs(out['margin']-saved)
    if err<=.05:return dict(error=float(err),mode='saved')
    ex=plan.get('compatibility_exceptions',{}).get(p['id'])
    if ex is None:raise ValueError(f'Gate {label}: {err} > 0.05 for {p["id"]} {cell}; no frozen compatibility exception')
    if cell=='x00':
        ref=abs(out['margin']-ex['margin'])
        if ref>ex['reference_tolerance']:raise ValueError(f'Gate {label}: compatibility reference mismatch {ref} for {p["id"]}')
        return dict(error=float(err),mode='frozen_compatibility_exception',reference_error=float(ref),gpu=ex['gpu'])
    if err>2*.05:raise ValueError(f'Gate {label}: {err} > 0.10 for exception pair {p["id"]} {cell}')
    return dict(error=float(err),mode='frozen_compatibility_exception_corrupted_cell')

def replacement(e,p,cell,baseline,bank,twin_cache):
    sp=e.spec(p,cell);keys=p['cells'][cell]['mean_keys']
    if baseline=='mean':
        pos,v,info=bank.cell_vectors(p['family'],keys)
        if pos!=sp['positions']:raise ValueError('Mean positions differ from keyed positions')
        return e.t.from_numpy(v),info
    twin={'x00':'x10','x10':'x00','x01':'x11','x11':'x01'}[cell]
    if twin not in twin_cache:
        st=e.spec(p,twin)
        if st['positions']!=sp['positions']:raise ValueError('Donor twin positions unaligned')
        twin_cache[twin]=e.all_head_outputs(st['ids'],st['positions'],st['g'],st['d'])[0]
    return twin_cache[twin],{'donor':len(sp['positions'])}

def gates(e,pairs,plan,bank,s):
    """Fixed gate families; fail closed. Background types: full, all-live, C33, C33-h, C33+h, diagnostic."""
    reports=[];cand=hid(CANDIDATES[0]);cand2=hid(CANDIDATES[3]);ctrl=hid(plan['control_rosters'][attachment('T1',CANDIDATES[0])['key']]['control'])
    bg_types={'all_live':list(range(1024)),'C33':ids(C33),'C33-h':sorted(set(ids(C33))-{hid('L8H15')}),'C33+h':sorted(set(ids(C33))|{cand}),
              'diag':sorted(set(ids(C33))|{cand,ctrl})}
    for p in pairs:
        if p['family'] not in plan['gate_families']:continue
        errs={};specs={c:spec_checked(e,p,c) for c in CELLS};reps={}
        for c in ['x00','x10']:
            out=e.output(specs[c]['ids'],specs[c]['g'],specs[c]['d'])
            errs[c+'_saved']=baseline_check(plan,p,c,out,'saved baseline')
            cap=e.capture_s43(specs[c]['ids'],[545,595],specs[c]['g'],specs[c]['d'],full_z_layers=[18])
            errs[c+'_inert']=check(error(cap['output'],out),.001,'inert instrumentation')
            errs[c+'_sdpa']=check(cap['reconstruction_error'],.02,'SDPA reconstruction')
            if bank is not None:reps[c]=replacement(e,p,c,'mean',bank,{})[0]
        # Changing the question cannot alter anything before it: identical logits at all pre-question rows.
        for a,b in [('x00','x01'),('x10','x11')]:
            q=specs[a]['masks']['question'][0]
            if specs[a]['ids'][0,:q].tolist()!=specs[b]['ids'][0,:q].tolist():raise ValueError('Question change altered earlier tokens')
            with e.t.no_grad():
                la=e.model(specs[a]['ids'],use_cache=False).logits[0,:q].float();lb=e.model(specs[b]['ids'],use_cache=False).logits[0,:q].float()
            errs[f'prefix_invariance_{a}_{b}']=check(float((la-lb).abs().max()),.05,'question change altered earlier positions')
        sp=specs['x00'];sd=specs['x10'];full=e.output(sp['ids'],sp['g'],sp['d'])
        if bank is not None:
            out,_=e.behave(sp,bg_types['all_live'],reps['x00']);errs['all_live_equals_full']=check(error(out,full),.001,'all-live background')
            out,nrm=e.behave(sp,[],reps['x00']);errs['empty_mask_changes_model']=float(abs(out['margin']-full['margin']));errs['empty_replaced_norm']=nrm['replaced_norm']
            for name in ['C33','C33-h','C33+h','diag']:
                live=bg_types[name];cache={}
                for t in ['T4','T1']:
                    a=STRUCTURES[t]['anchor'];job=dict(source=hid(a['source']),site=a['site'],live=ids(a['live']),receiver=hid(a['receiver']),channel=a['channel'],direction='noise')
                    same=e.route(sp,sp,live,reps['x00'],reps['x00'],job,cache)
                    errs[f'self_donor_{name}_{t}']=check(abs(same['effect']),.001,'self-donor identity in background')
                    errs[f'self_donor_endpoint_{name}_{t}']=check(error(same['endpoint'],same['intact']),.001,'self-donor endpoint logits')
                    real=e.route(sp,sd,live,reps['x00'],reps['x10'],job,cache)
                    errs[f'route_{name}_{t}']=float(real['effect'])
                a=STRUCTURES['T1']['anchor'];job=dict(source=hid(a['source']),site=a['site'],live=[],receiver=hid(a['receiver']),channel='Q',direction='noise')
                zero=e.route(sp,sd,live,reps['x00'],reps['x10'],job,cache)
                errs[f'structural_zero_{name}']=check(abs(zero['effect']),.001,'no-intermediate cross-position Q must be zero')
                errs[f'structural_zero_norm_{name}']=check(zero['channel_norm'],.001,'no-intermediate Q channel')
            # A mean-clamped receiver silently returns a null: the diagnostic background must keep it live.
            att=attachment('T1',CANDIDATES[0]);job=dict(source=hid(att['source']),site=att['site'],live=[],receiver=ctrl,channel=att['channel'],direction='noise')
            clamped=e.route(sp,sd,bg_types['C33+h'],reps['x00'],reps['x10'],job,{})
            errs['clamped_control_is_null']=check(abs(clamped['effect']),.001,'mean-clamped receiver must be inert')
            livec=e.route(sp,sd,bg_types['diag'],reps['x00'],reps['x10'],job,{})
            errs['live_control_effect']=float(livec['effect'])
        # Full-background anchors, reported for the regression analysis (historical means are family-level).
        for t in ['T4','T5']:
            a=STRUCTURES[t]['anchor'];job=dict(source=hid(a['source']),site=a['site'],live=[],receiver=hid(a['receiver']),channel=a['channel'],direction='noise')
            errs[f'full_{t}']=float(e.route(sp,sd,None,None,None,job,{})['effect'])
        reports.append(dict(pair_id=p['id'],errors=errs));print('Gate passed:',p['id'],flush=True)
    if len(reports)!=2*len(plan['gate_families']):raise ValueError('Gate pair count')
    return dict(passed=True,pairs=reports,bank=bank.identity() if bank else None,
                scope='inert instrumentation, saved baselines, prefix invariance, all-live identity, self-donor identity per background type, structural zero, clamped-control null, full-background anchors')

def run(a):
    plan,sc,pairs=load(a.inputs,a.schedule);allpairs=list(read_lines(a.inputs/'pairs.jsonl.gz'))
    if json_read(ROOT/'model_lock_olmo2.json')!=plan['model'] and not os.environ.get('S45_TINY_MODEL'):raise ValueError('Model lock')
    mode='full' if sc['population']=='heldout' else 'lofo'
    ident=identity(a.inputs,a.schedule,a.shard,a.shards,a.bank,mode if a.bank else None);out=a.out
    if out.exists() and not a.resume:raise FileExistsError('Use --resume')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        mf=out/'manifest.json'
        if mf.exists() and json_read(mf)['identity']!=ident:raise ValueError('Resume identity changed')
        json_write(mf,dict(identity=ident,tiny_model=bool(os.environ.get('S45_TINY_MODEL'))))
        old=chunks(out,ident);done={key(r) for r in old};want=record_keys(sc,pairs,a.shard,a.shards)
        if len(done)!=len(old) or not done<=want:raise ValueError('Invalid partial coverage')
        cache,sources=prior_cache(a.inputs,json_read(a.prior) if a.prior else []);json_write(out/'reuse_sources.json',sources)
        e=start_model()
        from s45_means import MeanBank
        bank=MeanBank(a.bank,a.inputs,mode) if a.bank else None
        if any(sc['states'][j['state']]['live'] is not None and (j['kind']!='behavior' or j['baseline']=='mean') for j in sc['jobs']) and bank is None:raise ValueError('Mean bank required')
        gate=gates(e,allpairs,plan,bank,sc);json_write(out/'gate.json',gate)
        import transformers,platform
        json_write(out/'runtime.json',dict(torch=e.t.__version__,transformers=transformers.__version__,python=platform.python_version(),
            job=os.environ.get('SLURM_JOB_ID'),host=platform.node(),devices=str(getattr(e.model,'hf_device_map',{})),
            gpus=[e.t.cuda.get_device_name(i) for i in range(e.t.cuda.device_count())] if e.t.cuda.is_available() else []))
        if a.gate_only:return
        stop=[False]
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
        number=max([int(p.stem) for p in (out/'chunks').glob('*.json')]+[-1])+1;pending=[]
        bank_hash=ident['bank_hash']
        def flush():
            nonlocal number,pending
            if pending:write_chunk(out,ident,number,pending);number+=1;pending=[]
        for p in assigned(pairs,a.shard,a.shards):
            todo=[]
            for j in sc['jobs']:
                cells=j['cells'] if j['kind']=='behavior' else ['x00' if j['direction']=='noise' else 'x10']
                todo+=[(j,c) for c in cells if (p['id'],j['id'],c) not in done]
            if not todo:continue
            specs={c:spec_checked(e,p,c) for c in CELLS};reps={};twin={}
            compat={}
            for c in ['x00','x10']:
                if p['saved_baselines']:compat[c]=baseline_check(plan,p,c,e.output(specs[c]['ids'],specs[c]['g'],specs[c]['d']),'per-pair saved baseline')
            base=dict(pair_id=p['id'],family=p['family'],order=p['order'],split=p['split'],names=p['names'],prefix_len=len(p['shared_prefix']),bank_hash=bank_hash,origin='new',
                      baseline_replication=compat)
            by_state=collections.defaultdict(list)
            for j,c in todo:by_state[j['state']].append((j,c))
            for sid,items in by_state.items():
                st=sc['states'][sid];live=st['live'];route_cache={}
                rjobs=[j for j,c in items if j['kind']!='behavior']
                cheads=sorted({h for j in rjobs for h in [j['source'],j['receiver'],*j.get('live',[])]});clayers=sorted({h//32 for j in rjobs for h in j.get('live',[])})
                for j,c in items:
                    if stop[0]:flush();raise SystemExit(75)
                    k=(p['id'],j['id'],c);row=dict(base,kind=j['kind'],job_id=j['id'],state=sid,label=st['label'],cell=c,
                        live_count=None if live is None else len(live))
                    if j['kind']=='behavior':
                        row.update(baseline=j['baseline'],gold_side=p['cells'][c]['gold_side'],tok_a=specs[c]['g'],tok_b=specs[c]['d'])
                        hit=cache.get(reuse_key(row))
                        if hit is not None:row=dict(hit,job_id=j['id'],origin='prior:'+hit['origin'])
                        else:
                            if live is None:rep,info=None,{}
                            else:
                                if (c,j['baseline']) not in reps:reps[(c,j['baseline'])]=replacement(e,p,c,j['baseline'],bank,twin)
                                rep,info=reps[(c,j['baseline'])]
                            outp,nrm=e.behave(specs[c],live,rep);gold=specs[c]['g'] if row['gold_side']=='a' else specs[c]['d']
                            row.update(margin=outp['margin'],logit_a=outp['clean_logit'],logit_b=outp['corr_logit'],prob_a=outp['clean_prob'],prob_b=outp['corr_prob'],
                                top_token=outp['top_token'],gold_token=gold,top_is_gold=outp['top_token']==gold,
                                candidate_correct=(outp['margin']>0)==(row['gold_side']=='a'),replaced_norm=nrm['replaced_norm'],fallback=info)
                    else:
                        row.update(direction=j['direction'],key=j.get('key',j.get('anchor')),anchor=j.get('anchor'),role=j.get('role','anchor'),
                                   candidate=j.get('candidate'),structure=j.get('structure',j.get('anchor')),control_of=j.get('control_of'),
                                   source=j['source'],site=j['site'],receiver=j['receiver'],channel=j['channel'],live_heads=j.get('live',[]))
                        hit=cache.get(reuse_key(row))
                        if hit is not None:row=dict(hit,job_id=j['id'],origin='prior:'+hit['origin'])
                        else:
                            rc,dc=('x00','x10') if j['direction']=='noise' else ('x10','x00')
                            if live is None:rr=rd=None
                            else:
                                for cc in (rc,dc):
                                    if (cc,'mean') not in reps:reps[(cc,'mean')]=replacement(e,p,cc,'mean',bank,twin)
                                rr,rd=reps[(rc,'mean')][0],reps[(dc,'mean')][0]
                            res=e.route(specs[rc],specs[dc],live,rr,rd,j,route_cache,cheads,clayers)
                            check(res['reconstruction_error'],.02,'SDPA reconstruction')
                            row.update(effect=res['effect'],intact_margin=res['intact']['margin'],endpoint_margin=res['endpoint']['margin'],
                                hybrid_margin=res['hybrid_output']['margin'],intact=res['intact'],endpoint=res['endpoint'],channel_norm=res['channel_norm'],
                                source_norm=res['source_norm'],reconstruction_error=res['reconstruction_error'],
                                fallback=reps[(rc,'mean')][1] if live is not None else {})
                    pending.append(row);done.add(k)
                    if len(pending)>=16:flush()
                route_cache.clear()
            flush();print('Completed',sc['phase'],p['id'],len(done),'/',len(want),flush=True)
        if done!=want:raise ValueError('Coverage incomplete')
        json_write(out/'done.json',dict(complete=True,records=len(done)));verify(out,a.inputs,a.schedule)

def collect_means(a):
    from s45_means import accumulate
    plan,pairs=load(a.inputs)
    if any(p['split']!='discovery' for p in pairs):raise ValueError('Means come from discovery only')
    ident=mean_identity(a.inputs,a.shard,a.shards);out=a.out
    if out.exists() and not a.resume:raise FileExistsError('Use --resume')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        mf=out/'manifest.json'
        if mf.exists() and json_read(mf)['identity']!=ident:raise ValueError('Mean resume identity')
        json_write(mf,dict(identity=ident,tiny_model=bool(os.environ.get('S45_TINY_MODEL'))));e=start_model();vocab={k:i for i,k in enumerate(plan['mean_keys'])}
        grouped=collections.defaultdict(list)
        for p in assigned(pairs,a.shard,a.shards):grouped[p['family']].append(p)
        for fam,ps in grouped.items():
            path=out/'families'/f'{fam:03d}.npz';marker=path.with_suffix('.ok')
            if marker.exists():
                m=json_read(marker)
                if m['identity']!=ident or m['sha256']!=digest(path):raise ValueError('Mean checkpoint mismatch')
                continue
            if len(ps)!=2 or {p['order'] for p in ps}!={0,1}:raise ValueError('Expected both family orders')
            sums=np.zeros((e.L*e.H,len(vocab),e.DH),dtype=np.float64);counts=np.zeros(len(vocab),dtype=np.int64)
            for p in ps:
                for c in ['x00','x10']:
                    sp=spec_checked(e,p,c);z,outp=e.all_head_outputs(sp['ids'],sp['positions'],sp['g'],sp['d'])
                    baseline_check(plan,p,c,outp,'mean capture baseline')
                    keys=[p['cells'][c]['mean_keys'][j] for j in sp['positions']]
                    accumulate(sums,counts,keys,z.permute(1,0,2).double().numpy(),vocab)
            npz_write(path,family=fam,sums=sums,counts=counts);json_write(marker,dict(identity=ident,sha256=digest(path)))
            print('Mean family',fam,flush=True)
        json_write(out/'done.json',dict(complete=True))

def tokenizer(a):
    if a.tokenizer:
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast(tokenizer_file=str(a.tokenizer))
    from stage1_scan import STATE
    from transformers import AutoTokenizer
    lock=json_read(ROOT/'model_lock_olmo2.json');return AutoTokenizer.from_pretrained(STATE/'model'/lock['revision'],local_files_only=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare')
    for n in ['stage3-inputs','s42-inputs','candidates','out']:q.add_argument('--'+n,type=Path,required=True)
    q.add_argument('--dataset',type=Path);q.add_argument('--baseline',type=Path);q.add_argument('--tokenizer',type=Path);q.add_argument('--compatibility',type=Path)
    q=sub.add_parser('prepare-heldout')
    for n in ['inputs','freeze','dataset','baseline','out']:q.add_argument('--'+n,type=Path,required=True)
    q.add_argument('--tokenizer',type=Path)
    q=sub.add_parser('check');q.add_argument('--inputs',type=Path,required=True);q.add_argument('--schedule',type=Path)
    for name in ['run','means']:
        q=sub.add_parser(name)
        for n in ['inputs','out']:q.add_argument('--'+n,type=Path,required=True)
        q.add_argument('--shard',type=int,default=0);q.add_argument('--shards',type=int,default=1);q.add_argument('--resume',action='store_true')
        if name=='run':
            q.add_argument('--schedule',type=Path,required=True);q.add_argument('--prior',type=Path);q.add_argument('--bank',type=Path);q.add_argument('--gate-only',action='store_true')
    a=p.parse_args()
    if a.cmd=='prepare':
        from s45_plan import prepare
        prepare(a.stage3_inputs,a.s42_inputs,a.candidates,a.out,tokenizer(a),a.dataset,a.baseline,a.compatibility);print('Frozen S4.5 discovery inputs ready')
    elif a.cmd=='prepare-heldout':
        from s45_plan import prepare_heldout
        prepare_heldout(a.inputs,a.freeze,a.dataset,a.baseline,a.out,tokenizer(a));print('Held-out inputs sealed for the frozen validation suite')
    elif a.cmd=='check':
        plan,sc,pairs=load(a.inputs,a.schedule or a.inputs/'stage_a.json')
        print(dict(pairs=len(pairs),states=len(sc['states']),jobs=len(sc['jobs']),records=len(record_keys(sc,pairs,0,1))))
    elif a.cmd=='run':run(a)
    else:collect_means(a)

if __name__=='__main__':main()
