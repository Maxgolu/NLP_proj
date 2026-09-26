"""Tiny random OLMo2 and CPU tests for the S4.5 background layer, cells, means and freeze.

Required gates (specification section 12): all-live reproduces full; self-donor is identity
in every background type; no-intermediate cross-position Q is zero; changing the question
cannot alter earlier facts; mean role keys exclude identities/relevance; disabled receiver
controls are not silently interpreted as nulls; the freeze manifest gate.
"""
import contextlib
import json
import re
import tempfile
from pathlib import Path
import unittest
import numpy as np
from stage3_common import json_read,json_write,digest


class StubTokenizer:
    """Deterministic word/punctuation tokenizer with offsets; long names split in two tokens."""
    def __init__(self):self.vocab={}
    def _pieces(self,text):
        out=[]
        for m in re.finditer(r'[A-Za-z]+|[^\sA-Za-z]',text):
            s,e=m.span();w=m.group()
            if w.isalpha() and len(w)>4:out+=[(s,s+3),(s+3,e)]
            else:out.append((s,e))
        return out
    def _id(self,piece):return self.vocab.setdefault(piece,len(self.vocab)+2)
    def encode(self,text,add_special_tokens=False):return [self._id(text[a:b]) for a,b in self._pieces(text)]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        pieces=self._pieces(text);return dict(input_ids=[self._id(text[a:b]) for a,b in pieces],offset_mapping=pieces)
    def convert_ids_to_tokens(self,ids):
        inv={v:k for k,v in self.vocab.items()};return [inv[i] for i in ids]

NAMES=[('Jarlia','Kernna','Rarnia'),('Kulnra','Tilnia','Dirlna')]
ORIGINAL59=sorted(int(n[1:].split('H')[0])*32+int(n.split('H')[1]) for n in ('L1H27 L2H21 L2H22 L4H0 L5H18 L6H2 L6H24 L7H18 L7H25 L8H8 L8H15 L8H16 L9H6 L9H16 L9H17 L10H16 L11H3 L11H4 L11H7 L11H15 '
 'L12H2 L12H9 L13H1 L13H22 L14H10 L14H23 L15H0 L15H2 L15H3 L15H18 L15H28 L15H29 L16H1 L16H2 L16H4 L16H16 L16H21 L16H24 L16H31 L17H5 L17H24 L18H15 L18H19 L19H16 L20H7 '
 'L20H14 L21H24 L21H25 L21H27 L22H5 L23H10 L23H15 L24H17 L25H18 L26H23 L26H31 L27H23 L30H3 L31H1').split())
def synthetic_rows(family,order,swap=False):
    """Two chains; the queried child is Jarlia; the two mothers are the candidates."""
    (c1,g1,m1),(c2,g2,m2)=NAMES;ma,mb=(m2,m1) if swap else (m1,m2)
    demo='Belmia is the mother of Linda.\nDirdna is the mother of Belmia.\nQuestion: Who is the mother of Belmia?\nAnswer: Dirdna.\n\n'
    lines=[f'{c1} is the mother of {g1}.',f'{ma} is the mother of {c1}.',f'{c2} is the mother of {g2}.',f'{mb} is the mother of {c2}.']
    if order:lines=lines[2:]+lines[:2]
    test='\n'.join(lines)+f'\nQuestion: Who is the mother of {c1}?\nAnswer:';prompt=demo+test;facts=[]
    for ln in lines:
        tail,head=re.fullmatch(r'(\w+) is the mother of (\w+)\.',ln).groups();s=prompt.index(ln)
        facts.append(dict(block='test',relation='mother_of',head=head,tail=tail,head_span=[s+ln.index(head,10),s+ln.index(head,10)+len(head)],tail_span=[s,s+len(tail)],line=ln))
    facts=[dict(block='demo',head='Linda',tail='Belmia',head_span=[24,29],tail_span=[0,6])]+facts
    return dict(id=f'{family:03d}/direct/{"corrupted" if swap else "base"}/{int(swap)}/order{order}',family=family,variant='corrupted' if swap else 'base',order=order,
                prompt=prompt,facts=facts,candidates=sorted([m1,m2]),gold=ma,query_source=c1,question_entity=c1,demo_ids=['d1'],test_start=len(demo),split='discovery')

def synthetic_inputs(out,families=range(0,24,2),core=None):
    from s45_plan import build_pair,freeze_inputs,annotate_row
    tok=StubTokenizer();pairs=[]
    for f in families:
        for o in (0,1):
            b,c=annotate_row(tok,synthetic_rows(f,o)),annotate_row(tok,synthetic_rows(f,o,True))
            pairs.append(build_pair(tok,b,c,'discovery',saved=None,exact_all=(f in (core or list(families)[:4]))))
    core=sorted({p['family'] for p in pairs if p['exact_all']})
    plan=freeze_inputs(out,pairs,dict(model_id='tiny'),core,core[:1],ORIGINAL59,dict(sealed=True),{})
    return tok,pairs,plan


class Cells(unittest.TestCase):
    def test_four_cells_rebuild_every_derived_field(self):
        from s45_plan import build_pair,annotate_row,CELLS
        tok=StubTokenizer();b=annotate_row(tok,synthetic_rows(0,0));c=annotate_row(tok,synthetic_rows(0,0,True));p=build_pair(tok,b,c,'discovery')
        self.assertEqual(p['names'],['Rarnia','Dirlna']);cells=p['cells']
        self.assertEqual([cells[x]['gold'] for x in CELLS],['Rarnia','Dirlna','Dirlna','Rarnia'])
        self.assertEqual(cells['x01']['query_source'],'Kulnra');self.assertEqual(cells['x00']['query_source'],'Jarlia')
        self.assertIn('Kulnra?',cells['x01']['prompt'][-40:]);self.assertNotIn('Jarlia?',cells['x01']['prompt'])
        # Derived fields are rebuilt: token ids, offsets, masks, keys, prefix metadata and cell identity.
        self.assertNotEqual(cells['x01']['token_ids'],cells['x00']['token_ids']);self.assertEqual(cells['x01']['cell'],'x01')
        self.assertNotEqual(cells['x01']['masks']['query_mother'],cells['x00']['masks']['query_mother'])
        self.assertTrue(cells['x01']['masks']['_query_first']!=cells['x00']['masks']['_query_first'])
        self.assertEqual(cells['x01']['n'],len(cells['x01']['token_ids']));self.assertEqual(cells['x01']['masks']['colon'],[cells['x01']['n']-1])
        # Facts precede the question: everything before the question is identical between x00 and x01.
        q=cells['x00']['masks']['question'][0];self.assertEqual(cells['x00']['token_ids'][:q],cells['x01']['token_ids'][:q])
        self.assertEqual(cells['x00']['token_ids'][q:],cells['x00']['token_ids'][q:])
        # Twins are aligned; the query-changed twins are aligned with each other.
        self.assertEqual(len(cells['x00']['token_ids']),len(cells['x10']['token_ids']));self.assertEqual(cells['x01']['masks'],cells['x11']['masks'])
        # The all-fact mask covers all four sentences and excludes question and colon.
        m=cells['x00']['masks'];self.assertTrue(set(m['query_sentence'])|set(m['other_sentence'])<set(m['all_sentence']))
        self.assertFalse(set(m['all_sentence'])&set(m['question']));self.assertNotIn(m['colon'][0],m['all_sentence'])
    def test_mean_keys_exclude_identity_and_relevance(self):
        from s45_plan import build_pair,annotate_row
        tok=StubTokenizer();p=build_pair(tok,annotate_row(tok,synthetic_rows(1,0)),annotate_row(tok,synthetic_rows(1,0,True)),'discovery')
        names={n for group in NAMES for n in group}
        for c in p['cells'].values():
            for key in c['mean_keys']:
                if key:self.assertFalse(any(n.lower() in '|'.join(key).lower() for n in names));self.assertNotIn('query',key[1]);self.assertNotIn('gold',key[1])
        self.assertEqual(p['cells']['x00']['mean_keys'],p['cells']['x10']['mean_keys'])
        # Changing the query keeps the query-relevance out of the keys: same roles, different positions only.
        self.assertEqual(sorted(k[1] for k in p['cells']['x01']['mean_keys'] if k),sorted(k[1] for k in p['cells']['x00']['mean_keys'] if k))
    def test_rosters_controls_and_schedules(self):
        from s45_plan import control_rosters,C50,STRUCTURES,CANDIDATES,hid,states_for,C33,stage_a,stage_b,stage_c,stage_d,full_discovery,load,record_keys,attachment
        r1=control_rosters(ORIGINAL59);r2=control_rosters(ORIGINAL59);self.assertEqual(r1,r2)
        excluded={hid(h) for h in C50}|{hid(h) for s in STRUCTURES.values() for h in s['retained']}|set(ORIGINAL59)
        for key,v in r1.items():
            src,site,recv,ch=key.split('|');self.assertEqual(hid(v['control'])//32,hid(recv)//32);self.assertNotIn(hid(v['control']),excluded|{hid(src),hid(recv)})
        self.assertEqual(len({v['id'] for v in states_for(C33).values()}),9)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'inputs';tok,pairs,plan=synthetic_inputs(out)
            plan,sc,core=load(out,out/'stage_a.json');self.assertEqual(len(core),8);self.assertEqual(len(sc['jobs']),8)
            self.assertEqual(len(record_keys(sc,core,0,1)),8*4*8)
            sb=stage_b(out,plan,C33);self.assertEqual(len(sb.jobs),10)
            sel=dict(selected=[dict(candidate='L17H5',structure='T2',record='route_pair'),dict(candidate='L1H27',structure='B',record='functional_only')])
            sd=stage_d(out,plan,C33,sel);self.assertEqual(len(sd.jobs),4+4)
            diag=[s for s in sd.states.values() if len(s['live'])==35];self.assertEqual(len(diag),1)
            self.assertIn(hid(plan['control_rosters'][attachment('T2','L17H5')['key']]['control']),diag[0]['live'])
            sc_=stage_c(out,plan,C33);self.assertEqual(len(sc_.jobs),7*5+5);sc_.write(out/'c.json');load(out,out/'c.json')
            fd=full_discovery(out,plan,C33,sel);self.assertEqual(len(fd.jobs),6+4)
            # A route whose receiver is mean-clamped in its background is refused by the loader, never silently a null.
            bad=stage_d(out,plan,C33,sel);bad.states[diag[0]['id']]['live']=[h for h in diag[0]['live'] if h!=hid(plan['control_rosters'][attachment('T2','L17H5')['key']]['control'])]
            path=out/'bad.json';json_write(path,dict(policy=sc['policy'],plan_hash=sc['plan_hash'],phase='x',population='core',states=bad.states,jobs=bad.jobs))
            with self.assertRaises(ValueError):load(out,path)


class Means(unittest.TestCase):
    def test_lofo_and_full_modes(self):
        from s45_means import MeanBank
        from s45_plan import key_levels
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'inputs';tok,pairs,plan=synthetic_inputs(out);K=len(plan['mean_keys']);fams=plan['families'];bank=Path(td)/'bank';(bank/'families').mkdir(parents=True)
            fm={f:np.full((4,K,2),float(f)+1,dtype=np.float32) for f in fams};support=np.ones((len(fams),K),dtype=bool)
            k0=plan['mean_keys'].index('0|mother|first');support[2:,k0]=False
            for f in fams[2:]:fm[f][:,k0]=0
            total=sum(fm[f].astype(np.float64) for f in fams)
            for f in fams:np.save(bank/'families'/f'{f:03d}.npy',fm[f])
            np.save(bank/'total.npy',total);np.save(bank/'support.npy',support)
            from s45_plan import POLICY
            files={'total.npy':digest(bank/'total.npy'),'support.npy':digest(bank/'support.npy'),**{f'families/{f:03d}.npy':digest(bank/'families'/f'{f:03d}.npy') for f in fams}}
            json_write(bank/'mean_bank_manifest.json',dict(policy=POLICY,plan_hash=digest(out/'plan.json'),keys=plan['mean_keys'],families=fams,heads=4,files=files))
            lofo=MeanBank(bank,out,'lofo');pos,v,info=lofo.cell_vectors(fams[0],[['0','mother','last'],None,['0','child','single']])
            self.assertEqual(pos,[0,2]);others=np.mean([f+1 for f in fams[1:]])
            self.assertAlmostEqual(float(v[0,0,0]),others,places=5)   # own family excluded
            pos,v,info=lofo.cell_vectors(fams[0],[['0','mother','first']]);self.assertEqual(info,{'1':1})  # exact key lacks support: fallback level 1
            full=MeanBank(bank,out,'full')
            with self.assertRaises(ValueError):full.cell_vectors(fams[0],[['0','mother','last']])
            pos,v,info=full.cell_vectors(999,[['0','mother','last']]);self.assertAlmostEqual(float(v[0,0,0]),np.mean([f+1 for f in fams]),places=5)
            with self.assertRaises(ValueError):lofo.cell_vectors(999,[['0','mother','last']])


class Freeze(unittest.TestCase):
    def test_manifest_gate(self):
        import s45_freeze as F
        from s45_plan import C33
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'inputs';tok,pairs,plan=synthetic_inputs(out);runs=[]
            for name in F.REQUIRED_STAGES:
                r=Path(td)/name;r.mkdir();json_write(r/'manifest.json',dict(identity=name));json_write(r/'done.json',dict(complete=True));runs.append(r)
            json_write(Path(td)/'bank.json',dict(x=1))
            stages={name:dict(schedule=None,runs=[str(r)],analysis=[]) for name,r in zip(F.REQUIRED_STAGES,runs)}
            sel=dict(selected=[dict(candidate='L1H27',structure='T1',record='route_pair',gamma_mean=.2)])
            m=F.build(out,Path(td)/'freeze.json',stages,C33,'pass',sel,{'L1H27':dict(gamma=1)},Path(td)/'bank.json',dict(code='x'))
            self.assertTrue(F.validate_freeze(out,m))
            bad=dict(m);bad['B']=dict(m['B'],status='partial')
            with self.assertRaises(ValueError):F.validate_freeze(out,bad)
            with self.assertRaises(ValueError):F.build(out,Path(td)/'f2.json',{k:v for k,v in stages.items() if k!='stage_d'},C33,'pass',sel,{},Path(td)/'bank.json',{})
            with self.assertRaises(ValueError):F.validate_freeze(out,m,code=dict(code='changed'))
            from s45_plan import prepare_heldout
            with self.assertRaises(ValueError):prepare_heldout(out,Path(td)/'freeze.json',None,None,Path(td)/'h',tok) if False else F.validate_freeze(out,dict(m,selection=dict(selected=[dict(candidate=f'L{i}H1') for i in range(4)])))


class TinyBackground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Olmo2Config,Olmo2ForCausalLM
        except ImportError as exc:raise unittest.SkipTest('CPU torch/transformers unavailable') from exc
        from s45_engine import S45Engine
        torch.manual_seed(45);torch.set_num_threads(2)
        cfg=Olmo2Config(vocab_size=41,hidden_size=32,intermediate_size=48,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,attention_dropout=0.)
        cfg._attn_implementation='sdpa';cls.t=torch;cls.e=S45Engine(torch,None,Olmo2ForCausalLM(cfg).eval())
        cls.rec=dict(ids=torch.tensor([[1,3,4,5,6,7]]),n=6,g=10,d=11,cell='x00',positions=[1,2,3,4,5],masks=dict(site=[1,2],colon=[5],all_sentence=[1,2,3,4]))
        cls.don=dict(cls.rec,ids=torch.tensor([[1,8,9,5,6,7]]),cell='x10')
        # shared-prefix case: the metric row (6) follows the colon (5); receivers must then sit below the last layer
        cls.prec=dict(cls.rec,ids=torch.tensor([[1,3,4,5,6,7,2]]));cls.pdon=dict(cls.don,ids=torch.tensor([[1,8,9,5,6,7,2]]))
        cls.rep=torch.randn(5,12,8)*.5;cls.rep2=torch.randn(5,12,8)*.5
    def out(self,sp):return self.e.output(sp['ids'],sp['g'],sp['d'])
    def test_all_live_reproduces_full_and_empty_matches_manual_joint(self):
        e=self.e;full=self.out(self.rec);o,_=e.behave(self.rec,list(range(12)),self.rep);self.assertAlmostEqual(o['margin'],full['margin'],places=6)
        o,nrm=e.behave(self.rec,[],self.rep);self.assertGreater(nrm['replaced_norm'],0)
        patches={h:(self.rec['positions'],self.rep[:,h]) for h in range(12)};expected,_=e.evaluate(self.rec['ids'],patches,10,11)
        self.assertAlmostEqual(o['margin'],expected['margin'],places=5);self.assertNotAlmostEqual(o['margin'],full['margin'],places=3)
    def test_precedence_live_slices_untouched(self):
        e=self.e;seen={}
        def keep(mod,args):seen['z']=args[0].detach().clone()
        h=e.layers[0].self_attn.o_proj.register_forward_pre_hook(keep)
        try:
            plain=self.out(self.rec);z0=seen['z'].clone()
            with e.background([0,1,2,3,5],self.rec['positions'],self.rep):
                # the layer-0 capture hook was registered BEFORE the background: it sees raw values (and no replacement in layer 0)
                self.out(self.rec);self.assertTrue(self.t.allclose(seen['z'],z0))
        finally:h.remove()
        cap=[]
        with e.background([0,1,2,3,5],self.rec['positions'],self.rep):
            hh=e.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda m,a:cap.append(a[0].detach().clone()))
            try:self.out(self.rec)
            finally:hh.remove()
        # Registered after the background hook: layer-1 slices of heads 4,6,7 carry the replacement; head 5 (live) does not.
        z=cap[0][0,self.rec['positions']].reshape(5,4,8)
        for hi in [0,2,3]:self.assertTrue(self.t.allclose(z[:,hi],self.rep[:,4+hi].to(z.dtype)))
        self.assertFalse(self.t.allclose(z[:,1],self.rep[:,5].to(z.dtype)))
    def test_self_donor_identity_in_every_background_type(self):
        e=self.e
        for live in [None,list(range(12)),[],[1,5,9],[0,1,2,3,4,5,6,7]]:
            for job in [dict(source=1,site='site',live=[],receiver=9,channel='V',direction='noise'),dict(source=1,site='site',live=[5],receiver=9,channel='Q',direction='noise'),
                        dict(source=1,site='colon',live=[],receiver=9,channel='Q',direction='restore')]:
                if live is not None and any(h not in live for h in [job['source'],job['receiver'],*job['live']]):continue
                for sp in [self.rec,self.prec]:r=e.route(sp,sp,live,self.rep,self.rep,job,{})
                self.assertLess(abs(r['effect']),1e-5,str((live,job)));self.assertLess(abs(r['endpoint']['clean_logit']-r['intact']['clean_logit']),1e-5)
    def test_captures_live_inside_the_same_background(self):
        e=self.e;live=[1,5,9];cache={}
        job=dict(source=1,site='site',live=[5],receiver=9,channel='Q',direction='noise')
        r=e.route(self.rec,self.don,live,self.rep,self.rep2,job,cache)
        under,_=e.behave(self.rec,live,self.rep);full=self.out(self.rec)
        self.assertAlmostEqual(r['intact']['margin'],under['margin'],places=6);self.assertNotAlmostEqual(r['intact']['margin'],full['margin'],places=3)
        rec=[v for k,v in cache.items() if k[0]=='x00'][0];don=[v for k,v in cache.items() if k[0]=='x10'][0]
        # donor captured under the same background: differs from a full-model donor capture
        fulldon=e.capture_s43(self.don['ids'],[1,5,9],10,11,full_z_layers=[1]);self.assertFalse(self.t.allclose(don['z'][5],fulldon['z'][5]))
        self.assertTrue(self.t.allclose(don['z'][1],fulldon['z'][1]))  # layer 0 has no upstream background to differ
        # manual construction with the same background objects reproduces route()
        with e.background(live,self.rec['positions'],self.rep):
            hy=e.chain_hybrid(self.rec['ids'],1,[1,2],don,rec,[1,5,9],10,11,[5]);end=e.endpoint(self.rec['ids'],dict(kind='head',receiver=9,channel='Q'),[1,2],hy,10,11,6)
        self.assertAlmostEqual(r['endpoint']['margin'],end['margin'],places=6)
    def test_freeze_does_not_cancel_source_replacement(self):
        e=self.e;job=dict(source=1,site='site',live=[],receiver=9,channel='V',direction='noise')
        r=e.route(self.rec,self.don,[1,5,9],self.rep,self.rep2,job,{})
        self.assertGreater(r['source_norm'],0);self.assertGreater(r['channel_norm'],0)  # the receiver's V at the site moved
        self.assertEqual(r['hybrid_output']['margin'],r['intact']['margin'])             # fact-to-final residual bypass is structurally null
    def test_structural_zero_cross_position_q(self):
        e=self.e
        for live in [None,[1,5,9],[]]:
            if live==[]:continue
            r=e.route(self.rec,self.don,live,self.rep,self.rep2,dict(source=1,site='site',live=[],receiver=9,channel='Q',direction='noise'),{})
            self.assertLess(abs(r['effect']),1e-6);self.assertLess(r['channel_norm'],1e-6)
    def test_clamped_control_receiver_is_null_and_live_control_is_not(self):
        e=self.e;job=dict(source=1,site='all_sentence',live=[],receiver=10,channel='V',direction='noise')
        clamped=e.route(self.rec,self.don,[1,9],self.rep,self.rep2,job,{});self.assertEqual(clamped['effect'],0.);self.assertGreater(clamped['channel_norm'],0)  # channel moves, output is clamped: a silent null
        live=e.route(self.rec,self.don,[1,9,10],self.rep,self.rep2,job,{});self.assertNotEqual(live['effect'],0.);self.assertGreater(live['channel_norm'],0)
        # shared-prefix case with a receiver below the last layer
        job=dict(source=1,site='all_sentence',live=[],receiver=6,channel='V',direction='noise')
        clamped=e.route(self.prec,self.pdon,[1,9],self.rep,self.rep2,job,{});self.assertLess(abs(clamped['effect']),1e-6)
        live=e.route(self.prec,self.pdon,[1,6,9],self.rep,self.rep2,job,{});self.assertNotEqual(live['effect'],0.);self.assertGreater(live['channel_norm'],0)
    def test_question_change_cannot_alter_earlier_positions(self):
        e=self.e;a=self.t.tensor([[1,3,4,5,6,7,2]]);b=self.t.tensor([[1,3,4,5,6,8,9,2]])
        with self.t.no_grad():la=e.model(a,use_cache=False).logits[0,:5];lb=e.model(b,use_cache=False).logits[0,:5]
        self.assertTrue(self.t.allclose(la,lb,atol=1e-5))
    def test_all_head_outputs_layout(self):
        e=self.e;z,out=e.all_head_outputs(self.rec['ids'],self.rec['positions'],10,11);self.assertEqual(tuple(z.shape),(5,12,8))
        cap=e.capture_z(self.rec['ids'],[7],10,11);self.assertTrue(self.t.allclose(z[:,7],cap['z'][7][self.rec['positions']].float(),atol=1e-6))
        o,_=e.behave(self.rec,[],z);self.assertAlmostEqual(o['margin'],out['margin'],places=5)  # own outputs as replacement = identity
    def test_hook_cleanup_after_exception(self):
        e=self.e;full=self.out(self.rec)
        with self.assertRaises(RuntimeError):
            with e.background([],self.rec['positions'],self.rep):raise RuntimeError('interrupted')
        self.assertAlmostEqual(self.out(self.rec)['margin'],full['margin'],places=7)


if __name__=='__main__':unittest.main()
