"""CPU tests of the S4.5 analysis formulas on synthetic records with known answers.

Covers: g/F/L/accuracy guards and B selection (Stage A), d_b and the retention rule (Stage B),
Gamma matrix, eligibility and deterministic selection (Stage C), bidirectional Gamma and the
attachment-versus-control paired difference (Stage D), and the final labels (validation).
"""
import unittest
import numpy as np
import s45_analyze as A
from s45_plan import CELLS,STRUCTURES,CANDIDATES,C33,states_for,state,ids,hid,rb,attachment
from stage4_plan import retained

FAMS=list(range(0,40,2));ORIG59=[hid('L1H27'),hid('L8H15'),hid('L6H24'),hid('L9H16'),hid('L23H10')]

def beh(sid,cell,fam,order,margin,baseline='mean',correct=None):
    gold='a' if cell in ['x00','x11'] else 'b'
    return dict(kind='behavior',pair_id=f'{fam}/{order}',family=fam,order=order,split='discovery',cell=cell,state=sid,baseline=baseline,
                margin=margin,logit_a=margin,logit_b=0.,gold_side=gold,candidate_correct=((margin>0)==(gold=='a')) if correct is None else correct,top_is_gold=True)

def four_cell_rows(sid,fn,baseline='mean'):
    """fn(fam,order,cell)->margin."""
    return [beh(sid,c,f,o,fn(f,o,c),baseline) for f in FAMS for o in (0,1) for c in CELLS]

class StageA(unittest.TestCase):
    def test_guards_and_interaction(self):
        full=lambda f,o,c:{'x00':10.,'x10':-10.,'x01':-8.,'x11':8.}[c]+0.1*o
        good=lambda f,o,c:full(f,o,c)*1.1                       # F=1.1, L=0.1
        bad=lambda f,o,c:full(f,o,c)*0.5                        # F=0.5
        noisy=lambda f,o,c:full(f,o,c)*(1.3 if f%4==0 else 0.7)  # F=1.0 but L=0.3
        states={'full':state(None,'full'),'g':state([1],'good'),'b':state([2],'bad'),'n':state([3],'noisy')}
        rows=four_cell_rows('full',full)+four_cell_rows(states['g']['id'],good)+four_cell_rows(states['b']['id'],bad)+four_cell_rows(states['n']['id'],noisy)
        summ,fam=A.behavior_summary(rows,{s['id']:s for s in states.values()});A.guards(summ)
        by={s['label']:s for s in summ}
        self.assertAlmostEqual(by['full']['g_mean'],20.);self.assertAlmostEqual(by['good']['F'],1.1);self.assertAlmostEqual(by['good']['L'],.1)
        self.assertTrue(by['good']['passes']);self.assertEqual(by['bad']['guard'],'fail:F,L');self.assertFalse(by['noisy']['passes']);self.assertIn('L',by['noisy']['guard'])
        # b = (M00-M10-M01+M11)/4 = (10+10+8+8)/4 = 9 for full
        self.assertAlmostEqual(by['full']['b_mean'],9.);self.assertAlmostEqual(by['full']['query_contrast_original_facts'],18.)
        self.assertEqual(by['full']['accuracy']['x01_order0'],1.);self.assertEqual(len(fam),4*len(FAMS))
    def test_accuracy_guard(self):
        full=lambda f,o,c:{'x00':10.,'x10':-10.,'x01':-8.,'x11':8.}[c]
        wrong=lambda f,o,c:full(f,o,c)*(1. if not (c=='x01' and f<4) else -1.)  # two families wrong in x01: -10 points
        s=state([1],'w');rows=four_cell_rows('full',full)+four_cell_rows(s['id'],wrong)
        summ,_=A.behavior_summary(rows,{'full':state(None,'full'),s['id']:s});A.guards(summ);w=next(x for x in summ if x['label']=='w')
        self.assertFalse(w['passes']);self.assertIn('accuracy:x01_order0',w['guard'])

class StageBC(unittest.TestCase):
    def rows_for(self,B,plus_shift):
        st=states_for(B);rows=[]
        base=lambda f,o,c:{'x00':10.,'x10':-10.,'x01':-8.,'x11':8.}[c]
        for k,s in st.items():
            shift=plus_shift.get(k,0.)
            rows+=four_cell_rows(s['id'],lambda f,o,c,sh=shift:base(f,o,c)+(sh if c=='x00' else -sh if c=='x10' else 0.))
        rows+=four_cell_rows(state(set(ids(B))-set(rb(B,ORIG59)),'B-R')['id'],base)+four_cell_rows('full',base)
        return rows,st
    def test_d_b_and_rule(self):
        import tempfile
        from pathlib import Path
        rows,st=self.rows_for(C33,{'B+L1H27':1.,'B+L11H4':-1.})   # L1H27: plus raises x00 by 1 and lowers x10 by 1 -> d_b=(1+1)/4=0.5; L11H4 (not in B): B+h=B, B-h shifted -> d_b=-0.5
        import s45_plan as P
        with tempfile.TemporaryDirectory() as td:
            import test_s45 as T
            out=Path(td)/'i';tok,pairs,plan=T.synthetic_inputs(out);plan['original59']=ORIG59
            sc=P.stage_b(out,plan,C33).write(out/'b.json')
            # monkeypatch gather to feed synthetic rows
            orig=A.gather;A.gather=lambda *a,**k:rows
            try:table=A.analyze_stage_b(out,sc,[],Path(td)/'an',C33)
            finally:A.gather=orig
        by={x['candidate']:x for x in table}
        self.assertAlmostEqual(by['L1H27']['d_b']['mean'],.5);self.assertEqual(by['L1H27']['d_b']['classification'],'coherent');self.assertTrue(by['L1H27']['functional'])
        self.assertAlmostEqual(by['L11H4']['d_b']['mean'],-.5);self.assertAlmostEqual(by['L11H4']['gold_diff_x10']['mean'],-1.)
        self.assertEqual(by['L17H5']['functional_rule'],'below_rule');self.assertEqual(by['R']['kind'],'collective_RI')
    def test_gamma_matrix_and_selection(self):
        st=states_for(C33);rows=[]
        def route(sid,t,f,o,eff):return dict(kind='route',pair_id=f'{f}/{o}',family=f,order=o,split='discovery',cell='x00',state=sid,anchor=t,key=t,direction='noise',
            effect=eff,intact_margin=10.,endpoint_margin=10.-eff,hybrid_margin=10.,channel_norm=1.,source_norm=1.,role='anchor')
        # Route T1 retained everywhere (+1.6). L1H27 raises it by +0.5 (coherent); L17H5 changes it heterogeneously (+/-0.3 by family);
        # L23H10 tiny (0.02). T5 negative route (-0.24) unaffected. T2 below rule everywhere (0.02) -> ineligible.
        for k,s in st.items():
            if k.startswith('B-L18H19') or k.startswith('B-L8H15'):continue
            for f in FAMS:
                for o in (0,1):
                    g={'T1':1.6,'T2':.02,'T3':.9,'T4':.5,'T5':-.24}
                    if k=='B+L1H27':g['T1']+=.5;g['T4']+=.2
                    if k=='B+L17H5':g['T1']+=(.3 if f%4==0 else -.3);g['T2']+=.4
                    if k=='B+L23H10':g['T3']+=.02
                    for t in STRUCTURES:rows.append(route(s['id'],t,f,o,g[t]))
        for t in STRUCTURES:
            for f in FAMS:
                for o in (0,1):rows.append(route('full',t,f,o,STRUCTURES[t]['historical']['I']+.01))
        import tempfile,test_s45 as T,s45_plan as P
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'i';tok,pairs,plan=T.synthetic_inputs(out);sc=P.stage_c(out,plan,C33).write(out/'c.json')
            stage_b=[dict(candidate=h,functional=h in ['L1H27','L25H18'],functional_rule='d_b',d_b=dict(retained=h in ['L1H27','L25H18'],classification='coherent',mean=.3,mean_absolute=.3)) for h in CANDIDATES]
            orig=A.gather;A.gather=lambda *a,**k:rows
            try:sel=A.analyze_stage_c(out,sc,[],Path(td)/'an',C33,stage_b)
            finally:A.gather=orig
            import csv
            with open(Path(td)/'an'/'candidate_structure_matrix.csv') as f:rowsm=list(csv.DictReader(f))
        chosen=[(s['candidate'],s['structure'],s['record']) for s in sel['selected']]
        # coherent pairs first (L1H27/T1 +0.5, L17H5/T2 +0.4, L1H27/T4 +0.2), one per candidate in pass 1, then pass 2 fills the third slot
        self.assertEqual(chosen,[('L1H27','T1','route_pair'),('L17H5','T2','route_pair'),('L1H27','T4','route_pair')])
        self.assertTrue(all(r['within_0_05'] for r in sel['full_background_regression']))
        t2=next(r for r in rowsm if r['candidate']=='L17H5' and r['structure']=='T2');self.assertEqual(t2['status'],'eligible')  # B+h retains T2 (0.42)
        t2b=next(r for r in rowsm if r['candidate']=='L23H10' and r['structure']=='T2');self.assertEqual(t2b['gamma_class'],'ineligible')
        t1h=next(r for r in rowsm if r['candidate']=='L17H5' and r['structure']=='T1');self.assertEqual(t1h['gamma_class'],'heterogeneous')

    def test_functional_only_fill_and_zero_selection(self):
        st=states_for(C33);rows=[]
        def route(sid,t,f,o,eff):return dict(kind='route',pair_id=f'{f}/{o}',family=f,order=o,split='discovery',cell='x00',state=sid,anchor=t,key=t,direction='noise',
            effect=eff,intact_margin=10.,endpoint_margin=10.-eff,hybrid_margin=10.,channel_norm=1.,source_norm=1.,role='anchor')
        for k,s in list(st.items())+[('full',state(None,'full'))]:
            if k.startswith('B-L18H19') or k.startswith('B-L8H15'):continue
            for f in FAMS:
                for o in (0,1):
                    for t in STRUCTURES:rows.append(route(s['id'],t,f,o,STRUCTURES[t]['historical']['I']))
        import tempfile,test_s45 as T,s45_plan as P
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'i';tok,pairs,plan=T.synthetic_inputs(out);sc=P.stage_c(out,plan,C33).write(out/'c.json')
            fb=lambda h,r,m:dict(candidate=h,functional=r,functional_rule='d_b' if r else 'below_rule',d_b=dict(retained=r,classification='coherent' if r else 'below_rule',mean=m,mean_absolute=abs(m)))
            stage_b=[fb('L1H27',True,.2),fb('L11H4',False,0.),fb('L17H5',True,-.4),fb('L23H10',False,0.),fb('L25H18',True,.3),fb('L9H16',True,.1)]
            orig=A.gather;A.gather=lambda *a,**k:rows
            try:sel=A.analyze_stage_c(out,sc,[],Path(td)/'an',C33,stage_b)
            finally:A.gather=orig
        self.assertEqual([(s['candidate'],s['record']) for s in sel['selected']],[('L17H5','functional_only'),('L25H18','functional_only'),('L1H27','functional_only')])

class StageD(unittest.TestCase):
    def test_bidirectional_and_attachment_control(self):
        import tempfile,test_s45 as T,s45_plan as P
        from pathlib import Path
        h,t='L17H5','T2';st=states_for(C33);plus,minus=st[f'B+{h}'],st[f'B-{h}']
        def route(sid,key,d,f,o,eff,role='anchor'):return dict(kind='route' if role=='anchor' else 'attachment',pair_id=f'{f}/{o}',family=f,order=o,split='discovery',cell='x00' if d=='noise' else 'x10',
            state=sid,anchor=t if role=='anchor' else None,key=key,direction=d,effect=eff,intact_margin=0.,endpoint_margin=-eff,hybrid_margin=0.,channel_norm=1.,source_norm=1.,role=role)
        rows=[]
        for f in FAMS:
            for o in (0,1):
                rows+=[route(plus['id'],t,'noise',f,o,.6),route(minus['id'],t,'noise',f,o,.2),route(plus['id'],t,'restore',f,o,.5),route(minus['id'],t,'restore',f,o,.2)]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'i';tok,pairs,plan=T.synthetic_inputs(out)
            att=attachment(t,h);ctrl=plan['control_rosters'][att['key']]['control'];diag=state(set(ids(C33))|{hid(h),hid(ctrl)},'diag')
            for f in FAMS:
                for o in (0,1):
                    for d in ['noise','restore']:
                        rows.append(route(diag['id'],att['key'],d,f,o,.4 if f%4==0 else -.4,'target'));rows.append(route(diag['id'],att['key']+'|control='+ctrl,d,f,o,.05,'control'))
            sel=dict(selected=[dict(candidate=h,structure=t,record='route_pair')]);sd=P.stage_d(out,plan,C33,sel).write(out/'d.json')
            orig=A.gather;A.gather=lambda *a,**k:rows
            try:res=A.analyze_stage_d(out,sd,[],Path(td)/'an',C33,sel,[])
            finally:A.gather=orig
        g=res['gamma'][0];self.assertAlmostEqual(g['noise']['gamma']['mean'],.4);self.assertAlmostEqual(g['restore']['gamma']['mean'],.3);self.assertEqual(g['bidirectional'],'coherent')
        a=res['attachments'][0];self.assertEqual(a['noise']['target']['classification'],'heterogeneous');self.assertAlmostEqual(a['noise']['abs_paired_difference']['mean'],.35)
        self.assertTrue(a['noise']['receiver_selectivity']);self.assertEqual(a['bidirectional'],'heterogeneous');self.assertEqual(a['control_receiver'],ctrl)

class Validation(unittest.TestCase):
    def test_final_labels(self):
        import tempfile,test_s45 as T,s45_plan as P
        from pathlib import Path
        import s45_freeze as F
        h,t='L17H5','T2';st=states_for(C33);plus,minus=st[f'B+{h}'],st[f'B-{h}'];HF=list(range(1,41,2))
        def route(sid,key,d,f,o,eff,role='anchor',anchor=t):return dict(kind='route' if role=='anchor' else 'attachment',pair_id=f'{f}/{o}',family=f,order=o,split='heldout',cell='x00' if d=='noise' else 'x10',
            state=sid,anchor=anchor if role=='anchor' else None,key=key,direction=d,effect=eff,intact_margin=0.,endpoint_margin=-eff,hybrid_margin=0.,channel_norm=1.,source_norm=1.,role=role)
        base=lambda c:{'x00':10.,'x10':-10.,'x01':-8.,'x11':8.}[c]
        rows=[]
        for f in HF:
            for o in (0,1):
                for c in CELLS:
                    sh=1. if c=='x00' else -1. if c=='x10' else 0.   # L17H5 raises binding contrast: d_b=+0.5
                    rows.append(beh(plus['id'],c,f,o,base(c)+sh));rows.append(beh(minus['id'],c,f,o,base(c)))
                    for lab in ['full','empty']:rows.append(beh(state(None,'full')['id'] if lab=='full' else state([],'empty')['id'],c,f,o,base(c)))
                rows+=[route(plus['id'],t,'noise',f,o,.6),route(minus['id'],t,'noise',f,o,.2),route(plus['id'],t,'restore',f,o,.5),route(minus['id'],t,'restore',f,o,.2)]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'i';tok,pairs,plan=T.synthetic_inputs(out)
            att=attachment(t,h);ctrl=plan['control_rosters'][att['key']]['control'];diag=state(set(ids(C33))|{hid(h),hid(ctrl)},'diag')
            for f in HF:
                for o in (0,1):
                    for d in ['noise','restore']:rows.append(route(diag['id'],att['key'],d,f,o,.5,'target'));rows.append(route(diag['id'],att['key']+'|control='+ctrl,d,f,o,.05,'control'))
            sel=dict(selected=[dict(candidate=h,structure=t,record='route_pair',gamma_mean=.4),dict(candidate='L1H27',structure='B',record='functional_only',d_b_mean=.2)])
            freeze=dict(B=dict(members=C33,status='pass'),selection=sel)
            hplan=dict(plan,families=HF,core_families=[],discovery_plan_hash='x');hout=Path(td)/'h';hout.mkdir();A.json_write(hout/'plan.json',hplan)
            import gzip,json
            with gzip.open(hout/'pairs.jsonl.gz','wt') as fz:
                for p in pairs[:0]:fz.write(json.dumps(p)+'\n')
            hplan['pair_hash']=A.digest(hout/'pairs.jsonl.gz');A.json_write(hout/'plan.json',hplan)
            vs=P.validation(hout,hplan,freeze).write(hout/'v.json')
            orig=A.gather;A.gather=lambda *a,**k:rows;origload=A.load;A.load=lambda i,s=None:(hplan,A.json_read(s),[])
            try:labels=A.analyze_validation(hout,vs,[],Path(td)/'an',freeze,dict(signs={h:dict(d_b=1,gamma=1,attachment=1),'L1H27':dict(d_b=1)}))
            finally:A.gather=orig;A.load=origload
        by={l['candidate']:l for l in labels}
        self.assertTrue(by[h]['functional_reproduced'] and by[h]['gamma_reproduced'] and by[h]['attachment_resolved']);self.assertTrue(by[h]['final_label'].startswith('(a)'))
        self.assertTrue(by['L1H27']['final_label'].startswith('(e)'))   # retained in discovery, no held-out state rows -> not reproduced

class Rule(unittest.TestCase):
    def test_retention_rule(self):
        self.assertEqual(retained([.2]*7+[-.1]*3)['classification'],'coherent')
        self.assertEqual(retained([.5,-.5,.5,-.5,0,0,0,0,0,0])['classification'],'heterogeneous')
        self.assertEqual(retained([.05]*10)['classification'],'below_rule')
        ci=A.bootstrap(np.array([1.,2.,3.,4.]));self.assertTrue(ci[0]<2.5<ci[1])

if __name__=='__main__':unittest.main()
