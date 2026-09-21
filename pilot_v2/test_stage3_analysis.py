"""End-to-end CPU report/schema regression on clearly synthetic measurements."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from stage3_common import *
from stage3_run import finish_checkpoint,pair_filename
from stage3_analyze import analyze


class AnalysisTests(unittest.TestCase):
    def test_complete_report_and_missing_checkpoint_rejection(self):
        root=Path(__file__).resolve().parents[1];source=root/'results/stage3_inputs_v1'
        tokenizer=root/'pilot_v3/olmo2_tokenizer/tokenizer.json'
        if not source.exists() or not tokenizer.exists():self.skipTest('Local fixture sources not in upload')
        from transformers import PreTrainedTokenizerFast
        tok=PreTrainedTokenizerFast(tokenizer_file=str(tokenizer))
        pairs=[p for p in read_lines(source/'pairs.jsonl.gz') if p['family']==0]
        prompts=[p for p in read_lines(source/'prompts.jsonl.gz') if p['family']==0]
        ev=copy.deepcopy(next(read_lines(source/'events.jsonl.gz')));h=ev['layer']*32+ev['head']
        ev['scores']={'first':ev['scores']['first']};plan=dict(heads=[h],common_families=[0],reverse_heads=[])
        with tempfile.TemporaryDirectory() as folder:
            inputs=Path(folder)/'inputs';out=Path(folder)/'run';base=out/'replica_0';inputs.mkdir();base.mkdir(parents=True)
            json_write(inputs/'plan.json',plan);json_write(inputs/'inventory.json',dict(groups={'test fixture':[head_name(h)]}))
            write_lines(inputs/'events.jsonl.gz',[ev])
            npz_write(inputs/'references.npz',heads=[h],scopeP_delta=np.full((2,1),-.2),scopeF_delta=np.full((2,1),np.nan),
                      scopeF_readout=np.full((2,1,3),np.nan),baselines=np.tile([2.,-1.],(2,1)),families=[0,0],common=[True,True])
            json_write(out/'manifest.json',dict(input_hash=digest(inputs/'plan.json'),gate_only=False,shards=[[h]],identity='fixture'))
            for name,value in [('gate_passed.json',{'passed':True}),('gate_0.json',{'passed':True}),('done_0.json',{'complete':True})]:json_write(out/name,value)
            b=np.array([2.,3.,1.]);f=np.array([1.8,2.8,1.]);m=np.array([2.,1.9,1.95,1.8]);av=np.stack([m,m+1,np.ones(4)],axis=-1)[None]
            byid={r['id']:r for r in prompts}
            for p in pairs:
                a,c=[byid[p[k]['id']]['token_ids'] for k in ['clean','corr']];n=len(a);first=next(j for j in range(n) if a[j]!=c[j]);pos=list(range(first,n))
                path=base/'pairs'/pair_filename(p,0)
                npz_write(path,pair_id=p['id'],heads=[h],n=n,first_diff=first,positions=pos,baseline=b,corrupt_baseline=[-1.,1.,2.],
                          scopeF_readout=f[None],scopeF_reused=[False],av_readout=av,position_readout=np.tile(f,(1,len(pos),1)),
                          reverse_heads=[],reverse_readout=np.zeros((0,2,3)))
                side=path.with_suffix('.roles.json');json_write(side,[dict(j=j,text='fixture',final=j==n-1,question=False,punctuation=False,entities=[]) for j in pos])
                finish_checkpoint(path,'fixture:0:'+p['id']+':0',[side])
            for r in prompts+[changed_query(p['clean']) for p in pairs]:
                ann=annotate(tok,r);rows=ann['test_positions'];tokens=sorted(set(ann['ids']));n=len(ann['ids'])
                records=[dict(kind='metadata',id=r['id'],core=True,annotation=ann,candidate_margin=1.)]
                for j in rows:
                    facts=[]
                    for fact in ann['facts']:
                        item=dict(fact=fact['index'],query=fact['query'])
                        for role in ['source','target']:
                            item.update({role+'_visible':fact[role+'_positions'][-1]<=j,role+'_mass':.1,role+'_first':.03,role+'_last':.04})
                        facts.append(item)
                    records.append(dict(kind='anatomy',head=h,id=r['id'],family=0,order=r['order'],variant=r['variant'],j=j,final=j==n-1,
                                        self_attention=.1,previous_attention=.2,facts=facts,projection_tokens=tokens,
                                        projections={k:dict(logits=[0.]*len(tokens)) for k in ['raw','output']}))
                if r['id']==ev['id']:
                    records.append(dict(kind='contextual_ri',head=h,id=r['id'],event_id=ev['event_id'],anchor='first',family=0,
                                        query_fact=True,position='earlier',scores={k:dict(target=.3,names_gap=.1,words_gap=.2) for k in ['raw','contextual']}))
                path=base/'prompts'/(slug(r['id'])+'.jsonl.gz');write_lines(path,records)
                side=path.with_name(path.name.replace('.jsonl.gz','.attention.npz'))
                pattern=np.zeros((1,len(rows),n));pattern[:,:,0]=1
                npz_write(side,heads=[h],rows=rows,pattern=pattern);finish_checkpoint(path,'fixture:0:'+r['id'],[side])
            p=base/'copying_weights.npz';npz_write(p,heads=[h],tokens=[2,3],logits=np.zeros((1,2,2)))
            side=base/'copying_summary.json';json_write(side,[dict(head=h,population=k,tokens=2,self_minus_others=0.) for k in ['name','nonname']])
            finish_checkpoint(p,'fixture:0:weights',[side])
            records=[]
            for kind,n in [('repeated_sequence',31),('key_value_retrieval',1)]:
                for trial in range(32):
                    for j in range(n):records.append(dict(head=h,kind=kind,trial=trial,model_correct=False,source_attention=.2,
                                                         source_argmax=False,self_attention=.1,previous_attention=.1,output_gold_minus_others=0.))
            p=base/'synthetic.jsonl.gz';write_lines(p,records);finish_checkpoint(p,'fixture:0:synthetic')
            with patch('stage3_run.validate_inputs',return_value=(plan,pairs,prompts,{})):
                analyze(out,inputs)
                self.assertTrue(json_read(out/'summary.json')['complete'])
                import csv
                with (out/'analysis/head_profiles.csv').open() as f:profile=next(csv.DictReader(f))
                self.assertAlmostEqual(float(profile['stage3_F']),.2)
                self.assertAlmostEqual(float(profile['AV_interaction_importance']),.05)
                with (out/'analysis/synthetic_support.csv').open() as f:
                    self.assertTrue(all('insufficient' in x['conditional_status'] for x in csv.DictReader(f)))
                # A missing required artifact cannot be accepted as completed work.
                (base/'synthetic.jsonl.gz.ok.json').unlink()
                with self.assertRaisesRegex(ValueError,'Missing diagnostic'):analyze(out,inputs)


if __name__=='__main__':unittest.main()
