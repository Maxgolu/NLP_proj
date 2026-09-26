"""Exact head-output patching and matched first-order / input-path IG estimates."""
import contextlib
import time
import numpy as np
from stage1_audit import first_divergence


class Engine:
    def __init__(self,torch,tok,model):
        self.t=torch;self.tok=tok;self.model=model;self.layers=model.model.layers
        self.H=model.config.num_attention_heads;self.L=len(self.layers)
        self.DH=model.config.hidden_size//self.H
        self.device=model.get_input_embeddings().weight.device
        for p in model.parameters():p.requires_grad_(False)
        model.eval()

    def sync(self):
        if self.t.cuda.is_available():
            for i in range(self.t.cuda.device_count()):self.t.cuda.synchronize(i)

    def encode(self,pair):
        a=first_divergence(self.tok,pair['clean']);b=first_divergence(self.tok,pair['corr'])
        if a['prompt_length']!=b['prompt_length'] or len(a['input_ids'])!=len(b['input_ids']):
            raise ValueError('Unaligned prompt lengths: '+pair['id'])
        if a['shared_prefix']!=b['shared_prefix'] or (a['gold_token'],a['other_token'])!=(b['other_token'],b['gold_token']):
            raise ValueError('Inconsistent divergent-token contrast: '+pair['id'])
        if sum(x!=y for x,y in zip(a['input_ids'],b['input_ids']))>6:
            raise ValueError('More than six differing tokens: '+pair['id'])
        return dict(clean=self.t.tensor([a['input_ids']],device=self.device),
                    corr=self.t.tensor([b['input_ids']],device=self.device),
                    n=a['prompt_length'],g=a['gold_token'],d=a['other_token'],
                    shared_prefix=a['shared_prefix'])

    def raw_metric(self,ids,g,d):
        logits=self.model(ids,use_cache=False).logits[0,-1]
        return logits[g].float()-logits[d].float()

    def metric(self,ids,g,d):
        with self.t.no_grad():return float(self.raw_metric(ids,g,d))

    def cache(self,ids):
        saved={};handles=[]
        def keep(li):
            def hook(mod,inp):saved[li]=inp[0].detach().clone()
            return hook
        try:
            for li,layer in enumerate(self.layers):
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(keep(li)))
            with self.t.no_grad():self.model(ids,use_cache=False)
        finally:
            for h in handles:h.remove()
        return saved

    @contextlib.contextmanager
    def patch(self,li,hi,donor,scale=1.):
        def hook(mod,inp):
            z=inp[0];n=donor.shape[1]
            if n>z.shape[1]:raise ValueError('Donor must contain only prompt positions')
            result=z.clone();sl=slice(hi*self.DH,(hi+1)*self.DH)
            target=donor[:,:n,sl].to(z.device,z.dtype)
            result[:,:n,sl]=target if scale==1 else z[:,:n,sl]+scale*(target-z[:,:n,sl])
            return (result,)
        h=self.layers[li].self_attn.o_proj.register_forward_pre_hook(hook)
        try:yield
        finally:h.remove()

    def exact(self,spec,donor,heads):
        values={};self.sync();start=time.monotonic()
        for index,(li,hi) in enumerate(heads):
            with self.patch(li,hi,donor[li]):
                value=self.metric(spec['clean'],spec['g'],spec['d'])
            if not np.isfinite(value):raise ValueError('Non-finite patched metric')
            values[li*self.H+hi]=value
            if len(heads)>=128 and (index+1)%128==0:
                print(f'  exact heads {index+1}/{len(heads)}',flush=True)
        self.sync()
        return values,time.monotonic()-start

    def attribution(self,spec,clean_cache,corr_cache,steps=1):
        """steps=1: clean gradient. steps>1: midpoint input-embedding IG.

        In either case, contract with the SAME clean->corrupted head-output
        delta at original prompt positions. The IG result is an approximation,
        not an exact single-head path integral or an exact causal effect.
        """
        if steps<1:raise ValueError('Need positive interpolation steps')
        t=self.t;embed=self.model.get_input_embeddings()
        with t.no_grad():
            e0=embed(spec['clean']).detach();e1=embed(spec['corr']).detach()
        result=np.zeros((self.L,self.H),dtype=float)
        alphas=[0.] if steps==1 else [(i+.5)/steps for i in range(steps)]
        for alpha in alphas:
            saved={};handles=[]
            def replace(mod,inp,out):
                return (e0+alpha*(e1-e0)).detach().requires_grad_(True)
            def keep(li):
                def hook(mod,inp):saved[li]=inp[0]
                return hook
            try:
                handles.append(embed.register_forward_hook(replace))
                for li,layer in enumerate(self.layers):
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(keep(li)))
                with t.enable_grad():
                    metric=self.raw_metric(spec['clean'],spec['g'],spec['d'])
                    gradients=t.autograd.grad(metric,[saved[li] for li in range(self.L)],allow_unused=False)
                for li,grad in enumerate(gradients):
                    delta=(corr_cache[li]-clean_cache[li]).float()
                    product=delta*grad[:,:spec['n']].float()
                    values=product.reshape(1,spec['n'],self.H,self.DH).sum(dim=(0,1,3))
                    result[li]+=values.detach().cpu().numpy()/steps
                del gradients,metric
            finally:
                for h in handles:h.remove()
                saved.clear()
        if not np.isfinite(result).all():raise ValueError('Non-finite attribution')
        return result.reshape(-1)

    def prepare(self,pair):
        spec=self.encode(pair)
        clean=self.cache(spec['clean'][:,:spec['n']]);corr=self.cache(spec['corr'][:,:spec['n']])
        m0=self.metric(spec['clean'],spec['g'],spec['d'])
        mc=self.metric(spec['corr'],spec['g'],spec['d'])
        if not np.isfinite([m0,mc]).all():raise ValueError('Non-finite baseline metric')
        return spec,clean,corr,m0,mc

    def gate(self,pairs,references,runner_score,tolerance=.05):
        """Do not gate on head usefulness, effect sign or approximation quality."""
        reports=[]
        probe_heads=sorted({(3,11),(9,22),(16,4),(0,0),(16,0),(31,31)})
        probe_heads=[h for h in probe_heads if h[0]<self.L and h[1]<self.H]
        for pair in pairs:
            self.sync();start=time.monotonic()
            spec,clean,corr,m0,mc=self.prepare(pair)
            drift=[]
            for row in (pair['clean'],pair['corr']):
                other=next(c for c in row['candidates'] if c!=row['gold'])
                score=runner_score(self.t,self.tok,self.model,self.device,row['prompt'],row['gold'])-runner_score(self.t,self.tok,self.model,self.device,row['prompt'],other)
                drift.append(abs(score-references[row['id']]))
            if max(drift)>=tolerance:raise ValueError('Behavioral replication failed: '+pair['id'])
            self_effect=[]
            for li,hi in probe_heads:
                with self.patch(li,hi,clean[li]):value=self.metric(spec['clean'],spec['g'],spec['d'])
                self_effect.append(abs(value-m0))
            if max(self_effect,default=0)>=.001:raise ValueError('Self patch failed: '+pair['id'])
            # Endpoint test independent of candidate token identities: transplant
            # the complete corrupted input embeddings and recover its SAME-sign metric.
            with self.t.no_grad():donor_emb=self.model.get_input_embeddings()(spec['corr']).detach()
            hook=self.model.get_input_embeddings().register_forward_hook(lambda mod,inp,out:donor_emb)
            try: endpoint=self.metric(spec['clean'],spec['g'],spec['d'])
            finally:hook.remove()
            if abs(endpoint-mc)>=tolerance:raise ValueError('Embedding endpoint failed: '+pair['id'])
            attr=self.attribution(spec,clean,corr)
            exact,seconds=self.exact(spec,corr,probe_heads)
            # Identical donor and recipient must give exactly zero Taylor delta.
            self_attr=self.attribution(spec,clean,clean)
            if np.max(np.abs(self_attr))!=0:raise ValueError('Self attribution failed')
            self.sync()
            reports.append(dict(id=pair['id'],shared_prefix=spec['shared_prefix'],m_clean=m0,
                m_corr_fixed_sign=mc,runner_max_drift=max(drift),self_patch_max=max(self_effect,default=0),
                embedding_endpoint_drift=abs(endpoint-mc),finite_gradient=True,
                exact_delta={str(h):v-m0 for h,v in exact.items()},
                attribution_preview={str(h):float(attr[h]) for h in exact},
                patched_forward_s=seconds/len(exact),elapsed_s=time.monotonic()-start))
            print(f'Gate pair passed: {pair["id"]}; shared prefix {spec["shared_prefix"]}',flush=True)
        return dict(passed=True,pairs=reports,patch_scope='all original prompt positions only',
                    metric='first divergent token contrast; clean-gold sign held fixed')
