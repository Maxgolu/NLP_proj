"""Exact OLMo2 residual-route interventions. No gradients, eager SDPA, one worker/process.

Caches are CPU tensors. Branches are frozen AFTER shared normalization, never at
the sum of unnormalized heads. Only the selected receiver output row is released
when injecting a changed K/V channel; other rows keep their unmodified SDPA output.
"""
import contextlib
import numpy as np
from stage3_engine import Stage3Engine


class Stage4Engine(Stage3Engine):
    def __init__(self, *args):
        super().__init__(*args)
        if self.model.config.model_type != 'olmo2':
            raise ValueError('Stage 4 requires OLMo2 normalization boundaries')
        if self.model.config._attn_implementation != 'sdpa':
            raise ValueError('SDPA backend required')
        if self.model.config.num_key_value_heads != self.H:
            raise ValueError('Only the pinned MHA architecture is supported')
        for layer in self.layers:
            for name in ['post_attention_layernorm', 'post_feedforward_layernorm']:
                if not hasattr(layer, name): raise ValueError('Unknown residual branch topology')

    def output(self, ids, g, d):
        with self.t.no_grad(): v = self.model(ids, use_cache=False).logits[0, -1].float()
        p = v.softmax(-1)
        return dict(margin=float(v[g]-v[d]), clean_logit=float(v[g]), corr_logit=float(v[d]),
                    clean_prob=float(p[g]), corr_prob=float(p[d]), top_token=int(v.argmax()))

    @contextlib.contextmanager
    def instrument(self, heads=(), collect=False, freeze=None, source_layer=None,
                   injection=None, mlp_input=None, live_branches=(), profile_positions=None):
        """Capture real SDPA tensors and normalized branch increments, or intervene.

        freeze: intact recipient cache; source attention branch remains live.
        injection: head, positions, row, channel, qkv tensor dictionary.
        mlp_input: layer, positions, values. Used only in the fresh endpoint run.
        """
        t=self.t; F=t.nn.functional; original=F.scaled_dot_product_attention
        heads=set(heads); active=[None]; handles=[]; calls=set(); seen_injection=[]
        cache=dict(qkv={}, z={}, branches={}, mlp_inputs={}, residual_profiles={}, reconstruction_error=0.)
        live=set(tuple(x) for x in live_branches)
        def enter(li):
            def hook(mod,args,kwargs): active[0]=li
            return hook
        def branch(li,kind):
            def hook(mod,args,out):
                if collect: cache['branches'][(li,kind)]=out.detach().cpu().clone()
                frozen=freeze is not None and li>=source_layer and (li,kind) not in live
                if li==source_layer and kind=='attn': frozen=False
                return freeze['branches'][(li,kind)].to(out.device,out.dtype) if frozen else out
            return hook
        def mlp(li):
            def hook(mod,args):
                x=args[0]
                if collect: cache['mlp_inputs'][li]=x.detach().cpu().clone()
                if mlp_input is not None and li==mlp_input['layer']:
                    z=x.clone(); ix=mlp_input['positions']
                    z[0,ix]=mlp_input['values'].to(x.device,x.dtype)
                    return (z,)+args[1:]
            return hook
        def residual(li):
            def hook(mod,args,out):
                x=out[0] if isinstance(out,tuple) else out
                cache['residual_profiles'][li]={name:x[0,ix].detach().float().cpu().clone() for name,ix in profile_positions.items()}
            return hook
        def sdpa(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,scale=None,**kw):
            li=active[0]
            if li is None or li in calls: raise ValueError('Unexpected SDPA dispatch')
            calls.add(li)
            if q.shape[0]!=1 or q.shape[1]!=self.H or k.shape[1]!=self.H or dropout_p:
                raise ValueError('Requires deterministic batch-one MHA')
            kwargs=dict(attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,scale=scale,**kw)
            base=original(q,k,v,**kwargs)
            if collect:
                for h in sorted(heads):
                    if h//self.H!=li: continue
                    hi=h%self.H
                    cache['qkv'][h]={n:a[0,hi].detach().cpu().clone() for n,a in zip('QKV',(q,k,v))}
                    cache['z'][h]=base[0,hi].detach().cpu().clone()
                    # Explicit attention reconstruction at the final row (actual mask/RoPE/QK norm).
                    scores=q[0,hi,-1:].float()@k[0,hi].float().T
                    scores*=float(scale) if scale is not None else self.DH**-.5
                    if attn_mask is not None:
                        mask=attn_mask
                        if mask.ndim==4: mask=mask[0,hi if mask.shape[1]>1 else 0]
                        mask=mask[-1:,:k.shape[-2]]
                        scores=scores.masked_fill(~mask,float('-inf')) if mask.dtype==t.bool else scores+mask.float()
                    recon=(scores.softmax(-1)@v[0,hi].float()).to(base.dtype)
                    error=float((recon-base[0,hi,-1:]).abs().max())
                    cache['reconstruction_error']=max(cache['reconstruction_error'],error)
            if injection is not None and injection['head']//self.H==li:
                hi=injection['head']%self.H; ix=injection['positions']; abc=[]
                for name,a in zip('QKV',(q,k,v)):
                    z=a.clone()
                    if name in injection['channel']:
                        z[0,hi,ix]=injection['qkv'][name][ix].to(a.device,a.dtype)
                    abc.append(z)
                changed=original(*abc,**kwargs)
                result=base.clone(); row=injection['row']
                result[0,hi,row]=changed[0,hi,row]
                seen_injection.append(li)
                return result
            return base
        try:
            for li,layer in enumerate(self.layers):
                handles.append(layer.self_attn.register_forward_pre_hook(enter(li),with_kwargs=True))
                handles.append(layer.post_attention_layernorm.register_forward_hook(branch(li,'attn')))
                handles.append(layer.post_feedforward_layernorm.register_forward_hook(branch(li,'mlp')))
                handles.append(layer.mlp.register_forward_pre_hook(mlp(li)))
                if collect and profile_positions is not None:
                    handles.append(layer.register_forward_hook(residual(li)))
            F.scaled_dot_product_attention=sdpa
            yield cache
            if calls!=set(range(self.L)): raise ValueError('Not all layers used expected SDPA')
            if injection is not None and len(seen_injection)!=1: raise ValueError('Injection missed')
            if collect and set(cache['qkv'])!=heads: raise ValueError('Missing requested capture')
        finally:
            F.scaled_dot_product_attention=original
            for h in handles: h.remove()

    def capture4(self, ids, heads, g, d, **kwargs):
        with self.instrument(heads,collect=True,**kwargs) as cache:
            result=self.output(ids,g,d)
        cache['output']=result
        return cache

    def hybrid(self, ids, source, positions, donor, recipient, heads, g, d):
        with self.vector_patch(source,positions,donor['z'][source][positions]):
            return self.capture4(ids,heads,g,d,freeze=recipient,source_layer=source//self.H)

    def endpoint(self, ids, config, positions, hybrid, g, d, n):
        if config['kind']=='head':
            receiver=config['receiver']; channel=config['channel']
            ix=[n-1] if channel=='Q' else positions
            inj=dict(head=receiver,positions=ix,row=n-1,channel=channel,qkv=hybrid['qkv'][receiver])
            with self.instrument(injection=inj): return self.output(ids,g,d)
        if config['kind']=='mlp':
            li=config['receiver']
            inj=dict(layer=li,positions=positions,values=hybrid['mlp_inputs'][li][0,positions])
            with self.instrument(mlp_input=inj): return self.output(ids,g,d)
        if config['kind']=='bypass': return hybrid['output']
        raise ValueError('Unknown receiver kind')

    def channel_norm(self, config, positions, hybrid, recipient, n):
        if config['kind']=='head':
            h=config['receiver'];ix=[n-1] if config['channel']=='Q' else positions
            return float(sum((hybrid['qkv'][h][ch][ix].float()-recipient['qkv'][h][ch][ix].float()).square().sum()
                             for ch in config['channel']).sqrt())
        if config['kind']=='mlp':
            h=config['receiver']
            return float((hybrid['mlp_inputs'][h][0,positions].float()-recipient['mlp_inputs'][h][0,positions].float()).norm())
        return None
