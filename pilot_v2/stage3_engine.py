"""Head-local interventions; capture actual SDPA inputs after QK norm and RoPE.

Never reconstruct OLMo Q/K from raw weights. Normal forwards keep the installed
Transformers/PyTorch SDPA implementation; only an o_proj input slice is patched.
"""
import contextlib
import numpy as np
from stage2_engine import Engine


class Stage3Engine(Engine):
    def readout(self, ids, g, d):
        with self.t.no_grad(): v=self.model(ids,use_cache=False).logits[0,-1].float()
        return np.array([float(v[g]-v[d]),float(v[g]),float(v[d])])

    @contextlib.contextmanager
    def vector_patch(self, head, positions, vectors):
        li,hi=divmod(head,self.H); positions=list(positions)
        if not positions: raise ValueError('Empty patch')
        def replace(mod, inp):
            z=inp[0]; out=z.clone(); ix=self.t.tensor(positions,device=z.device)
            v=vectors.to(z.device,z.dtype)
            if v.shape!=(len(positions),self.DH): raise ValueError('Patch vector shape mismatch')
            out[0,ix,hi*self.DH:(hi+1)*self.DH]=v
            return (out,)
        h=self.layers[li].self_attn.o_proj.register_forward_pre_hook(replace)
        try: yield
        finally: h.remove()

    def patched(self, spec, head, positions, vectors, reverse=False):
        with self.vector_patch(head,positions,vectors):
            return self.readout(spec['corr'] if reverse else spec['clean'],spec['g'],spec['d'])

    def capture(self, ids, heads, rows, context_inputs=False):
        """CPU caches for selected heads/rows; includes exact SDPA outputs.

        Reconstructed attention is float32 softmax of actual Q/K, with the actual
        mask and scale. Its product with actual V is gated against the real SDPA
        output before use in attention/value interventions.
        """
        t=self.t; F=t.nn.functional; original=F.scaled_dot_product_attention
        chosen={li:[h%self.H for h in heads if h//self.H==li] for li in range(self.L)}
        handles=[]; active=[None]; data={}; inputs={}; calls=[]
        def enter(li):
            def hook(mod,args,kwargs): active[0]=li
            return hook
        def keep_input(li):
            def hook(mod,args): inputs[li]=args[0][0].detach().cpu().clone()
            return hook
        def sdpa(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None,**kw):
            result=original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,scale=scale,**kw)
            li=active[0]; hs=chosen.get(li,[])
            if not hs: return result
            if q.shape[0]!=1 or q.shape[1]!=self.H or k.shape[1]!=self.H or dropout_p:
                raise ValueError('Stage 3 requires eval-mode MHA with batch size one')
            if li in calls: raise ValueError('Multiple SDPA calls in one attention layer')
            calls.append(li)
            ix=t.tensor(rows,device=q.device); hi=t.tensor(hs,device=q.device)
            Q=q[0,hi][:,ix].float(); K=k[0,hi].float(); V=v[0,hi]
            scores=(Q@K.transpose(-1,-2))*(float(scale) if scale is not None else q.shape[-1]**-0.5)
            if is_causal:
                allowed=t.arange(k.shape[-2],device=q.device)[None,:]<=ix[:,None]
                scores=scores.masked_fill(~allowed[None],float('-inf'))
            if attn_mask is not None:
                mask=attn_mask
                if mask.ndim==4:
                    mask=mask[0]
                    if mask.shape[0]==self.H: mask=mask[hi]
                if mask.shape[-2]>1: mask=mask[...,ix,:k.shape[-2]]
                else: mask=mask[...,:k.shape[-2]]
                scores=scores.masked_fill(~mask,float('-inf')) if mask.dtype==t.bool else scores+mask.float()
            pattern=t.softmax(scores,dim=-1)
            actual=result[0,hi][:,ix]
            recon=(pattern@V.float()).to(actual.dtype)
            for j,h in enumerate(hs):
                data[li*self.H+h]=dict(pattern=pattern[j].detach().cpu(),values=V[j].detach().cpu(),
                                       z=actual[j].detach().cpu(),rows=list(rows),
                                       reconstruction_error=float((recon[j]-actual[j]).abs().max()))
            return result
        try:
            for li,layer in enumerate(self.layers):
                handles.append(layer.self_attn.register_forward_pre_hook(enter(li),with_kwargs=True))
                if context_inputs and chosen[li]:handles.append(layer.self_attn.v_proj.register_forward_pre_hook(keep_input(li)))
            F.scaled_dot_product_attention=sdpa
            with t.no_grad(): logits=self.model(ids,use_cache=False).logits[0,-1].float().cpu()
        finally:
            F.scaled_dot_product_attention=original
            for h in handles:h.remove()
        if set(data)!=set(heads): raise ValueError('SDPA interception did not capture every requested head; unsupported attention runtime')
        return dict(heads=data,value_inputs=inputs,logits=logits)

    def mix_vectors(self, clean, corr, head, position):
        a,b=clean['heads'][head],corr['heads'][head]
        ia,ib=a['rows'].index(position),b['rows'].index(position)
        # CPU float32 matmul keeps scratch memory off GPUs. One position only.
        return [a['pattern'][ia:ia+1]@a['values'].float(),
                b['pattern'][ib:ib+1]@a['values'].float(),
                a['pattern'][ia:ia+1]@b['values'].float(),
                b['pattern'][ib:ib+1]@b['values'].float()]

    def projected_logits(self, head, x, from_value_input=False, token_ids=None):
        """Project with the exact fp16 matrix-product convention of the RI run."""
        t=self.t;li,hi=divmod(head,self.H);att=self.layers[li].self_attn
        with t.no_grad():
            if from_value_input:
                if att.v_proj.bias is not None: raise ValueError('Unexpected value bias')
                x=x.to(att.v_proj.weight.device,att.v_proj.weight.dtype)
                z=x@att.v_proj.weight[hi*self.DH:(hi+1)*self.DH].T
            else:z=x
            w=att.o_proj.weight[:,hi*self.DH:(hi+1)*self.DH]
            out=z.to(w.device,w.dtype)@w.T
            u=self.model.get_output_embeddings().weight
            if token_ids is not None:u=u[t.tensor(token_ids,device=u.device)]
            return (out.to(u.device,u.dtype)@u.T).float().cpu().numpy()
