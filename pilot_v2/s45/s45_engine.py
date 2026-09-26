"""Retained-head backgrounds over the frozen S4.1/S4.3 route executors.

Precedence (specification section 12): the background hook is registered FIRST on every
o_proj input, so mean or donor masks define the state; source replacement, selective
live-head clamps and captures are registered afterwards and override only their declared
coordinates. Recipient, donor, hybrid and endpoint runs of one route all execute inside
the SAME background object; full-model captures are never mixed into a retained state.
"""
import contextlib
import numpy as np
from s43_engine import S43Engine


class S45Engine(S43Engine):
    def spec(self,pair,cell):
        c=pair['cells'][cell];ids=self.t.tensor([c['input_ids']],device=self.device)
        return dict(ids=ids,n=c['n'],g=c['tok_a'],d=c['tok_b'],cell=cell,masks=c['masks'],positions=[j for j,k in enumerate(c['mean_keys']) if k is not None])

    @contextlib.contextmanager
    def background(self,live,positions,replacement,norms=None):
        """live=None: full model. Otherwise every head outside `live` is replaced at `positions`.

        replacement: (positions, 1024, DH) CPU tensor of role means or paired-donor outputs.
        """
        if live is None:yield;return
        positions=list(positions)
        if not positions or len(set(positions))!=len(positions):raise ValueError('Background positions')
        if tuple(replacement.shape)!=(len(positions),self.L*self.H,self.DH):raise ValueError('Replacement shape')
        live=set(int(h) for h in live);handles=[];seen=set();P=len(positions)
        def hook(li):
            keep=self.t.tensor([hi in {h%self.H for h in live if h//self.H==li} for hi in range(self.H)])
            rep=replacement[:,li*self.H:(li+1)*self.H,:]
            def replace(mod,args):
                if li==0:seen.clear()
                if li in seen:raise ValueError('Background hook repeated in one forward')
                seen.add(li);z=args[0]
                if z.shape[0]!=1 or max(positions)>=z.shape[1]:raise ValueError('Background requires batch one and prompt positions')
                out=z.clone();ix=self.t.tensor(positions,device=z.device);sel=(~keep).to(z.device)
                if bool(sel.any()):
                    blk=out[0,ix].reshape(P,self.H,self.DH);r=rep.to(z.device,z.dtype)
                    if norms is not None:norms[li]=float((r[:,sel].float()-blk[:,sel].float()).norm())
                    blk[:,sel]=r[:,sel];out[0,ix]=blk.reshape(P,self.H*self.DH)
                return (out,)+args[1:]
            return replace
        try:
            for li in range(self.L):handles.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(hook(li)))
            yield
            if seen and seen!=set(range(self.L)):raise ValueError('Background hook missed a layer')
        finally:
            for h in handles:h.remove()

    def all_head_outputs(self,ids,positions,g,d):
        """Full-model o_proj inputs at `positions` for all heads: (positions, 1024, DH) float32 CPU."""
        saved={};handles=[];positions=list(positions)
        def keep(li):
            def hook(mod,args):
                z=args[0];saved[li]=z[0,self.t.tensor(positions,device=z.device)].detach().float().cpu().reshape(len(positions),self.H,self.DH)
            return hook
        try:
            for li in range(self.L):handles.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(keep(li)))
            out=self.output(ids,g,d)
        finally:
            for h in handles:h.remove()
        if set(saved)!=set(range(self.L)):raise ValueError('Missing head-output capture')
        return self.t.cat([saved[li] for li in range(self.L)],dim=1),out

    def behave(self,sp,live,rep):
        norms={}
        with self.background(live,sp['positions'],rep,norms):out=self.output(sp['ids'],sp['g'],sp['d'])
        return out,dict(replaced_norm=float(np.sqrt(sum(v*v for v in norms.values()))) if norms else 0.)

    def route(self,sp_rec,sp_don,live,rep_rec,rep_don,job,cache=None,capture_heads=None,capture_layers=None):
        """One route/attachment endpoint inside a declared background; all components share it.

        Recipient/donor captures may cover a superset of heads/layers (shared across anchors of
        one background state); they are always captured under this same background.
        """
        heads=sorted({job['source'],job['receiver'],*job.get('live',[])});layers=sorted({h//self.H for h in job.get('live',[])})
        cheads=sorted(set(heads)|set(capture_heads or []));clayers=sorted(set(layers)|set(capture_layers or []))
        site=job['site'];n=sp_rec['n'];g,d=sp_rec['g'],sp_rec['d'];pos=sp_rec['masks'][site]
        if not pos or max(pos)>=n:raise ValueError('Site outside prompt')
        cache=cache if cache is not None else {}
        def captured(sp,rep):
            key=(sp['cell'],tuple(cheads),tuple(clayers))
            if key not in cache:
                with self.background(live,sp['positions'],rep):
                    cache[key]=self.capture_s43(sp['ids'],cheads,g,d,full_z_layers=clayers)
            return cache[key]
        rec=captured(sp_rec,rep_rec);don=captured(sp_don,rep_don)
        with self.background(live,sp_rec['positions'],rep_rec):
            if job.get('live'):hy=self.chain_hybrid(sp_rec['ids'],job['source'],pos,don,rec,heads,g,d,job['live'])
            else:hy=self.local_hybrid(sp_rec['ids'],job['source'],pos,don,rec,heads,g,d,live_branches=())
            cfg=dict(kind='head',receiver=job['receiver'],channel=job['channel'])
            end=self.endpoint(sp_rec['ids'],cfg,pos,hy,g,d,n)
        norm=self.channel_norm(cfg,pos,hy,rec,n)
        effect=rec['output']['margin']-end['margin'] if job['direction']=='noise' else end['margin']-rec['output']['margin']
        return dict(effect=effect,intact=rec['output'],endpoint=end,hybrid_output=hy['output'],channel_norm=norm,
                    reconstruction_error=max(rec['reconstruction_error'],hy['reconstruction_error']),
                    source_norm=float((don['z'][job['source']][pos].float()-rec['z'][job['source']][pos].float()).norm()))
