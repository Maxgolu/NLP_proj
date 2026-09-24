"""Simultaneous head-output replacement before the shared OLMo2 normalization."""
import contextlib
import collections
from stage4_engine import Stage4Engine


class S42Engine(Stage4Engine):
    def capture_z(self,ids,heads,g,d):
        cache={};handles=[]
        def hook(li):
            def keep(mod,args):
                z=args[0]
                for h in heads:
                    if h//self.H==li:
                        j=h%self.H;cache[h]=z[0,:,j*self.DH:(j+1)*self.DH].detach().cpu().clone()
            return keep
        try:
            for li in sorted({h//self.H for h in heads}):
                handles.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(hook(li)))
            out=self.output(ids,g,d)
        finally:
            for h in handles:h.remove()
        if set(cache)!=set(heads):raise ValueError('Missing head output capture')
        return dict(z=cache,output=out)

    @contextlib.contextmanager
    def joint(self,patches):
        """patches: head -> (unique positions, replacement vectors). One hook/layer."""
        handles=[];seen=set();layers=collections.defaultdict(list)
        for h,(pos,values) in patches.items():
            if not pos or len(pos)!=len(set(pos)) or values.shape!=(len(pos),self.DH):raise ValueError('Bad patch shape/positions')
            if not 0<=h<self.L*self.H:raise ValueError('Head outside model')
            layers[h//self.H].append((h,pos,values))
        def hook(li):
            def replace(mod,args):
                if li in seen:raise ValueError('Repeated layer patch')
                seen.add(li);z=args[0];out=z.clone()
                for h,pos,values in layers[li]:
                    if min(pos)<0 or max(pos)>=z.shape[1]:raise ValueError('Patch outside sequence')
                    j=h%self.H;out[0,pos,j*self.DH:(j+1)*self.DH]=values.to(z.device,z.dtype)
                return (out,)+args[1:]
            return replace
        try:
            for li in layers:handles.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(hook(li)))
            yield
            if seen!=set(layers):raise ValueError('Patch hook not called')
        finally:
            for h in handles:h.remove()

    def patches(self,p,c,rec,don,mode,bank=None):
        selected=collections.defaultdict(set);donors={}
        for a in c['atoms']:
            h=a['head'];selected[h].update(p['masks'][a['site']]);donors[h]=a['donor']
        patches={};norms={};fallback=collections.Counter()
        for h,positions in selected.items():
            ix=sorted(positions)
            if donors[h]=='self':v=rec['z'][h][ix]
            elif mode=='donor':v=don['z'][h][ix]
            elif mode=='mean':
                # Means have no demonstration or teacher-forced-answer support.
                ix=[j for j in ix if p['mean_keys'][j] is not None]
                if not ix:raise ValueError('Mean intervention has no test support')
                vv,info=bank.vectors(h,p['family'],[p['mean_keys'][j] for j in ix])
                v=self.t.from_numpy(vv);fallback.update(info)
            else:raise ValueError('Unknown replacement mode')
            patches[h]=(ix,v);norms[str(h)]=float((v.to(rec['z'][h].dtype).float()-rec['z'][h][ix].float()).norm())
        return patches,norms,dict(fallback)

    def evaluate(self,ids,patches,g,d,observe_head=None,row=None):
        with self.joint(patches):
            if observe_head is None:return self.output(ids,g,d),None
            cap=self.capture(ids,[observe_head],[row]);x=cap['heads'][observe_head];v=cap['logits'];prob=v.softmax(-1)
            out=dict(margin=float(v[g]-v[d]),clean_logit=float(v[g]),corr_logit=float(v[d]),
                     clean_prob=float(prob[g]),corr_prob=float(prob[d]),top_token=int(v.argmax()))
            diag=dict(pattern=x['pattern'][0].tolist(),z=x['z'][0].float().tolist(),
                      reconstruction_error=x['reconstruction_error'],row=row,head=observe_head,
                      scope='under background, before own o_proj replacement')
            return out,diag
