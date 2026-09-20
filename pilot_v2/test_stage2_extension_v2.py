"""CPU tests for stage2_extension_v2 (tiny model from test_stage2.py)."""
import unittest
import numpy as np
from stage2_extension_v2 import plan, first_diff

try:
    import torch
    from types import SimpleNamespace
    from stage2_extension_v2 import EngineV2
except ImportError:
    torch = None


if torch is not None:
    # Same tiny model as test_stage2.py (copied to avoid its analysis-module imports).
    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.v_proj = torch.nn.Linear(8, 8, bias=False); self.o_proj = torch.nn.Linear(8, 8, bias=False)
        def forward(self, x):
            n = x.shape[1]; a = torch.ones(n, n, device=x.device).tril(); a = a / a.sum(dim=-1, keepdim=True)
            return self.o_proj(a @ self.v_proj(x))

    class TinyLayer(torch.nn.Module):
        def __init__(self, nonlinear):
            super().__init__(); self.self_attn = TinyAttention(); self.nonlinear = nonlinear
        def forward(self, x):
            y = x + self.self_attn(x)
            return y + 0.1 * torch.tanh(y) if self.nonlinear else y

    class TinyModel(torch.nn.Module):
        def __init__(self, nonlinear=False):
            super().__init__(); torch.manual_seed(42)
            self.embedding = torch.nn.Embedding(12, 8); self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([TinyLayer(nonlinear) for _ in range(2)])
            self.unembed = torch.nn.Linear(8, 12, bias=False)
            self.config = SimpleNamespace(num_attention_heads=2, hidden_size=8)
        def get_input_embeddings(self): return self.embedding
        def forward(self, ids, use_cache=False):
            x = self.embedding(ids)
            for l in self.model.layers: x = l(x)
            return SimpleNamespace(logits=self.unembed(x))


class PlanTests(unittest.TestCase):
    def test_whole_families_and_balanced_forwards(self):
        pairs = [dict(family=f, order=o, id=f'{f}/{o}', exact_all=(f % 4 == 0)) for f in range(89) for o in range(2)]
        planned, loads = plan(pairs, list(range(56)), list(range(69)), 3)
        for f in range(89):
            self.assertEqual(len({p['worker'] for p in planned if p['family'] == f}), 1)
        expected = sum((0 if p['exact_all'] else 56) + 69 for p in pairs)
        self.assertEqual(sum(loads), expected)
        self.assertLessEqual(max(loads) - min(loads), 2 * 125)


@unittest.skipIf(torch is None, 'CPU PyTorch required')
class EngineV2Tests(unittest.TestCase):
    def setup(self, nonlinear=True):
        e = EngineV2(torch, None, TinyModel(nonlinear))
        # positions 0..2 are the prompt; token 7 at position 3 is a shared answer prefix (never patched).
        spec = dict(clean=torch.tensor([[1, 2, 3, 7]]), corr=torch.tensor([[1, 4, 3, 7]]), n=3, g=8, d=9)
        clean = e.cache(spec['clean'][:, :3]); corr = e.cache(spec['corr'][:, :3])
        return e, spec, clean, corr

    def test_first_diff(self):
        e, s, a, b = self.setup()
        self.assertEqual(first_diff(s), 1)

    def test_all_positions_equals_engine_patch(self):
        e, s, a, b = self.setup(); m0 = e.metric(s['clean'], s['g'], s['d'])
        heads = [(0, 0), (0, 1), (1, 0), (1, 1)]
        exact, _ = e.exact(s, b, heads)
        pos, _ = e.exact_positions(s, b, heads, [0, 1, 2], 5)
        np.testing.assert_allclose(pos[:, 0], [exact[h] for h in range(4)], atol=1e-6)
        np.testing.assert_allclose(pos[:, 0], pos[:, 1] - pos[:, 2], atol=1e-6)

    def test_final_position_patch_touches_only_last_prompt_position(self):
        e, s, a, b = self.setup(); seen = {}
        with e.positional_patch(1, 0, b[1], [2]):
            hook = e.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda mod, inp: seen.update(z=inp[0].detach().clone()))
            try: e.metric(s['clean'], s['g'], s['d'])
            finally: hook.remove()
        full = e.cache(s['clean'])[1]
        torch.testing.assert_close(seen['z'][:, :2], full[:, :2], rtol=0, atol=0)      # earlier positions untouched
        torch.testing.assert_close(seen['z'][:, 3:], full[:, 3:], rtol=0, atol=0)      # answer prefix untouched
        torch.testing.assert_close(seen['z'][:, 2, :4], b[1][:, 2, :4], rtol=0, atol=0)  # head 0 slice replaced at position 2
        torch.testing.assert_close(seen['z'][:, 2, 4:], full[:, 2, 4:], rtol=0, atol=0)  # head 1 slice untouched

    def test_prefix_patch_is_neutral_and_self_patch_exact(self):
        e, s, a, b = self.setup(); m0 = e.metric(s['clean'], s['g'], s['d']); k = first_diff(s)
        heads = [(0, 0), (1, 1)]
        pre, _ = e.exact_positions(s, b, heads, list(range(k)), 5)
        np.testing.assert_allclose(pre[:, 0], m0, atol=1e-6)         # identical prefix activations -> no change
        for li in range(e.L):
            torch.testing.assert_close(a[li][:, :k], b[li][:, :k], rtol=0, atol=0)
        selfF, _ = e.exact_positions(s, a, heads, [2], 5)
        for v in selfF[:, 0]: self.assertEqual(v, m0)

    def test_final_scope_differs_from_all_scope_when_head_writes_earlier(self):
        e, s, a, b = self.setup(); heads = [(0, 0), (0, 1), (1, 0), (1, 1)]
        allp, _ = e.exact_positions(s, b, heads, [0, 1, 2], 5)
        fin, _ = e.exact_positions(s, b, heads, [2], 5)
        self.assertTrue(np.any(np.abs(allp[:, 0] - fin[:, 0]) > 1e-6))


if __name__ == '__main__': unittest.main()
