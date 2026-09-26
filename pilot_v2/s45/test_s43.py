"""Tiny random OLMo2 tests for S4.3 selective mediation."""

import unittest
import numpy as np


class TinyS43(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import (
                Olmo2Config,
                Olmo2ForCausalLM,
            )
        except ImportError as exc:
            raise unittest.SkipTest(
                "torch/transformers unavailable"
            ) from exc

        from s43_engine import S43Engine

        torch.manual_seed(143)
        torch.set_num_threads(2)

        cfg = Olmo2Config(
            vocab_size=41,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )

        cfg._attn_implementation = "sdpa"

        cls.t = torch

        cls.e = S43Engine(
            torch,
            None,
            Olmo2ForCausalLM(cfg).eval(),
        )

        cls.ids = torch.tensor(
            [[1, 3, 4, 5, 6, 7]]
        )

        cls.donor_ids = torch.tensor(
            [[1, 8, 9, 5, 6, 7]]
        )

        cls.heads = list(range(12))

        cls.g = 10
        cls.d = 11

    def captures(self):
        e = self.e

        recipient = e.capture_s43(
            self.ids,
            self.heads,
            self.g,
            self.d,
            full_z_layers=[1],
        )

        donor = e.capture_s43(
            self.donor_ids,
            self.heads,
            self.g,
            self.d,
            full_z_layers=[1],
        )

        return recipient, donor

    # ========================================================
    # 1. Backwards compatibility with Stage4Engine
    # ========================================================

    def test_no_release_matches_stage4_hybrid(self):
        """S4.3 direct comparator must equal old Stage4 hybrid."""

        e = self.e
        c, r = self.captures()

        source = 1
        pos = [1, 2]

        old = e.hybrid(
            self.ids,
            source,
            pos,
            r,
            c,
            self.heads,
            self.g,
            self.d,
        )

        new = e.local_hybrid(
            self.ids,
            source,
            pos,
            r,
            c,
            self.heads,
            self.g,
            self.d,
            live_branches=[],
        )

        self.assertAlmostEqual(
            old["output"]["margin"],
            new["output"]["margin"],
            places=6,
        )

        for h in self.heads:
            for channel in "QKV":
                np.testing.assert_allclose(
                    old["qkv"][h][channel].numpy(),
                    new["qkv"][h][channel].numpy(),
                    atol=0,
                    rtol=0,
                )

    # ========================================================
    # 2. Effective MLP branch release
    # ========================================================

    def test_released_mlp_is_only_named_branch(self):
        """Named MLP release changes only the intended effective branch.

        Stage4Engine caches ``branches`` before its freeze hook returns
        the recipient value. We therefore verify effective freezing from
        decoder-layer input/output rather than assuming that the cached
        raw branch itself has already been overwritten.
        """

        e = self.e
        c, r = self.captures()

        source = 1
        pos = [1, 2]

        def run_and_observe(call, layers=(0, 1)):
            observed = {}
            handles = []

            def pre(li):
                def hook(mod, args):
                    observed[(li, "input")] = (
                        args[0]
                        .detach()
                        .cpu()
                        .clone()
                    )
                return hook

            def post(li):
                def hook(mod, args, out):
                    x = (
                        out[0]
                        if isinstance(out, tuple)
                        else out
                    )

                    observed[(li, "output")] = (
                        x
                        .detach()
                        .cpu()
                        .clone()
                    )
                return hook

            try:
                for li in layers:
                    handles.append(
                        e.layers[li]
                        .register_forward_pre_hook(
                            pre(li)
                        )
                    )

                    handles.append(
                        e.layers[li]
                        .register_forward_hook(
                            post(li)
                        )
                    )

                result = call()

            finally:
                for handle in handles:
                    handle.remove()

            return result, observed

        # ----------------------------------------------------
        # Direct comparator:
        # source attention live, source-layer MLP frozen.
        # ----------------------------------------------------

        direct, direct_obs = run_and_observe(
            lambda: e.local_hybrid(
                self.ids,
                source,
                pos,
                r,
                c,
                self.heads,
                self.g,
                self.d,
                live_branches=[],
            )
        )

        # ----------------------------------------------------
        # Same hybrid, but MLP0 released.
        # ----------------------------------------------------

        released, released_obs = run_and_observe(
            lambda: e.local_hybrid(
                self.ids,
                source,
                pos,
                r,
                c,
                self.heads,
                self.g,
                self.d,
                live_branches=[
                    (0, "mlp"),
                ],
            )
        )

        # Both conditions compute the same raw candidate MLP.
        # Only the freeze decision differs afterward.
        np.testing.assert_allclose(
            direct["branches"][(0, "mlp")].numpy(),
            released["branches"][(0, "mlp")].numpy(),
            atol=0,
            rtol=0,
        )

        # Ensure this random test actually has a nontrivial MLP response.
        self.assertGreater(
            float(
                (
                    released["branches"][(0, "mlp")]
                    - c["branches"][(0, "mlp")]
                )
                .abs()
                .max()
            ),
            1e-8,
        )

        # ----------------------------------------------------
        # Direct effective layer-0 residual update:
        #
        # hybrid attention + recipient/frozen MLP.
        # ----------------------------------------------------

        direct_delta = (
            direct_obs[(0, "output")]
            - direct_obs[(0, "input")]
        )

        expected_direct_delta = (
            direct["branches"][(0, "attn")]
            + c["branches"][(0, "mlp")]
        )

        np.testing.assert_allclose(
            direct_delta.numpy(),
            expected_direct_delta.numpy(),
            atol=1e-6,
            rtol=1e-6,
        )

        # ----------------------------------------------------
        # Released effective layer-0 residual update:
        #
        # hybrid attention + hybrid/live MLP.
        # ----------------------------------------------------

        released_delta = (
            released_obs[(0, "output")]
            - released_obs[(0, "input")]
        )

        expected_released_delta = (
            released["branches"][(0, "attn")]
            + released["branches"][(0, "mlp")]
        )

        np.testing.assert_allclose(
            released_delta.numpy(),
            expected_released_delta.numpy(),
            atol=1e-6,
            rtol=1e-6,
        )

        # Difference between released and direct outputs must be
        # precisely the extra MLP contribution.
        observed_extra = (
            released_obs[(0, "output")]
            - direct_obs[(0, "output")]
        )

        expected_extra = (
            released["branches"][(0, "mlp")]
            - c["branches"][(0, "mlp")]
        )

        np.testing.assert_allclose(
            observed_extra.numpy(),
            expected_extra.numpy(),
            atol=1e-6,
            rtol=1e-6,
        )

        # Layer 1 was not released. Its effective branch increments
        # must therefore still be the intact-recipient branches.
        layer1_delta = (
            released_obs[(1, "output")]
            - released_obs[(1, "input")]
        )

        expected_layer1_delta = (
            c["branches"][(1, "attn")]
            + c["branches"][(1, "mlp")]
        )

        np.testing.assert_allclose(
            layer1_delta.numpy(),
            expected_layer1_delta.numpy(),
            atol=1e-6,
            rtol=1e-6,
        )

    # ========================================================
    # 3. Selective live-head shared normalization
    # ========================================================

    def test_selective_live_head_uses_shared_normalization(self):
        """Only named intermediate head is live before shared normalization."""

        e = self.e
        t = self.t
        c, r = self.captures()

        source = 1
        live = 5
        pos = [1, 2]

        # Whole layer released, used only to obtain the actual
        # source-conditioned value of the selected intermediate head.
        with e.vector_patch(
            source,
            pos,
            r["z"][source][pos],
        ):
            whole_layer_live = e.capture_s43(
                self.ids,
                self.heads,
                self.g,
                self.d,
                freeze=c,
                source_layer=0,
                live_branches=[
                    (1, "attn"),
                ],
            )

        live_z = whole_layer_live["z"][live]

        # Actual selective S4.3 chain.
        selective = e.chain_hybrid(
            self.ids,
            source,
            pos,
            r,
            c,
            self.heads,
            self.g,
            self.d,
            live_heads=[live],
        )

        # Independent expected construction:
        # clean recipient all-head bank + only one changed head.
        z = (
            c["full_z"][1]
            .clone()
            .reshape(
                1,
                6,
                4,
                8,
            )
        )

        z[
            0,
            :,
            live % 4,
            :
        ] = live_z

        z = z.reshape(
            1,
            6,
            32,
        )

        with t.no_grad():
            expected = (
                e.layers[1]
                .post_attention_layernorm(
                    e.layers[1]
                    .self_attn
                    .o_proj(z)
                )
            )

        np.testing.assert_allclose(
            selective[
                "branches"
            ][(1, "attn")].numpy(),
            expected.numpy(),
            atol=1e-6,
            rtol=1e-6,
        )

    # ========================================================
    # 4. Non-live intermediate heads remain recipient
    # ========================================================

    def test_nonlive_intermediate_heads_are_clamped(self):
        """The 3 non-live heads at the intermediate layer equal recipient."""

        e = self.e
        c, r = self.captures()

        source = 1
        live = 5
        pos = [1, 2]

        selective = e.chain_hybrid(
            self.ids,
            source,
            pos,
            r,
            c,
            self.heads,
            self.g,
            self.d,
            live_heads=[live],
        )

        mixed = (
            selective["full_z"][1]
            .reshape(
                6,
                4,
                8,
            )
        )

        clean = (
            c["full_z"][1]
            .reshape(
                6,
                4,
                8,
            )
        )

        live_local = live % 4

        for hi in range(4):
            if hi == live_local:
                continue

            np.testing.assert_allclose(
                mixed[:, hi].numpy(),
                clean[:, hi].numpy(),
                atol=0,
                rtol=0,
            )

    # ========================================================
    # 5. Releasing all heads degenerates to whole-layer release
    # ========================================================

    def test_all_live_heads_matches_whole_attention_release(self):
        """If every head is live, selective clamp becomes whole-layer release."""

        e = self.e
        c, r = self.captures()

        source = 1
        pos = [1, 2]

        layer1_heads = [
            4,
            5,
            6,
            7,
        ]

        with e.vector_patch(
            source,
            pos,
            r["z"][source][pos],
        ):
            whole = e.capture_s43(
                self.ids,
                self.heads,
                self.g,
                self.d,
                freeze=c,
                source_layer=0,
                live_branches=[
                    (1, "attn"),
                ],
            )

        selective = e.chain_hybrid(
            self.ids,
            source,
            pos,
            r,
            c,
            self.heads,
            self.g,
            self.d,
            live_heads=layer1_heads,
        )

        np.testing.assert_allclose(
            selective[
                "branches"
            ][(1, "attn")].numpy(),
            whole[
                "branches"
            ][(1, "attn")].numpy(),
            atol=0,
            rtol=0,
        )

    # ========================================================
    # 6. Self donor must be an intervention identity
    # ========================================================

    def test_self_donor_chain_is_identity(self):
        """Self donor must remain an exact intervention null."""

        e = self.e
        c, _ = self.captures()

        source = 1
        live = 5
        pos = [1, 2]

        hy = e.chain_hybrid(
            self.ids,
            source,
            pos,
            c,
            c,
            self.heads,
            self.g,
            self.d,
            live_heads=[live],
        )

        self.assertAlmostEqual(
            hy["output"]["clean_logit"],
            c["output"]["clean_logit"],
            places=6,
        )

        self.assertAlmostEqual(
            hy["output"]["corr_logit"],
            c["output"]["corr_logit"],
            places=6,
        )

    # ========================================================
    # 7. Empty group must equal direct comparator
    # ========================================================

    def test_empty_live_group_is_direct_comparator(self):
        """No-D/no-P comparator is exactly the fully frozen hybrid."""

        e = self.e
        c, r = self.captures()

        source = 1
        pos = [1, 2]

        direct = e.local_hybrid(
            self.ids,
            source,
            pos,
            r,
            c,
            self.heads,
            self.g,
            self.d,
            live_branches=[],
        )

        chain_zero = e.chain_hybrid(
            self.ids,
            source,
            pos,
            r,
            c,
            self.heads,
            self.g,
            self.d,
            live_heads=[],
        )

        self.assertAlmostEqual(
            direct["output"]["margin"],
            chain_zero["output"]["margin"],
            places=6,
        )

        for h in self.heads:
            for channel in "QKV":
                np.testing.assert_allclose(
                    direct["qkv"][h][channel].numpy(),
                    chain_zero["qkv"][h][channel].numpy(),
                    atol=0,
                    rtol=0,
                )

    # ========================================================
    # 8. Hooks must always be cleaned up
    # ========================================================

    def test_hook_cleanup_after_exception(self):
        """No selective-head hooks may leak after an interrupted run."""

        e = self.e
        c, _ = self.captures()

        with self.assertRaises(RuntimeError):
            with e.selective_live_heads(
                c,
                [5],
            ):
                raise RuntimeError(
                    "deliberate failure"
                )

        out = e.output(
            self.ids,
            self.g,
            self.d,
        )

        self.assertTrue(
            np.isfinite(out["margin"])
        )


if __name__ == "__main__":
    unittest.main()
