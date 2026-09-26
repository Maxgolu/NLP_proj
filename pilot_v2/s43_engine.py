"""Stage 4.3 intervention engine.

Extends the existing S4.1/S4.2 machinery with selective intermediate-head
propagation for composed-chain tests.

Scientific semantics
--------------------
Local mediator tests:
    Replace the source head at the selected source positions, freeze all
    downstream normalized residual-branch increments to the recipient, and
    selectively release only the named MLPs / whole blocks.

Composed-chain tests:
    Replace the source head, then allow only a named intermediate head group
    to respond in its attention layer. Every non-live head in that layer is
    clamped to the intact recipient head output before the shared o_proj and
    post-attention normalization. The intermediate MLP branch remains frozen
    unless explicitly released elsewhere.

This preserves the shared OLMo2 attention-output normalization. A "live head"
therefore means a live pre-o_proj head slice inside one jointly normalized
attention branch, not an independently normalized residual contribution.
"""

import contextlib
import collections

from s42_engine import S42Engine


class S43Engine(S42Engine):

    # ------------------------------------------------------------------
    # Recipient / hybrid capture with optional complete head-output banks
    # ------------------------------------------------------------------

    def capture_s43(
        self,
        ids,
        heads,
        g,
        d,
        full_z_layers=(),
        **instrument_kwargs,
    ):
        """Capture normal Stage4 quantities plus full o_proj inputs.

        ``full_z_layers`` identifies attention layers for which all head-output
        slices are needed. The saved tensor for one layer has shape:

            [sequence, hidden_size]

        and is the actual input to that layer's attention o_proj.

        This is required for composed-chain interventions because all
        non-live heads must be restored to their recipient values before the
        shared o_proj and post-attention normalization.
        """

        full_z_layers = sorted(set(int(x) for x in full_z_layers))

        for li in full_z_layers:
            if not 0 <= li < self.L:
                raise ValueError(
                    f"Requested full-z layer outside model: {li}"
                )

        full_z = {}
        handles = []

        def keep_full_z(li):
            def hook(mod, args):
                if li in full_z:
                    raise ValueError(
                        f"Repeated o_proj call in layer {li}"
                    )

                z = args[0]

                if z.ndim != 3 or z.shape[0] != 1:
                    raise ValueError(
                        "S4.3 requires batch-one attention output"
                    )

                if z.shape[-1] != self.H * self.DH:
                    raise ValueError(
                        "Unexpected o_proj input width"
                    )

                full_z[li] = (
                    z[0]
                    .detach()
                    .cpu()
                    .clone()
                )

            return hook

        try:
            for li in full_z_layers:
                handles.append(
                    self.layers[li]
                    .self_attn
                    .o_proj
                    .register_forward_pre_hook(
                        keep_full_z(li)
                    )
                )

            with self.instrument(
                heads=heads,
                collect=True,
                **instrument_kwargs,
            ) as cache:
                result = self.output(
                    ids,
                    g,
                    d,
                )

        finally:
            for handle in handles:
                handle.remove()

        if set(full_z) != set(full_z_layers):
            raise ValueError(
                "Did not capture every requested full-z layer"
            )

        cache["output"] = result
        cache["full_z"] = full_z

        return cache

    # ------------------------------------------------------------------
    # Selective intermediate-head release
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def selective_live_heads(
        self,
        recipient,
        live_heads,
    ):
        """Allow only selected heads to change in intermediate attention.

        At each live layer:

        1. SDPA computes the current head outputs under the upstream source
           intervention.
        2. Immediately before ``o_proj``, every NON-live head slice is
           replaced by its intact-recipient value.
        3. The selected live slices remain current.
        4. ``o_proj`` and the single shared post-attention normalization are
           then evaluated normally.

        Thus releasing {h1,h2} does NOT release all heads in that layer.
        """

        live_heads = sorted(set(int(h) for h in live_heads))

        if not live_heads:
            yield
            return

        for h in live_heads:
            if not 0 <= h < self.L * self.H:
                raise ValueError(
                    f"Live head outside model: {h}"
                )

        by_layer = collections.defaultdict(set)

        for h in live_heads:
            by_layer[h // self.H].add(
                h % self.H
            )

        handles = []
        seen = set()

        for li in by_layer:
            if li not in recipient.get(
                "full_z",
                {},
            ):
                raise ValueError(
                    "Recipient cache lacks full head-output bank "
                    f"for live layer {li}"
                )

        def clamp_hook(li, live_local):
            def hook(mod, args):
                if li in seen:
                    raise ValueError(
                        f"Repeated selective-head hook at layer {li}"
                    )

                seen.add(li)

                z = args[0]

                if z.ndim != 3 or z.shape[0] != 1:
                    raise ValueError(
                        "Selective live-head propagation "
                        "requires batch size one"
                    )

                seq = z.shape[1]

                reference = recipient["full_z"][li]

                if reference.shape != (
                    seq,
                    self.H * self.DH,
                ):
                    raise ValueError(
                        "Recipient/current sequence shape mismatch "
                        f"at layer {li}: "
                        f"{tuple(reference.shape)} vs "
                        f"{tuple(z.shape[1:])}"
                    )

                current = z.reshape(
                    1,
                    seq,
                    self.H,
                    self.DH,
                )

                ref = (
                    reference
                    .to(
                        z.device,
                        z.dtype,
                    )
                    .reshape(
                        1,
                        seq,
                        self.H,
                        self.DH,
                    )
                )

                out = current.clone()

                frozen_local = [
                    hi
                    for hi in range(self.H)
                    if hi not in live_local
                ]

                if frozen_local:
                    out[
                        :,
                        :,
                        frozen_local,
                        :
                    ] = ref[
                        :,
                        :,
                        frozen_local,
                        :
                    ]

                out = out.reshape_as(z)

                return (
                    out,
                    *args[1:],
                )

            return hook

        try:
            for li, live_local in sorted(
                by_layer.items()
            ):
                handles.append(
                    self.layers[li]
                    .self_attn
                    .o_proj
                    .register_forward_pre_hook(
                        clamp_hook(
                            li,
                            live_local,
                        )
                    )
                )

            yield

            if seen != set(by_layer):
                raise ValueError(
                    "Not every selective live-head layer executed"
                )

        finally:
            for handle in handles:
                handle.remove()

    # ------------------------------------------------------------------
    # Local S4.3 hybrid
    # ------------------------------------------------------------------

    def local_hybrid(
        self,
        ids,
        source,
        positions,
        donor,
        recipient,
        heads,
        g,
        d,
        live_branches=(),
    ):
        """Source replacement with only named residual branches released."""

        source_layer = source // self.H

        live_branches = [
            tuple(x)
            for x in live_branches
        ]

        for li, kind in live_branches:
            if kind not in ("attn", "mlp"):
                raise ValueError(
                    f"Unknown residual branch kind: {kind}"
                )

            if not source_layer <= li < self.L:
                raise ValueError(
                    "Live branch precedes source layer"
                )

        with self.vector_patch(
            source,
            positions,
            donor["z"][source][positions],
        ):
            return self.capture_s43(
                ids,
                heads,
                g,
                d,
                freeze=recipient,
                source_layer=source_layer,
                live_branches=live_branches,
            )

    # ------------------------------------------------------------------
    # Composed-chain hybrid
    # ------------------------------------------------------------------

    def chain_hybrid(
        self,
        ids,
        source,
        positions,
        donor,
        recipient,
        heads,
        g,
        d,
        live_heads,
    ):
        """Source replacement with only selected intermediate heads live.

        The attention branch of each intermediate layer must remain live so
        that the selected head output can pass through o_proj + shared
        normalization. All non-live heads in that branch are clamped to the
        intact recipient by ``selective_live_heads``.

        MLPs and all other downstream residual branches remain frozen.
        """

        source_layer = source // self.H

        live_heads = sorted(
            set(int(h) for h in live_heads)
        )

        live_layers = sorted({
            h // self.H
            for h in live_heads
        })

        for li in live_layers:
            if li <= source_layer:
                raise ValueError(
                    "Intermediate live-head layer must be "
                    "strictly downstream of the source"
                )

        # No live group is the matched direct comparator.
        if not live_heads:
            return self.local_hybrid(
                ids=ids,
                source=source,
                positions=positions,
                donor=donor,
                recipient=recipient,
                heads=heads,
                g=g,
                d=d,
                live_branches=(),
            )

        # Releasing the attention branch is necessary, but the o_proj-input
        # clamp below ensures that only the named heads inside it can differ.
        live_branches = [
            (li, "attn")
            for li in live_layers
        ]

        with self.vector_patch(
            source,
            positions,
            donor["z"][source][positions],
        ):
            with self.selective_live_heads(
                recipient,
                live_heads,
            ):
                return self.capture_s43(
                    ids,
                    heads,
                    g,
                    d,
                    full_z_layers=live_layers,
                    freeze=recipient,
                    source_layer=source_layer,
                    live_branches=live_branches,
                )

    # ------------------------------------------------------------------
    # Registry dispatch
    # ------------------------------------------------------------------

    def hybrid_for_config(
        self,
        ids,
        config,
        positions,
        donor,
        recipient,
        heads,
        g,
        d,
    ):
        """Construct the source-replaced hybrid specified by one S4.3 row."""

        source = int(
            config["source_id"]
        )

        section = config[
            "registry_section"
        ]

        if section in (
            "chain",
            "direct_comparator_chain",
        ):
            return self.chain_hybrid(
                ids=ids,
                source=source,
                positions=positions,
                donor=donor,
                recipient=recipient,
                heads=heads,
                g=g,
                d=d,
                live_heads=[
                    int(h)
                    for h in config.get(
                        "live_head_ids",
                        [],
                    )
                ],
            )

        return self.local_hybrid(
            ids=ids,
            source=source,
            positions=positions,
            donor=donor,
            recipient=recipient,
            heads=heads,
            g=g,
            d=d,
            live_branches=config.get(
                "live_branches",
                [],
            ),
        )

    def endpoint_for_config(
        self,
        ids,
        config,
        positions,
        hybrid,
        g,
        d,
        n,
    ):
        """Fresh-recipient endpoint for an S4.3 registry row."""

        if config["channel"] == "bypass":
            return hybrid["output"]

        endpoint_config = {
            "kind": "head",
            "receiver": int(
                config["receiver_id"]
            ),
            "channel": config["channel"],
        }

        return self.endpoint(
            ids,
            endpoint_config,
            positions,
            hybrid,
            g,
            d,
            n,
        )

    def channel_norm_for_config(
        self,
        config,
        positions,
        hybrid,
        recipient,
        n,
    ):
        """Norm of the propagated endpoint channel relative to recipient."""

        if config["channel"] == "bypass":
            return None

        endpoint_config = {
            "kind": "head",
            "receiver": int(
                config["receiver_id"]
            ),
            "channel": config["channel"],
        }

        return self.channel_norm(
            endpoint_config,
            positions,
            hybrid,
            recipient,
            n,
        )
