"""
Transformer components for JAX predictive coding networks.

Decomposed transformer pipeline — each stage is a separate PC node:

  tokens → Embedding → MhaResidual(+) → LnMlp1 → Mlp2Residual(+) → VocabProjection → logits
                        │ (skip)   ↑              │ (skip)    ↑
                        └──────────┘              └───────────┘
"""

from typing import Dict, Any, Tuple, Optional
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.initializers import (
    InitializerBase,
    initialize,
    NormalInitializer,
    OnesInitializer,
    XavierInitializer,
    KaimingInitializer,
)
from fabricpc.graph_initialization import FeedforwardStateInit
from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb
from fabricpc.utils.helpers import layernorm

# Builder Imports
from fabricpc.nodes.linear import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.activations import (
    GeluActivation,
    IdentityActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceBase
from fabricpc.core.mupc import MuPCConfig


def _non_default_options(**options: Tuple[Any, Any]) -> Dict[str, Any]:
    """Return the ``name: (value, default)`` options whose value is not the default.

    Default options stay out of the node config, so configs and the training
    identities built from them are unchanged for existing graphs.
    """
    return {
        name: value for name, (value, default) in options.items() if value != default
    }


def _apply_dropout_mask(
    x: jnp.ndarray,
    inputs: Dict[str, jnp.ndarray],
    slot_suffix: str = ":dropout_mask",
) -> jnp.ndarray:
    """Apply an externally clamped inverted-dropout mask if present.

    Dropout masks are represented as source nodes so the same mask is held
    fixed throughout a PC inference solve. Training helpers clamp Bernoulli /
    keep_prob masks, while evaluation/generation clamps ones.
    """
    mask_key = next((k for k in inputs if k.endswith(slot_suffix)), None)
    if mask_key is None:
        return x
    return x * inputs[mask_key]


# ==============================================================================
# EMBEDDING NODE
# ==============================================================================


class EmbeddingNode(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        vocab_size,
        embed_dim,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            weight_init=weight_init or NormalInitializer(std=0.02),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or GaussianEnergy(),
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        # Discrete token indices, not a continuous signal — keep muPC scaling off.
        return {
            "in": SlotSpec(name="in", is_multi_input=False, is_variance_scalable=False)
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Dict[str, Any],
    ) -> NodeParams:
        vocab_size = config["vocab_size"]
        embed_dim = config["embed_dim"]

        weights = {"embeddings": initialize(key, (vocab_size, embed_dim), weight_init)}
        return NodeParams(weights=weights, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        edge_key = list(inputs.keys())[0]
        indices = inputs[edge_key]

        if indices.ndim == 3 and indices.shape[-1] == 1:
            indices = jnp.squeeze(indices, axis=-1)

        # Token indices must reach this node as an integer dtype.
        # Source-node clamps now propagate dtype through state init, so callers
        # should clamp with int (e.g. jnp.int32), not float.
        z_mu = params.weights["embeddings"][indices]

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)

        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state

    @staticmethod
    def forward_and_latent_grads(params, inputs, state, node_info, is_clamped=False):
        _, new_state = node_info.node_class.forward(params, inputs, state, node_info)
        # Discrete indices: no gradient flows back through the input edge.
        input_grads = {
            edge_key: jnp.zeros_like(inp) for edge_key, inp in inputs.items()
        }
        # Self-grad anchors z_latent to the lookup z_mu. Computed explicitly
        # because this override skips the base autodiff path.
        energy_obj = node_info.energy
        self_grad = type(energy_obj).grad_latent(
            new_state.z_latent, new_state.z_mu, energy_obj.config
        )
        return new_state, input_grads, self_grad


# ==============================================================================
# MULTI-HEAD ATTENTION + RESIDUAL NODE
# ==============================================================================


class MhaResidualNode(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        num_heads,
        use_rope=True,
        rope_theta=10000.0,
        is_causal=True,
        dropout_rate=0.0,
        norm_position="pre",
        ln_eps=1e-5,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        """Attention block with a residual add.

        ``norm_position="pre"`` computes ``skip + Attn(LN(x))``; ``"post"``
        computes ``LN(skip + Attn(x))`` as in the original BERT. ``ln_eps`` is
        the LayerNorm epsilon.
        """
        if norm_position not in ("pre", "post"):
            raise ValueError(
                f"norm_position must be 'pre' or 'post', got {norm_position!r}"
            )
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            is_causal=is_causal,
            dropout_rate=dropout_rate,
            weight_init=weight_init or XavierInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or GaussianEnergy(),
            **_non_default_options(
                norm_position=(norm_position, "pre"), ln_eps=(ln_eps, 1e-5)
            ),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", False),
            "skip": SlotSpec(
                "skip", False, is_variance_scalable=False, is_skip_connection=True
            ),
            "dropout_mask": SlotSpec("dropout_mask", False, is_variance_scalable=False),
            "attention_mask": SlotSpec(
                "attention_mask", False, is_variance_scalable=False
            ),
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        dim = config["embed_dim"]
        keys = jax.random.split(key, 6)

        def init_w(k, s):
            return initialize(k, s, weight_init)

        weights = {
            "ln_gamma": jnp.ones((dim,)),
            "W_q": init_w(keys[0], (dim, dim)),
            "W_k": init_w(keys[1], (dim, dim)),
            "W_v": init_w(keys[2], (dim, dim)),
            "W_o": init_w(keys[3], (dim, dim)),
        }
        biases = {
            "ln_beta": jnp.zeros((dim,)),
            "b_q": jnp.zeros((dim,)),
            "b_k": jnp.zeros((dim,)),
            "b_v": jnp.zeros((dim,)),
            "b_o": jnp.zeros((dim,)),
        }
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[next(k for k in inputs if k.endswith(":in"))]
        skip_key = next((k for k in inputs if k.endswith(":skip")), None)
        skip = inputs[skip_key] if skip_key else x

        cfg = node_info.node_config
        B, L, D = x.shape
        num_heads = cfg["num_heads"]
        head_dim = D // num_heads
        post_ln = cfg.get("norm_position", "pre") == "post"
        ln_eps = cfg.get("ln_eps", 1e-5)

        if post_ln:
            x_norm = x
        else:
            x_norm = layernorm(
                x, params.weights["ln_gamma"], params.biases["ln_beta"], ln_eps
            )

        def proj(h, w_name, b_name):
            return jnp.dot(h, params.weights[w_name]) + params.biases[b_name]

        Q = proj(x_norm, "W_q", "b_q").reshape(B, L, num_heads, head_dim)
        K = proj(x_norm, "W_k", "b_k").reshape(B, L, num_heads, head_dim)
        V = proj(x_norm, "W_v", "b_v").reshape(B, L, num_heads, head_dim)

        if cfg.get("use_rope"):
            freqs_cis = precompute_freqs_cis(
                head_dim, L, theta=cfg.get("rope_theta", 10000.0)
            )
            Q, K = apply_rotary_emb(Q, K, freqs_cis)

        Q, K, V = (
            Q.transpose(0, 2, 1, 3),
            K.transpose(0, 2, 1, 3),
            V.transpose(0, 2, 1, 3),
        )
        scores = jnp.matmul(Q, K.swapaxes(-1, -2)) / jnp.sqrt(head_dim)

        if cfg.get("is_causal", True):
            causal_mask = jnp.tril(jnp.ones((L, L)))
            scores = jnp.where(causal_mask == 0, -1e9, scores)

        mask_key = next((k for k in inputs if k.endswith(":attention_mask")), None)
        if mask_key is not None:
            attention_mask = inputs[mask_key].astype(bool)
            scores = jnp.where(attention_mask[:, None, None, :], scores, -1e9)

        attn = jax.nn.softmax(scores, axis=-1)
        mha = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, L, D)
        mha = proj(mha, "W_o", "b_o")
        mha = _apply_dropout_mask(mha, inputs)
        if mask_key is not None:
            mha = jnp.where(attention_mask[:, :, None], mha, 0.0)

        z_mu = skip + mha
        if post_ln:
            z_mu = layernorm(
                z_mu, params.weights["ln_gamma"], params.biases["ln_beta"], ln_eps
            )
        error = state.z_latent - z_mu

        state = state._replace(z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ==============================================================================
# LAYERNORM + MLP1 NODE
# ==============================================================================


class LnMlp1Node(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = GeluActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        ff_dim,
        pre_ln=True,
        exact_gelu=False,
        ln_eps=1e-5,
        activation=None,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        """First feed-forward layer, ``act(W_ff1 LN(x) + b_ff1)``.

        ``pre_ln=False`` drops the input LayerNorm and its parameters.
        ``exact_gelu=True`` uses the erf GELU in place of the activation object,
        whose derivative then comes from autodiff.
        """
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            activation=activation or GeluActivation(),
            weight_init=weight_init or KaimingInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or GaussianEnergy(),
            **_non_default_options(
                pre_ln=(pre_ln, True),
                exact_gelu=(exact_gelu, False),
                ln_eps=(ln_eps, 1e-5),
            ),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]
        keys = jax.random.split(key, 2)

        weights = {"W_ff1": initialize(keys[0], (embed_dim, ff_dim), weight_init)}
        biases = {"b_ff1": jnp.zeros((ff_dim,))}
        if config.get("pre_ln", True):
            weights = {"ln_gamma": jnp.ones((embed_dim,)), **weights}
            biases = {"ln_beta": jnp.zeros((embed_dim,)), **biases}
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        cfg = node_info.node_config
        x = inputs[list(inputs.keys())[0]]
        if cfg.get("pre_ln", True):
            x = layernorm(
                x,
                params.weights["ln_gamma"],
                params.biases["ln_beta"],
                cfg.get("ln_eps", 1e-5),
            )
        h = jnp.dot(x, params.weights["W_ff1"]) + params.biases["b_ff1"]

        if cfg.get("exact_gelu", False):
            z_mu = jax.nn.gelu(h, approximate=False)
        else:
            act_obj = node_info.activation
            z_mu = type(act_obj).forward(h, act_obj.config)

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ==============================================================================
# MLP2 + RESIDUAL NODE
# ==============================================================================


class Mlp2ResidualNode(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        ff_dim,
        dropout_rate=0.0,
        post_ln=False,
        ln_eps=1e-5,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        """Second feed-forward layer with a residual add.

        ``post_ln=True`` normalises the sum, ``LN(res + W_ff2 in + b_ff2)``, as
        in the original BERT, with LayerNorm epsilon ``ln_eps``.
        """
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            weight_init=weight_init or XavierInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or GaussianEnergy(),
            **_non_default_options(post_ln=(post_ln, False), ln_eps=(ln_eps, 1e-5)),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", False),
            "residual": SlotSpec(
                "residual", False, is_variance_scalable=False, is_skip_connection=True
            ),
            "dropout_mask": SlotSpec("dropout_mask", False, is_variance_scalable=False),
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]
        weights = {"W_ff2": initialize(key, (ff_dim, embed_dim), weight_init)}
        biases = {"b_ff2": jnp.zeros((embed_dim,))}
        if config.get("post_ln", False):
            weights["ln_gamma"] = jnp.ones((embed_dim,))
            biases["ln_beta"] = jnp.zeros((embed_dim,))
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        mlp1_in = next(val for key, val in inputs.items() if key.endswith(":in"))
        res_in = next(val for key, val in inputs.items() if key.endswith(":residual"))

        mlp2 = jnp.dot(mlp1_in, params.weights["W_ff2"]) + params.biases["b_ff2"]
        mlp2 = _apply_dropout_mask(mlp2, inputs)
        z_mu = res_in + mlp2
        cfg = node_info.node_config
        if cfg.get("post_ln", False):
            z_mu = layernorm(
                z_mu,
                params.weights["ln_gamma"],
                params.biases["ln_beta"],
                cfg.get("ln_eps", 1e-5),
            )

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ==============================================================================
# VOCAB PROJECTION NODE
# ==============================================================================


class VocabProjectionNode(NodeBase):
    DEFAULT_ENERGY = CrossEntropyEnergy
    DEFAULT_ACTIVATION = SoftmaxActivation

    def __init__(
        self,
        shape,
        name,
        vocab_size,
        embed_dim,
        activation=None,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            activation=activation or SoftmaxActivation(),
            weight_init=weight_init or XavierInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or CrossEntropyEnergy(),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        vocab, dim = config["vocab_size"], config["embed_dim"]
        weights = {"W_out": initialize(key, (dim, vocab), weight_init)}
        return NodeParams(weights, {"b_out": jnp.zeros((vocab,))})

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[list(inputs.keys())[0]]
        logits = jnp.dot(x, params.weights["W_out"]) + params.biases["b_out"]

        act_obj = node_info.activation
        z_mu = type(act_obj).forward(logits, act_obj.config)

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ==============================================================================
# MODEL BUILDER
# ==============================================================================


def create_deep_transformer(
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    inference: InferenceBase,
    weight_init: Optional[Dict[str, Any]] = None,
    dropout_rate: float = 0.0,
):
    """
    Creates a deep transformer graph using the new class-based builder API.

    Note on initialization: the embedding and output-projection nodes
    deliberately OVERRIDE their class-default initializers. The embedding uses
    unit-normal (std=1.0) instead of the node default std=0.02, and the output
    projection uses Normal(std=sqrt(1/embed_dim)) instead of the node default
    Xavier. Both choices keep activations and logits at O(1) variance given that
    muPC scaling is disabled on these two nodes (embedding = discrete lookup,
    output = include_output=False). Changing these without accounting for the
    muPC interaction can cause embedding variance collapse or softmax saturation.
    """
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

    if weight_init is None:
        # Transformer block weights default to std=0.02 (GPT-style); embedding
        # and output projection set their own init below (see those nodes)
        w_init_obj = NormalInitializer(std=0.02)
    else:
        init_type = weight_init.get("type", "normal")
        if init_type == "normal":
            w_init_obj = NormalInitializer(std=weight_init.get("std", 0.05))
        elif init_type == "xavier":
            w_init_obj = XavierInitializer()
        else:
            w_init_obj = KaimingInitializer()

    nodes = []
    edges = []
    dropout_tasks = {}

    input_node = Linear(
        shape=(seq_len,), activation=IdentityActivation(), name="input_ids"
    )
    nodes.append(input_node)

    # Embedding init: unit-normal (std=1.0), NOT the small std=0.02 used for
    # dense layers. EmbeddingNode is a table lookup with muPC scaling disabled
    # (discrete token indices, not a continuous signal). A Linear+one-hot+muPC
    # path would collapse embedding variance to ~1/vocab_size because muPC
    # assumes dense input with fan_in active features. Unit-normal keeps each
    # token's embedding at O(1) variance going into the first attention block.
    embed_node = EmbeddingNode(
        name="embed",
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=1.0),
    )
    nodes.append(embed_node)
    edges.append(Edge(source=input_node, target=embed_node.slot("in")))

    previous_residual = embed_node

    def add_dropout_mask(name: str, target_node) -> None:
        if dropout_rate <= 0.0:
            return
        mask_node = Linear(
            shape=(seq_len, embed_dim),
            activation=IdentityActivation(),
            name=name,
            use_bias=False,
            latent_init=OnesInitializer(),
        )
        nodes.append(mask_node)
        edges.append(Edge(source=mask_node, target=target_node.slot("dropout_mask")))
        dropout_tasks[name] = mask_node

    for i in range(depth):
        mha = MhaResidualNode(
            name=f"L{i}_mha",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            weight_init=w_init_obj,
        )
        nodes.append(mha)
        edges.append(Edge(source=previous_residual, target=mha.slot("in")))
        edges.append(Edge(source=previous_residual, target=mha.slot("skip")))
        add_dropout_mask(f"dropout_L{i}_mha", mha)

        mlp1 = LnMlp1Node(
            name=f"L{i}_mlp1",
            shape=(seq_len, mlp_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            activation=GeluActivation(),
            weight_init=w_init_obj,
        )
        nodes.append(mlp1)
        edges.append(Edge(source=mha, target=mlp1.slot("in")))

        mlp2 = Mlp2ResidualNode(
            name=f"L{i}_mlp2",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            dropout_rate=dropout_rate,
            weight_init=w_init_obj,
        )
        nodes.append(mlp2)
        edges.append(Edge(source=mlp1, target=mlp2.slot("in")))
        edges.append(Edge(source=mha, target=mlp2.slot("residual")))
        add_dropout_mask(f"dropout_L{i}_mlp2", mlp2)

        previous_residual = mlp2

    # Output projection init: std = sqrt(1/embed_dim). This keeps pre-softmax
    # logits at O(1) variance regardless of model width — each logit is a dot
    # product over embed_dim features, so scaling by 1/sqrt(embed_dim) prevents
    # logit magnitudes from growing with width. Large initial logits would
    # saturate the softmax and produce near-zero CE gradients at the start of
    # training. muPC is disabled on the output (include_output=False), so this
    # explicit init is what controls output-layer variance.
    logits = VocabProjectionNode(
        name="logits",
        shape=(seq_len, vocab_size),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=float(jnp.sqrt(1.0 / embed_dim))),
    )
    nodes.append(logits)
    edges.append(Edge(source=previous_residual, target=logits.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=logits, **dropout_tasks),
        inference=inference,
        scaling=MuPCConfig(include_output=False),
        graph_state_initializer=FeedforwardStateInit(),
    )
    return structure._replace(
        config={**structure.config, "dropout_rate": float(dropout_rate)}
    )
