"""
nptorch.nn
==========
Layer building blocks, analogous to torch.nn, all implemented on top of
nptorch.tensor.Tensor. Every layer exposes .parameters() so the optimizer
can find its learnable weights.
"""
import numpy as np
from .tensor import Tensor, concat, zeros, randn


class Module:
    """Base class: tracks sub-modules and parameters, like nn.Module."""
    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """y = xW + b, matching nn.Linear. x: (..., in_dim) -> (..., out_dim)."""
    def __init__(self, in_dim, out_dim, bias=True):
        scale = np.sqrt(2.0 / in_dim)
        self.W = randn(in_dim, out_dim, scale=scale)
        self.b = zeros((out_dim,), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.W)
        if self.b is not None:
            out = out + self.b
        return out


class LayerNorm(Module):
    """Normalizes the last dimension, matching nn.LayerNorm."""
    def __init__(self, dim, eps=1e-5):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = zeros((dim,), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        std = (var + self.eps).pow(0.5)
        normed = centered / std
        return normed * self.gamma + self.beta


class MultiHeadAttention(Module):
    """
    Implements the paper's MHCA (Eqs. 4-9). Used both as:
      - cross-attention (CAE): query from one modality, key/value from another
      - self-attention (SAE): query = key = value, same modality

    Shapes: query (B, Lq, D), key/value (B, Lk, D) -> output (B, Lq, D).
    """
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        x = x.reshape(B, L, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (B, H, L, d_head)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, L, d = x.shape
        x = x.transpose(0, 2, 1, 3)  # (B, L, H, d)
        return x.reshape(B, L, H * d)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        Q = self._split_heads(self.Wq(query))
        K = self._split_heads(self.Wk(key))
        V = self._split_heads(self.Wv(value))

        scale = 1.0 / np.sqrt(self.d_head)
        scores = Q.matmul(K.transpose(0, 1, 3, 2)) * scale   # (B,H,Lq,Lk)
        attn = scores.softmax(axis=-1)
        out = attn.matmul(V)                                  # (B,H,Lq,d_head)
        out = self._merge_heads(out)                          # (B,Lq,D)
        return self.Wo(out)


class FeedForward(Module):
    """Position-wise FFN used inside the HFE Transformer block."""
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


class CrossAttentionEncoder(Module):
    """
    CAE from the paper: X_align = LN(MHCA(Q,K,V) + Q)   (Eq. 9-10)
    """
    def __init__(self, d_model, n_heads):
        self.mhca = MultiHeadAttention(d_model, n_heads)
        self.ln = LayerNorm(d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        attn_out = self.mhca(query, key, value)
        return self.ln(attn_out + query)


class SelfAttentionEncoder(Module):
    """SAE: same as CAE but query=key=value=x (Eq. 14-16)."""
    def __init__(self, d_model, n_heads):
        self.cae = CrossAttentionEncoder(d_model, n_heads)

    def forward(self, x: Tensor) -> Tensor:
        return self.cae(x, x, x)


class TransformerBlock(Module):
    """
    One HFE branch from Fig. 1's left panel: Multi-Head Self-Attention
    (Add&Norm) followed by Multi-Head Cross-Attention (Add&Norm) followed
    by FeedForward (Add&Norm). Used inside the HFE module.
    """
    def __init__(self, d_model, n_heads, d_ff):
        self.self_attn = SelfAttentionEncoder(d_model, n_heads)
        self.cross_attn = CrossAttentionEncoder(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln_ff = LayerNorm(d_model)

    def forward(self, x: Tensor, cross_kv: Tensor) -> Tensor:
        x = self.self_attn(x)
        x = self.cross_attn(x, cross_kv, cross_kv)
        return self.ln_ff(self.ff(x) + x)


class GatedFusion(Module):
    """
    Gating unit from Eqs. 26-29:
      alpha = sigmoid(W_gate [x_cross ; x_aug] + b_gate)
      O_gate = alpha * x_aug + (1-alpha) * x_cross
    Applied twice in the paper (alpha then beta); this module does ONE
    gate application - CMDF composes two of these.
    """
    def __init__(self, d_model):
        self.gate = Linear(2 * d_model, d_model)

    def forward(self, x_primary: Tensor, x_secondary: Tensor) -> Tensor:
        gate_in = concat([x_secondary, x_primary], axis=-1)
        alpha = self.gate(gate_in).sigmoid()
        return alpha * x_primary + (Tensor(1.0) - alpha) * x_secondary


class LSTMCell(Module):
    """A single-layer, single-direction LSTM cell, matching the role of
    nn.LSTM(X_mfcc) in Eq. 3 (kept lightweight: batch loop unrolled in numpy)."""
    def __init__(self, in_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        # combine the 4 gates (i, f, g, o) into one big linear for both x and h
        self.Wx = Linear(in_dim, 4 * hidden_dim)
        self.Wh = Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x_seq: Tensor) -> Tensor:
        """x_seq: (B, L, in_dim) -> returns (B, L, hidden_dim) of hidden states."""
        B, L, _ = x_seq.shape
        H = self.hidden_dim
        h = zeros((B, H))
        c = zeros((B, H))
        outputs = []
        for t in range(L):
            xt = x_seq[:, t, :]
            gates = self.Wx(xt) + self.Wh(h)          # (B, 4H)
            i = gates[:, 0:H].sigmoid()
            f = gates[:, H:2 * H].sigmoid()
            g = gates[:, 2 * H:3 * H].tanh()
            o = gates[:, 3 * H:4 * H].sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            outputs.append(h.reshape(B, 1, H))
        return concat(outputs, axis=1)
