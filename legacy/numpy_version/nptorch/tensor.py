"""
nptorch.tensor
==============
A minimal reverse-mode autodiff engine over NumPy arrays.

This plays the role that torch.Tensor + autograd plays in PyTorch, but is
implemented from scratch: every Tensor remembers which Tensors produced it
and how, so calling .backward() on a scalar loss walks that graph in
reverse topological order and accumulates gradients everywhere.

Design notes:
- Values are always plain numpy.ndarray (dtype float64 by default).
- Broadcasting is supported: if an op broadcasts shapes together, the
  backward pass sums the incoming gradient back down to the original
  (pre-broadcast) shape via `_unbroadcast`.
- Only the ops actually needed by the HRDF-MER model are implemented.
  Add more as needed by following the existing patterns.
"""
from __future__ import annotations
import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum-reduce `grad` down to `shape`, undoing whatever broadcasting
    happened during the forward op."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---- basic properties -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, op={self._op or 'leaf'})"

    def zero_grad(self):
        self.grad = None

    # ---- graph plumbing -----------------------------------------------------
    def _accumulate(self, g):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + g

    def backward(self, grad=None):
        """Standard reverse-mode backprop. Call on a scalar (or pass an
        explicit upstream gradient matching self.shape).

        Uses an ITERATIVE (stack-based) post-order traversal to build the
        topological order, rather than recursive DFS - a recursive version
        hits Python's default recursion limit (1000) on real models, since
        e.g. the MFCC LSTM alone unrolls one graph node per timestep, and
        that stacks on top of HCFR/CMDF's own depth."""
        if grad is None:
            assert self.data.size == 1, "backward() with no args requires a scalar tensor"
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=np.float64)

        topo, visited = [], set()
        # each stack entry is (tensor, iterator over its not-yet-visited parents)
        stack = [(self, iter(self._prev))]
        visited.add(id(self))

        while stack:
            node, children_iter = stack[-1]
            advanced = False
            for child in children_iter:
                if id(child) not in visited:
                    visited.add(id(child))
                    stack.append((child, iter(child._prev)))
                    advanced = True
                    break
            if not advanced:
                topo.append(node)
                stack.pop()

        self._accumulate(grad)
        for t in reversed(topo):
            t._backward()

    # ---- elementwise arithmetic --------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "add")

        def _backward():
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad, other.shape))
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self._accumulate(-out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "mul")

        def _backward():
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad * self.data, other.shape))
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def pow(self, p: float):
        out = Tensor(self.data ** p, self.requires_grad, (self,), f"pow{p}")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (p * self.data ** (p - 1)))
        out._backward = _backward
        return out

    # ---- matmul -------------------------------------------------------------
    def matmul(self, other: "Tensor"):
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
                self._accumulate(_unbroadcast(grad_self, self.shape))
            if other.requires_grad:
                grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
                other._accumulate(_unbroadcast(grad_other, other.shape))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self, *axes):
        axes = axes if axes else tuple(reversed(range(self.ndim)))
        out = Tensor(np.transpose(self.data, axes), self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                inv = np.argsort(axes)
                self._accumulate(np.transpose(out.grad, inv))
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # ---- reductions -----------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
                self._accumulate(np.broadcast_to(g, self.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in
              ((axis,) if isinstance(axis, int) else axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- shape ops --------------------------------------------------------
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.shape))
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                g = np.zeros_like(self.data)
                g[idx] = out.grad
                self._accumulate(g)
        out._backward = _backward
        return out

    # ---- nonlinearities -----------------------------------------------------
    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * e)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / (self.data + 1e-12))
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * s * (1 - s))
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (1 - t * t))
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                # d L/d x_i = s_i * (dL/ds_i - sum_j dL/ds_j * s_j)
                dot = np.sum(out.grad * s, axis=axis, keepdims=True)
                self._accumulate(s * (out.grad - dot))
        out._backward = _backward
        return out

    def l2_normalize(self, axis=-1, eps=1e-8):
        norm = np.sqrt((self.data ** 2).sum(axis=axis, keepdims=True)) + eps
        out = Tensor(self.data / norm, self.requires_grad, (self,), "l2norm")

        def _backward():
            if self.requires_grad:
                x = self.data
                n = norm
                # d(x/||x||)/dx = I/n - x x^T / n^3
                g = out.grad
                dot = np.sum(g * x, axis=axis, keepdims=True)
                self._accumulate(g / n - x * dot / (n ** 3))
        out._backward = _backward
        return out


def concat(tensors, axis=-1):
    datas = [t.data for t in tensors]
    out = Tensor(np.concatenate(datas, axis=axis), any(t.requires_grad for t in tensors),
                 tuple(tensors), "concat")
    sizes = [d.shape[axis] for d in datas]

    def _backward():
        splits = np.cumsum(sizes)[:-1]
        grads = np.split(out.grad, splits, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t._accumulate(g)
    out._backward = _backward
    return out


def stack(tensors, axis=0):
    return concat([t.reshape(*(t.shape[:axis] + (1,) + t.shape[axis:])) for t in tensors], axis=axis)


def zeros(shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad)


def randn(*shape, scale=0.02, requires_grad=True):
    return Tensor(np.random.randn(*shape) * scale, requires_grad)
