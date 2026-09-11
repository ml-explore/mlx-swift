# Copyright © 2026 Apple Inc.

"""
A tiny numpy-backed stand-in for `mlx.core`, used only by
`generate.py --self-test`.

It exists so the *generator* can be exercised (case tables, python evaluation,
summaries, Swift emission) on a machine with no GPU / no mlx.  The values it
produces are **not** mlx values, so its output must never be checked in.

Unsupported ops raise `Unsupported`; `--self-test` reports and skips them.
"""

from __future__ import annotations

import numpy as np


class Unsupported(Exception):
    pass


class FakeDType:
    def __init__(self, name: str, np_type):
        self.name = name
        self.np = np_type

    def __repr__(self) -> str:
        return f"mlx.core.{self.name}"


DTYPES = {
    "bool_": FakeDType("bool_", np.bool_),
    "uint8": FakeDType("uint8", np.uint8),
    "uint16": FakeDType("uint16", np.uint16),
    "uint32": FakeDType("uint32", np.uint32),
    "uint64": FakeDType("uint64", np.uint64),
    "int8": FakeDType("int8", np.int8),
    "int16": FakeDType("int16", np.int16),
    "int32": FakeDType("int32", np.int32),
    "int64": FakeDType("int64", np.int64),
    "float16": FakeDType("float16", np.float16),
    "float32": FakeDType("float32", np.float32),
    # numpy has no bfloat16; float16 is close enough for a plumbing test
    "bfloat16": FakeDType("bfloat16", np.float16),
    "float64": FakeDType("float64", np.float64),
    "complex64": FakeDType("complex64", np.complex64),
}

_NP_TO_NAME = {
    np.dtype(np.bool_): "bool_",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.uint64): "uint64",
    np.dtype(np.int8): "int8",
    np.dtype(np.int16): "int16",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
    np.dtype(np.float16): "float16",
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
    np.dtype(np.complex64): "complex64",
    np.dtype(np.complex128): "complex64",
}


def _unwrap(value):
    if isinstance(value, FakeArray):
        return value.value
    if isinstance(value, FakeDType):
        return value.np
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(v) for v in value)
    return value


def _wrap(value):
    if isinstance(value, np.ndarray):
        return FakeArray(value)
    if isinstance(value, (np.generic, int, float, bool)):
        return FakeArray(np.asarray(value))
    return value


class FakeArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    # ---- properties the generator uses
    @property
    def shape(self):
        return tuple(self.value.shape)

    @property
    def size(self) -> int:
        return int(self.value.size)

    @property
    def dtype(self) -> FakeDType:
        return DTYPES[_NP_TO_NAME.get(self.value.dtype, "float32")]

    @property
    def real(self):
        return FakeArray(np.real(self.value))

    @property
    def imag(self):
        return FakeArray(np.imag(self.value))

    @property
    def T(self):
        return FakeArray(self.value.T)

    def item(self):
        return self.value.reshape(-1)[0].item() if self.value.size else float("nan")

    def astype(self, dtype):
        return FakeArray(self.value.astype(_unwrap(dtype)))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return FakeArray(self.value.reshape(shape))

    # ---- method spellings used by cases
    def _reduce(self, name, axis=None, keepdims=False, **kwargs):
        fn = getattr(np, name)
        return _wrap(fn(self.value, axis=axis, keepdims=keepdims, **kwargs))

    def sum(self, axis=None, keepdims=False):
        return self._reduce("sum", axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return self._reduce("mean", axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return self._reduce("min", axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return self._reduce("max", axis, keepdims)

    def prod(self, axis=None, keepdims=False):
        return self._reduce("prod", axis, keepdims)

    def abs(self):
        return _wrap(np.abs(self.value))

    def exp(self):
        return _wrap(np.exp(self.value))

    def sqrt(self):
        return _wrap(np.sqrt(self.value))

    def round(self, decimals=0):
        return _wrap(np.round(self.value, decimals))

    def logsumexp(self, axis=None, keepdims=False):
        raise Unsupported("logsumexp")

    def __getitem__(self, key):
        return _wrap(self.value[_unwrap(key)])

    # ---- operators
    def _binary(self, other, op):
        return _wrap(op(self.value, _unwrap(other)))

    def __add__(self, o):
        return self._binary(o, lambda a, b: a + b)

    def __radd__(self, o):
        return _wrap(_unwrap(o) + self.value)

    def __sub__(self, o):
        return self._binary(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return _wrap(_unwrap(o) - self.value)

    def __mul__(self, o):
        return self._binary(o, lambda a, b: a * b)

    def __rmul__(self, o):
        return _wrap(_unwrap(o) * self.value)

    def __truediv__(self, o):
        return self._binary(o, lambda a, b: a / b)

    def __rtruediv__(self, o):
        return _wrap(_unwrap(o) / self.value)

    def __mod__(self, o):
        return self._binary(o, lambda a, b: a % b)

    def __rmod__(self, o):
        return _wrap(_unwrap(o) % self.value)

    def __pow__(self, o):
        return self._binary(o, lambda a, b: a**b)

    def __neg__(self):
        return _wrap(-self.value)

    def __eq__(self, o):  # type: ignore[override]
        return self._binary(o, lambda a, b: a == b)

    def __ne__(self, o):  # type: ignore[override]
        return self._binary(o, lambda a, b: a != b)

    def __lt__(self, o):
        return self._binary(o, lambda a, b: a < b)

    def __le__(self, o):
        return self._binary(o, lambda a, b: a <= b)

    def __gt__(self, o):
        return self._binary(o, lambda a, b: a > b)

    def __ge__(self, o):
        return self._binary(o, lambda a, b: a >= b)


class _Submodule:
    """maps `mx.fft.foo` / `mx.linalg.foo` onto numpy"""

    def __init__(self, module, unsupported=()):
        self.module = module
        self.unsupported = set(unsupported)

    def __getattr__(self, name):
        if name in self.unsupported:
            raise Unsupported(name)
        fn = getattr(self.module, name, None)
        if fn is None:
            raise Unsupported(name)

        def wrapper(*args, **kwargs):
            args = tuple(_unwrap(a) for a in args)
            kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
            try:
                return _wrap(fn(*args, **kwargs))
            except TypeError as e:
                raise Unsupported(f"{name}: {e}") from e

        return wrapper


class _Random:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def normal(self, shape=(), dtype=None, loc=0.0, scale=1.0):
        value = self.rng.normal(loc, scale, tuple(shape))
        return FakeArray(value.astype(_unwrap(dtype) or np.float32))

    def uniform(self, low=0.0, high=1.0, shape=(), dtype=None):
        value = self.rng.uniform(low, high, tuple(shape))
        return FakeArray(value.astype(_unwrap(dtype) or np.float32))

    def randint(self, low, high, shape=(), dtype=None):
        value = self.rng.integers(low, high, tuple(shape))
        return FakeArray(value.astype(_unwrap(dtype) or np.int32))

    def bernoulli(self, p=0.5, shape=None):
        value = self.rng.uniform(0, 1, tuple(shape or ())) < p
        return FakeArray(value)

    def gumbel(self, shape=(), dtype=None, **kw):
        value = self.rng.gumbel(0, 1, tuple(shape))
        return FakeArray(value.astype(_unwrap(dtype) or np.float32))

    def laplace(self, shape=(), dtype=None, loc=0.0, scale=1.0, **kw):
        value = self.rng.laplace(loc, scale, tuple(shape))
        return FakeArray(value.astype(_unwrap(dtype) or np.float32))

    def truncated_normal(self, lower, upper, shape=None, dtype=None, **kw):
        value = np.clip(self.rng.normal(0, 1, tuple(shape or ())), lower, upper)
        return FakeArray(value.astype(_unwrap(dtype) or np.float32))

    def categorical(self, logits, axis=-1, shape=None, num_samples=None, **kw):
        count = num_samples or 1
        picks = self.rng.integers(0, _unwrap(logits).shape[axis], (len(_unwrap(logits)), count))
        picks = picks if num_samples else picks[:, 0]
        return FakeArray(picks.astype(np.uint32))

    def permutation(self, x, axis=0, **kw):
        if isinstance(x, int):
            return FakeArray(self.rng.permutation(x).astype(np.int32))
        return FakeArray(self.rng.permutation(_unwrap(x), axis=axis))

    def multivariate_normal(self, mean, cov, shape=(), dtype=None, **kw):
        value = self.rng.multivariate_normal(_unwrap(mean), _unwrap(cov), tuple(shape))
        return FakeArray(value.astype(_unwrap(dtype) or np.float32))

    def key(self, seed):
        return FakeArray(np.array([0, seed], dtype=np.uint32))


# mlx name -> numpy name, where they differ
RENAMES = {
    "arctan2": "arctan2",
    "power": "power",
    "logical_not": "logical_not",
    "bitwise_invert": "invert",
    "flip": "flip",
    "repeat": "repeat",
    "pad": "pad",
    "take_along_axis": "take_along_axis",
    "sort": "sort",
    "flatten": "reshape",  # handled specially below
    "var": "var",
    "median": "median",
    "where": "where",
    "concatenate": "concatenate",
    "expand_dims": "expand_dims",
    "broadcast_to": "broadcast_to",
    "atleast_2d": "atleast_2d",
    "stop_gradient": None,  # identity
}

UNSUPPORTED = {
    "rsqrt",
    "erf",
    "erfinv",
    "sigmoid",
    "logsumexp",
    "softmax",
    "logcumsumexp",
    "cummax",
    "cummin",
    "topk",
    "addmm",
    "unflatten",
    "std",
    "degrees",
    "radians",
    "expm1",
    "vecdot",
}


class _Module:
    is_fake = True
    fft = _Submodule(np.fft)
    linalg = _Submodule(
        np.linalg, unsupported=("tri_inv", "cholesky_inv", "solve_triangular")
    )
    __version__ = "fake-numpy"
    random = _Random()
    inf = float("inf")
    nan = float("nan")

    def __getattr__(self, name: str):
        if name in DTYPES:
            return DTYPES[name]
        if name in UNSUPPORTED:
            raise Unsupported(name)
        if name == "stop_gradient":
            return lambda a, **kw: a
        if name == "array":
            return lambda values, dtype=None: FakeArray(
                np.asarray(_unwrap(values), dtype=_unwrap(dtype))
            )
        if name == "flatten":

            def flatten(a, start=0, end=-1, **kw):
                shape = list(a.shape)
                end = end if end >= 0 else len(shape) + end
                merged = int(np.prod(shape[start : end + 1]))
                return FakeArray(a.value.reshape(*shape[:start], merged, *shape[end + 1 :]))

            return flatten
        if name == "arange":

            def arange(*args, dtype=None, **kw):
                return FakeArray(np.arange(*args, dtype=_unwrap(dtype)))

            return arange

        np_name = RENAMES.get(name, name)
        fn = getattr(np, np_name, None)
        if fn is None:
            raise Unsupported(name)

        def wrapper(*args, **kwargs):
            args = tuple(_unwrap(a) for a in args)
            kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
            if "keepdims" in kwargs and np_name in ("argmin", "argmax"):
                kwargs.pop("keepdims")
            try:
                return _wrap(fn(*args, **kwargs))
            except TypeError as e:
                raise Unsupported(f"{name}: {e}") from e

        return wrapper


mx = _Module()
