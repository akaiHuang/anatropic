"""
Backend abstraction layer for the Anatropic numerical kernels.

This module exposes a NumPy-like array module ``xp`` and FFT module ``xpfft``
that map either to ``numpy`` (CPU, float64, the historical default) or to
``mlx.core`` (Apple Silicon GPU via Metal, float32). The choice is made once
at import time from the environment variable ``ANATROPIC_BACKEND``:

    ANATROPIC_BACKEND=numpy   (default)
    ANATROPIC_BACKEND=mlx     (Apple Silicon GPU; requires `pip install mlx`)

Design goals
------------
* Backward-compatible: NumPy path is the default and is left untouched, so
  every test that runs on plain NumPy keeps running the same float64 code.
* Backend-agnostic call sites: the hot-path modules (``gravity3d``, ``euler3d``,
  ``simulation3d``, ``eos``) ``import xp, xpfft, ...`` from here and write
  generic code that runs on both backends.
* Honest precision story: MLX restricts GPU work to float32 / complex64. The
  helpers here surface the working dtypes so callers can stay consistent.

What this layer is NOT
----------------------
* It is not a full NumPy shim. Only the operators actually used by the
  Anatropic hot path are exposed. If a new operator is needed, add it here
  rather than sprinkling backend checks throughout the solvers.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

BACKEND = os.environ.get("ANATROPIC_BACKEND", "numpy").lower()
if BACKEND not in ("numpy", "mlx"):
    raise ValueError(
        f"ANATROPIC_BACKEND must be 'numpy' or 'mlx', got {BACKEND!r}"
    )


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

if BACKEND == "numpy":
    import numpy as _np
    import numpy.fft as _npfft

    xp = _np
    xpfft = _npfft

    # Real / complex working dtypes for this backend
    real_dtype = _np.float64
    complex_dtype = _np.complex128

    def to_numpy(a):
        """Return a NumPy view of ``a`` (no-op on the NumPy backend)."""
        return _np.asarray(a)

    def from_numpy(a):
        """Convert a NumPy array into the backend array type (no-op here)."""
        return _np.asarray(a)

    def asarray(a, dtype=None):
        return _np.asarray(a, dtype=dtype)

    def evaluate(*arrays):
        """No-op fence on the NumPy backend (NumPy is eager)."""
        return None

    def empty(shape, dtype=None):
        return _np.empty(shape, dtype=dtype if dtype is not None else real_dtype)

    def zeros(shape, dtype=None):
        return _np.zeros(shape, dtype=dtype if dtype is not None else real_dtype)

    def full(shape, value, dtype=None):
        return _np.full(shape, value, dtype=dtype if dtype is not None else real_dtype)

    def full_like(a, value):
        return _np.full_like(a, value)

    def fftfreq(n, d=1.0):
        return _np.fft.fftfreq(n, d=d)

    def to_real(a):
        """Real part as the backend's real dtype."""
        return _np.real(a).astype(real_dtype, copy=False)

    def stack(arrays, axis=0):
        return _np.stack(arrays, axis=axis)


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

else:  # BACKEND == "mlx"
    import numpy as _np
    import mlx.core as _mx
    import mlx.core.fft as _mxfft

    xp = _mx
    xpfft = _mxfft

    # MLX GPU only supports float32 / complex64. Stick to those everywhere
    # so we never silently promote and fall off the GPU.
    real_dtype = _mx.float32
    complex_dtype = _mx.complex64

    def to_numpy(a):
        """Return a NumPy ``ndarray`` copy of ``a``.

        MLX arrays are zero-copy convertible via ``np.asarray``; we trigger
        eval first so the host sees the materialised values.
        """
        if isinstance(a, _mx.array):
            _mx.eval(a)
            return _np.asarray(a)
        return _np.asarray(a)

    def from_numpy(a):
        """Convert a NumPy array into an ``mlx.core.array`` (real_dtype)."""
        if isinstance(a, _mx.array):
            return a
        arr = _np.asarray(a)
        if arr.dtype.kind == "c":
            return _mx.array(arr.astype(_np.complex64))
        if arr.dtype.kind == "f":
            return _mx.array(arr.astype(_np.float32))
        return _mx.array(arr)

    def asarray(a, dtype=None):
        if isinstance(a, _mx.array):
            return a if dtype is None else a.astype(dtype)
        arr = _np.asarray(a)
        if dtype is None:
            if arr.dtype.kind == "c":
                arr = arr.astype(_np.complex64)
            elif arr.dtype.kind == "f":
                arr = arr.astype(_np.float32)
            return _mx.array(arr)
        return _mx.array(arr).astype(dtype)

    def evaluate(*arrays):
        """Force MLX to materialise the given arrays (graph fence point)."""
        if arrays:
            _mx.eval(*arrays)

    def empty(shape, dtype=None):
        # MLX has no `empty`; zero-init is cheap and keeps semantics safe
        # against accidental NaN propagation.
        return _mx.zeros(shape, dtype=dtype if dtype is not None else real_dtype)

    def zeros(shape, dtype=None):
        return _mx.zeros(shape, dtype=dtype if dtype is not None else real_dtype)

    def full(shape, value, dtype=None):
        return _mx.full(shape, value, dtype=dtype if dtype is not None else real_dtype)

    def full_like(a, value):
        # MLX has no `full_like`; build via shape + dtype.
        return _mx.full(a.shape, value, dtype=a.dtype)

    def fftfreq(n, d=1.0):
        """NumPy-compatible fftfreq for MLX (returns a real_dtype array).

        np.fft.fftfreq(n, d) = [0, 1, ..., n/2-1, -n/2, ..., -1] / (n * d).
        """
        idx = _mx.arange(n, dtype=_mx.int32)
        half = (n + 1) // 2
        shifted = _mx.where(idx < half, idx, idx - n)
        return shifted.astype(real_dtype) / (n * d)

    def to_real(a):
        """Real part of a (possibly complex) MLX array, as real_dtype."""
        if a.dtype == complex_dtype:
            return _mx.real(a).astype(real_dtype)
        return a.astype(real_dtype)

    def stack(arrays, axis=0):
        return _mx.stack(list(arrays), axis=axis)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "BACKEND",
    "xp",
    "xpfft",
    "real_dtype",
    "complex_dtype",
    "asarray",
    "to_numpy",
    "from_numpy",
    "evaluate",
    "empty",
    "zeros",
    "full",
    "full_like",
    "fftfreq",
    "to_real",
    "stack",
]
