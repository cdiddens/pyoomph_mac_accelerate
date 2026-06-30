"""
pyoomph_mac_accelerate
===========

pybind11 wrapper around Apple's Accelerate "Sparse Solvers" framework,
exposing a factorize / solve / resolve workflow for real (generally
unsymmetric) sparse matrices, driven from scipy.sparse CSR matrices.

Note on the underlying algorithm: Accelerate does not expose a documented
general sparse LU for unsymmetric matrices. This package uses Accelerate's
sparse QR factorization (``SparseFactorizationQR``), which supports any
matrix shape, including square unsymmetric systems -- for a nonsingular
square system the least-squares QR solve reduces to the exact solution.

Example
-------
>>> import numpy as np
>>> import scipy.sparse as sp
>>> from pyoomph_mac_accelerate import SparseLU
>>>
>>> A = sp.csr_matrix(np.array([[4.0, 0.0, 1.0],
...                              [0.0, 3.0, 0.0],
...                              [2.0, 0.0, 5.0]]))
>>> b = np.array([1.0, 2.0, 3.0])
>>>
>>> solver = SparseLU()
>>> solver.factorize(A)
>>> x = solver.solve(b)
>>> np.allclose(A @ x, b)
True
>>> x2 = solver.resolve(b * 2)   # reuse the factorization for a new rhs
"""

from __future__ import annotations

import numpy as np

try:
    import scipy.sparse as _sp
except ImportError:  # pragma: no cover
    _sp = None

from ._pyoomph_mac_accelerate import SparseSolver as _SparseSolver

__all__ = ["SparseLU"]
__version__ = "0.1.0"


class SparseLU:
    """
    High-level factorize/solve/resolve interface for real sparse matrices
    on top of Apple Accelerate's Sparse Solvers (sparse QR under the hood;
    see module docstring for why QR is used instead of LU).

    Parameters
    ----------
    None at construction time; call :meth:`factorize` with a CSR matrix.
    """

    def __init__(self) -> None:
        self._solver = _SparseSolver()
        self._shape = None

    def factorize(self, A) -> "SparseLU":
        """
        Factorize a sparse matrix ``A``.

        Parameters
        ----------
        A : scipy.sparse matrix or anything scipy.sparse.csr_matrix(A) accepts
            The matrix to factorize. Will be converted to CSR (and to
            float64 / canonical form) if it is not already.

        Returns
        -------
        self, to allow chaining: ``SparseLU().factorize(A).solve(b)``
        """
        if _sp is None:
            raise ImportError("scipy is required to use pyoomph_mac_accelerate.SparseLU")

        A_csr = _sp.csr_matrix(A, dtype=np.float64)
        A_csr.sort_indices()
        A_csr.eliminate_zeros()

        n_rows, n_cols = A_csr.shape
        indptr = A_csr.indptr.astype(np.int64, copy=False)
        indices = A_csr.indices.astype(np.int64, copy=False)
        data = A_csr.data.astype(np.float64, copy=False)

        self._solver.factorize(n_rows, n_cols, indptr, indices, data)
        self._shape = (n_rows, n_cols)
        return self

    def solve(self, b) -> np.ndarray:
        """
        Solve ``A x = b`` for ``x`` using the cached factorization.

        Parameters
        ----------
        b : array_like, shape (n_rows,) or (n_rows, k)
            Right-hand side vector, or a 2D array of column rhs vectors.

        Returns
        -------
        x : np.ndarray, shape (n_cols,) or (n_cols, k)
        """
        self._require_factorized()
        b = np.asarray(b, dtype=np.float64)
        if b.ndim == 1:
            return self._solver.solve(b)
        if b.ndim == 2:
            return np.column_stack([self._solver.solve(b[:, j]) for j in range(b.shape[1])])
        raise ValueError("b must be 1D or 2D")

    def resolve(self, b) -> np.ndarray:
        """
        Re-solve against a new right-hand side, reusing the factorization
        computed by the last call to :meth:`factorize`. Equivalent to
        :meth:`solve`, provided as an explicit name for workflows that
        distinguish the first solve from subsequent re-solves.
        """
        self._require_factorized()
        b = np.asarray(b, dtype=np.float64)
        if b.ndim == 1:
            return self._solver.resolve(b)
        if b.ndim == 2:
            return np.column_stack([self._solver.resolve(b[:, j]) for j in range(b.shape[1])])
        raise ValueError("b must be 1D or 2D")

    @property
    def shape(self):
        return self._shape

    @property
    def is_factorized(self) -> bool:
        return self._solver.is_factorized()

    def _require_factorized(self):
        if not self._solver.is_factorized():
            raise RuntimeError("Call factorize(A) before solve()/resolve()")
