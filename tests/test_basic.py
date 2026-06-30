import numpy as np
import scipy.sparse as sp
import pytest

from pyoomph_mac_accelerate import SparseLU


def test_unsymmetric_square_solve():
    A = sp.csr_matrix(np.array([
        [4.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [2.0, 0.0, 5.0],
    ]))
    b = np.array([1.0, 2.0, 3.0])

    solver = SparseLU()
    solver.factorize(A)
    x = solver.solve(b)

    assert np.allclose(A @ x, b, atol=1e-10)


def test_resolve_reuses_factorization():
    rng = np.random.default_rng(0)
    n = 20
    A = sp.random(n, n, density=0.3, random_state=rng, format="csr")
    A = A + sp.eye(n) * n  # diagonally dominant -> nonsingular

    solver = SparseLU()
    solver.factorize(A)

    for _ in range(3):
        b = rng.standard_normal(n)
        x = solver.resolve(b)
        assert np.allclose(A @ x, b, atol=1e-8)


def test_multiple_rhs():
    n = 10
    rng = np.random.default_rng(1)
    A = sp.random(n, n, density=0.4, random_state=rng, format="csr") + sp.eye(n) * n

    B = rng.standard_normal((n, 4))
    solver = SparseLU().factorize(A)
    X = solver.solve(B)

    assert np.allclose(A @ X, B, atol=1e-8)


def test_solve_before_factorize_raises():
    solver = SparseLU()
    with pytest.raises(RuntimeError):
        solver.solve(np.zeros(3))
