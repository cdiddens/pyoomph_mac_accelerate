# pyoomph_mac_accelerate

**Note: The entire package was vibe-coded with Claude.ai**

Required to use Apple's `Accelerate` linear solver `pyoomph`

## Important note on the algorithm

Accelerate does **not** expose a documented general sparse LU factorization
for unsymmetric matrices in its public API. The supported direct sparse
factorizations are:

- `SparseFactorizationCholesky` / the LDLT family — **symmetric only**
- `SparseFactorizationQR` — works for **any shape**, including square
  unsymmetric systems

This package therefore factorizes using `SparseFactorizationQR`. For a
square, nonsingular, unsymmetric system this gives the exact solution (QR
least-squares reduces to the direct solve in that case); for a rectangular
system it gives the least-squares solution. If your matrices happen to be
symmetric positive definite, Cholesky/LDLT would be faster, but that is a
different code path than what's wired up here (open an issue / ask if you
want that variant added too).

## Requirements

- macOS (this links directly against the system `Accelerate.framework`,
  so it cannot build or run on Linux/Windows)
- Xcode command line tools (`xcode-select --install`)
- Python >= 3.8
- `pybind11`, `numpy`, `scipy`

## Installation

From the package directory:

```bash
pip install .
```

or, for an editable/development install:

```bash
pip install -e .
```

`setup.py` will refuse to build on non-macOS platforms with a clear error.

## Usage within pyoomph (currently only in the development version)

```python

from pyoomph import *
problem=Problem()
problem.set_linear_solver("mac_accelerate")
...
```

## Usage as standalone

```python
import numpy as np
import scipy.sparse as sp
from pyoomph_mac_accelerate import SparseLU

# Build an unsymmetric sparse matrix
A = sp.csr_matrix(np.array([
    [4.0, 0.0, 1.0],
    [0.0, 3.0, 0.0],
    [2.0, 0.0, 5.0],
]))
b = np.array([1.0, 2.0, 3.0])

solver = SparseLU()
solver.factorize(A)

x = solver.solve(b)
print(x, A @ x)          # A @ x ~= b

# Reuse the cached factorization for a new rhs (cheap, no re-factorization)
b2 = np.array([5.0, 1.0, 0.0])
x2 = solver.resolve(b2)

# Multiple right-hand sides at once (2D array, columns are RHS vectors)
B = np.column_stack([b, b2])
X = solver.solve(B)
```

## API

### `pyoomph_mac_accelerate.SparseLU`

- `factorize(A)` — factorize a sparse matrix `A` (any scipy.sparse format;
  converted internally to CSR / float64). Returns `self` for chaining.
- `solve(b)` — solve `A x = b` for a 1D rhs or a 2D array of rhs columns.
- `resolve(b)` — identical to `solve`, provided as an explicit name for
  reusing a previously computed factorization against a new rhs.
- `.shape` — `(n_rows, n_cols)` of the last factorized matrix.
- `.is_factorized` — whether `factorize` has been called successfully.

### Low-level module

`pyoomph_mac_accelerate._pyoomph_mac_accelerate.SparseSolver` is the raw pybind11 binding
(`factorize(n_rows, n_cols, indptr, indices, data)`, `solve(b)`,
`resolve(b)`) if you want to bypass the scipy convenience wrapper, e.g. to
hand it raw CSR arrays directly.

## How it works internally

1. Python passes a scipy CSR matrix's `indptr` / `indices` / `data` arrays
   into the C++ extension.
2. The extension converts CSR (row-major) into Accelerate's native CSC
   (column-major) layout via a counting-sort transpose.
3. A `SparseMatrix_Double` / `SparseMatrixStructure` is built over that CSC
   data and factorized once with `SparseFactor(SparseFactorizationQR, ...)`.
4. `solve`/`resolve` build a `DenseVector_Double` over the rhs (padded to
   `max(n_rows, n_cols)` as Accelerate's solve routine requires) and call
   `SparseSolve`, then copy the leading `n_cols` entries back into a numpy
   array.
5. The factorization (and the CSC backing arrays it points into) are kept
   alive for the lifetime of the `SparseLU`/`SparseSolver` object, and
   released via `SparseCleanup` on destruction.

## License

GPL v3.0
