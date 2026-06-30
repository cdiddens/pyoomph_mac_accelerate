// bindings.cpp
//
// pybind11 wrapper around Apple's Accelerate "Sparse Solvers" C API.
//
// Apple's Accelerate framework does not expose a documented general
// sparse LU factorization for unsymmetric matrices. The supported direct
// factorizations are:
//   - SparseFactorizationCholesky / LDLT family  (symmetric only)
//   - SparseFactorizationQR                      (any shape, incl. square
//                                                  unsymmetric; reduces to
//                                                  the exact solve for a
//                                                  nonsingular square system)
//
// This wrapper therefore uses SparseFactorizationQR as the workhorse for
// "unsymmetric" real sparse matrices, exposed to Python as factorize() /
// solve() / resolve().
//
// Accelerate's native sparse storage is CSC (compressed sparse column),
// column-major. The Python side hands us a CSR matrix (row-major), so we
// transpose-convert CSR -> CSC internally before building the
// SparseMatrix_Double.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Accelerate/Accelerate.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>

namespace py = pybind11;

namespace {

std::string statusToString(SparseStatus_t status) {
    switch (status) {
        case SparseStatusOK:                return "SparseStatusOK";
        case SparseFactorizationFailed:      return "SparseFactorizationFailed";
        case SparseMatrixIsSingular:         return "SparseMatrixIsSingular";
        case SparseInternalError:            return "SparseInternalError";
        case SparseParameterError:           return "SparseParameterError";
        case SparseStatusReleased:           return "SparseStatusReleased";
        default: {
            std::ostringstream oss;
            oss << "Unknown SparseStatus_t(" << static_cast<int>(status) << ")";
            return oss.str();
        }
    }
}

void checkStatus(SparseStatus_t status, const char* where) {
    if (status != SparseStatusOK) {
        throw std::runtime_error(std::string(where) + " failed: " + statusToString(status));
    }
}

} // namespace

// -----------------------------------------------------------------------
// SparseSolver: holds a factorized matrix and lets you solve against one
// or more right-hand sides.
// -----------------------------------------------------------------------
class SparseSolver {
public:
    SparseSolver() = default;

    ~SparseSolver() {
        releaseFactorization();
    }

    // Disallow copies (the Accelerate handles own native memory).
    SparseSolver(const SparseSolver&) = delete;
    SparseSolver& operator=(const SparseSolver&) = delete;

    // factorize(n_rows, n_cols, indptr, indices, data)
    //
    // CSR arrays as produced by scipy.sparse.csr_matrix:
    //   indptr  : int32/int64 array of length n_rows+1
    //   indices : int32/int64 array of length nnz (column indices)
    //   data    : double array of length nnz
    void factorize(int n_rows,
                   int n_cols,
                   py::array_t<long long, py::array::c_style | py::array::forcecast> indptr,
                   py::array_t<long long, py::array::c_style | py::array::forcecast> indices,
                   py::array_t<double, py::array::c_style | py::array::forcecast> data) {
        releaseFactorization();

        m_n_rows = n_rows;
        m_n_cols = n_cols;

        auto ip = indptr.unchecked<1>();
        auto ind = indices.unchecked<1>();
        auto dat = data.unchecked<1>();

        if (static_cast<long long>(ip.shape(0)) != n_rows + 1) {
            throw std::invalid_argument("indptr must have length n_rows + 1");
        }
        const long long nnz = ip(n_rows);
        if (static_cast<long long>(ind.shape(0)) != nnz ||
            static_cast<long long>(dat.shape(0)) != nnz) {
            throw std::invalid_argument("indices/data length must equal indptr[-1] (nnz)");
        }

        // ---- Convert CSR (row-major) -> CSC (column-major) ----
        // Standard counting-sort transpose-conversion, O(nnz + n).
        m_colStarts.assign(n_cols + 1, 0);
        for (long long i = 0; i < nnz; ++i) {
            long long col = ind(i);
            if (col < 0 || col >= n_cols) {
                throw std::invalid_argument("column index out of range in CSR data");
            }
            m_colStarts[col + 1]++;
        }
        for (int c = 0; c < n_cols; ++c) {
            m_colStarts[c + 1] += m_colStarts[c];
        }

        m_rowIndices.assign(nnz, 0);
        m_values.assign(nnz, 0.0);
        std::vector<long long> writeCursor(m_colStarts.begin(), m_colStarts.end());

        for (int row = 0; row < n_rows; ++row) {
            const long long rowStart = ip(row);
            const long long rowEnd = ip(row + 1);
            for (long long k = rowStart; k < rowEnd; ++k) {
                const long long col = ind(k);
                const double val = dat(k);
                const long long dest = writeCursor[col]++;
                m_rowIndices[dest] = static_cast<int32_t>(row);
                m_values[dest] = val;
            }
        }

        // Accelerate wants int32_t column starts (length n_cols+1)
        m_colStarts32.assign(m_colStarts.begin(), m_colStarts.end());

        // ---- Build SparseMatrixStructure / SparseMatrix_Double ----
        SparseAttributes_t attributes{};
        attributes.transpose = false;
        attributes.triangle = SparseUpperTriangle;   // ignored for general matrices
        attributes.kind = SparseOrdinary;            // general (unsymmetric) matrix
        attributes._reserved = 0;

        SparseMatrixStructure structure{};
        structure.rowCount = n_rows;
        structure.columnCount = n_cols;
        structure.columnStarts = m_colStarts32.data();
        structure.rowIndices = m_rowIndices.data();
        structure.attributes = attributes;
        structure.blockSize = 1;

        m_matrix = SparseMatrix_Double{};
        m_matrix.structure = structure;
        m_matrix.data = m_values.data();

        // ---- Factorize with sparse QR (handles square unsymmetric too) ----
        m_factorization = SparseFactor(SparseFactorizationQR, m_matrix);
        checkStatus(m_factorization.status, "SparseFactor (QR)");
        m_hasFactorization = true;
    }

    // solve(b) -> x.  b must have length n_rows; returns vector of length n_cols.
    py::array_t<double> solve(py::array_t<double, py::array::c_style | py::array::forcecast> b) {
        if (!m_hasFactorization) {
            throw std::runtime_error("solve() called before factorize()");
        }
        auto bp = b.unchecked<1>();
        if (static_cast<int>(bp.shape(0)) != m_n_rows) {
            throw std::invalid_argument("rhs length must equal number of rows of factorized matrix");
        }

        // Accelerate's SparseSolve for QR expects the rhs/solution buffer
        // sized to max(n_rows, n_cols) and overwrites it in place with the
        // solution in the first n_cols entries.
        const int workLen = std::max(m_n_rows, m_n_cols);
        std::vector<double> work(workLen, 0.0);
        for (int i = 0; i < m_n_rows; ++i) work[i] = bp(i);

        DenseVector_Double xb{};
        xb.count = workLen;
        xb.data = work.data();

        SparseSolve(m_factorization, xb);

        py::array_t<double> result(m_n_cols);
        auto rp = result.mutable_unchecked<1>();
        for (int i = 0; i < m_n_cols; ++i) rp(i) = work[i];
        return result;
    }

    // resolve(b): re-solve against a new rhs reusing the cached
    // factorization. Functionally identical to solve(); provided as a
    // distinct, explicit entry point for API parity with other sparse
    // solver libraries (e.g. UMFPACK-style factorize/solve/resolve).
    py::array_t<double> resolve(py::array_t<double, py::array::c_style | py::array::forcecast> b) {
        if (!m_hasFactorization) {
            throw std::runtime_error("resolve() called before factorize()");
        }
        return solve(b);
    }

    bool isFactorized() const { return m_hasFactorization; }
    int rows() const { return m_n_rows; }
    int cols() const { return m_n_cols; }

private:
    void releaseFactorization() {
        if (m_hasFactorization) {
            SparseCleanup(m_factorization);
            m_hasFactorization = false;
        }
    }

    int m_n_rows = 0;
    int m_n_cols = 0;

    // Backing storage for the CSC representation; must outlive m_matrix
    // and m_factorization since Accelerate keeps raw pointers into it.
    std::vector<long long> m_colStarts;
    std::vector<long> m_colStarts32;
    std::vector<int32_t> m_rowIndices;
    std::vector<double> m_values;

    SparseMatrix_Double m_matrix{};
    SparseOpaqueFactorization_Double m_factorization{};
    bool m_hasFactorization = false;
};

PYBIND11_MODULE(_pyoomph_mac_accelerate, m) {
    m.doc() = "Low-level pybind11 bindings to Apple Accelerate's Sparse Solvers";

    py::class_<SparseSolver>(m, "SparseSolver")
        .def(py::init<>())
        .def("factorize", &SparseSolver::factorize,
             py::arg("n_rows"), py::arg("n_cols"),
             py::arg("indptr"), py::arg("indices"), py::arg("data"),
             "Factorize a CSR matrix (sparse QR, supports unsymmetric square systems)")
        .def("solve", &SparseSolver::solve, py::arg("b"),
             "Solve A x = b using the cached factorization")
        .def("resolve", &SparseSolver::resolve, py::arg("b"),
             "Re-solve A x = b against a new rhs, reusing the cached factorization")
        .def("is_factorized", &SparseSolver::isFactorized)
        .def_property_readonly("rows", &SparseSolver::rows)
        .def_property_readonly("cols", &SparseSolver::cols);
}
