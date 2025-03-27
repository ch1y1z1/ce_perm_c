from warnings import warn

import numpy as np
from numpy import asarray
from scipy.sparse import (
    isspmatrix_csc,
    isspmatrix_csr,
    isspmatrix,
    SparseEfficiencyWarning,
    csc_matrix,
)
from scipy.sparse.linalg import MatrixRankWarning, factorized
from scipy.sparse.sputils import is_pydata_spmatrix

import _superlu


# same as scipy.sparse.linalg.spsolve
def spsolve(A, b, permc_spec=None, use_umfpack=True):
    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

    if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
        A = csc_matrix(A)
        warn("spsolve requires A be CSC or CSR matrix format", SparseEfficiencyWarning)

    # b is a vector only if b have shape (n,) or (n, 1)
    b_is_sparse = isspmatrix(b) or is_pydata_spmatrix(b)
    if not b_is_sparse:
        b = asarray(b)
    b_is_vector = (b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1)

    # sum duplicates for non-canonical format
    A.sum_duplicates()
    A = A.asfptype()  # upcast to a floating point format
    result_dtype = np.promote_types(A.dtype, b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)

    # validate input shapes
    M, N = A.shape
    if M != N:
        raise ValueError("matrix must be square (has shape %s)" % ((M, N),))

    if M != b.shape[0]:
        raise ValueError(
            "matrix - rhs dimension mismatch (%s - %s)" % (A.shape, b.shape[0])
        )

    if b_is_vector and b_is_sparse:
        b = b.toarray()
        b_is_sparse = False

    if not b_is_sparse:
        if isspmatrix_csc(A):
            flag = 1  # CSC format
        else:
            flag = 0  # CSR format

        options = dict(ColPerm=permc_spec)
        x, info = _superlu.gssv(
            N, A.nnz, A.data, A.indices, A.indptr, b, flag, options=options
        )
        if info != 0:
            warn("Matrix is exactly singular", MatrixRankWarning)
            x.fill(np.nan)
        if b_is_vector:
            x = x.ravel()
    else:
        # b is sparse
        Afactsolve = factorized(A)

        if not (isspmatrix_csc(b) or is_pydata_spmatrix(b)):
            warn(
                "spsolve is more efficient when sparse b is in the CSC matrix format",
                SparseEfficiencyWarning,
            )
            b = csc_matrix(b)

        # Create a sparse output matrix by repeatedly applying
        # the sparse factorization to solve columns of b.
        data_segs = []
        row_segs = []
        col_segs = []
        for j in range(b.shape[1]):
            bj = np.asarray(b[:, j].todense()).ravel()
            xj = Afactsolve(bj)
            w = np.flatnonzero(xj)
            segment_length = w.shape[0]
            row_segs.append(w)
            col_segs.append(np.full(segment_length, j, dtype=int))
            data_segs.append(np.asarray(xj[w], dtype=A.dtype))
        sparse_data = np.concatenate(data_segs)
        sparse_row = np.concatenate(row_segs)
        sparse_col = np.concatenate(col_segs)
        x = A.__class__(
            (sparse_data, (sparse_row, sparse_col)), shape=b.shape, dtype=A.dtype
        )

        if is_pydata_spmatrix(b):
            x = b.__class__(x)

    return x


class Todo(Exception):
    pass


def sp_perm_c(A, b, permc_spec=None):
    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

    if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
        A = csc_matrix(A)
        warn("spsolve requires A be CSC or CSR matrix format", SparseEfficiencyWarning)

    # b is a vector only if b have shape (n,) or (n, 1)
    b_is_sparse = isspmatrix(b) or is_pydata_spmatrix(b)
    if not b_is_sparse:
        b = asarray(b)
    b_is_vector = (b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1)

    # sum duplicates for non-canonical format
    A.sum_duplicates()
    A = A.asfptype()  # upcast to a floating point format
    result_dtype = np.promote_types(A.dtype, b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)

    # validate input shapes
    M, N = A.shape
    if M != N:
        raise ValueError("matrix must be square (has shape %s)" % ((M, N),))

    if M != b.shape[0]:
        raise ValueError(
            "matrix - rhs dimension mismatch (%s - %s)" % (A.shape, b.shape[0])
        )

    if b_is_vector and b_is_sparse:
        b = b.toarray()
        b_is_sparse = False

    if not b_is_sparse:
        if isspmatrix_csc(A):
            flag = 1  # CSC format
        else:
            flag = 0  # CSR format

        options = dict(ColPerm=permc_spec)
        perm_c = _superlu.zgssv_1(
            N, A.nnz, A.data, A.indices, A.indptr, b, flag, options=options
        )

    else:
        raise Todo
        # b is sparse

    return perm_c


def sp_solve_with_perm_c(A, b, perm_c, permc_spec=None):
    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

    if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
        A = csc_matrix(A)
        warn("spsolve requires A be CSC or CSR matrix format", SparseEfficiencyWarning)

    # b is a vector only if b have shape (n,) or (n, 1)
    b_is_sparse = isspmatrix(b) or is_pydata_spmatrix(b)
    if not b_is_sparse:
        b = asarray(b)
    b_is_vector = (b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1)

    # sum duplicates for non-canonical format
    A.sum_duplicates()
    A = A.asfptype()  # upcast to a floating point format
    result_dtype = np.promote_types(A.dtype, b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)

    # validate input shapes
    M, N = A.shape
    if M != N:
        raise ValueError("matrix must be square (has shape %s)" % ((M, N),))

    if M != b.shape[0]:
        raise ValueError(
            "matrix - rhs dimension mismatch (%s - %s)" % (A.shape, b.shape[0])
        )

    if b_is_vector and b_is_sparse:
        b = b.toarray()
        b_is_sparse = False

    if not b_is_sparse:
        if isspmatrix_csc(A):
            flag = 1  # CSC format
        else:
            flag = 0  # CSR format

        options = dict(ColPerm=permc_spec)
        x, info = _superlu.zgssv_2(
            N,
            A.nnz,
            A.data,
            A.indices,
            A.indptr,
            b,
            flag,
            options=options,
            perm_c=perm_c,
        )

        if info != 0:
            warn("Matrix is exactly singular", MatrixRankWarning)
            x.fill(np.nan)
        if b_is_vector:
            x = x.ravel()
    else:
        # b is sparse
        Afactsolve = factorized(A)

        if not (isspmatrix_csc(b) or is_pydata_spmatrix(b)):
            warn(
                "spsolve is more efficient when sparse b is in the CSC matrix format",
                SparseEfficiencyWarning,
            )
            b = csc_matrix(b)

        # Create a sparse output matrix by repeatedly applying
        # the sparse factorization to solve columns of b.
        data_segs = []
        row_segs = []
        col_segs = []
        for j in range(b.shape[1]):
            bj = np.asarray(b[:, j].todense()).ravel()
            xj = Afactsolve(bj)
            w = np.flatnonzero(xj)
            segment_length = w.shape[0]
            row_segs.append(w)
            col_segs.append(np.full(segment_length, j, dtype=int))
            data_segs.append(np.asarray(xj[w], dtype=A.dtype))
        sparse_data = np.concatenate(data_segs)
        sparse_row = np.concatenate(row_segs)
        sparse_col = np.concatenate(col_segs)
        x = A.__class__(
            (sparse_data, (sparse_row, sparse_col)), shape=b.shape, dtype=A.dtype
        )

        if is_pydata_spmatrix(b):
            x = b.__class__(x)

    return x
