'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from curses.ascii import ETX
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    # N = A.shape[1]
    # x = np.zeros((N, ))
    x = inv(A.T @ A) @ A.T @ b

    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    # N = A.shape[1]
    # x = np.zeros((N, ))
    # U = eye(N)

    lu = splu(A.T @ A, permc_spec='NATURAL')
    x = lu.solve(A.T @ b)
    U = lu.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    # N = A.shape[1]
    # x = np.zeros((N, ))
    # U = eye(N)

    lu = splu(A.T @ A, permc_spec='COLAMD')
    x = lu.solve(A.T @ b)
    U = lu.U
    return x, U



def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    # N = A.shape[1]
    # x = np.zeros((N, ))
    # R = eye(N)

    z, R, E, rank = rz(A, b, permc_spec='NATURAL')
    x = spsolve_triangular(R.tocsr(),z,lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    # N = A.shape[1]
    # x = np.zeros((N, ))
    # R = eye(N)

    z, R, E, rank = rz(A,b,permc_spec='COLAMD')
    ETx = spsolve_triangular(R.tocsr(),z,lower=False) # E^T x
    x = permutation_vector_to_matrix(E) @ ETx
    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matirx
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
        # 'lu_bonus': solve_lu_bonus,
    }

    return fn_map[method](A, b)
