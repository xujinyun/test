'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

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
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)
    LU_decomposition = splu(A.T @ A, permc_spec='NATURAL')
    x = LU_decomposition.solve(A.T @ b)
    U = LU_decomposition.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)
    LU_decomposition = splu(A.T @ A, permc_spec='COLAMD')
    x = LU_decomposition.solve(A.T @ b)
    U = LU_decomposition.U
    return x, U

def plu(A):
    
    #Get the number of rows
    n = A.shape[1]
    
    #Allocate space for P, L, and U
    U = eye(n)
    L = eye(n, format='csc')
    P = eye(n, format='csc')
    
    # Loop over rows
    for i in range(n):
        
        # Permute rows if needed
        for k in range(i, n):
            print(U)
            # if ~np.isclose(U[i, i], 0.0):
            #     break
            # U is a dia_matrix, so we can't use the fancy indexing
            # Find the row with the largest magnitude element in column i of matrix U
            
            # j = np.argmax(np.abs(U.data[i])) + i
            # # Swap rows j and i
            # U[:, [i, j]] = U[:, [j, i]]
            # P[:, [i, j]] = P[:, [j, i]]
            
        # Eliminate entries below i with row operations on U and
        # reverse the row operations to manipulate L
        row_indices = U.indices[U.indptr[i]:U.indptr[i+1]]
        row_data = U.data[U.indptr[i]:U.indptr[i+1]]
        for j, factor in zip(row_indices, row_data / U[i, i]):
            if j > i:
                U[:, j] -= factor * U[:, i]
                L[j, i] = factor
                
    return P, L, U

def forward_substitution(L, b):
    
    #Get number of rows
    n = L.shape[0]
    
    #Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double);
    
    #Here we perform the forward-substitution.  
    #Initializing  with the first row.
    y[0] = b[0] / L[0, 0]
    
    #Looping over rows in reverse (from the bottom  up),
    #starting with the second to last row, because  the 
    #last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]
        
    return y

def back_substitution(U, y):
    
    #Number of rows
    n = U.shape[0]
    
    #Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double);

    #Here we perform the back-substitution.  
    #Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]
    
    #Looping over rows in reverse (from the bottom up), 
    #starting with the second to last row, because the 
    #last row solve was completed in the last step.
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]
        
    return x

def solve_my_splu(A, B):
    # N = A.shape[1]
    # x = np.zeros((N, ))
    # U = eye(N)
    
    # convert sparse matrix A in csc format to regular matrix
    # A = A.todense()
    
    P, L, U = plu(A)
    y = forward_substitution(L, P @ B)
    x = back_substitution(U, y)
    

    return x, U





    


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)
    z, R, E, rank = rz(A, b, tolerance = None, permc_spec='NATURAL')
    x = spsolve_triangular(R.tocsr(),z,lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)
    z, R, E, rank = rz(A, b, permc_spec='COLAMD')
    x_ = spsolve_triangular(R.tocsr(),z,lower=False)
    x = permutation_vector_to_matrix(E) @ x_
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
        'my_splu': solve_my_splu,
    }

    return fn_map[method](A, b)
