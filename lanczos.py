import numpy as np
from util import norm


def lanczos_bidiagonalization(A, p0, k):
    """
    Lanczos bidiagonalization as presented as (algorithm 1)
    in the Kokiopoulou paper. 

    inputs:

    Linear Operator : as defined in util
    p0 : start vector 
    k : number of iters

    output: 
    B_k :  bidiagonal real matrix, size (k+1) x k
    U_kp1  : Orthogonal bases U_{k+1} \in C, m x (k+1) 
    V_k  : Orthogonal bases V_{k} \in C, n x (k) 
    
    Note that we infer the correct dtype based on p0

    FIXME: Don't return dense B_k

    """
    
    dtype = p0.dtype

    betas = np.zeros(k+1)
    alphas = np.zeros(k)
    m, n = A.dims()
    U = np.zeros((m, k+1), dtype=dtype)
    V = np.zeros((n, k+1), dtype=dtype)

    # init
    betas[0] = norm(p0)
    U[:, 0] = p0/betas[0]
    V[:, 0] = 0 
    for i in range(k):
        r = A.compute_ATx(U[:, i]) 
        if i > 0: 
            r -= np.dot(betas[0], V[:, i]) 
        alphas[i] = norm(r)
        V[:, i+1]  = r/alphas[i]
        p = A.compute_Ax(V[:, i+1]) - alphas[i]*U[:, i]
        betas[i+1] = norm(p)
        U[:, i+1] = p / betas[i+1]

    B = np.zeros((k+1, k), dtype = dtype)

    for i in range(k):
        B[i, i] = alphas[i]
        B[i+1, i] = betas[i+1]

    return B, U, V[:, 1:]

def partial_lanczos_bidiagonalization(A, p1, m):
    """
    Are we implementing the same algo as before? 
    Does this do the same reortho? 

    from BR05 algo 2.1
    FIXME : Add recent docs
    """
    
    l, n = A.dims()
    assert len(p1) == l
    
    P = np.zeros((n, m))
    Q = np.zeros((l, m))
    B = np.zeros((m, m))

    beta = np.zeros(m-1)
    alpha = np.zeros(m)

    P[:, 0] = p1; 
    q_cur = A.compute_Ax(p1)
    
    alpha[0] = norm(q_cur)
    q_cur = q_cur / alpha[0]
    Q[:, 0] = q_cur
    for j in range(m):
        q_cur = Q[:, j]
        rj = A.compute_ATx(q_cur) - alpha[j]*P[:, j]
        #reorthogonalize

        rj = rj - np.dot(P[:, j], np.dot(P[:, j].T, rj))
        if j < (m-1):
            beta[j] = norm(rj)
            P[:, j+1] = rj/beta[j]
            q_next = A.compute_Ax(P[:, j+1]) - beta[j]*q_cur
            
            # reortho:
            q_next = q_next - np.dot(Q[:, j], np.dot(Q[:, j].T, q_next))
            alpha[j+1] = norm(q_next)
            q_next = q_next / alpha[j+1]
            Q[:, j+1] = q_next
    rm = rj
    
    for i in range(m):
        B[i, i] = alpha[i]
        if i < (m-1):
            B[i, i+1] = beta[i]

    return P, Q, B, rm
                

    
