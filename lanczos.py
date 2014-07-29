import numpy as np
from util import norm
import util

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
    assert l >= n
    assert len(p1) == n 
    
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
                

    
def ablanzbd(A, V, W, F, B, K, 
             interchange, m_b, n, m,  
             SVTol, two_reorth=False, iteration = 0):
    """
    a fairly close port of the lanczos bidiagonalization step 
    """
    def mat_prod(x):
        if interchange:
            return A.compute_ATx(x)
        else:
            return A.compute_Ax(x)

    def adjoint_prod(x):
        if interchange:
            return A.compute_Ax(x)
        else:
            return A.compute_ATx(x)
        
    J = 0

    # noramlization of starting vector
    if iteration == 0:
        V[:, 0] = V[:, 0] / norm(V[:, 0])
        B = []
    else:
        J = K

    W[:, J] =  mat_prod(V[:, J])
    
    # input vectors are singular vectors and AV[:, J] which must be 
    # orthogonalized
    
    if iteration > 0:
        W[:, J] = util.orthog(W[:, J], W[:, :J]) # FIXME possible idnex error

    S = norm(W[:, J])
    
    # check for linearly-dependent vectors
    if S < SVTol:
        W[:, J] = np.random.normal(0, 1, W.shape[1])
        W[:, J] = util.orthog(W[:, J], W[:, :J])
        W[:, J] = W[:, J]/ norm(W[:, J])
        
    else:
        W[:, J] = W[:, J]/ S

    # begin main iteration loop for block lanczos
    while J  < m_b:

        F =  adjoint_prod(W[:, J])

        # One step of the block classical Gram-Schmidt process
        F = F - V[:, J] * S
        
        # full reorthogonalization, short vectors
        F = util.orthog(F, V[:, :J])

        
        if (J+1) < m_b:
            R = norm(F)

            if R <= SVTol:
                F = np.random.normal(0, 1, (V.shape[0], 1))
                F = util.orthog(F, V[:, :J])

                V[:, J+1] = F.flatten() / norm(F)
                R = 0
            else:
                V[:, J+1] = F / R

            # compute block diagonal B
            if len(B) == 0:
                B = np.array([[S, R]])
            else:
                Bnew = np.zeros((B.shape[0] + 1, 
                                 B.shape[1] + 1))
                Bnew[:B.shape[0], 
                  :B.shape[1]] = B
                Bnew[-1, -2] = S
                Bnew[-1, -1] = R
                B = Bnew

            # W = A*V
            W[:, J+1] = mat_prod(V[:, J+1])
    
            # one step of block classical Gram-Schmidt process:
            W[:, J+1] = W[:, J+1] - np.dot( W[:, J] , R)
            
            if two_reorth:
                W[:, J+1] = util.orthog(W[:, J+1], W[:, :J]) # FIXME I THINK THESE INCDES ARE OFF
                
            # compute norm of W
            S = norm(W[:, J+1])
            
            if S <= SVTol:
                W[:, J+1] = np.random.normal(0, 1, (W.shape[0], 1))
                W[:, J+1] = util.orthog(W[:, J+1], W[:, :J])
                W[:, J+1] = W[:, J+1] / norm(W[:, J+1])
                S = 0
            else:
                W[:, J+1] = W[:, J+1]/S
                
                
        else:
            # Add last block to matrix B
            Bnew = np.zeros((B.shape[0]+1,  B.shape[1]))
            Bnew[:-1, :] = B

            Bnew[-1, -1:] = S
            B = Bnew

        J += 1

    return V, W, F, B, m


    
    
