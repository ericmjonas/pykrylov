import numpy as np
import util
import lanczos
from util import norm

EPS = np.finfo(float).eps

def ordered_svd(X):
    """
    returns the svd ordered by singular value largest to smallest
    """
    u, s, v = np.linalg.svd(X)
    if (np.sort(s) != s[::-1]).any():
        raise Exception("NTO SORTED"); 
    return u, s, v.T

def irlba(A, K=6, largest=True,  adjust = 3, aug=None, disps=0, maxit=1000, m_b=20, 
          reorth_two = False, tol=1e-6, V0=None):
    """
    This is a port of IRLBA.m following the code there very closely
    """

    # FIXME do interchange stuff
    m, n = A.dims()
    
    interchange = False
    if n > m and largest == False:
        t = m
        m = n
        n = t
        interchange = True
        raise Exception("Don't do interchange yet")


    W = np.zeros((m, m_b))
    F = np.zeros((n, 1))
    V = np.zeros((n, m_b)) # Preallocate for V
    if V0 == None:
        V[:, 0] = np.arange(1, n+1) 
        V[:, 0] = V[:, 0] / np.sum(V[:, 0]) # np.random.normal(0, 1, n)
    else:
        V[:, :V0.shape[1]] = V0
    # increase the number of desired values by adjust to help increase convergence. 
    # K is re-adjusted as vectors converged. This is only an initial value of K
    K_org = K
    K += adjust

    # sanity checking for input values
    if K <= 0:
        raise Exception("K must be a positive Value")
    if K > min(n, m):
        raise Exception("K must be less than min(n, m) + %d" % adjust)
    if m_b <= 1:
        raise Exception("M_B must be > 1")

    # FIXME clean up parameters A LOT
    if aug == None:
        if largest == True:
            aug = 'RITZ'
        else:
            aug = 'HARM'

    # set tolerance to machine precision 
    tol = max(tol, EPS)
    
    # begin initialization
    B = []
    Bsz = []
    conv = False
    EPS23 = EPS**(2/3.0)
    iteration = 0
    J = 0
    mprod = 0
    R_F = []
    SQRTEPS = np.sqrt(EPS)
    Smax = 1 # holds the maximum value of all computed singular values of B est. ||A||_2
    Smin = [] # holds the minimum value of all computed singular values of B est. cond(A)
    SVTol = min(SQRTEPS, tol) # tolerance to determine whether a singular value has converged
    
    S_B = [] # singular values of B 
    U_B = [] # Left singular vectors of B
    V_B = [] # right singular vectors of B
    V_B_last = [] # holds the last row of the modified V_B
    S_B2 = [] # singular values of [B ||F||]
    U_B2 = [] # left singular vectors of [B ||F||]
    V_B2 = [] # right signular vectors of [b || F||] 

    
    
    while iteration < maxit:

        V, W, F, B, mprod = lanczos.ablanzbd(A, V, W, F, B, K, 
                                             interchange, m_b, n, m, SVTol*Smax, 
                                             reorth_two, iteration)

        # determine the size of the bidiagonal matrix B
        Bsz = B.shape[0]

        # compute the norm of the vector F, and normalize F
        R_F = norm(F)
        F = F/R_F

        # compute singular triplets of B
        U_B, S_B, V_B = ordered_svd(B)

        # estimate ||A|| using the largest singular value ofer all 
        # all iterations and estimate the cond(A) using approximations
        # to the largest and smallest singular values. If a small singular value
        # is less than sqrteps use only Rtiz vectors
        # to augment and require two-sided reorthogonalization 

        if iteration == 0:
            Smax = S_B[0]; 
            Smin = S_B[-1]
        else:
            Smax = max(Smax, S_B[0])
            Smin = min(Smin, S_B[-1])

        Smax = max(EPS23, Smax)

        if Smin/Smax < SQRTEPS:
            reorth_two = True
            aug = 'RITZ'
            
        # re-order the singular values if we're looking for the smallest ones
        if not largest:
            raise NotImplementedError()
            
        # compute the residuals

        R = np.dot(R_F, U_B[-1,:])

        # convergest tests and displays
        conv, U_B, S_B, V_B, K = convtests(Bsz, disps, tol, K_org, U_B, S_B, V_B, 
                                           abs(R), iteration, 
                                           K, SVTol, Smax)

        if conv: # all singular values within tolerance, return ! 
            break # yay
        if iteration > maxit:
            break # boo
            
        # compute starting vectors and first block:
        if aug == "HARM":
            # update the SVD of B to be the SVD of [B ||F||F E_m] 
            U_B2, S_B, V_B2 = ordered_svd(np.c_[np.diag(S_B), R.T])

            if not largest:
                # pick the last ones
                raise Exception("NOT DONE")
            U_B = np.dot(UB, U_B2)
            #V_B = 
            
            raise NotImplementedError()
        else:
            V[:, :K] = np.dot(V, V_B[:, :K])
            V[:, K] = F
            B = np.c_[np.diag(S_B[:K]), R[:K]]

        # compute left approximate singular values
        W[:, :K] = np.dot(W, U_B[:, :K])

        iteration += 1

    # results

    if interchange:
        u = np.dot(V, V_B[:, :K_org])
        s = S_B[:K_org]
        v = np.dot(W, U_B[:, :K_org])

    else:

        u = np.dot(W, U_B[:, :K_org])    
        s = S_B[:K_org]
        v = np.dot(V, V_B[:, :K_org])

    return u, s, v

            
            
def convtests(Bsz, disps, tol, K_org, U_B, S_B, V_B, 
              residuals, iter, K, SVTol, Smax):
    converged = False

    len_res = np.sum(residuals[:K_org] < (tol*Smax))

    if len_res == K_org:
        return True, U_B[:, :K_org], S_B[:K_org], V_B[:, :K_org],  K
    
    else:
        len_res = np.sum(residuals[:K_org] < (SVTol*Smax))
        K = max(K, K_org + len_res)
        if K > Bsz-3:
            K = Bsz-3
            
        return False, U_B, S_B, V_B, K
