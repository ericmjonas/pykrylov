"""
The coverage of givens rotations in GVL chapter 5 is quite good, 
far better than wikipedia. Adequately explains in the context of
Householder reflections too. 


Algorithm 8.6.1 is the Golub-Kahan SVD step

"""

import util
import numpy as np

def givens(a, b):
    """
    From GVL 5.1.3
    FXME: We're supposed to do something smart to prevent underflow
    """
    print "Calling givens with", a, b
    if util.close_zero(b):
        c = 1
        s = 0
    else:
        if np.abs(b) > np.abs(a):
            tau = -a/b
            s = 1/np.sqrt(1 + tau**2)
            c = s * tau
        else:
            tau = -b/a
            c = 1/np.sqrt(1 + tau**2)
            s = c * tau
    print "givens c=", c, "s=", s        
    return c, s

def apply_Givens_rotation(i, k, M,  direction="L"):
    """
    using GVR 5.1.9
    direction : L means return M' = G M 
                R means return M' = M G

    FIXME: perform in-place? 
    FIXME: handle sparse case
    
    """
    ## hilariously build up full matrix because why not? 

    m, n = M.shape
    newM = M.copy()

    # FIXME vectorize

    if direction == "L":
        c, s, = givens(M[i-1, k], M[i, k])
        for j in range(n):
            t2 = M[i, j]
            t1 = M[k, j]
            a = c*t1 - s*t2
            b = s*t1 + c * t2
            print "j=", j, "t1=", t1, "t2=", t2, "a=", a, "b=", b
            newM[k, j] = a 
            newM[i, j] = b

    elif direction == 'R':
        c, s, = givens(M[i, k], M[i, k-1])
        for j in range(m):
            t1 = M[j, i]
            t2 = M[j, k]
            newM[j, i] = c*t1 - s*t2
            newM[j, k] = s*t1 + c*t2
    return newM


def create_full_Givens_matrix(i, j, theta_params, size):
    ## hilariously build up full matrix because why not? 

    N = size
    G = np.eye(N)

    c, s = theta_params
    G[i, i] = c
    G[j, j] = c
    G[j, i] = -s
    G[i, j] = s
    
    return G
    
def apply_Givens_rotation_f(i, j, M,  direction="L"):
    """
    
    FIXME: perform in-place? 
    FIXME: handle sparse case
    
    """
    ## hilariously build up full matrix because why not? 

    N = len(M)

    x = M[i-1, j]
    y = M[i, j]
    tp = givens(x, y)


    G = create_full_Givens_matrix(i, j, tp, N)

    return np.dot(G, M)

    
