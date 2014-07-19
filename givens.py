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

def apply_Givens_rotation(i, k, (c, s), M,  direction="L"):
    """

    using GVR 5.1.9
    if direction = L :
          
         return M' = G M  
    
    FIXME: perform in-place? 
    FIXME: handle sparse case
    
    """
    ## hilariously build up full matrix because why not? 
    
    m, n = M.shape
    newM = M.copy()

    # FIXME vectorize

    if direction == "L":
        print "applying givens, row i = ", i, "row k=", k
        for j in range(n):
            t1 = M[k, j]
            t2 = M[i, j]
            a = c*t1 - s*t2
            b = s*t1 + c * t2
            newM[i, j] = b
            newM[k, j] = a

    elif direction == 'R':
        print "applying givens, col i = ", i, "col j=", k
        for j in range(m):
            t1 = M[j, k]
            t2 = M[j, i]
            a = c*t1 - s*t2
            b = s*t1 + c * t2
            newM[j, i] = b
            newM[j, k] = a

            
    return newM


def create_full_Givens_matrix(i, j, (c, s), size):
    ## hilariously build up full matrix because why not? 

    N = size
    G = np.eye(N)

    G[i, i] = c
    G[j, j] = c
    G[j, i] = -s
    G[i, j] = s
    
    return G
    
def apply_Givens_rotation_f(i, j, (c, s),  M,  direction="L"):
    """
    
    FIXME: perform in-place? 
    FIXME: handle sparse case
    
    """
    ## hilariously build up full matrix because why not? 

    N = len(M)

    
    G = create_full_Givens_matrix(i, j, (c, s), N)
    if direction == "L":
        return np.dot(G, M)
    else:
        return np.dot(M, G.T)
        


    
