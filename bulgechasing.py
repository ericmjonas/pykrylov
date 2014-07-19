import numpy as np
import util
import givens


def bulgechasing_gk_svd_step(B, shift):
    """
    An implementing of GVL4's 8.6.1 bulge-chasing SVD step
    
    input: 
    shift : an eigenvalue to shift 
    B : real m x n bidiagonal matrix with no zeros on diagonal/superdiagonal
    
    output: 
    B' : a bidiagonal matrix  B' = U^T B V where U and V are orthogonal and V is essentially the orthogonal matrix that would have been obtained by applying 8.3.2 to T=B^T B 

    """

    l = B.shape[0]
    
    y = B[0, 0] - shift
    z = B[0, 1]
    for i in range(l-1):
        print "ITER", i, "="*60
        c, s = givens.givens(y, z)
        B = givens.apply_Givens_rotation(i, i+1, (c, s), B, direction='R')
        print B
        y = B[i, i]
        z = B[i+1, i]
        
        c, s = givens.givens(y, z)
        B = givens.apply_Givens_rotation(i, i+1, (c, s), B, direction='L')
        print B
        if i < l-2:
            y = B[i, i+1]
            z = B[i, i+2]

    return B
