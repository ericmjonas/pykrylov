import numpy as np
from pykrylov import givens
from pykrylov import util
from nose.tools import * 


def test_simple_L():
    """
    Example from wikipedia
    """
    A = np.array([[6, 5, 9, 3], 
                  [5, 1, 4, 2], 
                  [8, 4, 3, 6], 
                  [1, 2, 3, 4]], dtype=float)
    
    # try and zero (2, 1)
    row_a = 3 # change the values in row 2 as our target
    row_b = 2 # manipulating the values in row 3
    col_tgt = 1 # we want row_a, col_tgt to be zero 
    
    
    c, s = givens.givens(A[row_b, col_tgt], 
                         A[row_a, col_tgt]) # this is the value we want to zero

    Anew = givens.apply_Givens_rotation_f(row_a, row_b, (c, s), A)
    assert util.close_zero(Anew[row_a, col_tgt])
    # Anew2 = givens.apply_Givens_rotation(row_a, row_b, (c, s), A)
    # assert util.close_zero(Anew2[row_a, col_tgt])
    

def test_simple_R():
    """
    """
    A = np.array([[6, 5, 9, 3], 
                  [5, 1, 4, 2], 
                  [8, 4, 3, 6], 
                  [1, 2, 3, 4]], dtype=float)
    
    # try and zero (2, 1)
    col_a = 3 # change the values in col 3 as our target
    col_b = 2 # manipulating the values in row 2
    row_tgt = 1 # we want row_a, col_tgt to be zero 
    
    
    c, s = givens.givens(A[row_tgt, col_b], 
                         A[row_tgt, col_a]) # this is the value we want to zero

    Anew = givens.apply_Givens_rotation_f(col_a, col_b, (c, s), A, "R")
    print A
    print Anew

    assert util.close_zero(Anew[row_tgt, col_a])
    # Anew2 = givens.apply_Givens_rotation(row_a, row_b, (c, s), A)
    # assert util.close_zero(Anew2[row_a, col_tgt])
    
