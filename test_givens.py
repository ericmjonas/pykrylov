import numpy as np
import givens
import util
from nose.tools import * 


def test_simple():
    """
    Example from wikipedia
    """
    A = np.array([[6, 5, 0], 
                  [5, 1, 4], 
                  [0, 4, 3]], dtype=float)
    
    # try and zero (2, 1)
    Anew = givens.apply_Givens_rotation_f(1, 0, A)
    Anew2 = givens.apply_Givens_rotation(1, 0, A)

    Anew3 = givens.apply_Givens_rotation(2, 1, Anew)
    print A
    print Anew
    print Anew2
    print Anew3
    #util.assert_upper_triangular(Anew)
    
