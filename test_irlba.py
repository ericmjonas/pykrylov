import numpy as np
import util
import irlba2
import lanczos


def test_simple():
    A =np.diag(np.arange(1, 3000))

    opA = util.BasicLinOp(A)

    u, s, v= irlba2.irlba(opA)
    
    print s
