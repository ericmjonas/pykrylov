import numpy as np

def norm(x):
    return np.sqrt(np.sum(x**2.0))

class BasicLinOp(object):
    """ 
    Turn a matrix into a simple linear operator
    """

    def __init__(self, A):
        self.A = A
        self.AT = A.T

    def compute_Ax(self, x):
        return np.dot(self.A, x)
        
    def compute_ATx(self, x):
        return np.dot(self.AT, x)
    
    def dims(self):
        """
        returns the output shapes
        of the operator
        """
        return self.A.shape[0], self.A.shape[1]

def min_dist_in(x, a):
    """
    for each element of x, how L1 close is it to any element in A? 
    """
    d = np.zeros(len(x))
    for xi, xv in enumerate(x):
        d[xi] = np.min(np.abs(a - xv))
    return d

def min_dist_in_frac(x, a):
    """
    for each element of x, how close is it to any element in A, 
    in fractions
    """
    d = np.zeros(len(x))
    for xi, xv in enumerate(x):
        delta = np.abs(a - xv)
        di = np.argmin(delta)
        d[xi] = np.abs(delta[di] / a[di])
    return d


def assert_upper_triangular(X):
    Xtu = np.tril(X, -1)
    np.testing.assert_array_almost_equal(Xtu, np.zeros_like(X))

def close_zero(x):
    if np.abs(x) < np.finfo(np.float).eps:
        return True
    return False
    
