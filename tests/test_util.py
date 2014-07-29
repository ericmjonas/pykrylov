import numpy as np
from nose.tools import * 
from pykrylov import util

def test_orthog():


    Y = np.random.rand(5, 3)
    X = np.random.rand(5, 1)
    for i in range(X.shape[1]):
        X[:, i] = X[:, i] / util.norm(X[:, i])
        print np.dot(X[:, i], X[:, i])
    Z = util.orthog(Y, X)
    assert_equal(Z.shape, Y.shape)

    # now each col vector should be orthogonal to each other vector
    for i in range(Z.shape[1]):
        for j in range(X.shape[1]):
            print Z[:, i]
            print X[:, j]
            print np.dot(Z[:, i], X[:, j])

            np.testing.assert_almost_equal(np.dot(Z[:, i], X[:, j]), 0.0)
    
