import numpy as np
import util
import irlba
import lanczos


def test_simple_biggest_default():
    A =np.diag(np.arange(1, 1000))

    opA = util.BasicLinOp(A)

    t_u, t_s, t_v = np.linalg.svd(A)

    for K in range(1, 10):
        u, s, v= irlba.irlba(opA, K=K)

        # s should be the sorted top K e
        max_delta = np.max(np.abs(s -  t_s[:K]))
        print max_delta
        assert max_delta < 1e-8




def test_simple_biggest_default_2():
    A = np.random.normal(0, 1, (1000, 100))

    opA = util.BasicLinOp(A)

    t_u, t_s, t_v = np.linalg.svd(A)

    for K in range(1, 10):
        u, s, v= irlba.irlba(opA, K=K)

        # s should be the sorted top K e
        max_delta = np.max(np.abs(s -  t_s[:K]))
        print max_delta
        assert max_delta < 1e-8




def test_simple_smallest_default():

    A =np.diag(np.arange(1, 30))

    opA = util.BasicLinOp(A)

    t_u, t_s, t_v = np.linalg.svd(A)

    for K in range(3, 10):
        u, s, v= irlba.irlba(opA, K=K, largest=False)
        print s
        # s should be the sorted top K e
        max_delta = np.max(np.abs(s -  t_s[-K:][::-1]))

        assert max_delta < 1e-8



def test_simple_smallest_default_2():

    A = np.random.normal(0, 1, (1000, 300))
    A[:, 1] = A[:, 0]
    
    opA = util.BasicLinOp(A)

    t_u, t_s, t_v = np.linalg.svd(A)
    print t_s
    for K in range(3, 4):
        u, s, v= irlba.irlba(opA, K=K, largest=False)
        print s
        # s should be the sorted top K e
        max_delta = np.max(np.abs(s -  t_s[-K:][::-1]))
        print max_delta
        assert max_delta < 1e-6



