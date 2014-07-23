import numpy as np
import scipy.linalg
import lanczos
from util import norm

def assert_small(x):
    assert np.abs(x) < 1e-10

def pick_extreme_svd(k, U, s, V, largest):
    
    ai = np.argsort(s)
    if largest:
        ai = ai[::-1]
    print s, s[ai[:k]]
    return U[:, ai[:k]], s[ai[:k]], V[ai[:k], :]
        
def sort_svd(U, s, V):
    ai = np.argsort(s)
    ai = ai[::-1]
    return U[:, ai], s[ai], V[ai, :]

def svd(A, p1, k, m, tol, harmonic=False, largest=True):
    """
    Implementation of BR05 for computing singular triplets. 

    A: linear operator that can compute Ax and A*x, effective size l x n
    p1 : init vector of unit length, n elements
    k : number of desired singular triplets
    m : number of bidiagonalization steps
    tol : tolerance for accepting approximated singular triplet
    harmonic : type of augmentation to use (Harmonic = True
    
    largest : get the largest k triplets; if false return the smallest

    output: compute set of approximate singular triplets, [(sigma, u, v), ...]
    
    """
    
    
    
    l, n = A.dims()

    Pm, Qm, Bm, rm  = lanczos.partial_lanczos_bidiagonalization(A, p1, m)
    print "At the beginning", Pm.shape, Qm.shape, Bm.shape
    assert_small(np.sum(np.abs(A.compute_Ax(Pm) - np.dot(Qm, Bm))))
    assert_small(np.sum(np.abs(A.compute_ATx(Qm) - np.dot(Pm, Bm.T) - np.outer(rm, np.eye(m)[:, m-1]))))

    iteration = 0
    while True:
        print "ITERATION", iteration, "-"*40
        U, s, V = np.linalg.svd(Bm, full_matrices=True)
        #U, s, V = pick_extreme_svd(k, U, s, V, largest)
        U, s, V = sort_svd(U, s, V)

        B_m_val = norm(rm)
        p_m_plus_1 = rm/B_m_val
        print "B_m_val, ||rm|| = ", B_m_val
            
        # 3. check convergence
        if iteration > 100:
            return U, s, V.T
        # 4. compute augmenting vectors
        # approx singular triplets of A from singular triplets of Bm
        u_A = np.dot(Qm, U)
        v_A = np.dot(Pm, V.T)

        # let's do some sanity checking here
        for i in range(k+1):
            
            delta = np.sum(np.abs(A.compute_Ax(v_A[:, i]) - s[i]*u_A[:, i]))
            #print "2.12 DELTA=", delta
            assert_small(delta)
            #dbg = A.compute_ATx(u_A[:, i]) - s[i]*v_A[:, i] - rm * np.dot(np.eye(m)[-1], U[:, i])
            #delta = np.sum(np.abs(dbg))
            
            #assert_small(delta)

        print "Q", Qm.shape, U.shape, u_A.shape
        print "V", Pm.shape, V.shape, v_A.shape
        if not harmonic or np.linalg.cond(B) > 1./np.sqrt(epsilon):
            # if not otherwise indicated matrices are with tildes
            # determine new matrices
            P = np.zeros((n, k+1))
            print "Pm.shape=", Pm.shape, "P.shape", P.shape

            P[:, :k] = v_A[:, :k]
            P[:, k] = p_m_plus_1
            
            # DEBUG sanity check -- use eqn 3.3
            dbg = A.compute_Ax(P)
            for i in range(k):
                delta = np.sum(np.abs(dbg[:, i] - s[i]*u_A[:, i]))
                #print "subdelta i=", i, delta
                assert_small(delta)

                
            Q = np.zeros((l, k+1))
            Q[:, :k] = u_A[:, :k]

            # compute the rhos: # FIXME use the optimized trick
            rho = np.zeros(k)

            for i in range(k):
                #print "rho vals for", i
                # hilariously we're going to walk through 
                #rho_1 = np.dot(u_A[:, i].T, A.compute_Ax(p_m_plus_1)) # from first part
                #rho_2 = np.dot(p_m_plus_1.T, A.compute_ATx(u_A[:, i]))
                #rho_3 = np.dot(p_m_plus_1.T, s[i]*v_A[:, i] + rm * U[m-1, i])
                rho_4 =  B_m_val*U[m-1, i]

                #print rho[i], rho2, Bm[m-1, m-1]

                rho[i] = rho_4

            run_tot = A.compute_Ax(p_m_plus_1)
            for i in range(k):
                run_tot -= rho[i] * u_A[:, i]

            r_tilde_k = run_tot
            Q[:, k] = r_tilde_k / norm(r_tilde_k)

            B = np.zeros((k+1, k+1))
            for i in range(k):
                B[i, i] = s[i] 
                B[i, k] = rho[i] 
            B[k, k] = norm(r_tilde_k) # INFERENCE FROM EQUATIONS

            # DEBUG check the identity from 3.7
            dbg = A.compute_Ax(P) - np.dot(Q, B)
            for i in range(k+1):
                print "2 subdelta i=", i, np.sum(np.abs(dbg[:, i]))
            print A.compute_Ax(P)[:, -1]
            print np.dot(Q, B)[:, -1]
            
            f_k_plus_1 = A.compute_ATx(r_tilde_k) / norm(r_tilde_k) - norm(r_tilde_k)*p_m_plus_1
            r = f_k_plus_1
            # now a santiy check
            print "DELTA2=", np.sum(np.abs(A.compute_Ax(P) - np.dot(Q, B)))
            
        else:
            # harmonic and Kappa(B) <= 1/sqrt(epsilon)
            # compute svd of Bm,m+1 and qr factorization

            # update matrices
            raise NotImplementedError() 
            pass
        # P, Q, B, # append columns and rows and call the new matrices whatever
        print P.shape, np.zeros((l, m-k)).shape
        Pm = np.column_stack([P, np.zeros((n, m-k))])
        Qm = np.column_stack([Q, np.zeros((l, m-k))])
        Bm = np.zeros((B.shape[0] + m-k, 
                       B.shape[1] + m-k))
        Bm[:k+1, :k+1] = B
        rm = r

        iteration += 1
