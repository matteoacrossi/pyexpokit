"""
A Python implementation of expm from Expokit
"""

import numpy as np
from scipy.sparse.linalg import norm as spnorm

from numpy.linalg import norm
from scipy.linalg import expm

def expmv(t, A, v, tol=1e-7, krylov_dim=30):
    """
        Evaluates exp(t * A) @ v efficiently using Krylov subspace projection
        techniques and matrix-vector product operations.

    This function implements the expv function of the Expokit library
    (https://www.maths.uq.edu.au/expokit). It is in particular suitable for 
    large sparse matrices.

    Args:
    t (float): real or complex time-like parameter
    A (array or sparse): an numpy.array or scipy.sparse square matrix
    v (array): a vector with size compatible with A
    tol (real): the required tolerance (default 1e-7)
    krylov_dim (int): dimension of the Krylov subspace 
                      (typically a number between 15 and 50, default 30)

    Returns:
    The array w(t) = exp(t * A) @ v.
    """
    assert A.shape[1] == A.shape[0], "A must be square"
    assert A.shape[1] == v.shape[0], "v and A must have compatible shapes"

    n = A.shape[0]
    m = min(krylov_dim, n)

    anorm = spnorm(A, ord=np.inf)

    out_type = np.result_type(type(t), A.dtype, v.dtype)

    # safety factors
    gamma = 0.9
    delta = 1.2

    btol = 1e-7     # tolerance for "happy-breakdown"
    maxiter = 10    # max number of time-step refinements

    rndoff = anorm*np.spacing(1)

    # estimate first time-step and round to two significant digits
    beta = norm(v)
    r = 1/m
    fact = (((m+1)/np.exp(1.0)) ** (m+1))*np.sqrt(2.0*np.pi*(m+1))
    tau = (1.0/anorm) * ((fact*tol)/(4.0*beta*anorm)) ** r

    outvec = np.zeros(v.shape, dtype=out_type)

    # storage for Krylov subspace vectors
    vm = np.zeros((m + 1, len(v)), dtype=out_type)
    # for i in range(1, m + 2):
    #     vm.append(np.empty_like(outvec))
    hm = np.zeros((m+2, m+2), dtype=outvec.dtype)

    tf = np.abs(t)

    # For some reason numpy.sign has a different definition than Julia or MATLAB
    if isinstance(t, complex):
        tsgn = t / np.abs(t)
    else:
        tsgn = np.sign(t)

    tk = 0. * tf
    w = np.array(v, dtype=out_type)
    p = np.empty_like(w)

    mx = m
    while tk < tf:
        tau = min(tf - tk, tau)
        # Arnoldi procedure
        vm[0] = w / beta
        mx = m

        for j in range(m):
            p = A.dot(vm[j])

            hm[:j+1, j] = vm[:j+1, :].conj() @ p
            tmp = hm[:j+1, j][:, np.newaxis] * vm[:j+1]
            p -= np.sum(tmp, axis=0)

            s = norm(p)
            if s < btol: # happy-breakdown
                tau = tf - tk
                err_loc = btol

                F = expm(tsgn * tau * hm[:j+1, :j+1])

                tmp = beta * F[:j+1, 0][:, np.newaxis] * vm[:j+1,:]
                w = np.sum(tmp, axis=0)

                mx = j
                break

            hm[j+1, j] = s
            vm[j+1] = p / s

        hm[m + 1, m] = 1.

        if mx == m:
            avnorm = norm(A @ vm[m])

        # propagate using adaptive step size
        it = 1
        while it < maxiter and mx == m:
            F = expm(tsgn * tau * hm)

            err1 = abs(beta * F[m, 0])
            err2 = abs(beta * F[m+1, 0] * avnorm)

            if err1 > 10*err2:	# err1 >> err2
                err_loc = err2
                r = 1/m
            elif err1 > err2:
                err_loc = (err1*err2)/(err1-err2)
                r = 1/m
            else:
                err_loc = err1
                r = 1/(m-1)

            # is time step sufficient?
            if err_loc <= delta * tau * (tau*tol/err_loc) ** r:
                tmp = beta * F[:m+1, 0][:, np.newaxis] * vm[:m+1,:]
                w = np.sum(tmp, axis=0)
                break

            # estimate new time-step
            tau = gamma * tau * (tau * tol / err_loc) ** r 
            it += 1

        if it == maxiter:
            raise(RuntimeError("Number of iteration exceeded maxiter. "
                               "Requested tolerance might be too high."))

        beta = norm(w)
        tk += tau
        tau = gamma * tau * (tau * tol / err_loc) ** r # estimate new time-step
        err_loc = max(err_loc, rndoff)
        
        hm.fill(0.)

    return w