import cython
from libc.math cimport log, exp, log1p
from scipy.optimize.cython_optimize cimport brentq
from .loss_base cimport LossBase


ctypedef struct alpha_beta:
    double alpha
    double beta


cdef double qprime(double s, void* args) noexcept:
    cdef alpha_beta* ab = <alpha_beta*>args
    return -ab.alpha * s + ab.beta + log1p(-s) - log(s)


cdef class Logistic(LossBase):
    cdef double solve_dual(self, double alpha, double beta):
        # compute [l,u] containing a point with zero qprime
        cdef alpha_beta ab
        ab.alpha = alpha
        ab.beta = beta

        cdef double l = 0.5
        while qprime(l, &ab) <= 0:
            l /= 4

        cdef double u = 0.5
        while qprime(1 - u, &ab) >= 0:
            u /= 4
        u = 1 - u

        cdef double XTOL = 2e-12
        cdef double RTOL = 8.881784197001252e-16
        cdef int MAX_ITER = 100

        return brentq(qprime, l, u, &ab, XTOL, RTOL, MAX_ITER, NULL)

    cdef double eval(self, double z):
        return log1p(exp(z))