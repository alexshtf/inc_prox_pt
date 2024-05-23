cimport cython
from .loss_base cimport LossBase

@cython.cdivision(True)
cdef class HalfSquared(LossBase):
    cdef double solve_dual(self, double alpha, double beta):
        return beta / (1 + alpha)

    cdef double eval(self, double z):
        return 0.5 * (z * z)