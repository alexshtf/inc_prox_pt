import cython
from .loss_base cimport LossBase

@cython.cdivision(True)
cdef class Hinge(LossBase):
    cdef double solve_dual(self, double alpha, double beta):
        cdef double result = beta / alpha
        if result > 1:
            return 1
        elif result < 0:
            return 0
        else:
            return result

    cdef double eval(self, double z):
        cdef double result = max(0, z)
        return result