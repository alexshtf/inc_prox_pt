cimport cython

@cython.cdivision(True)
cdef class HalfSquared:
    cpdef cython.floating solve_dual(self, cython.floating alpha, cython.floating beta):
        return beta / (1 + alpha)

    cpdef cython.floating eval(self, cython.floating z):
        return 0.5 * (z * z)