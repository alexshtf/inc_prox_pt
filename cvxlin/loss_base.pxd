cimport cython


cdef class LossBase:
    cdef double solve_dual(self, double alpha, double beta)
    cdef double eval(self, double z)