cdef class LossBase:
    cdef double solve_dual(self, double alpha, double beta):
        raise NotImplementedError()

    cdef double eval(self, double z):
        raise NotImplementedError()