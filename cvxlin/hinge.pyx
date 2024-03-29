cimport cython

@cython.cdivision(True)
cdef class Hinge:
    cpdef cython.floating solve_dual(self, cython.floating alpha, cython.floating beta):
        cdef cython.floating result = beta / alpha
        if result > 1:
            return 1
        elif result < 0:
            return 0
        else:
            return result

    cpdef cython.floating eval(self, cython.floating z):
        cdef cython.floating result = max(0, z)
        return result