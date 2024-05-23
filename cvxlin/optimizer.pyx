cimport cython
import torch
from .loss_base cimport LossBase


cdef class IncConvexOnLinear:
    cdef LossBase _h
    cdef object _x

    def __init__(self, x, LossBase h not None):
        self._h = h
        self._x = x


    cpdef double step(self, double eta, a, b):
        """
        Performs the optimizer's step, and returns the loss incurred.
        """
        x = self._x

        # compute the dual problem's coefficients
        cdef double alpha = (eta * torch.sum(a ** 2)).item()
        cdef double beta = (torch.dot(a, self._x) + b).item()

        # solve the dual problem
        cdef double s_star = self._h.solve_dual(alpha, beta)

        # update x
        x.sub_(eta * s_star * a)

        return self._h.eval(beta)

    def x(self):
        return self._x
