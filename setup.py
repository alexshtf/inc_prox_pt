from setuptools import setup
from Cython.Build import cythonize

cvxlin = cythonize("cvxlin/*.pyx")

setup(
    name='inc_prox_pt',
    version='1.0',
    author='Alex Shtoff',
    ext_modules=cvxlin
)