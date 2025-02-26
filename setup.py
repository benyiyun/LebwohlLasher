from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("lebwohl_lasher.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
