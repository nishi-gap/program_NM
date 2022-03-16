from setuptools import setup
from Cython.Build import cythonize
import numpy

sourcefiles = ["./src/readEXRImage.pyx"]
setup(
    ext_modules = cythonize(sourcefiles),
    include_dirs = [numpy.get_include()]
)