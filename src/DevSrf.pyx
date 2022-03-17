from dataclasses import dataclass
from cpython cimport array
import numpy as np
cimport numpy as np 

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float32

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t

# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

#@dataclass
cdef class DevSrf:
    cdef int rulingNum
    cdef int modelWidth
    cdef int modelHeight
    cdef int MapWidth
    cdef int MapHeight
    cdef np.ndarray[:] foldAngles #うまくいかないのでlist or c++のvectorで表す
    #cdef double[:] foldAngles, xl, xr
    #xr:np.array

    def __cinit__(self, int rulingNum, int modelWidth, int modelHeight, 
    double foldAngle, int MapWidth, int MapHeight):
        self.rulingNum = rulingNum
        self.modelWidth = modelWidth
        self.modelHeight = modelHeight   
        self.foldAngles = np.arange(np.full(rulingNum,foldAngle), dtype = np.dtype("f"))
        self.MapWidth = MapWidth
        self.MapHeight = MapHeight
        cdef double step = modelWidth/(rulingNum + 1)
        self.xl = np.linspace(step, modelWidth - step, rulingNum)
        self.xr = np.linspace(step, modelWidth - step, rulingNum)
        print(type(self.xl))


