#参考にしているサイト
#https://qiita.com/neruoneru/items/6c0fc0496620d2968b57 ##cythonでnumpyを使う
#https://qiita.com/syukan3/items/6ff633e2e0e4c80c6839   ##cythonでの型指定
import OpenEXR, Imath, os
from numpy import linalg as LA

import numpy as np
cimport numpy as np # コンパイル（コツ1）
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function


def C2V(np.ndarray[DTYPE_t, ndim=3] img , int X, int Y):
    cdef np.ndarray nm = 2 * img - 1
    for y in range(Y):
        for x in range(X):
            nm[x][y] /= LA.norm(nm[x][y])
            nm[x][y] = (nm[x][y] + 1)/2
            norm = LA.norm(2 * nm[x][y] - 1)
            nm[x][y] = (2 * nm[x][y] - 1)/norm
    return img


def readEXR(str filename):
    print("readEXR")
    if not os.path.isfile(filename):
        raise FileNotFoundError('Image file not found!')
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT) #型あってるか確かめる
    file = OpenEXR.InputFile(filename)

    dw = file.header()['dataWindow']
    cdef tuple size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    x, y = size[0], size[1]
    cdef bytes r_str = file.channel('R', pt)
    cdef bytes g_str = file.channel('G',pt)
    cdef bytes b_str = file.channel('B', pt)
    cdef np.ndarray r = np.fromstring(r_str, dtype = np.float32)
    cdef np.ndarray g = np.fromstring(g_str, dtype = np.float32)
    cdef np.ndarray b = np.fromstring(b_str, dtype = np.float32)
    cdef np.ndarray img = (np.array([[r, g, b] for r, g, b in zip(r, g, b)])).reshape(size[1], size[0], 3)
    cdef np.ndarray nm = C2V(img, y, x) #色ベクトルから単位法線ベクトル
    
    return nm

