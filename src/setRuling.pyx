from scipy.optimize import minimize, Bounds
from numpy import linalg as LA
import numpy as np
cimport numpy as np
np.import_array()
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)

cdef float dx_t
cdef int rulingNum, MapWidth, MapHeight, modelWidth, modelHeight
cdef np.ndarray[DTYPE_t, ndim=3] img

def getSD(np.ndarray[DTYPE_t, ndim=1] x):
    #print("getSD ", img.shape)
    f = 0.0
    
    for i in range(rulingNum + 1):
        slicedImage = np.empty((3,),dtype=float)

        if i == 0:                
            for y in range(0, MapHeight):
                t =  round(y * (x[i] - x[i + rulingNum])/ modelHeight + x[i] * dx_t)
                
                for x in range(0, round(y * (x[i] - x[i + rulingNum])/ modelHeight + x[i] * dx_t)):
                    if(0 <= x and x < MapWidth):
                        slicedImage = np.append(slicedImage,img[y][x],axis = 0)
        elif i == rulingNum:
            for y in range(0, MapHeight):
                for x in range(round(y * (x[i - 1] - x[i + rulingNum - 1])/ modelHeight + x[i - 1] * dx_t), MapWidth):
                    if(0 <= x and x < MapWidth):
                        slicedImage = np.append(slicedImage,img[y][x],axis = 0)
        else:
            for y in range(0, MapHeight):
                for x in range(round(y * (x[i - 1] - xr[i + rulingNum - 1])/ modelHeight + x[i + rulingNum - 1] * dx_t),
                 round(y * (x[i] - x[i + rulingNum])/ modelHeight + x[i] * dx_t)):
                    if(0 < x and x < MapWidth):
                        slicedImage = np.append(slicedImage,img[y][x],axis = 0)  
       
        slicedImage = slicedImage.reshape([3,int(slicedImage.size/3)])
        slicedImage = slicedImage.T          
        slicedImage = np.delete(slicedImage,0,0)
        #print("slicedImage size : ", slicedImage.shape)          
        Vave = np.mean(slicedImage, axis = 0)
        for n in range(slicedImage.shape[0]):
            val = 1 - np.dot(slicedImage[n,:],Vave)
            f += val
            #print(1 - np.dot(slicedImage[n,:],Vave), "  ", Vave)
       
        #print("f = ", f)
    return f

def setRuling(int _rulingNum, int _modelWidth, int _modelHeight,
int _MapWidth, int _MapHeight, np.ndarray[DTYPE_t, ndim=3] _img):
    print("setRuling")
    
    rulingNum = _rulingNum
    modelWidth = _modelWidth
    modelHeight = _modelHeight
    MapWidth = _MapWidth
    MapHeight = _MapHeight
    img = _img

    cdef float step = modelWidth/(rulingNum + 1)
    cdef np.ndarray xl  = np.linspace(step, modelWidth - step, rulingNum)
    cdef np.ndarray xl  = np.linspace(step, modelWidth - step, rulingNum)
    cdef np.ndarray x = np.concatenate([xl, xr],0)

    dx_t = ds.MapWidth/ds.modelWidth

    #res = minimize(optimization, x, args = img, method='trust-exact', jac=Func_Der, hess=Func_Hess, options={'gtol': 1e-8, 'disp': True})

    return

