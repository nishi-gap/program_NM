import numpy as np
from scipy.optimize import minimize, Bounds

from src.pylib import DevSrf

##標準偏差により評価
##各ベクトルのとる範囲が-1から1のため十分小さくなれば
def getSD(x: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, dx:float):
    f = 0.0

    for i in range(ds.rulingNum + 1):
        slicedImage = np.empty((3,),dtype=float)

        if i == 0:                
            for y in range(0, ds.MapHeight):
                t =  round(y * (x[i] - x[i + ds.rulingNum])/ ds.modelHeight + x[i] * dx) 
                print(x[i + ds.rulingNum])             
                for x in range(0, round(y * (x[i] - x[i + ds.rulingNum])/ ds.modelHeight + x[i] * dx)):
                    if(0 <= x and x < ds.MapWidth):
                        slicedImage = np.append(slicedImage,img[y][x],axis = 0)
        elif i == ds.rulingNum:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (x[i - 1] - x[i + ds.rulingNum - 1])/ ds.modelHeight + x[i - 1] * dx), ds.MapWidth):
                    if(0 <= x and x < ds.MapWidth):
                        slicedImage = np.append(slicedImage,img[y][x],axis = 0)
        else:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (x[i - 1] - x[i + ds.rulingNum - 1])/ ds.modelHeight + x[i + ds.rulingNum - 1] * dx),
                 round(y * (x[i] - x[i + ds.rulingNum])/ ds.modelHeight + x[i] * dx)):
                    if(0 < x and x < ds.MapWidth):
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
       
        print("f = ", f)
    return f

eps = 1e-3 #仮置き
def Func_Der(ds: DevSrf.DevSrf,img:np.array):
    f_der = np.zeros(ds.rulingNum * 2)
    der_a, der_b = ds
    
    for i in range(ds.rulingNum):
        f1, f2 = 0.0
        der_a.xl[i] += eps
        f1 = getSD(der_a, img)
        der_a.xl[i] -= 2 * eps
        f2 = getSD(der_a, img)
        der_a.xl[i] += eps
        f_der[i] = (f1 - f2) / (2 * eps)
        
    for i in range(ds.rulingNum):
        f1, f2 = 0.0
        der_b.xr[i] += eps
        f1 = getSD(der_b, img)
        der_b.xr[i] -= 2 * eps
        f2 = getSD(der_b, img)
        f_der[i + ds.rulingNum] = (f1 - f2) / (2 * eps)

    return f_der

"""
f1 = f(x+h,y+h), f2 = f(x+h,y-h), f3 = f(x-h,y+h), f4 = f(x-h,y+h)
df/dxdy = (f1 + f3 - f4 - f2)/h^2
"""
eps_list = [1e-3, -1e-3,]
def diff(i:int, j:int, ds:DevSrf.DevSrf, img:np.array):
    f = np.zeros(4)
    der = ds
    for n in range(2):
        for m in range(2):
            if(i < ds.rulingNum):
                der.xl[i] += eps_list[n]
                if(j < ds.rulingNum):
                    der.xl[j] += eps_list[m]
                else:
                    der.xr[j] += eps_list[m]
            else:
                der.xr[i] += eps_list[n]
                if(j < ds.rulingNum):
                    der.xl[j] += eps_list[m]
                else:
                    der.xr[j] += eps_list[m]
            f[2*m + n] = getSD(der,img)
    return (f[0] - f[1] + f[2] - f[3])/(eps * eps)

def Func_Hess(ds: DevSrf.DevSrf,img:np.array):
    f1, f2, f3, f4 = 0.0
    der = ds
    H = np.zeros((ds.rulingNum * 2, ds.rulingNum * 2, ))
    for i in range(2 * ds.rulingNum):
        for j in range(2 * ds.rulingNum):
            H[i][j] = diff(i, j, ds, img)
    return H

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
def optimization(ds: DevSrf.DevSrf,img:np.array):
    return getSD(ds, img)

def setRuling(ds:DevSrf.DevSrf, img: np.array):  
    dx = ds.MapWidth/ds.modelWidth

    step = ds.modelWidth/(ds.rulingNum + 1)
    xl  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    xr  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    x = np.concatenate([xl, xr],0)
    
    #最適化
    #res = minimize(optimization, x, method='trust-exact', jac=Func_Der, hess=Func_Hess, options={'gtol': 1e-8, 'disp': True})\
    getSD(x, ds, img, dx)

    return
