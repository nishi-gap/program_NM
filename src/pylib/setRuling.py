import numpy as np
from scipy.optimize import minimize, LinearConstraint

from src.pylib import DevSrf

##標準偏差により評価
##各ベクトルのとる範囲が-1から1のため十分小さくなれば
def getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, dx:float):
    f = 0.0
    for i in range(ds.rulingNum + 1):
        slicedImage = np.empty((3,),dtype=float)
        r = g = b = 0.0
        cnt = 0
        if i == 0:                
            for y in range(0, ds.MapHeight):            
                for x in range(0, round(y * (p[i] - p[i + ds.rulingNum])/ ds.modelHeight + p[i] * dx)):
                    if(0 <= x and x < ds.MapWidth):
                        r += img[x][y][0]
                        g += img[x][y][1]
                        b += img[x][y][2]
                        cnt += 1
            r /= cnt
            g /= cnt
            b /= cnt
            Vave = np.array([r,g,b])
            for y in range(0, ds.MapHeight):            
                for x in range(0, round(y * (p[i] - p[i + ds.rulingNum])/ ds.modelHeight + p[i] * dx)):
                    if(0 <= x and x < ds.MapWidth):
                        f += 1 - np.dot(Vave, img[x][y])
        elif i == ds.rulingNum:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i - 1] * dx), ds.MapWidth):
                    if(0 <= x and x < ds.MapWidth):
                        r += img[x][y][0]
                        g += img[x][y][1]
                        b += img[x][y][2]
                        cnt += 1
            r /= cnt
            g /= cnt
            b /= cnt
            Vave = np.array([r,g,b])
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i - 1] * dx), ds.MapWidth):
                    if(0 <= x and x < ds.MapWidth):
                        f += 1 - np.dot(Vave, img[x][y])
        else:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i + ds.rulingNum - 1] * dx),
                 round(y * (p[i] - p[i + ds.rulingNum])/ ds.modelHeight + p[i] * dx)):
                    if(0 < x and x < ds.MapWidth):
                        r += img[x][y][0]
                        g += img[x][y][1]
                        b += img[x][y][2]
                        cnt +=1 
            r /= cnt
            g /= cnt
            b /= cnt
            Vave = np.array([r,g,b])
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i + ds.rulingNum - 1] * dx),
                 round(y * (p[i] - p[i + ds.rulingNum])/ ds.modelHeight + p[i] * dx)):
                    if(0 < x and x < ds.MapWidth):
                        f += 1 - np.dot(Vave, img[x][y])
    print("f",f)
    return f

eps = 1e-2 #仮置き
def Func_Der(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, dx:float):
    f_der = np.zeros(p.size)
    x = p
    for i in range(p.size):
        x[i] += eps
        f1 = getSD(x,ds,img,dx)
        x[i] -= 2*eps
        f2 = getSD(x,ds,img,dx)
        x[i] += eps
        f_der[i] = (f1 - f2)/(2 * eps)

    return f_der

"""
f1 = f(i+h,j+h), f2 = f(i+h,j-h), f3 = f(i-h,j+h), f4 = f(i-h,j+h)
df/dxdy = (f1 + f3 - f4 - f2)/h^2
"""
eps_list = [eps, -eps,]
def diff(i:int, j:int, p:np.ndarray, ds:DevSrf.DevSrf, img:np.array, dx:float):
    f = np.zeros(4)
    for n in range(2):
        for m in range(2):
            p[i] += eps_list[n]
            p[j] += eps_list[m]
            f[2 * n + m] = getSD(p,ds,img,dx)
    return (f[0] - f[2] + f[3] - f[1])/(eps * eps)

def Func_Hess(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, dx:float):
    H = np.zeros((p.size, p.size, ))
    for i in range(p.size):
        for j in range(p.size):
            H[i][j] = diff(i, j, p, ds, img, dx)
    return H

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
#https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
def optimization(x:np.ndarray, ds: DevSrf.DevSrf,img:np.array, dx:float):
    return getSD(x,ds,img, dx)

#n　rulingの数
#A　パラメータの関係を表す行列(m * n) mは制約の数？->　2 * (n - 1)
#lb, ub　下限、上限を表すベクトル ... 下限= -∞ 上限= 0
## xL,i-1 < xL,i  && xR,i-1 < xR,i　の制約より
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html
def setLinearConstrait(n:int):
    A = np.zeros((2 * (n - 1), 2 * n))
    lb = np.full(2 * (n - 1), -np.inf)
    lu = np.zeros(2 * (n - 1))

    m = 0
    while m < 2 * (n - 1):
        l = m
        if m >= n - 1:
            l += 1
        A[m][l] = 1
        A[m][l + 1] = -1
        m += 1

    return A, lb, lu

def setRuling(ds:DevSrf.DevSrf, img: np.array):  
    dx = ds.MapWidth/ds.modelWidth
    step = ds.modelWidth/(ds.rulingNum + 1)
    xl  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    xr  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    p = np.concatenate([xl, xr],0)
    A, lb, lu = setLinearConstrait(ds.rulingNum)
    linearConstrait = LinearConstraint(A, lb, lu)

    #最適化 <-パラメータの与え方と制約を与えればとりあえずは動くはず
    res = minimize(optimization, p, args = (ds,img, dx), method='trust-constr', 
    jac=Func_Der, hess=Func_Hess, constraints = linearConstrait, options={'gtol': 1e-2, 'disp': True})
    
    return res
