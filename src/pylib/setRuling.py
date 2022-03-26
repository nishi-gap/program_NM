import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.misc import derivative

from src.pylib import DevSrf

eps = 1e-2 #仮置き
eps_list = [eps, -eps,]

def getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, dx:float):
    f = 0.0
    for i in range(ds.rulingNum + 1):
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
    return f

#行列を作成して、行列積の解として得られたベクトルの要素の和
def getSD2(p:np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, dx:float, n:list):
    f = 0.0

    for i in n:
        slicedImage = np.empty(0,dtype=float)

        if i == 0:                
            for y in range(0, ds.MapHeight):            
                for x in range(0, round(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.modelHeight + p[i + 1] * dx)):
                    if(0 <= x and x < ds.MapWidth):
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])

        elif i == ds.rulingNum - 1:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 2] - p[i + ds.rulingNum - 2])/ ds.modelHeight + p[i - 2] * dx), ds.MapWidth):
                    if(0 <= x and x < ds.MapWidth):
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])

        else:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i + ds.rulingNum - 1] * dx),
                    round(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.modelHeight + p[i + 1] * dx)):
                    if(0 <= x and x < ds.MapWidth):
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])

        Vave = slicedImage.mean(axis = 0)
        val = np.ones(len(slicedImage)) - np.dot(slicedImage, Vave)
        f += sum(val)
    print("f ",f)
    return f

def cb_getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, dx:float, l:list):
    if len(l) != 0:
        return  getSD2(p,ds,img,dx,l)
    else:
        return getSD(p,ds,img,dx)

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.derivative.html
#https://github.com/tttamaki/lecture_code/blob/main/MachineLearningMath/differentiation_scipy.ipynb
#https://home.hirosaki-u.ac.jp/jupyter/python-sk/
#x:微分するパラメータ, eps:微小区間, ndim:何階微分
def grad(x:float, ndim:int, p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, dx:float):
    f = derivative(func = getSD, x0 = x, args = (p,ds,img,dx), dx = eps, n = ndim)
    return f

def Func_Der(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, dx:float):
    print("called First derivative")
    f_der = np.zeros(p.size)
    if 0:
        #scipyの数値微分を使ったversion
        for i in range(p.size):
            f_der[i] = grad(p[i], 1, p, ds, img, dx)
        return f_der
    else:
        x = p
        for i in range(p.size):
            x[i] += eps
            f1 = cb_getSD(x,ds,img,dx, [i % ds.rulingNum])
            x[i] -= 2*eps
            f2 = cb_getSD(x,ds,img,dx, [i % ds.rulingNum])
            x[i] += eps
            f_der[i] = (f1 - f2)/(2 * eps)

    return f_der

#f1 = f(i+h,j+h), f2 = f(i+h,j-h), f3 = f(i-h,j+h), f4 = f(i-h,j+h)
#df/dxdy = (f1 + f3 - f4 - f2)/h^2
def diff(i:int, j:int, p:np.ndarray, ds:DevSrf.DevSrf, img:np.array, dx:float):
    f = np.zeros(4)
    for n in range(2):
        for m in range(2):
            p[i] += eps_list[n]
            p[j] += eps_list[m]
            f[2 * n + m] = cb_getSD(p,ds,img,dx, [i % ds.rulingNum,j % ds.rulingNum])
    return (f[0] - f[2] + f[3] - f[1])/(eps * eps)


#ヘッセ行列の2階微分のやり方間違っているかも
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.html#scipy.optimize.HessianUpdateStrategy
#scipy.optimize.HessianUpdateStrategy　使えないか試してみる
def Func_Hess(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, dx:float):
    print("called Hessian")
    H = np.zeros((p.size, p.size, ))
    for i in range(p.size):
        for j in range(i, p.size):
            H[i][j] = diff(i, j, p, ds, img, dx)
            H[j][i] = H[i][j] #ヘッセ行列は対角行列であるため
    return H

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
#https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
def optimization(x:np.ndarray, ds: DevSrf.DevSrf,img:np.array, dx:float):
    f = cb_getSD(x,ds,img, dx, [])
    print("called optimization ", f)
    return f

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

def set_SLSQP_ineq(p:np.ndarray, rulingNum:int):
    cons = np.zeros(p.size - 2)
    jac_cons, lb, lu = setLinearConstrait(rulingNum)
    print("size ", cons.shape, jac_cons.shape)
    if rulingNum != 1:
        for i in range(rulingNum - 1):
            cons[i] = p[i] - p[i + 1]
            cons[i + rulingNum - 1] = p[i + rulingNum  - 1] - p[i + rulingNum]

    return cons, jac_cons

def cb_optimization(x:np.ndarray, state):

    print("callback")
    print("------------------")
    print("iteration : ", state.nit)
    print("parametor x :", state.x)
    print("------------------")

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def setRuling(ds:DevSrf.DevSrf, img: np.array):  
    dx = ds.MapWidth/ds.modelWidth
    step = ds.modelWidth/(ds.rulingNum + 1)
    xl  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    xr  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    p = np.concatenate([xl, xr],0)
    
    #最適化 <-パラメータの与え方と制約を与えればとりあえずは動くはず
    #A, lb, lu = setLinearConstrait(ds.rulingNum)
    #linearConstrait = LinearConstraint(A, lb, lu)
    #res = minimize(optimization, p, args = (ds,img, dx), method='trust-constr', 
    #jac=Func_Der, hess=Func_Hess, constraints = linearConstrait, callback= cb_optimization, options={'gtol': 1e-2, 'disp': True})

    SLSQP_fun, SLSQP_jac = set_SLSQP_ineq(p,ds.rulingNum)
    cons_SLSQP = ({'type':'ineq', 'fun':SLSQP_fun, 'jac':SLSQP_jac})
    res = minimize(optimization, x0 = p, args = (ds, img, dx), 
    method = 'SLSQP', jac = Func_Der, 
     constraints = cons_SLSQP, callback = cb_optimization, options = {'gtol':1e-2, 'disp':True, 'eps':eps, 'maxiter':50})
    print("result")
    print("Is success : ", res.key)
    print("parameter : ", res.x)

    return res
