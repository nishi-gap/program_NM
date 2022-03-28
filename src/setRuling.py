import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.misc import derivative
import sys
import DevSrf

eps = 1 #仮置き
eps_list = [eps, -eps,]

IS_DEBUG_MODE = 0

def SmoothImg(img:np.ndarray, ds:DevSrf.DevSrf, x:float, y:float):
    xi = int(x)
    yi = int(y)
    xd = x - xi
    yd = y - yi
    im = np.zeros(3)
    if xd < sys.float_info.epsilon:
        im = xd * img[xi][yi]
        if 0 < xi < ds.MapWidth - 1:
            im += (1 - xd) * img[xi + 1][yi]
    else:
        im = yd * img[xi][yi]
        if 0 < yi < ds.MapHeight - 1:
            im += (1 - yd) * img[xi][yi + 1]
    return im

def getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, cmpNum:list):
    f = 0.0

    for i in range(ds.rulingNum + 1):
        slicedImage = np.empty(0,dtype=float)

        if i == 0:                
            for y in range(0, ds.MapHeight):   
                x = 0.0
                xr = y * (p[i] - p[i + ds.rulingNum])/ ds.MapHeight + p[i + ds.rulingNum]    
                while x < xr:
                    if(0 <= x and x < ds.MapWidth):
                        im = SmoothImg(img,ds,x,y)
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])
                    x += 1.0

        elif i == ds.rulingNum:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ds.MapHeight + p[i + ds.rulingNum - 1]), ds.MapWidth):
                    if(0 <= x and x < ds.MapWidth):
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])

        else:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ds.MapHeight + p[i + ds.rulingNum - 1]),
                    round(y * (p[i] - p[i + ds.rulingNum])/ds.MapHeight + p[i + ds.rulingNum])):
                    if(0 <= x and x < ds.MapWidth):
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])

        Vave = slicedImage.mean(axis = 0)
        val = np.ones(len(slicedImage)) - np.dot(slicedImage, Vave)
        f += sum(val)
    return f

#行列を作成して、行列積の解として得られたベクトルの要素の和
def getSD2(p:np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, n:list, cmpNum:list):
    f = 0.0

    for i in n:
        for j in range(2):
            slicedImage = np.empty(0,dtype=float)

            if i == 0 and j == 0: 
                for y in range(0, ds.MapHeight):            
                    for x in range(0, round(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.MapHeight + p[i + ds.rulingNum + 1])):
                        if(0 <= x and x < ds.MapWidth):
                            if(len(slicedImage) == 0):
                                slicedImage = np.hstack([slicedImage,img[x][y]])
                            else:
                                slicedImage = np.vstack([slicedImage,img[x][y]])

            elif i == ds.rulingNum - 1 and j == 1:
                for y in range(0, ds.MapHeight):
                    for x in range(round(y * (p[i - 2] - p[i + ds.rulingNum - 2])/ ds.modelHeight + p[i - 2]), ds.MapWidth):
                        if(0 <= x and x < ds.MapWidth):
                            if(len(slicedImage) == 0):
                                slicedImage = np.hstack([slicedImage,img[x][y]])
                            else:
                                slicedImage = np.vstack([slicedImage,img[x][y]])

            else:
                for y in range(0, ds.MapHeight):
                    for x in range(round(y * (p[i - j + 1] - p[i - j + 1 + ds.rulingNum])/ ds.modelHeight + p[i - (j - 1) + ds.rulingNum]),
                        round(y * (p[i + j] - p[i + j + ds.rulingNum])/ ds.modelHeight + p[i + j])):
                        if(0 <= x and x < ds.MapWidth):
                            if(len(slicedImage) == 0):
                                slicedImage = np.hstack([slicedImage,img[x][y]])
                            else:
                                slicedImage = np.vstack([slicedImage,img[x][y]])

            Vave = slicedImage.mean(axis = 0)
            val = np.ones(len(slicedImage)) - np.dot(slicedImage, Vave)
            f += sum(val)
    #print("f ",f)
    return f

def cb_getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, l:list, cmpNum:list):
    if len(l) != 0:
        return  getSD2(p,ds,img,l, cmpNum)
    else:
        return getSD(p,ds,img, cmpNum)

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.derivative.html
#https://github.com/tttamaki/lecture_code/blob/main/MachineLearningMath/differentiation_scipy.ipynb
#https://home.hirosaki-u.ac.jp/jupyter/python-sk/
#x:微分するパラメータ, eps:微小区間, ndim:何階微分
def grad(x:float, ndim:int, p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray):
    f = derivative(func = getSD, x0 = x, args = (p,ds,img), n = ndim)
    return f

def Func_Der(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, cmpNum:list):
    if IS_DEBUG_MODE == 1:
        print("called First derivative")
    f_der = np.zeros(p.size)
    if 0:
        #scipyの数値微分を使ったversion
        for i in range(p.size):
            f_der[i] = grad(p[i], 1, p, ds, img)
        return f_der
    else:
        x = p
        for i in range(p.size):
            x[i] += eps
            f1 = cb_getSD(x,ds,img, [i % ds.rulingNum], cmpNum)
            x[i] -= 2*eps
            f2 = cb_getSD(x,ds,img, [i % ds.rulingNum], cmpNum)
            x[i] += eps
            f_der[i] = (f1 - f2)/(2 * eps)
                    
    if IS_DEBUG_MODE == 1:
        print(f_der)
            
    return f_der

#f1 = f(i+h,j+h), f2 = f(i+h,j-h), f3 = f(i-h,j+h), f4 = f(i-h,j+h)
#df/dxdy = (f1 + f3 - f4 - f2)/h^2
def diff(i:int, j:int, p:np.ndarray, ds:DevSrf.DevSrf, img:np.array):
    f = np.zeros(4)
    for n in range(2):
        for m in range(2):
            p[i] += eps_list[n]
            p[j] += eps_list[m]
            f[2 * n + m] = cb_getSD(p,ds,img, [i % ds.rulingNum,j % ds.rulingNum])
    return (f[0] - f[2] + f[3] - f[1])/(eps * eps)

#ヘッセ行列の2階微分のやり方間違っているかも
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.html#scipy.optimize.HessianUpdateStrategy
#scipy.optimize.HessianUpdateStrategy　使えないか試してみる
def Func_Hess(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array):
    print("called Hessian")
    H = np.zeros((p.size, p.size, ))
    for i in range(p.size):
        for j in range(i, p.size):
            H[i][j] = diff(i, j, p, ds, img)
            H[j][i] = H[i][j] #ヘッセ行列は対角行列であるため
    return H

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
#https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
def optimization(x:np.ndarray, ds: DevSrf.DevSrf,img:np.array, cmpNum:list):
    f = cb_getSD(x,ds,img, cmpNum, [])
    if IS_DEBUG_MODE == 1:
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

IterCnt = 0
def cb_optimization(x:np.ndarray):
    global IterCnt
    IterCnt += 1
    print("callback")
    print("------------------")
    print("iteration : ", IterCnt)
    print("parametor x :")
    for i in range(int(x.size/4)):
        print(i, " :  xL{%f}, xR{%f}" %(x[i], x[i + int(x.size/2)]))
    print("------------------")

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def setRuling(ds:DevSrf.DevSrf, img: np.array):  

    global IS_DEBUG_MODE
    IS_DEBUG_MODE = 0 #0->false, 1 = true

    step = ds.MapWidth/(ds.rulingNum + 1)
    cmpNum = [ds.modelWidth * 10, ds.modelHeight * 10]
    #最適化するパラメータを一次元(xL ->(), xR->())から二次元(xL->(,), xR->(,))へとする→パラメータ数 4 * rulingNum
    x_w  = np.linspace(step, ds.MapWidth - step, ds.rulingNum)
    #xl_h = np.full(ds.rulingNum, ds.modelHeight)
    #xr_h = np.zeros(ds.rulingNum)
    p = np.concatenate([x_w, x_w],0)

    #最適化 <-パラメータの与え方と制約を与えればとりあえずは動くはず
    #A, lb, lu = setLinearConstrait(ds.rulingNum)
    #linearConstrait = LinearConstraint(A, lb, lu)
    #res = minimize(optimization, p, args = (ds,img, dx), method='trust-constr', 
    #jac=Func_Der, hess=Func_Hess, constraints = linearConstrait, callback= cb_optimization, options={'gtol': 1e-2, 'disp': True})

    #SLSQPの制約の与え方
    #https://towardsdatascience.com/optimization-with-scipy-and-application-ideas-to-machine-learning-81d39c7938b8
    #https://teratail.com/questions/181787
    #パラメータが変わったためここも修正必須
    cons = ()
    if ds.rulingNum != 1:
        for j in range(2):
            for i in range(ds.rulingNum - 1):
                cons = cons + ({'type':'ineq', 'fun' : lambda p, n = i + j * ds.rulingNum: (p[n + 1] - p[i])},)
    
    bnds = ()
    for i in range(4):
        for j in range(ds.rulingNum):
            if i < 2:
                bnds = bnds + ((0,ds.modelWidth),)
            else:
                bnds = bnds + ((0,ds.modelHeight),)
    
    res = minimize(optimization, x0 = p, args = (ds, img, cmpNum), method = 'COBYLA', jac = Func_Der, 
    constraints = cons, callback = cb_optimization,
     options = {'gtol':1e-1, 'disp':True, 'eps':eps, 'maxiter':100})

    #print(res)
    for i in range(ds.rulingNum):
        print(i, " :  xL{%f}, xR{%f}" %(res.x[i], res.x[i + ds.rulingNum]))
    print("===========================")
    return res
