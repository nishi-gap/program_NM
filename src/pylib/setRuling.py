import numpy as np
from scipy.optimize import minimize, LinearConstraint
from numpy import linalg as LA
from src.pylib import DevSrf
from src.pylib import DrawRuling

eps = 1e-2 #仮置き
eps_list = [eps, -eps,]

xstep = 0.1
ystep = 0.1

IS_DEBUG_MODE = 0

def Smoothing(img:np.ndarray,x:float,y:float, X:int,Y:int, ratio:list):
    xi = int(x * ratio[0])
    yi = int(y * ratio[1])
    xf = x - int(x)
    yf = y - int(y)
    Ism = img[xi][yi]
    if xi + int(ratio[0] * xstep)< X:
        Ism += xf * (img[xi + int(ratio[0] * xstep)][yi] - img[xi][yi])
        if yi + int(ratio[1] * ystep) < Y:
            Ism += yf * (img[xi][yi + int(ratio[1] * ystep)] - img[xi][yi])
    Ism = 2 * Ism - 1
    Ism /= LA.norm(Ism)
    Ism = (Ism + 1)/2
    return Ism

def C2V(img:np.ndarray):
    img = 2 *img - 1
    v /= LA.norm(img)
    return v

def getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, ratio:list):
    f = 0.0

    for i in range(ds.rulingNum + 1):
        slicedImage = np.empty(0,dtype=float)
        y = 0
        yt = ds.modelHeight
        if i == 0: 
            while y < yt:
                x = 0
                xr = min(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.modelHeight + p[i + 1], ds.modelWidth)   
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight,ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += xstep
                y += ystep
        elif i == ds.rulingNum:
            while y < yt:
                x = max(0,y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i - 1])
                xr = ds.modelWidth
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight,ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += xstep
                y += ystep
        else:
            while y < yt:
                x = max(0, y * (p[i - 1] - p[i + ds.rulingNum - 1])/ds.modelHeight + p[i + ds.rulingNum - 1])
                xr = min(y * (p[i] - p[i + ds.rulingNum])/ds.modelHeight + p[i],ds.modelWidth)
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight,ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += xstep
                y += ystep
                        
        Vave = slicedImage.mean(axis = 0)
        for v in slicedImage:
            f += LA.norm(v - Vave)

    return f

def getSD2(p:np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, ratio:list, n:list):
    f = 0.0
    slicedImage = np.empty(0,dtype=float)
    i = n
    while i < n + 2:
        slicedImage = np.empty(0,dtype=float)
        y = 0
        yt = ds.modelHeight
        if i == 0: 
            while y < yt:
                x = 0
                xr = min(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.modelHeight + p[i + 1], ds.modelWidth)   
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight, ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += xstep
                y += ystep

        elif i == ds.rulingNum:
            while y < yt:
                x = max(0,y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i - 1])
                xr = ds.modelWidth
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight, ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += xstep
                y += ystep

        else:
            while y < yt:
                x = min(max(0,y * (p[i - 1] - p[i + ds.rulingNum - 1])/ds.modelHeight + p[i + ds.rulingNum - 1]),ds.modelWidth)
                xr = min(max(0,y * (p[i] - p[i + ds.rulingNum])/ds.modelHeight + p[i],0),ds.modelWidth)
                if x > xr: x, xr = xr, x
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight,ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += xstep
                y += ystep
                        
        Vave = slicedImage.mean(axis = 0)
        f1 = 0.0

        for v in slicedImage:
            f1 += LA.norm(v - Vave)
        f += f1
        i += 1
    return f

def Func_Der(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    f_der = np.zeros(p.size)
    x = p
    ind = 0
    for i in range(p.size):
        ind = i % ds.rulingNum
        x[i] += eps
        f1 = getSD2(x,ds,img,ratio,ind) 
        x[i] = p[i]
        x[i] -= eps
        f2 = getSD2(x,ds,img,ratio,ind)
        x[i] = p[i]
        f_der[i] = (f1 - f2)/(2 * eps)
            
    return f_der

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
#https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
def optimization(x:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    f = getSD(x,ds,img, ratio)
    return f

#https://www.delftstack.com/ja/howto/matplotlib/how-to-automate-plot-updates-in-matplotlib/
#matplotlibを使ってのグラフ更新
IterCnt = 0
RuledLines = np.empty(0,dtype=float)
def cb_optimization(x:np.ndarray):
    global IterCnt, RuledLines
    IterCnt += 1
    print("callback")
    print("------------------")
    print("iteration : ", IterCnt)
    print("parametor x :")
    for i in range(int(x.size/2)):
        #r.append([[x[i], x[i + int(x.size/2)]],[x[i + int(x.size/4)], x[i + 3 * int(x.size/4)]]])  
        print(i, " :  xL{%f}, xR{%f}" %(x[i], x[i + int(x.size/2)]))
    print("------------------")
    if(len(RuledLines) == 0):
        RuledLines = np.hstack([RuledLines,x])
    else: RuledLines = np.vstack([RuledLines,x])

def setCons(p:np.ndarray, ds:DevSrf.DevSrf):
    cons = ()
    if ds.rulingNum != 1:
        for j in range(2):
            for i in range(ds.rulingNum - 1):
                cons = cons + ({'type':'ineq', 'fun' : lambda p, n = i + j * ds.rulingNum: -(p[n] - p[n + 1])},)
    for i in range(2 * ds.rulingNum):
        cons = cons + ({'type':'eq', 'fun' : lambda p, n = i: (p[n] - 0) * (p[n] - ds.modelWidth) * 
        (p[n + 2 * ds.rulingNum] - 0) * (p[n + 2 * ds.rulingNum] - ds.modelHeight)},)
    return cons

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def setRuling(ds:DevSrf.DevSrf, img: np.array):  

    dx = ds.MapWidth/ds.modelWidth
    dy = ds.MapHeight/ds.modelHeight
    ratio = [dx,dy]
    step = ds.modelWidth/(ds.rulingNum + 1)
    
    #最適化するパラメータを一次元(xL ->(), xR->())から二次元(xL->(,), xR->(,))へとする→パラメータ数 4 * rulingNum
    x_w  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    x_h = np.full(ds.rulingNum, ds.modelHeight)
    p = np.concatenate([x_w, x_w, x_h, np.zeros(ds.rulingNum)],0)

    #SLSQPの制約の与え方
    #https://towardsdatascience.com/optimization-with-scipy-and-application-ideas-to-machine-learning-81d39c7938b8
    #https://teratail.com/questions/181787
    #パラメータが変わったためここも修正必須
    bnds = ()
    for i in range(2):
        for j in range(ds.rulingNum):
            if i < 2:
                bnds = bnds + ((0,ds.modelWidth),)
            else:
                bnds = bnds + ((0,ds.modelHeight),)

    maxiter = 20
    res = minimize(optimization, x0 = p, args = (ds, img, ratio), method = 'SLSQP', jac = Func_Der, 
    constraints = setCons(p, ds), bounds = bnds, callback = cb_optimization,
     options = {'gtol':1e-1, 'disp':True, 'eps':eps, 'maxiter':maxiter})
    
    #print(res)
    for i in range(ds.rulingNum):
        print(i, " :  xL{%f}, xR{%f}" %(res.x[i], res.x[i + ds.rulingNum]))
    print("===========================")

    
    DrawRuling.dispResult(RuledLines)
    return RuledLines
     #最適化で折り線がどう動いているか見れるようにする
