import numpy as np
from scipy.optimize import minimize, Bounds
from numpy import linalg as LA
from src.pylib import DevSrf
from src.pylib import DrawRuling

eps = 1e-2 #仮置き
eps_list = [eps, -eps,]

xstep = 0.1
ystep = 0.1

IS_DEBUG_MODE = 0

#正則化を外してみている
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
    #Ism = 2 * Ism - 1
    #Ism /= LA.norm(Ism)
    #Ism = (Ism + 1)/2
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
                pr1 = np.array([p[i + 1], p[i + 2 * ds.rulingNum + 1]])
                pr2 = np.array([p[i + ds.rulingNum + 1], p[i + 3 * ds.rulingNum + 1]])
                xr = min((y - pr2[1]) * (pr1[0] - pr2[0])/ (pr1[1] - pr2[1]) + pr2[0], ds.modelWidth) 
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
                pl1 = np.array([p[i - 1], p[i + 2 * ds.rulingNum - 1]])
                pl2 = np.array([p[i + ds.rulingNum - 1], p[i + 3 * ds.rulingNum - 1]])
                x = max(0,(y - pl2[1]) * (pl1[0] - pl2[0])/ (pl1[1] - pl2[1]) + pl2[0])
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
                pl1 = np.array([p[i - 1], p[i + 2 * ds.rulingNum - 1]])
                pl2 = np.array([p[i + ds.rulingNum - 1], p[i + 3 * ds.rulingNum - 1]])
                x = min(max(0,(y - pl2[1]) * (pl1[0] - pl2[0])/(pl1[1] - pl2[1]) + pl2[0]),ds.modelWidth)

                pr1 = np.array([p[i], p[i + 2 * ds.rulingNum]])
                pr2 = np.array([p[i + ds.rulingNum], p[i + 3 * ds.rulingNum]])
                xr = min(max(0,(y - pr2[1]) * (pr1[0] - pr2[1])/(pr1[1] - pr2[1]) + pr2[0],0),ds.modelWidth)
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
                pr1 = np.array([p[i + 1], p[i + 2 * ds.rulingNum + 1]])
                pr2 = np.array([p[i + ds.rulingNum + 1], p[i + 3 * ds.rulingNum + 1]])
                xr = min((y - pr2[1]) * (pr1[0] - pr2[0])/ (pr1[1] - pr2[1]) + pr2[0], ds.modelWidth)   
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
                pl1 = np.array([p[i - 1], p[i + 2 * ds.rulingNum - 1]])
                pl2 = np.array([p[i + ds.rulingNum - 1], p[i + 3 * ds.rulingNum - 1]])
                x = max(0,(y - pl2[1]) * (pl1[0] - pl2[0])/ (pl1[1] - pl2[1]) + pl2[0])
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
                pl1 = np.array([p[i - 1], p[i + 2 * ds.rulingNum - 1]])
                pl2 = np.array([p[i + ds.rulingNum - 1], p[i + 3 * ds.rulingNum - 1]])
                x = min(max(0,(y - pl2[1]) * (pl1[0] - pl2[0])/(pl1[1] - pl2[1]) + pl2[0]),ds.modelWidth)

                pr1 = np.array([p[i], p[i + 2 * ds.rulingNum]])
                pr2 = np.array([p[i + ds.rulingNum], p[i + 3 * ds.rulingNum]])
                xr = min(max(0,(y - pr2[1]) * (pr1[0] - pr2[1])/(pr1[1] - pr2[1]) + pr2[0],0),ds.modelWidth)
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
rulingNum = 0
RuledLines = np.empty(0,dtype=float)
def cb_optimization(x:np.ndarray):
    global IterCnt, RuledLines
    IterCnt += 1
    print("callback")
    print("------------------")
    print("iteration : ", IterCnt)
    print("parametor x :")
    for i in range(rulingNum):
        #r.append([[x[i], x[i + int(x.size/2)]],[x[i + int(x.size/4)], x[i + 3 * int(x.size/4)]]])  
        print(i, " :  xL{%f, %f}, xR{%f, %f}" %(x[i], x[i + 2 * rulingNum], x[i + rulingNum], x[i + 3 * rulingNum]))
    print("------------------")
    if(len(RuledLines) == 0):
        RuledLines = np.hstack([RuledLines,x])
    else: RuledLines = np.vstack([RuledLines,x])

def setCons(p:np.ndarray, ds:DevSrf.DevSrf):
    cons = ()
    for j in range(2):
        for i in range(ds.rulingNum - 1):
            cons = cons + ({'type':'ineq', 'fun' : lambda p, n = i + j * ds.rulingNum: -(p[n] - p[n + 1])},)
    #領域内に点が乗るように制約を与える
    for j in range(2):
        for i in range(ds.rulingNum):
            cons = cons + ({'type':'eq', 'fun' : lambda p, k = j, n = i + j * ds.rulingNum: (p[n] - 0) * (p[n] - ds.modelWidth) * 
            (p[n + 2 * ds.rulingNum] - k * ds.modelHeight)},)
    return cons

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def setRuling(ds:DevSrf.DevSrf, img: np.array):  
    global rulingNum
    rulingNum = ds.rulingNum
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
    #https://docs.scipy.org/doc/scipy/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
    #パラメータが変わったためここも修正必須
    cons = ()
    #交差判定
    for j in range(2):
        for i in range(ds.rulingNum - 1):
            cons = cons + ({'type':'ineq', 'fun' : lambda p, n = i + j * ds.rulingNum: (p[n + 1] - p[n] - eps)},)
    #領域内に点が乗るように制約を与える
    for j in range(2):
        for i in range(ds.rulingNum):
            cons = cons + ({'type':'eq', 'fun' : lambda p, k = j, n = i + j * ds.rulingNum: (p[n] - 0) * (p[n] - ds.modelWidth) * 
            (p[n + 2 * ds.rulingNum] - k * ds.modelHeight)},)

    lb = np.zeros(4 * ds.rulingNum)
    ub = np.full(4 * ds.rulingNum, ds.modelHeight)
    for i in range(2 * ds.rulingNum): 
        ub[i] = (ds.modelWidth)
    bnds = Bounds(lb, ub)
    maxiter = 20
    res = minimize(optimization, x0 = p, args = (ds, img, ratio), method = 'SLSQP', jac = Func_Der, 
     constraints= cons, bounds = bnds, callback = cb_optimization,
     options = {'gtol':1e-1, 'disp':True, 'eps':eps, 'maxiter':maxiter})
    
    #print(res)
    for i in range(ds.rulingNum):
        print(i, " :  xL{%f, %f}, xR{%f, %f}" %(res.x[i], res.x[i + 2 * ds.rulingNum], res.x[i + ds.rulingNum], res.x[i + 3 * ds.rulingNum]))
    print("===========================")   
    DrawRuling.dispResult(RuledLines)
