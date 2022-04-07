import numpy as np
from scipy.optimize import minimize, LinearConstraint
from numpy import linalg as LA
from src.pylib import DevSrf
from src.pylib import DrawRuling

eps = 1e-2 #仮置き
eps_list = [eps, -eps,]

xstep = 0.1
ystep = 0.1
ff = 1e+6

IS_DEBUG_MODE = 0

def Smoothing(img:np.ndarray,x:float,y:float, X:int,Y:int, ratio:list): 
    xi = int(x)
    yi = int(y)
    xf = x - xi
    yf = y - yi
    Ism = img[xi][yi]
    if xi + int(ratio[0] * xstep)< X:
        Ism += (1 - xf) * img[xi + int(ratio[0] * xstep)][yi]
        if yi + int(ratio[1] * ystep) < Y:
            Ism += (1 - yf) * img[xi][yi + int(ratio[1] * ystep)]
    Ism = 2 * Ism - 1
    Ism /= LA.norm(Ism)
    Ism = (Ism + 1)/2
    return Ism

def C2V(img:np.ndarray):
    img = 2 *img - 1
    v = img /LA.norm(img)
    return v

def getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, ratio:list):
    f = 0.0

    for i in range(ds.rulingNum + 1):
        slicedImage = np.empty(0,dtype=float)
        y = 0
        yt = ds.MapHeight
        if i == 0: 
            while y < yt:
                x = 0
                xr = min(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.MapHeight + p[i + 1] * ratio[0], ds.MapWidth)   
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight, ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += ratio[0] * xstep
                y += ratio[1] * ystep

        elif i == ds.rulingNum:
            while y < yt:
                x = max(0,y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.MapHeight + p[i - 1] * ratio[0])
                xr = ds.MapWidth
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight, ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += ratio[0] * xstep
                y += ratio[1] * ystep

        else:
            while y < yt:
                x = min(max(0,y * (p[i - 1] - p[i + ds.rulingNum - 1])/ds.MapHeight + p[i + ds.rulingNum - 1] * ratio[0]),ds.MapWidth)
                xr = min(max(0,y * (p[i] - p[i + ds.rulingNum])/ds.MapHeight + p[i] * ratio[0],0),ds.MapWidth)
                if x > xr: x, xr = xr, x
                while x < xr:
                    im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight,ratio))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += ratio[0] * xstep
                y += ratio[1] * ystep
                        
        Vave = slicedImage.mean(axis = 0)
        f1 = 0.0
        f2 = 0.0
        for v in slicedImage:
            f1 += LA.norm(v - Vave)
        #f2 = sum(np.ones(len(slicedImage)) - np.dot(slicedImage, Vave))
        #print(i,f1,f2)
        f += f1

    return f

def Ffair(p:np.ndarray):
    f = 0.0
    n = int(len(p)/2)
    for i in range(n - 1):
        if p[i] > p[i + 1]: f += (p[i] - p[i + 1])
        if p[i + n] > p[i + n + 1]: f += (p[i + n] > p[i + n + 1])
    return ff * f

def Func_Der(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    if IS_DEBUG_MODE == 1:
        print("called First derivative")
    f_der = np.zeros(p.size)
    x = p
    for i in range(p.size):
        x[i] += eps
        f1 = getSD(x,ds,img,ratio) + Ffair(p)
        x[i] -= 2*eps
        f2 = getSD(x,ds,img,ratio) + Ffair(p)
        x[i] += eps
        f_der[i] = (f1 - f2)/(2 * eps)
                    
    if IS_DEBUG_MODE == 1:
        print(f_der)
            
    return f_der

#f1 = f(i+h,j+h), f2 = f(i+h,j-h), f3 = f(i-h,j+h), f4 = f(i-h,j+h)
#df/dxdy = (f1 + f3 - f4 - f2)/h^2
def diff(i:int, j:int, x:np.ndarray, ds:DevSrf.DevSrf, img:np.array, ratio:list):
    f = np.zeros(4) 
    p = x
    for n in range(2):
        for m in range(2):
            p[i] += eps_list[n]
            p[j] += eps_list[m]
            f[2 * n + m] = getSD(p,ds,img,ratio) + Ffair(p)
    return (f[0] - f[2] + f[3] - f[1])/(eps * eps)

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
#https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
def optimization(x:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    f = getSD(x,ds,img, ratio) + Ffair(x)
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

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def setRuling(ds:DevSrf.DevSrf, img: np.array):  
    IS_DEBUG_MODE = 1 #0->false, 1 = true
    dx = ds.MapWidth/ds.modelWidth
    dy = ds.MapHeight/ds.modelHeight
    ratio = [dx,dy]
    step = ds.modelWidth/(ds.rulingNum + 1)
    
    #最適化するパラメータを一次元(xL ->(), xR->())から二次元(xL->(,), xR->(,))へとする→パラメータ数 4 * rulingNum
    x_w  = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    p = np.concatenate([x_w, x_w],0)

    #SLSQPの制約の与え方
    #https://towardsdatascience.com/optimization-with-scipy-and-application-ideas-to-machine-learning-81d39c7938b8
    #https://teratail.com/questions/181787
    #パラメータが変わったためここも修正必須
    cons = ()
    if ds.rulingNum != 1:
        for j in range(2):
            for i in range(ds.rulingNum - 1):
                cons = cons + ({'type':'ineq', 'fun' : lambda p, n = i + j * ds.rulingNum: (p[n + 1] - p[i])},)
    
    maxiter = 50
    res = minimize(optimization, x0 = p, args = (ds, img, ratio), method = 'BFGS',  
    callback = cb_optimization,options = {'gtol':1e-2, 'disp':True, 'eps':eps, 'maxiter':maxiter})
    
    #print(res)
    for i in range(ds.rulingNum):
        print(i, " :  xL{%f}, xR{%f}" %(res.x[i], res.x[i + ds.rulingNum]))
    print("===========================")


    DrawRuling.dispResult(RuledLines, ds.modelWidth, ds.modelHeight)
    return RuledLines
     #最適化で折り線がどう動いているか見れるようにする
