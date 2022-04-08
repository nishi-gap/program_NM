import numpy as np
from scipy.optimize import minimize, LinearConstraint
from numpy import linalg as LA
from src.pylib import DevSrf
from src.pylib import DrawRuling

eps = 1e-2 #仮置き
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
        Ism += (1 - xf) * (img[xi + int(ratio[0] * xstep)][yi] - img[xi][yi])
        if yi + int(ratio[1] * ystep) < Y:
            Ism += (1 - yf) * (img[xi][yi + int(ratio[1] * ystep)] - img[xi][yi])
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

def Ffair(p:np.ndarray, W:int):
    f = 0.0
    n = int(len(p)/2)
    for i in range(n - 1):
        if p[i] > p[i + 1]: f += (p[i] - p[i + 1])
        if p[i + n] > p[i + n + 1]: f += (p[i + n] > p[i + n + 1])
        if p[i] < 0 and p[i + n] < 0:
            f += abs(p[i] + p[i + n])
        if p[i] > W and p[i + n] > W:
            f += abs(p[i] + p[i + n] - 2 * W)
    return ff * f

def Func_Der(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    if IS_DEBUG_MODE == 1:
        print("called First derivative")
    f_der = np.zeros(p.size)
    x = p
    for i in range(p.size):
        x[i] += eps
        f1 = getSD(x,ds,img,ratio) + Ffair(p, ds.modelWidth)
        x[i] -= 2*eps
        f2 = getSD(x,ds,img,ratio) + Ffair(p, ds.modelWidth)
        x[i] = p[i]
        f_der[i] = (f1 - f2)/(2 * eps)
            
    return f_der

RuledLines = np.empty(0,dtype=float)
def steepDescent(p:np.ndarray, ds:DevSrf.DevSrf,img:np.ndarray,ratio:list,step:float,maxItr:int, tol:float):
    global RuledLines
    RuledLines = np.hstack([RuledLines,p])
    itr = 0
    while itr < maxItr:
        print("iteration %d : "%itr)
        for i in range(ds.rulingNum):
            print("xL %f , xR %f "%(p[i], p[i + ds.rulingNum]))
        der = Func_Der(p,ds,img,ratio)
        p -= step * der
        if LA.norm(der) < tol:
            return
        #各パラメータが範囲を超えた場合に調整
        RuledLines = np.hstack([RuledLines,p])
        itr += 1

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def setRuling(ds:DevSrf.DevSrf, img: np.array):  
    dx = ds.MapWidth/ds.modelWidth
    dy = ds.MapHeight/ds.modelHeight
    ratio = [dx,dy]
    step = ds.modelWidth/(ds.rulingNum + 1)
    x_w = np.linspace(step, ds.modelWidth - step, ds.rulingNum)
    p = np.concatenate([x_w, x_w],0)
    gstep = 0.01
    maxItr = 100
    tol = 0.1
    steepDescent(p,ds,img,ratio,gstep,maxItr,tol)
    DrawRuling.dispResult(RuledLines, ds.modelWidth, ds.modelHeight)
    return 
