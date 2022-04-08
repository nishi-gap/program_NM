import numpy as np
from scipy.optimize import minimize, LinearConstraint
from numpy import linalg as LA
from src.pylib import DevSrf
from src.pylib import DrawRuling

eps = 1e-2 #仮置き
xstep = 0.1
ystep = 0.1
ff = 1e+3

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
    v = img /LA.norm(img)
    return v

def calRegion(X:list,Y:list, ds:DevSrf.DevSrf, img:np.ndarray, ratio:list):
    y = y[0]
    x = X[0]
    Image = np.empty(0,dtype=float)
    while y < Y[1]: 
        while x < X[1]:
            im = (Smoothing(img,x,y,ds.MapWidth,ds.MapHeight, ratio))
            if(len(Image) == 0):
                Image = np.hstack([Image,im])
            else:
                Image = np.vstack([Image,im])
            x += xstep
        y += ystep
    return Image

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
                    im = (Smoothing(img,x,y,ds.MaplWidth,ds.MapHeight,ratio))
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
        #f2 = sum(np.ones(len(slicedImage)) - np.dot(slicedImage, Vave))
        #print(i,f1,f2)
        f += f1

    return f

def getSD2(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, ratio:list, i:int):
    f = 0.0
    slicedImage = np.empty(0,dtype=float)
    y = 0
    yt = ds.modelHeight
    if i == 0: 
        while y < yt:
            x = 0
            xr = min(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.MapHeight + p[i + 1], ds.modelWidth)   
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
        if p[i + n] > p[i + n + 1]: f += (p[i + n] - p[i + n + 1])

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
    gstep = 1e-3
    maxItr = 100
    tol = 0.1
    steepDescent(p,ds,img,ratio,gstep,maxItr,tol)
    DrawRuling.dispResult(RuledLines, ds.modelWidth, ds.modelHeight)
    return 
