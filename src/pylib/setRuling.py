import numpy as np
from scipy.optimize import minimize, LinearConstraint
from numpy import linalg as LA
from src.pylib import DevSrf
from src.pylib import DrawRuling

eps = 1e-1 #仮置き
eps_list = [eps, -eps,]

xstep = 0.1
ystep = 0.1

IS_DEBUG_MODE = 0

def Smoothing(img:np.ndarray,x:float,y:float, X:int,Y:int):
    
    xi = int(x)
    yi = int(y)
    xf = x - xi
    yf = y - yi
    Ism = (xf + yf) * img[xi][yi]
    if x < X:
        Ism += (1 - xf) * img[xi + 1][yi]
        if y < Y:
            Ism += (1 - yf) * img[xi][yi + 1]
    Ism = 2 * Ism - 1
    Ism /= LA.norm(Ism)
    Ism = (Ism + 1)/2
    return Ism

def C2V(img:np.ndarray):
    img = 2 *img - 1
    v /= LA.norm(img)
    return v
#やること
#readEXRImageのC2Vを一旦無くす→Smoothingの処理が法線ベクトルとしてではなく色でやったほうがよい？可能性がある（両方試してみて結果の比較）
#C2Vを無くした場合→内積を求める前に色から法線への変換処理を追記
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
                    im = C2V(Smoothing(img,x,y,ds.MapWidth,ds.MapHeight))
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
                    im = C2V(Smoothing(img,x,y,ds.MapWidth,ds.MapHeight))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += ratio[0] * xstep
                y += ratio[1] * ystep

        else:
            while y < yt:
                x = max(0,y * (p[i - 1] - p[i + ds.rulingNum - 1])/ds.MapHeight + p[i + ds.rulingNum - 1] * ratio[0])
                xr = min(y * (p[i] - p[i + ds.rulingNum])/ds.MapHeight + p[i] * ratio[0],ds.MapWidth)
                while x < xr:
                    im = C2V(Smoothing(img,x,y,ds.MapWidth,ds.MapHeight))
                    if(len(slicedImage) == 0):
                        slicedImage = np.hstack([slicedImage,im])
                    else:
                        slicedImage = np.vstack([slicedImage,im])
                    x += ratio[0] * xstep
                y += ratio[1] * ystep
                        

        Vave = slicedImage.mean(axis = 0)
        val = np.ones(len(slicedImage)) - np.dot(slicedImage, Vave)
        f += sum(val)
    return f

#行列を作成して、行列積の解として得られたベクトルの要素の和
def getSD2(p:np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, ratio:list, n:list):
    f = 0.0

    for i in n:
        slicedImage = np.empty(0,dtype=float)

        if i == 0:                
            for y in range(0, ds.MapHeight):            
                for x in range(0, round(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.modelHeight + p[i + 1] * ratio[0])):
                    if(0 <= x and x < ds.MapWidth):
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])

        elif i == ds.rulingNum - 1:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 2] - p[i + ds.rulingNum - 2])/ ds.modelHeight + p[i - 2] * ratio[0]), ds.MapWidth):
                    if(0 <= x and x < ds.MapWidth):
                        if(len(slicedImage) == 0):
                            slicedImage = np.hstack([slicedImage,img[x][y]])
                        else:
                            slicedImage = np.vstack([slicedImage,img[x][y]])

        else:
            for y in range(0, ds.MapHeight):
                for x in range(round(y * (p[i - 1] - p[i + ds.rulingNum - 1])/ ds.modelHeight + p[i + ds.rulingNum - 1] * ratio[0]),
                    round(y * (p[i + 1] - p[i + ds.rulingNum + 1])/ ds.modelHeight + p[i + 1] * ratio[0])):
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

def cb_getSD(p: np.ndarray, ds:DevSrf.DevSrf, img:np.ndarray, ratio:list, l:list):
    return getSD(p,ds,img,ratio)

def Func_Der(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    if IS_DEBUG_MODE == 1:
        print("called First derivative")
    f_der = np.zeros(p.size)
    x = p
    for i in range(p.size):
        x[i] += eps
        f1 = cb_getSD(x,ds,img,ratio, [i % ds.rulingNum])
        x[i] -= 2*eps
        f2 = cb_getSD(x,ds,img,ratio, [i % ds.rulingNum])
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
            f[2 * n + m] = cb_getSD(p,ds,img,ratio, [i % ds.rulingNum,j % ds.rulingNum])
    return (f[0] - f[2] + f[3] - f[1])/(eps * eps)

#ヘッセ行列の2階微分のやり方間違っているかも
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.html#scipy.optimize.HessianUpdateStrategy
#scipy.optimize.HessianUpdateStrategy　使えないか試してみる
def Func_Hess(p:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    print("called Hessian")
    H = np.zeros((p.size, p.size, ))
    for i in range(p.size):
        for j in range(i, p.size):
            H[i][j] = diff(i, j, p, ds, img, ratio)
            H[j][i] = H[i][j] #ヘッセ行列は対角行列であるため
    return H

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
#https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
def optimization(x:np.ndarray, ds: DevSrf.DevSrf,img:np.array, ratio:list):
    f = cb_getSD(x,ds,img, ratio, [])
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
    for i in range(int(x.size/4)):
        #r.append([[x[i], x[i + int(x.size/2)]],[x[i + int(x.size/4)], x[i + 3 * int(x.size/4)]]])  
        print(i, " :  xL{%f, %f}, xR{%f, %f}" %(x[i], x[i + int(x.size/2)], x[i + int(x.size/4)], x[i + 3 * int(x.size/4)]))
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

    #最適化 <-パラメータの与え方と制約を与えればとりあえずは動くはず
    A, lb, lu = setLinearConstrait(ds.rulingNum)
    linearConstrait = LinearConstraint(A, lb, lu)
    res = minimize(optimization, p, args = (ds,img, ratio), method='trust-constr', 
    jac=Func_Der, hess=Func_Hess, constraints = linearConstrait, callback= cb_optimization, options={'gtol': 1e-2, 'disp': True})
    """
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
    
    maxiter = 20
    res = minimize(optimization, x0 = p, args = (ds, img, dx), method = 'SLSQP', jac = Func_Der, 
    constraints = cons, bounds = bnds, callback = cb_optimization,
     options = {'gtol':1e-1, 'disp':True, 'eps':eps, 'maxiter':maxiter})
    
    #print(res)
    for i in range(ds.rulingNum):
        print(i, " :  xL{%f, %f}, xR{%f, %f}" %(res.x[i], res.x[i + 2 * ds.rulingNum], res.x[i + ds.rulingNum], res.x[i + 3 * ds.rulingNum]))
    print("===========================")

    """
    DrawRuling.dispResult(RuledLines)
    return RuledLines
     #最適化で折り線がどう動いているか見れるようにする
