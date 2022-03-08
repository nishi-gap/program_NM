import numpy as np
import sys
from scipy.optimize import minimize, Bounds

##外部ファイルの読み込み
sys.path.append("DevSrf")
##
import DevSrf

def splitMesh(ds: DevSrf.DevSrf, ind: int):
    1

def extractArea(ds: DevSrf.DevSrf):
    1

##標準偏差により評価
##各ベクトルのとる範囲が-1から1のため十分小さくなれば
def getSD(img: np.array, area:np.array):
    img_l = img #必要なサイズ分スライスするようにしておく
    return np.std(img_l)

eps = 1e-3 #仮置き
def Func_Der(ds: DevSrf.DevSrf,img:np.array):
    f_der = np.zeros(ds.rulingNum * 2)
    der_a, der_b = ds
    
    for i in range(ds.rulingNum):
        f1, f2 = 0.0
        der_a.xl[i] += eps
        area = extractArea(der_a)
        f1 = getSD(img,area)
        der_a.xl[i] -= 2 * eps
        area = extractArea(der_a)
        f2 = getSD(img,area)
        f_der[i] = (f1 - f2) / (2 * eps)

    for i in range(ds.rulingNum):
        f1, f2 = 0.0
        der_b.xr[i] += eps
        area = extractArea(der_a)
        f1 = getSD(img,area)
        der_b.xr[i] -= 2 * eps
        area = extractArea(der_a)
        f2 = getSD(img,area)
        f_der[i + ds.rulingNum] = (f1 - f2) / (2 * eps)

    return f_der

def Func_Hess(ds: DevSrf.DevSrf,img:np.array):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)

#scipyによる最適化
#https://scipy.github.io/devdocs/tutorial/optimize.html
def optimization(ds: DevSrf.DevSrf,img:np.array):
    f = 0.0
    for i in range(ds.rulingNum):
        area = extractArea(ds) #区切るエリアを設定
        f += getSD(img, area) #目的関数

    return f
'''
手順
1. 法線マップNを2つに分割(N_l, N_r) 分割位置の初期状態はNを二等分にできる場所
2. N_l, N_rの各標準偏差の和が最小となるようにオリセンの位置調整
3. N_l, N_rそれぞれの標準偏差の値の差が十分小さい場合は分割終了.それ以外はN_l, N_rそれぞれを新たなNとして1に戻る(先に分割するのはN_l, N_rのうち標準偏差が大きい方)
'''
th = 5 #閾値
cnt = 0
def setRuling(ds: DevSrf.DevSrf, img: np.array, area:np.array, i: int, root:bool):

    """
        if(root == True):
        cnt = 0
        root = False
    if(cnt == ds.rulingNum): return
    """

    #i = math.floor(ds.rulingNum/2) #開始点
    #cnt = 0    

    #最適化
    #x = np.array() #最適化に使うパラメータ(今回はxl, xr)
    #res = minimize(optimization, x, method='trust-exact', jac=Func_Der, hess=Func_Hess, options={'gtol': 1e-8, 'disp': True})

    """
    SD_l, SD_r = optimization(ds, img, area)
    cnt += 1

    if(abs(SD_l - SD_r) > th): 
        area, i = extractArea() 
        setRuling(ds, img, area, i, root)      
        area, i = extractArea()
        setRuling(ds, img, area, i, root)
    else:
        return
    """
    return