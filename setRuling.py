import numpy as np
import sys
import math

##外部ファイルの読み込み
sys.path.append("DevSrf")
##
import DevSrf

def splitMesh(ds: DevSrf.DevSrf, ind: int):
    1

def extractArea():
    1

##標準偏差により評価
##各ベクトルのとる範囲が-1から1のため十分小さくなれば
def getSD(img: np.array, area:np.array):
    img_l = img #必要なサイズ分スライスするようにしておく
    img_r = img
    return np.std(img_l), np.std(img_r)


def optimization(ds: DevSrf.DevSrf,img:np.array, area:np.array, ind: int):
    
    #ここでSD_l, SD_rの和が最小となるように最適化処理を行う
    SD_l, SD_r = getSD(img, area) #目的関数

    splitMesh(ds, ind) #最適化により得られた折り線を使って分割する

    return SD_l, SD_r

'''
手順
1. 法線マップNを2つに分割(N_l, N_r) 分割位置の初期状態はNを二等分にできる場所
2. N_l, N_rの各標準偏差の和が最小となるようにオリセンの位置調整
3. N_l, N_rそれぞれの標準偏差の値の差が十分小さい場合は分割終了.それ以外はN_l, N_rそれぞれを新たなNとして1に戻る(先に分割するのはN_l, N_rのうち標準偏差が大きい方)
'''
th = 5 #閾値
cnt = 0
def setRuling(ds: DevSrf.DevSrf, img: np.array, area:np.array, i: int, root:bool):
    if(root == True):
        cnt = 0
        root = False
    if(cnt == ds.rulingNum): return

    #i = math.floor(ds.rulingNum/2) #開始点
    #cnt = 0    

    SD_l, SD_r = optimization(ds, img, area)
    cnt += 1

    if(abs(SD_l - SD_r) > th): 
        area, i = extractArea() 
        setRuling(ds, img, area, i, root)
        
        area, i = extractArea()
        setRuling(ds, img, area, i, root)
    else:
        return

    return