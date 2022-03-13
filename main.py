import sys
import numpy as np
import cv2
import os

##外部ファイルの読み込み
sys.path.append("DevSrf")
#sys.path.append("readEXRImage")
sys.path.append("setRuling")
###
import DevSrf
#import readEXRImage
import setRuling

if __name__ == "__main__":
    imFile = "./img/test.exr"
    if not os.path.isfile(imFile):
        raise FileNotFoundError('Image file not found!')
    img = cv2.imread(imFile, cv2.IMREAD_COLOR)#0~255の整数で表すため少数は切り捨てられる→exr形式が0~1のため改善が必要
    
    print(img.shape[0], "  ", img.shape[1])
    print(img[0][19])
    cv2.imshow('color', img)
    cv2.waitKey(0)#ここで初めてウィンドウが表示される
    ds = DevSrf.DevSrf(40,20,10,np.pi/3, img)
    setRuling.getSD(ds,img)

