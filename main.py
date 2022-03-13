import sys
import numpy as np
import matplotlib.pyplot as plt
##外部ファイルの読み込み
sys.path.append("DevSrf")
sys.path.append("readEXRImage")
sys.path.append("setRuling")
###
import DevSrf
import readEXRImage
import setRuling

if __name__ == "__main__":
    img = readEXRImage.readEXR("./img/test.exr")
    ds = DevSrf.DevSrf(40,20,10,np.pi/3, img)
    setRuling.getSD(ds,img)