import sys
import numpy as np

##外部ファイルの読み込み
sys.path.append("DevSrf")
sys.path.append("readEXRImage")
###
import DevSrf
import readEXRImage

if __name__ == "__main__":
    ds = DevSrf.DevSrf(10,20,10,np.pi/3)
    img = readEXRImage.readEXR("img/cylinder.exr")

    print(img.shape)