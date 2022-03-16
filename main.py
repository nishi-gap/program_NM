import sys
import numpy as np
import matplotlib.pyplot as plt

##外部ファイルの読み込み
sys.path.append("DevSrf")
sys.path.append("./src/readEXRImage")
sys.path.append("setRuling")
###
import DevSrf
import readEXRImage
import setRuling

if __name__ == "__main__":

    filename = './img/test.exr'
    img = readEXRImage.readEXR(filename)
    #plt.imshow(img)
    #plt.show()
    #x,y = img.shape[1], img.shape[0]
    #ds = DevSrf.DevSrf(40,20,10,np.pi/3, x, y)
    #setRuling.setRuling(ds,img)