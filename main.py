import sys
import numpy as np

##外部ファイルの読み込み
sys.path.append("DevSrf")

###
import DevSrf

if __name__ == "__main__":
    ds = DevSrf.DevSrf(10,20,10,np.pi/3)
    #ds.xl = np.resize(ds.xl,(20,1))
    print(ds.foldAngles.size)
    print(ds.xl.size)