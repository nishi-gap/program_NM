import numpy as np
import matplotlib.pyplot as plt

from src.pylib import DevSrf
from src.pylib import readEXRImage
from src.pylib import setRuling
#import setRuling

if __name__ == "__main__":

    filename = './img/one-sideFold.exr'
    img = readEXRImage.readEXR(filename)
    ds = DevSrf.DevSrf(20,20,10,np.pi/3, img.shape[0], img.shape[1])
    setRuling.setRuling(ds,img)

        
