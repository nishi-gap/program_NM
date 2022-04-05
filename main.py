import numpy as np

from src.pylib import DevSrf
from src.pylib import readEXRImage
from src.pylib import setRuling
#import setRuling

if __name__ == "__main__":

    filename = './img/simplePattern.exr'
    img = readEXRImage.readEXR(filename)
    ds = DevSrf.DevSrf(10,20,10,np.pi/3, img.shape[0], img.shape[1])
    l = setRuling.setRuling(ds,img)   
