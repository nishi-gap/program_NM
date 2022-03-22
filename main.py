import sys
import numpy as np
import matplotlib.pyplot as plt


from src.pylib import DevSrf
from src.pylib import readEXRImage
from src.pylib import setRuling
#import setRuling

if __name__ == "__main__":
    
    filename = './img/test.exr'
    img = readEXRImage.readEXR(filename)
    #print(img)
    #plt.imshow(img)
    #plt.show()
    ds = DevSrf.DevSrf(10,20,10,np.pi/3, img.shape[1], img.shape[0])
    setRuling.setRuling(ds,img)

