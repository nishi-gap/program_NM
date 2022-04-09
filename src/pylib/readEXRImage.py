import numpy as np
import OpenEXR, Imath, array
from numpy import linalg as LA

def C2V(img:np.array, X:int, Y:int):
    nm = 2 * img - 1
    for y in range(Y):
        for x in range(X):
            nm[x][y] /= LA.norm(nm[x][y])
            nm[x][y] = (nm[x][y] + 1)/2
            norm = LA.norm(2 * nm[x][y] - 1)
            nm[x][y] = (2 * nm[x][y] - 1)/norm
    return nm


def readEXR(file: str):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    img_exr = OpenEXR.InputFile(file)

    # 読んだEXRファイルからRGBの値を取り出し
    r_str, g_str, b_str = img_exr.channels('RGB', pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))

    # 画像サイズを取得
    dw = img_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # openCVで使えるように並べ替え
    img = np.array([[r, g, b] for r, g, b in zip(red, green, blue)])
    
    img = img.reshape(size[1], size[0], 3)
    img = img.transpose((1,0,2))
    #nm = C2V(img, size[0], size[1]) #色ベクトルから単位法線ベクトル
    
    return img
