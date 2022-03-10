import numpy as np
import OpenEXR, Imath, array

def readEXR(file: str):
    print("readEXR")
    # ファイルを読み込む
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

    return img