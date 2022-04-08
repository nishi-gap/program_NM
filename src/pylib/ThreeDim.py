from src.pylib import DevSrf
from dataclasses import dataclass
import numpy as np
#https://yttm-work.jp/model_render/model_render_0001.html　objファイルの中身

class OBJmodel:
    VerticesNum:int
    uvNum:int
    NormalNum:int
    FaceNum:int
    Vertices:np.ndarray
    uv:np.ndarray
    Normal:np.ndarray
    Face:list
    def __init__(self):
        self.VerticesNum = self.uvNum = self.NormalNum = self.FaceNum = 0
        self.Vertices = self.uv = self.Normal= np.empty(0,dtype=float)
        self.Face = ()
    def LoadOBJ(fp:str):
        with open(fp) as f:
            for l in f:
                if l[0] == "v":
                    if l[1] == "t": #テクスチャ座標
                        1
                    elif l[1] == "n":#法線
                        2
                    else:#頂点座標
                        3
                elif l[0] == "f":
                    4







