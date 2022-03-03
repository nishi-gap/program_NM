from dataclasses import dataclass
import numpy as np

#@dataclass
class DevSrf:
    rulingNum: int
    modelWidth: int
    modelHeight: int
    MapWidth: int
    MapHeight: int
    #foldAngle: float
    
    #xr:np.array

    def __init__(self, rulingNum: int, modelWidth: int, modelHeight: int, 
    foldAngle: float, MapWidth: int = 512,MapHeight:int = 256):
        self.rulingNum = rulingNum
        self.modelWidth = modelWidth
        self.modelHeight = modelHeight     
        self.foldAngles = np.full((rulingNum,),foldAngle)
        self.MapWidth = MapWidth
        self.MapHeight = MapHeight
        self.xl = np.zeros(rulingNum)
        self.xr = np.zeros(rulingNum)
