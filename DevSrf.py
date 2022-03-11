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
    foldAngle: float, img:np.array):
        self.rulingNum = rulingNum
        self.modelWidth = modelWidth
        self.modelHeight = modelHeight     
        self.foldAngles = np.full((rulingNum,),foldAngle)
        self.MapWidth = img.shape[0]
        self.MapHeight = img.shape[1]
        step = modelWidth/(rulingNum + 1) 
        self.xl = np.linspace(step, modelWidth - step,rulingNum)
        self.xr = np.linspace(step, modelWidth - step,rulingNum)
