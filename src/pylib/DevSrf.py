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
    foldAngle: float, MapWidth: int,MapHeight:int):
        self.rulingNum = rulingNum
        self.modelWidth = modelWidth
        self.modelHeight = modelHeight     
        self.foldAngles = np.full((rulingNum,),foldAngle)
        self.MapWidth = MapWidth
        self.MapHeight = MapHeight       
        step = modelWidth/(rulingNum + 1) 
        self.xl = np.linspace(step, modelWidth - step,rulingNum)