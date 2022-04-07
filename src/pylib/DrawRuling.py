import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [])
ruledLines = np.empty(0,dtype = float)
X,Y = 0

def init():
    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 15)
    return ln

def update(frame):
    p = ruledLines[frame]
    n = int(len(p)/2)
    ax.cla()
    plt.text(0,0,"iteration:%d"%frame)
    for i in range(n):  
        px = []
        py = []     
        for j in range(2):
            px[j] = p[i + j * n]
            py[j] = Y * j
            if px[j] < 0:
                px[j] = 0
                py[j] = p[i] * Y/(p[i + n] - p[i])
            elif px[j] > X:
                px[j] = X
                py[j] = Y * (X - p[i])/(p[i + n] - p[i])
        x = [px[0],px[1]]
        y = [py[0],py[1]]
        ax.plot(x,y)

def dispResult(_ruledLines, _X, _Y):
    global ruledLines, X, Y
    X = _X
    Y = _Y
    ruledLines = _ruledLines
    iter = ruledLines.shape[0]
    ani = FuncAnimation(fig,update,frames=iter)
    ani.save('img/result.gif', writer='pillow') 
    plt.show()
    