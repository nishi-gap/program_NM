import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [])
ruledLines = np.empty(0,dtype = float)

def init():
    ax.set_xlim(-5, 25)
    ax.set_ylim(-5, 15)
    return ln

def update(frame):
    p = ruledLines[frame]
    n = int(len(p)/4)
    ax.cla()
    plt.text(0,0,"iteration:%d"%frame)
    for i in range(n):
        x = [p[i],p[i + n]]
        y = [p[i + 2 * n], p[i + 3 * n]]
        ax.plot(x,y)

def dispResult(_ruledLines):
    global ruledLines
    ruledLines = _ruledLines
    iter = ruledLines.shape[0]
    print("iter ",iter)
    ani = FuncAnimation(fig,update,frames=iter)
    plt.show()
