import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

filePath = sys.argv[1]
x = np.zeros(1)
y = np.zeros(1)

with open(filePath + '.meta', 'r') as meta:
    md = meta.readline().split(',')
    nx = int(md[0])
    ny = int(md[1])
    xLen = float(md[2])
    yLen = float(md[3])
    x = np.linspace(0.0, xLen, nx)
    y = np.linspace(0.0, yLen, ny)

X, Y = np.meshgrid(y, x)
p = np.loadtxt(filePath + '_p.dat').T
u = np.loadtxt(filePath + '_u.dat').T
v = np.loadtxt(filePath + '_v.dat').T

fig, ax = plt.subplots(1, 2,  figsize=(11,7), dpi=100)

cf = ax[0].contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
fig.colorbar(cf, label='Pressure')
ax[0].streamplot(X, Y, u, v)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title("Pressure Gradient")

ax[1].quiver(X, Y, u, v)
ax[1].set_title("Quiver Flow Plot")

fig.suptitle(sys.argv[2])

plt.savefig(filePath + '.png')
