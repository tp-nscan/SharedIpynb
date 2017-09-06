import numpy as np
import random
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns; sns.set()

def plotOneMatrix(matrix, title = "", width:int = 4, height:int = 5):
    ''' a pcolormesh matrix with colorbar'''
    fig, ax = plt.subplots()
    ax.set_title(title)
    fig.gca().invert_yaxis()
    ax.figure.set_figwidth(width)
    ax.figure.set_figheight(height)
    pc = ax.pcolormesh(matrix, cmap='RdBu')
    fig.colorbar(pc, orientation='horizontal')
    plt.show()
    
 
def plotTwoMatricies(matrix1, matrix2, title1 = "", title2 = "", width:int = 4, height:int = 3):
    fig = plt.figure(figsize=(width * 2 + 2, height))
    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)

    p1 = plt.subplot(grid[0, 0])
    p1.set_title(title1)
    ee = p1.pcolormesh(matrix1, cmap='RdBu')
    fig.colorbar(ee, orientation='vertical');
    plt.gca().invert_yaxis()

    p2 = plt.subplot(grid[0, 1])
    p2.set_title(title2)
    ff = p2.pcolormesh(matrix2, cmap='RdBu')
    fig.colorbar(ff, orientation='vertical');
    plt.gca().invert_yaxis()
    plt.show()