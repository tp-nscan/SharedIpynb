import pandas as pd
import matplotlib
import numpy as np
import numpy.linalg as LA
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import Utils.TaggedRowVecs as trv
import Utils.TrvPlot as trvPlt



def youch():
    q = trv.Gaussians()
    tcvG = trv.Gaussians(center=q,
                         num_points=50,
                         covMatrix=np.matrix(np.diag([.01] * 2)))
    resG = trvPlt.PlotTrvs(tcvG,
                           figsize=(2, 2),
                           markersize=2,
                           tag_extractor=lambda x: x[0])
    plt.show()

if __name__ == '__main__':
    youch()