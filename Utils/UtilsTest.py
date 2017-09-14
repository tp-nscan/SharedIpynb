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
import Utils.TSNEext as tsneExt
from sklearn.manifold import TSNE


def testPlot():
    q = trv.Gaussians()
    tcvG = trv.Gaussians(center=q,
                         num_points=50,
                         covMatrix=np.matrix(np.diag([.01] * 2)))
    resG = trvPlt.PlotTrvs(tcvG,
                           figsize=(2, 2),
                           markersize=2,
                           tag_extractor=lambda x: x[0])
    plt.show()

def testTsne():
    pts = np.array([[1., 2., 3., 4., 5., 6.], [11., 2., 3., 4., 5., 6.], [21., 2., 3., 4., 5., 26.]])
    return TSNE(random_state=123).fit_transform(pts)

if __name__ == '__main__':
    print(testTsne())