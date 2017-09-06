import numpy as np
import random
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import Utils.TaggedRowVecs as trv
import seaborn as sns; sns.set()

def defaultFormat(tup):
    return str(tup)

def Format2Tuple(tup):
    return "({:.2f}, {:.2f})".format(*tup)

def firstLabel(tag_set:set, tag, lblformatter):
    if len(tag_set.intersection({tag})) > 0 :
        tag_set.remove(tag)
        return lblformatter(tag)
    return None

def MakeTagUniqueColorer(tags:list):
    ''' creates a color map based on the tag part that is extracted by f'''
    colorMap = {}
    for i, t in enumerate(set(tags)):
        colorMap[t] = i
    palette = sns.color_palette("hls", len(colorMap))
    def fRet(tag):
        return palette[colorMap[tag]]
    return fRet

def RedBlue2dPalette(span_x:int, span_y:int):
    reds = np.linspace(0, 1, span_x)
    blues = np.linspace(0, 1, span_y)
    return [(red, 0, blue) for red in reds for blue in blues]

def MakeRedBlueGridColorer(span_x:int, span_y:int):
    palette = RedBlue2dPalette(span_x=span_x, span_y=span_y)
    def fRet(tup):
        return palette[span_y * int(tup[0]) + int(tup[1])]
    return fRet

def PlotTrvs(tvs:trv.TaggedRowVecs, colorer=None, lblformatter=defaultFormat, figsize=(4,4),
             markersize:int=10, showLegend=False, tag_extractor=lambda x: x):
    downTagged = trv.TaggedRowVecs(row_vecs=tvs.row_vecs, tags=[tag_extractor(t) for t in tvs.tags])
    ''' plots a TaggedRowVecs struct'''
    plt.figure(figsize=figsize)
    if colorer==None:
        colorer = MakeTagUniqueColorer(downTagged.tags)
    fcs = []
    tag_set = set(downTagged.tags)
    for i, v in enumerate(downTagged.row_vecs):
        fcs.append(colorer(downTagged.tags[i]))
        plt.plot(v[0], v[1], 'o',
         markersize=markersize,
         markerfacecolor=colorer(downTagged.tags[i]),
         markeredgewidth=0,
         label=firstLabel(tag_set=tag_set, tag=downTagged.tags[i], lblformatter=lblformatter))
    
    minX = np.min(downTagged.row_vecs[:, [0]])
    spanX = np.max(downTagged.row_vecs[:, [0]]) - minX
    if showLegend:
        plt.xlim(minX - spanX * 0.1, minX + spanX * 1.6);
        plt.legend(numpoints=1)
    else:
        plt.xlim(minX - spanX * 0.1, minX + spanX * 1.1);
    return fcs

def PlotTrvsGrid(tvs:trv.TaggedRowVecs, span_x:int, span_y:int, figsize=(4,4), markersize=10, showLegend=False, tag_extractor=lambda x: x):
    ''' plots a TaggedNVectors struct, where the vectors correspond to points on a lattice'''
    colorer = MakeRedBlueGridColorer(span_x=span_x, span_y=span_y)
    return PlotTrvs(tvs=tvs, colorer=colorer, lblformatter=Format2Tuple, figsize=figsize, markersize=markersize, showLegend=showLegend)