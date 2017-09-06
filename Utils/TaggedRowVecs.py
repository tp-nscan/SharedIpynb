import numpy as np
from collections import namedtuple


# A set of K dim N vectors, with corresponding tags
# vectors: 2d numpy array with shape (K, N)
# tags: numpy array with shape (K)
TaggedRowVecs = namedtuple( 
    'TaggedRowVecs', 'row_vecs tags'
)

def Single(vec:np.ndarray, tag=None):
    assert len(vec.shape) == 1
    return TaggedRowVecs(row_vecs=np.array([vec.tolist()]), tags=tag)

def Indexes(seed=None, count:int = 2):
    ''' iterates the indexes, each appended to the seed object'''
    current = 0
    if seed is None:
        while current < count:
            yield current
            current += 1
    else:
        while current < count:
            yield (seed, current)
            current += 1

def Indexes2(seed=None, steps1:int = 2, steps2:int = 2):
    ''' iterates the indexes of a square lattice, each appended to the seed object'''
    current1 = 0
    if seed is None:
        while current1 < steps1:
            current2 = 0
            while current2 < steps2:
                yield (current1, current2)
                current2 += 1
            current1 += 1
    else:
        while current1 < steps1:
            current2 = 0
            while current2 < steps2:
                yield (seed, (current1, current2))
                current2 += 1
            current1 += 1
  
def Indexes3(seed=None, steps1:int = 2, steps2:int = 2, steps3:int = 2):
    ''' iterates the indexes of a square lattice, each appended to the seed object'''
    current1 = 0
    if seed is None:
        while current1 < steps1:
            current2 = 0
            while current2 < steps2:
                current3 = 0
                while current3 < steps3:
                    yield (current1, current2, current3)
                    current3 += 1
                current2 += 1
            current1 += 1
    else:
        while current1 < steps1:
            current2 = 0
            while current2 < steps2:
                current3 = 0
                while current3 < steps3:
                    yield (seed, (current1, current2, current3))
                    current3 += 1
                current2 += 1
            current1 += 1

def Gaussians(center:TaggedRowVecs=Single(vec=np.array([1.,2.]), tag='.'), 
              covMatrix:np.matrixlib.defmatrix.matrix=np.matrix(np.diag([1.] * 2)),
              num_points:int=2):
    ''' replaces each of the provided vectors with a gaussian cluster of vectors'''
    return TaggedRowVecs(row_vecs = np.vstack(np.random.multivariate_normal(
                                        center.row_vecs[i], 
                                        covMatrix,
                                        num_points) 
                                  for i in range(len(center.row_vecs))),
                         tags = [i for t in center.tags for i in Indexes(seed=t, count=num_points)])



def GaussianTwins(center:TaggedRowVecs=Single(vec=np.array([1.,2.]), tag='.'), 
              covMatrix1:np.matrixlib.defmatrix.matrix=np.matrix(np.diag([.1] * 2)),
              covMatrix2:np.matrixlib.defmatrix.matrix=np.matrix(np.diag([0.025] * 2)),
              num_points1:int=2,
              num_points2:int=2):
    ''' replaces each of the provided vectors with a pair of gaussian clusters'''
    rv1 = np.vstack(np.random.multivariate_normal(
                        center.row_vecs[i], 
                        covMatrix1,
                        num_points1) for i in range(len(center.row_vecs)))
    rv2 = np.vstack(np.random.multivariate_normal(
                        center.row_vecs[i], 
                        covMatrix2,
                        num_points2) for i in range(len(center.row_vecs)))
    
    tags1 = [(1, i) for t in center.tags for i in Indexes(seed=t, count=num_points1)]
    tags2 = [(2, i) for t in center.tags for i in Indexes(seed=t, count=num_points2)]
    
    return TaggedRowVecs(row_vecs =  np.vstack([rv1, rv2]),                                
                         tags = tags1 + tags2)


def d2Lattice(centers:TaggedRowVecs=Single(vec=np.array([1.,2.]), tag='.'),
                span1:np.ndarray=np.array([1,0]),
                span2:np.ndarray=np.array([0,1]),
                steps1:int=2, steps2:int=2):
    ''' replaces each of the provided vectors with a square lattice of vectors'''
    tics1 = span1 * np.linspace(-0.5, 0.5, steps1)[:, np.newaxis]
    tics2 = span2 * np.linspace(-0.5, 0.5, steps2)[:, np.newaxis]
    return TaggedRowVecs(row_vecs=np.array([c + tics1[x] + tics2[y]
                                       for x in range(len(tics1))
                                       for y in range(len(tics2))
                                       for c in centers.row_vecs]),
                         tags=[i for t in centers.tags for i in Indexes2(seed=t, steps1=steps1, steps2=steps2)])


def d3Lattice(centers:TaggedRowVecs=Single(vec=np.array([1., 2., 3.]), tag='.'),
                  span1:np.ndarray=np.array([1,0,0]),
                  span2:np.ndarray=np.array([0,1,0]),
                  span3:np.ndarray=np.array([0,0,1]),
                  steps1:int=2, steps2:int=2, steps3:int=2):
    tics1 = span1 * np.linspace(-0.5, 0.5, steps1)[:, np.newaxis]
    tics2 = span2 * np.linspace(-0.5, 0.5, steps2)[:, np.newaxis]
    tics3 = span3 * np.linspace(-0.5, 0.5, steps3)[:, np.newaxis]
    return TaggedRowVecs(row_vecs=np.array([c + tics1[x] + tics2[y] + tics3[z]
                                        for x in range(len(tics1))
                                        for y in range(len(tics2))
                                        for z in range(len(tics3))
                                        for c in centers.row_vecs]),
                         tags=[i for t in centers.tags for i in
                                     Indexes3(seed=t, steps1=steps1, steps2=steps2, steps3=steps3)])