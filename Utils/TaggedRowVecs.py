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

def Indexes(root=None, count:int = 2):
    ''' iterates the indexes, each appended to the seed object'''
    current = 0
    if root is None:
        while current < count:
            yield current
            current += 1
    else:
        while current < count:
            yield (root, current)
            current += 1

def Indexes2(root=None, steps1:int = 2, steps2:int = 2):
    ''' iterates the indexes of a square lattice, each appended to the seed object'''
    current1 = 0
    if root is None:
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
                yield (root, (current1, current2))
                current2 += 1
            current1 += 1
  

def Indexes3(root=None, steps1:int = 2, steps2:int = 2, steps3:int = 2):
    ''' iterates the indexes of a square lattice, each appended to the seed object'''
    current1 = 0
    if root is None:
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
                    yield (root, (current1, current2, current3))
                    current3 += 1
                current2 += 1
            current1 += 1

def Gaussians(centers:TaggedRowVecs=None, 
              covMatrix=None,
              num_points:int=2,
              dim:int=2):
    if centers is None:
        centers=Single(vec=np.array([0.] * dim), tag='.')
        covMatrix=np.matrix(np.diag([1.] * dim))                                   
    ''' replaces each of the provided vectors with a gaussian cluster of vectors'''
    return TaggedRowVecs(row_vecs = np.vstack(np.random.multivariate_normal(
                                        rv, 
                                        covMatrix,
                                        num_points) 
                                  for rv in centers.row_vecs),
                         tags = [i for t in centers.tags for i in Indexes(root=t, count=num_points)])
    
                                    
def GaussianTwins(centers:TaggedRowVecs=Single(vec=np.array([1.,2.]), tag='.'), 
              covMatrix1:np.matrixlib.defmatrix.matrix=np.matrix(np.diag([.1] * 2)),
              covMatrix2:np.matrixlib.defmatrix.matrix=np.matrix(np.diag([0.025] * 2)),
              num_points1:int=2,
              num_points2:int=2):
    ''' replaces each of the provided vectors with a pair of gaussian clusters'''
    rv1 = np.vstack(np.random.multivariate_normal(
                        rv,
                        covMatrix1,
                        num_points1) for rv in centers.row_vecs)
    rv2 = np.vstack(np.random.multivariate_normal(
                        rv, 
                        covMatrix2,
                        num_points2) for rv in centers.row_vecs)
    
    tags1 = [(1, i) for t in centers.tags for i in Indexes(root=t, count=num_points1)]
    tags2 = [(2, i) for t in centers.tags for i in Indexes(root=t, count=num_points2)]
    
    return TaggedRowVecs(row_vecs =  np.vstack([rv1, rv2]),                                
                         tags = tags1 + tags2)


def d2Lattice(centers:TaggedRowVecs=Single(vec=np.array([1.,2.]), tag='.'),
                span1:np.ndarray=np.array([1,0]),
                span2:np.ndarray=np.array([0,1]),
                steps1:int=2, steps2:int=2):
    ''' replaces each of the provided vectors with a square lattice of vectors'''
    tics1 = span1 * np.linspace(-0.5, 0.5, steps1)[:, np.newaxis]
    tics2 = span2 * np.linspace(-0.5, 0.5, steps2)[:, np.newaxis]
    return TaggedRowVecs(row_vecs=np.array([c + t1 + t2
                                       for t1 in tics1
                                       for t2 in tics2
                                       for c in centers.row_vecs]),
                         tags=[i for t in centers.tags for i in Indexes2(root=t, steps1=steps1, steps2=steps2)])


def d3Lattice(centers:TaggedRowVecs=Single(vec=np.array([1., 2., 3.]), tag='.'),
                  span1:np.ndarray=np.array([1,0,0]),
                  span2:np.ndarray=np.array([0,1,0]),
                  span3:np.ndarray=np.array([0,0,1]),
                  steps1:int=2, steps2:int=2, steps3:int=2):
    tics1 = span1 * np.linspace(-0.5, 0.5, steps1)[:, np.newaxis]
    tics2 = span2 * np.linspace(-0.5, 0.5, steps2)[:, np.newaxis]
    tics3 = span3 * np.linspace(-0.5, 0.5, steps3)[:, np.newaxis]
    return TaggedRowVecs(row_vecs=np.array([c + t1 + t2 + t3
                                        for t1 in tics1
                                        for t2 in tics2
                                        for t3 in tics3
                                        for c in centers.row_vecs]),
                         tags=[i for t in centers.tags for i in
                                     Indexes3(root=t, steps1=steps1, steps2=steps2, steps3=steps3)])