import numpy as np
from numpy import linalg as la
import random

def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    #num = float(inA.T*inB)
    num = np.dot(inA, inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)

def UT_uniform(dim:int=3, v:float=2.22):
    ''' makes upper triangular matrix with all the values=v'''
    return np.matrix(np.array([[0 if i < j else v for i in range(dim)] for j in range(dim)]))

def randnMatrix(rows:int=3, cols:int=3, seed:int=None):
    rng = np.random.RandomState(seed)
    m = np.matrix(rng.randn(rows, cols))
    return m

def gram_schmidt_columns(x):
    ''' makes the columns of x orthonormal'''
    q, r = np.linalg.qr(x)
    return q

def MatrixRowNorm(matrix):
    ''' makes the rows of x normal, but not orthogonal'''
    m = np.array(matrix.tolist())
    msq = m * m # np dot-product style multiplication
    row_sums = msq.sum(axis=1)
    return np.matrix(np.sqrt(msq / row_sums[:, np.newaxis]))

def MatrixColNorm(matrix):
    ''' makes the columns of x normal, but not orthogonal'''
    m = np.array(matrix.tolist())
    msq = m * m # np dot-product style multiplication
    col_sums = msq.sum(axis=0)
    return np.matrix(np.sqrt(msq / col_sums[np.newaxis, :]))

def randOrthoNormals(d:int=3, count:int=2, seed:int=None):
    ''' makes count orthonormal vectors of length d, uniformly distributed on the d-sphere'''
    ''' make sure that count < d'''   
    rng = np.random.RandomState(seed)
    colVecs = rng.randn(d, count)
    return gram_schmidt_columns(colVecs)

def randOrthoNormalMatrix(s:int=3):
    ''' returns a square matrix of size s with orthonormal columns''' 
    return np.matrix(randOrthoNormals(s,s))

def randnPdefMatrix(rows:int=4, cols:int=4, seed:int=None):
    ''' makes a positive definite square matrix of size rows'''
    m = randnMatrix(rows=rows, cols=cols, seed=seed)
    return m * np.transpose(m)

def randnPdefUnitMatrix(dim:int=4, seed:int=None):
    ''' makes a positive definite square matrix of size rows with unit determinant'''
    pDef = randnPdefMatrix(rows=dim, cols=dim, seed=seed)
    dt = np.linalg.det(pDef)
    scale = np.power(dt, 1/dim)
    return pDef / scale