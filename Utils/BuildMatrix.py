import numpy as np
import random

def UT_uniform(dim:int=3, v:float=2.22):
    ''' makes upper triangular matrix with all the values=v'''
    return np.matrix(np.array([[0 if i < j else v for i in range(dim)] for j in range(dim)]))

def randnMatrix(rows:int=3, cols:int=3):
    m = np.matrix(np.random.normal(0, 1, (rows, cols)))
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

def randOrthoNormals(d:int=3, count:int=2, seed:int=1234):
    ''' makes count orthonormal vectors of length d, uniformly distributed on the d-sphere'''
    ''' count must be < dim'''   
    random.seed(seed)
    colVecs = np.random.randn(d, count)
    return gram_schmidt_columns(colVecs)

def randOrthoNormalMatrix(s:int=3):
    ''' returns a square matrix of size s with orthonormal columns''' 
    return np.matrix(randOrthoNormals(s,s))

def randnPdefMatrix(dim:int):
    ''' makes a positive definite square matrix of size dim'''
    m = randnMatrix(rows=dim, cols=dim)
    return m * np.transpose(m)

def randnPdefUnitMatrix(dim:int):
    ''' makes a positive definite square matrix of size dim with unit determinant'''
    pDef = randnPdefMatrix(dim)
    dt = np.linalg.det(pDef)
    scale = np.power(dt, 1/dim)
    return pDef / scale