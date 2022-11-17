'''
A script of functions to put arrays back into Firedrake data structures
'''
#global
from firedrake import *
import numpy as np
import matplotlib.pylab as plt
import sys

#local
sys.path.insert(0,'../') 

#Convert numpy array to Firedrake function
def nptofd(prob,vec):

    #seperate vector components
    vec1, vec2, vec3 = np.split(vec, 3)

    #Set up spaces
    m = prob.mesh
    Z = prob.function_space(m)

    z = Function(Z)
    u, v, w = z.split()
    #Copy in external data
    u.dat.data[:] = vec1[:]
    v.dat.data[:] = vec2[:]
    w.dat.data[:] = vec3[:]

    return z


#Flatten Firedrake vectors
def flatten(vec):
    dim = len(vec)
    size = len(vec[0])
    newvec = np.zeros((size*dim,))

    for i in range(dim):
        newvec[i*size:size*(i+1)] = vec[i]


    return newvec

#Combine Firedrake vectors
def combine(vec):

    dim = len(vec)
    size = []
    for i in range(dim):
        size.append(len(vec[i]))

    totalsize = np.sum(size)

    newvec = np.zeros((totalsize,))

    total = 0
    for i in range(dim):
        newvec[total:total+size[i]] = vec[i]
        total = total + size[i]
    
    return newvec
