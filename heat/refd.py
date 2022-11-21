'''
A script of functions to put SCALAR objects back into Firedrake data structures (2D)
'''
#global
from firedrake import *


def nptofd(prob,vec):

    #Set up spaces
    m = prob.mesh
    Z = prob.function_space(m)

    #Initialise firedrake function
    z = Function(Z)

    #Copy in external data
    z.dat.data[:] = vec[:]
    
    return z
