"""
Here we wrap solvers and inform constraints where relevant
"""
#Global
import numpy as np
import sys

#Local
sys.path.insert(0,'../')
import solvers

def cgmresWrapper(dic,x0,k,tol=1e-50,pre=None,timing=None):

    A = dic['A']
    b = dic['b']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    e0 = dic['e0']

    
    #Define constraints
    class mass:
        def __init__(self):
            self.M = 0 * A
            self.v = np.transpose(omega)
            self.c = - m0

    class energy:
        def __init__(self):
            self.M = L
            self.v = np.zeros_like(x0)
            self.c = -e0
            
    #And stuff them in a list
    conlist = [mass(),energy()]

    #If tol is set to be unreasonably small, use prototypical
    #algorithm to enforce constraints one-by-one
    if tol<1e-20:
        out = solvers.cgmres_p(A=A,b=b,x0=x0,k=k,
                               conlist=conlist,
                               pre=pre)
    #Otherwise use algorithm with tolerance
    else:
        out = solvers.cgmres(A=A,b=b,x0=x0,k=k,tol=tol,
                             conlist=conlist,timing=timing,
                             pre=pre)
    return out


def gmresWrapper(dic,x0,k,tol=1e-50,pre=None):
    A = dic['A']
    b = dic['b']
    
    out = solvers.gmres(A=A,b=b,x0=x0,k=k,tol=tol,pre=pre)
    
    return out
