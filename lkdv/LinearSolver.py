"""

Here we wrap the solvers and inform constraints where relevant

"""
#Global
import numpy as np
import warnings
import scipy.sparse as sps
import sys

#Local
sys.path.insert(0,'../')
import solvers

def cgmresWrapper(dic,x0,k,tol=1e-50,contol=10,timing=None):

    A = dic['A']
    b = dic['b']
    M = dic['M']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    mo0 = dic['mo0']
    e0 = dic['e0']

    #Define constraints
    class mass:
        def __init__(self):
            self.M = 0 * A
            self.v = np.transpose(omega)
            self.c = - m0

    class momentum:
        def __init__(self):
            self.M = M
            self.v = np.zeros_like(x0)
            self.c = - mo0

    class energy:
        def __init__(self):
            self.M = L - M
            self.v = np.zeros_like(x0)
            self.c = - e0
        
    #And stuff them in an ordered list
    conlist = [mass(),momentum(),energy()]

    #If tolerance is not crazy small, use the cgmres with a tolerance
    if tol>1e-20:
        out = solvers.cgmres(A=A,b=b,x0=x0,k=k,tol=tol,contol=contol,
                             conlist=conlist,timing=timing)
    #If tolerance is set to be unrealistically small, use prototypical
    #CGMRES to enforce constraints one-by-one
    else:
        if timing!=None:
            raise NotImplementedError('Timings are not available for prototypical solver')
        out = solvers.cgmres_p(A=A,b=b,x0=x0,k=k,
                             conlist=conlist)
    return out

def gmresWrapper(dic,x0,k,tol=1e-50,contol=None):

    if contol!=None:
        warnings.warn('Contol is ignored as not used in GMRES')

    A = dic['A']
    b = dic['b']
    
    out = solvers.gmres(A=A,b=b,x0=x0,k=k,tol=tol)
    
    return out


#Wrapper for an "exact" linear solver
def exact(dic,x0=None,k=None,tol=None,prob=None,contol=None):
    
    A = dic['A']
    b = dic['b']

    out = sps.linalg.spsolve(A,b)
        
    return out, -1
