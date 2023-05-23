"""
Here we wrap the solvers and inform constraints where relevant
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
    dt = dic['dt']
    M = dic['M']
    L = dic['L']
    Lz0 = dic['Lz0']
    old_energy = dic['old_energy']
    omega = dic['omega']
    m0 = dic['m0']

    
    #Define constraints
    class mass:
        def __init__(self):
            self.M = 0 * A
            self.v = np.transpose(omega)
            self.c = - m0
    class energy:
        def __init__(self):
            self.M = M + 0.5 * dt * L
            self.v = 0.5 * dt * Lz0
            self.c = - old_energy

    #And stuff them in a list
    conlist = [mass(),energy()]

    #If tolerance is very small, use prototypical CGMRES to enforce
    #constraints one-by-one. This is what's being used in practice
    #here.
    if tol<1e-20:
        out = solvers.cgmres_p(A=A,b=b,x0=x0,k=k,
                               conlist=conlist,
                               pre=pre)
    else:
        out = solvers.cgmres(A=A,b=b,x0=x0,k=k,
                             tol=tol,conlist=conlist,
                             pre=pre,
                             timing=timing)
    return out


def gmresWrapper(dic,x0,k,tol=1e-50,pre=None):
    #there are no constraints, so ctol does not make sense here
    A = dic['A']
    b = dic['b']
    
    out = solvers.gmres(A=A,b=b,x0=x0,k=k,tol=tol,
                        pre=pre)
    
    return out
