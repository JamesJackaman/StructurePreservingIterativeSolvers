"""
Here we wrap solvers and inform constraints where relevant
"""
#Global
import numpy as np
import sys

#Local
sys.path.insert(0,'../')
import solvers

def cgmresWrapper(dic,x0,k,tol=1e-50,ctol=1e-14):

    A = dic['A']
    b = dic['b']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    e0 = dic['e0']

    
    #Define constraints
    def const_mass(z,x0,Q):
        X = x0 + Q @ z
        out = np.transpose(omega) @ X - m0
        return out
    
    def const_energy(z,x0,Q):#Depends on x0 and Q
        X = x0 + Q @ z
        out = 0.5 * X @ L @ X - e0
        return out


    #And stuff them in a list
    conlist = [const_mass,const_energy]

    #If tol is set to be unreasonably small, use prototypical
    #algorithm to enforce constraints one-by-one
    if tol<1e-20:
        out = solvers.cgmres_p(A=A,b=b,x0=x0,k=k,
                               conlist=conlist)
    #Otherwise use algorithm with tolerance
    else:
        out = solvers.cgmres(A=A,b=b,x0=x0,k=k,tol=tol,
                             conlist=conlist,
                             ctol=ctol)
    return out


def gmresWrapper(dic,x0,k,tol=1e-50,ctol=None):
    #there are no constraints, so ctol does not make sense here
    A = dic['A']
    b = dic['b']
    
    out = solvers.gmres(A=A,b=b,x0=x0,k=k,tol=tol)
    
    return out
