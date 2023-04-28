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
                               conlist=conlist,
                               pre=pre)
    #Otherwise use algorithm with tolerance
    else:
        out = solvers.cgmres(A=A,b=b,x0=x0,k=k,tol=tol,
                             conlist=conlist,timing=timing,
                             pre=pre)
    return out


def gmresWrapper(dic,x0,k,tol=1e-50,pre=None,timing=None):
    A = dic['A']
    b = dic['b']
    
    out = solvers.gmres(A=A,b=b,x0=x0,k=k,tol=tol,pre=pre,timing=timing)
    
    return out
