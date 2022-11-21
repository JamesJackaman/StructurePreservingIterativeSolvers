"""
Here we wrap the solvers and inform constraints where relevant
"""
#Global
import numpy as np
import sys

#Local
sys.path.insert(0,'../')
import solvers

def cgmresWrapper(dic,x0,k,tol=1e-50,pre=None):

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
    def const_mass(z,x0,Q):
        X = x0 + Q @ z
        out = np.transpose(omega) @ X - m0
        return out
    
    def const_energy(z,x0,Q):#Depends on x0 and Q
        X = x0 + Q @ z
        out = 0.5 * X @ M @ X + 0.25 * dt * X @ L @ X \
            + 0.5 * dt * X @ Lz0 - old_energy
        return out


    #And stuff them in a list
    conlist = [const_mass,const_energy]

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
                            pre=pre)
    return out


def gmresWrapper(dic,x0,k,tol=1e-50,pre=None):
    #there are no constraints, so ctol does not make sense here
    A = dic['A']
    b = dic['b']
    
    out = solvers.gmres(A=A,b=b,x0=x0,k=k,tol=tol,pre=pre)
    
    return out
