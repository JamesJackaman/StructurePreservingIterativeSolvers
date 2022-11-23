'''
Here we wrap the linear solvers and inform constraints where relevant
'''
#Global
import numpy as np
import warnings
import scipy.sparse as sps
import sys

#Local
sys.path.insert(0,'../')
import solvers
import lkdvRK

def cgmresWrapper(dic,x0,k,prob=None,pre=None,
                  tol=1e-50,contol=10):
    #Unpack
    A = dic['A']
    b = dic['b']
    M = dic['M']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    mo0 = dic['mo0']
    e0 = dic['e0']
    z0 = dic['z0']

    #Define constraints
    def const1(z,x0,Q):#Depends on x0 and Q
        X = x0 + Q @ z
        X = lkdvRK.z1calc(prob, X, z0)
        out = np.transpose(omega) @ X - m0
        return out

    def const2(z,x0,Q):
        X = x0 + Q @ z
        X = lkdvRK.z1calc(prob, X, z0)
        out = 0.5 * np.transpose(X) @ M @ X \
            - mo0
        return out
    
    def const3(z,x0,Q):
        X = x0 + Q @ z
        X = lkdvRK.z1calc(prob, X, z0)
        out = 0.5 * np.transpose(X) @ L @ X \
            - 0.5 * np.transpose(X) @ M @ X \
            - e0
        return out

    #And stuff them in a list
    conlist = [const1,const2,const3]

    #If tolerance is not very crazy small, use cgmres with a tolerance
    if tol>1e-20:
        out = solvers.cgmres(A=A,b=b,x0=x0,k=k,pre=pre,
                     tol=tol,contol=contol,
                     conlist=conlist)
    #If tolerance is set to be unrealistically small, use prototypical
    #GMRES to enforce constraints one-by-one
    else:
        out = solvers.cgmres_p(A=A,b=b,x0=x0,k=k,
                      pre=pre,conlist=conlist)
    return out


def gmresWrapper(dic,x0,k,tol=1e-50,pre=None,contol=None,prob=None):

    A = dic['A']
    b = dic['b']
    
    out = solvers.gmres(A=A,b=b,x0=x0,k=k,tol=tol,pre=pre)
    
    return out


#Wrapper for an "exact" linear solver
def exact(dic,x0,k=None,tol=None,prob=None,pre=None,contol=None):
    
    A = dic['A']
    b = dic['b']

    out = sps.linalg.spsolve(A,b)

    return out, -1
