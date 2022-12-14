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

def cgmresWrapper(dic,x0,k,tol=1e-50,contol=10):

    A = dic['A']
    b = dic['b']
    M = dic['M']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    mo0 = dic['mo0']
    e0 = dic['e0']

    #Define constraints
    def const1(z,x0,Q):#Depends on x0 and Q
        X = x0 + Q @ z
        out = np.transpose(omega) @ X - m0
        return out
    
    def const2(z,x0,Q):
        X = x0 + Q @ z
        out = 0.5*np.transpose(X) @ M @ X - mo0
        return out
    
    def const3(z,x0,Q):
        X = x0 + Q @ z
        out = 0.5 * np.transpose(X) @ L @ X \
            - 0.5 * np.transpose(X) @ M @ X \
            - e0
        return out

    #And stuff them in an ordered list
    conlist = [const1,const2,const3]

    #If tolerance is not crazy small, use the cgmres with a tolerance
    if tol>1e-20:
        out = solvers.cgmres(A=A,b=b,x0=x0,k=k,tol=tol,contol=contol,
                      conlist=conlist)
    #If tolerance is set to be unrealistically small, use prototypical
    #CGMRES to enforce constraints one-by-one
    else:
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
