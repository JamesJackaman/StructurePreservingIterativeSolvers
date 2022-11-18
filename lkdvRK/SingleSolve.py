#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt

#local
import lkdvRK
import LinearSolver as ls


if __name__=="__main__":

    tol = 1e-50
    k = 20
    
    params, prob = lkdvRK.linforms(degree=1,tstages=1)

    pre = spsla.spilu(params['A'], drop_tol=1e-4,
                      fill_factor=10) #Preconditioning important for
                                      #high order discretisations
    
    x, solvedict = ls.gmresWrapper(params,
                                   x0=np.zeros_like(params['b']),
                                   k=k,
                                   pre=pre,
                                   tol=tol)

    
    x_con, geodict = ls.cgmresWrapper(params,prob=prob,
                                      x0=np.zeros_like(params['b']),
                                      pre=pre,
                                      k=k, tol=tol)

    x_dir = spsla.spsolve(params['A'],params['b'])


    print('cgmres error =', np.max(np.abs(x_con-x_dir)/x_dir))
    print('gmres error =', np.max(np.abs(x-x_dir)/x_dir))
    
    inv = lkdvRK.compute_invariants(params,prob,x)
    print('gmres mass deviation =', inv['mass']-params['m0'])
    print('gmres momentum deviation =', inv['momentum']-params['mo0'])
    print('gmres energy deviation =', inv['energy']-params['e0'])
    
    invcon = lkdvRK.compute_invariants(params,prob,x_con)
    print('cgmres mass deviation =', invcon['mass']-params['m0'])
    print('cgmres momentum deviation =', invcon['momentum']-params['mo0'])
    print('cgmres energy deviation =', invcon['energy']-params['e0'])
    
    #compute invariants for direct solve
    invdir = lkdvRK.compute_invariants(params,prob,x_dir)
    print('direct solver mass deviation =', invdir['mass']-params['m0'])
    print('direct solver momentum deviation =', invdir['momentum']-params['mo0'])
    print('direct solver energy deviation =', invdir['energy']-params['e0'])
