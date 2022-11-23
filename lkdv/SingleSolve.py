#global
from firedrake import *
import numpy as np
import sys
import matplotlib.pylab as plt

#local
import lkdv
import refd
import LinearSolver as ls
import visualise as vis


if __name__=="__main__":

    #Specify tolerance to be so small prototypical CGMRES is used with
    #no stopping criteria
    tol=1e-50

    #Set up linear system
    params, prob = lkdv.linforms(degree=1)

    k = 20

    #GMRES solve
    x, solvedict = ls.gmresWrapper(params,
                                   x0=np.zeros_like(params['b']),
                                   k=k,
                                   tol=tol)
    #CGMRES solve
    x_con, geodict = ls.cgmresWrapper(params,
                            x0=np.zeros_like(params['b']),
                            k=k, tol=tol)

    #Direct solve
    x_dir, _ = ls.exact(params)


    print('cgmres error =', np.max(np.abs(x_con-x_dir)/x_dir))
    print('gmres error =', np.max(np.abs(x-x_dir)/x_dir))

    inv = lkdv.compute_invariants(prob,x)
    print('gmres mass deviation =', inv['mass']-params['m0'])
    print('gmres momentum deviation =', inv['momentum']-params['mo0'])
    print('gmres energy deviation =', inv['energy']-params['e0'])

    invcon = lkdv.compute_invariants(prob,x_con)
    print('cgmres mass deviation =', invcon['mass']-params['m0'])
    print('cgmres momentum deviation =', invcon['momentum']-params['mo0'])
    print('cgmres energy deviation =', invcon['energy']-params['e0'])

    #compute invariants for direct solve
    invdir = lkdv.compute_invariants(prob,x_dir)
    print('direct solver mass deviation =', invdir['mass']-params['m0'])
    print('direct solver momentum deviation =', invdir['momentum']-params['mo0'])
    print('direct solver energy deviation =', invdir['energy']-params['e0'])
    
    input('hit enter for result tabulation')

    #Generate table of results
    table = vis.tabulator(params,prob, [solvedict, geodict])

    #Save results as figure
    vis.convergence_plot(table)
