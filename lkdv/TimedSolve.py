#global
from firedrake import *
import numpy as np
import sys
import matplotlib.pylab as plt

#local
import lkdv
import refd
import LinearSolver as ls


if __name__=="__main__":

    #Specify tolerance to be so small prototypical CGMRES is used with
    #no stopping criteria
    tol=1e-8

    #Set up linear system
    params, prob = lkdv.linforms(degree=1,M=100)

    k = 100

    #GMRES solve
    x, solvedict = ls.gmresWrapper(params,
                                   x0=np.zeros_like(params['b']),
                                   k=k,
                                   tol=tol,
                                   timing=True)
    
    print('steps = ', solvedict['steps'])
    print('timings =', solvedict['timings'])
    input('GMRES solve end')
    
    #CGMRES solve
    x_con, geodict = ls.cgmresWrapper(params,
                                      x0=np.zeros_like(params['b']),
                                      k=k, tol=tol,
                                      timing=True)

    print('steps =', geodict['steps'])
    print('timings =', geodict['timings'])

    input('CGMRES solve end')


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

    #Generate table of results
    table = vis.tabulator(params,prob, [solvedict, geodict])
