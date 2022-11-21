#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt

#local
import swe
import refd
import LinearSolver as ls
import visualise as vis


if __name__=="__main__":

    params, prob = swe.linforms(degree=1)

    k = 20

    tol = 1e-50

    #Get initial conditions
    Z = prob.function_space(prob.mesh)
    z0 = Function(Z)
    x, y = SpatialCoordinate(prob.mesh)
    ic = prob.ic(x,y)
    z0.sub(0).assign(interpolate(ic[0],Z.sub(0)))
    z0.sub(1).assign(interpolate(ic[1],Z.sub(1)))

    #GMRES solve
    x, solvedict = ls.gmresWrapper(params,
                                   x0=np.zeros_like(params['b']),
                                   k=k,
                                   tol=tol)
    #CGMRES solve
    x_con, geodict = ls.cgmresWrapper(params,
                                      x0=np.zeros_like(params['b']),
                                      k=k,
                                      tol=tol)
    #Direct solve
    x_dir = spsla.spsolve(params['A'],params['b'])


    print('cgmres solver error =', np.max(np.abs(x_con-x_dir)/x_dir))
    print('gmres solver error =', np.max(np.abs(x-x_dir)/x_dir))

    inv = swe.compute_invariants(prob,x)
    print('gmres mass deviation =', inv['mass']-params['m0'])
    print('gmres energy deviation =', inv['energy']-params['e0'])

    invcon = swe.compute_invariants(prob,x_con)
    print('cgmres mass deviation =', invcon['mass']-params['m0'])
    print('cgmres energy deviation =', invcon['energy']-params['e0'])
    
    #compute invariants for direct solve
    invdir = swe.compute_invariants(prob,x_dir)
    print('direct solver mass deviation =', invdir['mass']-params['m0'])
    print('direct solver energy deviation =', invdir['energy']-params['e0'])
    
    input('hit enter for result tabulation')

    #Generate table of results
    table = vis.tabulator(params,prob,[solvedict,geodict])

    #Save results as figure
    vis.convergence_plot(table)
