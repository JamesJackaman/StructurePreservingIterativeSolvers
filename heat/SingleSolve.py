'''
Solve for a single time step of the heat equation and study
linear solver convergence
'''
#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt

#local
import heat
import refd
import LinearSolver as ls
import visualise as vis

if __name__=="__main__":

    params, prob = heat.linforms(degree=1)

    k = 20
    tol = 1e-50

    #Define preconditioner
    M = spsla.spilu(params['A'], drop_tol=1e-3,
                    fill_factor = 10)

    M = None #comment out this line to turn on preconditioning

    #Get initial conditions
    Z = prob.function_space(prob.mesh)
    z0 = Function(Z)
    x, y = SpatialCoordinate(prob.mesh)
    z0.assign(project(prob.ic(x,y),Z))

    #GMRES solve
    x, solvedict = ls.gmresWrapper(params,
                                   x0=np.zeros_like(params['b']),
                                   k=k,
                                   tol=tol,
                                   pre=M)
    #Append old value of z to dictionary (can be deleted?)
    solvedict['z0'] = z0

    #CGMRES solve
    x_con, geodict = ls.cgmresWrapper(params,
                                      x0=np.zeros_like(params['b']),
                                      k=k,
                                      tol=tol,
                                      pre=M)
    #Append old value of z to dictionary
    geodict['z0'] = z0

    #Direct solve
    x_dir = spsla.spsolve(params['A'],params['b'])


    print('cgmres error =', np.max(np.abs(x_con-x_dir)/x_dir))
    print('gmres error =', np.max(np.abs(x-x_dir)/x_dir))


    inv = heat.compute_invariants(prob,x,z0.dat.data)
    print('gmres mass deviation =', inv['mass']-params['m0'])
    print('gmres energy deviation =', inv['energy']-params['e0'])

    invcon = heat.compute_invariants(prob,x_con,z0.dat.data)
    print('cgmres mass deviation =', invcon['mass']-params['m0'])
    print('cgmres energy deviation =', invcon['energy']-params['e0'])

    invdir = heat.compute_invariants(prob,x_dir,z0.dat.data)
    print('direct solver mass deviation =', invdir['mass']-params['m0'])
    print('direct solver energy deviation =', invdir['energy']-params['e0'])
    
    input('hit enter for result tabulation')
    
    #Generate table of results
    table = vis.tabulator(params,prob,[solvedict,geodict])

    #Save results as figure
    vis.convergence_plot(table)
        
