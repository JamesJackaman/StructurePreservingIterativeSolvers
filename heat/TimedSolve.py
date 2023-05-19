#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import scipy.sparse as sps
import pandas as pd
import pyamg

#local
import heat
import refd
import LinearSolver as ls
import visualise as vis


#Get timings for single GMRES solve
def time_cgmres(M=2**3,degree=1,tol=1e-7,k=20):
    #Build system
    params, prob = heat.linforms(degree=degree,M=M)
    #preconditioner
    ml = pyamg.ruge_stuben_solver(params['A'])
    pre = ml.aspreconditioner(cycle='V')

    #Get initial conditions
    Z = prob.function_space(prob.mesh)
    z0 = Function(Z)
    x, y = SpatialCoordinate(prob.mesh)
    z0.assign(project(prob.ic(x,y),Z))
    
    #Run a regular (more optimised GMRES solve)
    gmres_x, solvedict = ls.gmresWrapper(params,
                                         x0=np.zeros_like(params['b']),
                                         k=k,
                                         tol=tol,
                                         pre=pre)
    
    #Run timed solve
    cgmres_x, geodict = ls.cgmresWrapper(params,
                                         x0=np.zeros_like(params['b']),
                                         k=k,
                                         tol=tol,
                                         pre=pre,
                                         timing=True)

    #Check for gain in constraint enforcement (i.e., that cgmres is
    #doing something)
    gmres_inv = heat.compute_invariants(prob,gmres_x,z0.dat.data)
    cgmres_inv = heat.compute_invariants(prob,cgmres_x,z0.dat.data)
    if not (abs(cgmres_inv['mass']-params['m0']) < 2 * abs(gmres_inv['mass']-params['m0'])):
        if not (abs(cgmres_inv['energy']-params['e0']) < 2*abs(gmres_inv['energy']-params['e0'])):
            warning('CGMRES does not lead to a significant improvement '
                    + 'in conservation with M=%s and tol=%s' % (M, tol))
    #Extract timings
    out = geodict['timings']
    #Append additional information
    out['unconstrained_steps'] = geodict['steps'] - out['constrained_steps']

    return out


if __name__=="__main__":

    #Initialise tables
    M = []
    runtime = []
    time_unconstrained = []
    time_constrained = []
    time_pre = []
    time_constraint = []
    steps_unconstrained = []
    steps_constrained = []
    
    #Compute timings
    for i in range(4,12):
        M.append(2**i)
        out = time_cgmres(M=M[-1])
        runtime.append(out['runtime'])
        time_unconstrained.append(out['iter_time_unconstrained'])
        time_constrained.append(out['iter_time_constrained'])
        time_constraint.append(out['constraint_building'])
        steps_constrained.append(out['constrained_steps'])
        steps_unconstrained.append(out['unconstrained_steps'])
        
    #Tabulate
    d = {'M': M,
         'Run time': runtime,
         'Average unconstrained iteration time': time_unconstrained,
         'Number of unconstrained iterations': steps_unconstrained,
         'Average overhead from building constraints': time_constraint,
         'Average constrained iteration time': time_constrained,
         'Number of constrained iterations': steps_constrained}
    df = pd.DataFrame(data=d)

    print(df.to_markdown())
