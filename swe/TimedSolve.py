#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import scipy.sparse as sps
import pandas as pd

#local
import swe
import refd
import LinearSolver as ls
import visualise as vis


#Get timings for single GMRES solve
def time_cgmres(M=2**3,degree=1,tol=1e-7,k=20):
    #Build system
    params, prob = swe.linforms(degree=degree,M=M)
    #preconditioner
    M = sps.linalg.spilu(params['A'], drop_tol=1e-4,
                         fill_factor=10)
    #Run a regular (more optimised GMRES solve)
    gmres_x, solvedict = ls.gmresWrapper(params,
                                         x0=np.zeros_like(params['b']),
                                         k=k,
                                         tol=tol,
                                         pre=M)
    
    #Run timed solve
    cgmres_x, geodict = ls.cgmresWrapper(params,
                                         x0=np.zeros_like(params['b']),
                                         k=k,
                                         tol=tol,
                                         pre=M,
                                         timing=True)
    #Check for gain in constraint enforcement (i.e., that cgmres is
    #doing something)
    gmres_inv = swe.compute_invariants(prob,gmres_x)
    cgmres_inv = swe.compute_invariants(prob,cgmres_x)
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
    time_constraint = []
    time_pre = []
    steps_unconstrained = []
    steps_constrained = []
    
    #Compute timings
    for i in range(3,10):
        M.append(2**i)
        out = time_cgmres(M=M[-1])
        runtime.append(out['runtime'])
        time_unconstrained.append(out['iter_time_unconstrained'])
        time_constrained.append(out['iter_time_constrained'])
        time_pre.append(out['pretime'])
        time_constraint.append(out['constraint_building'])
        steps_constrained.append(out['constrained_steps'])
        steps_unconstrained.append(out['unconstrained_steps'])

    #Tabulate
    d = {'M': M,
         'Run time': runtime,
         'Preconditioning time': time_pre,
         'Average unconstrained iteration time': time_unconstrained,
         'Number of unconstrained iterations': steps_unconstrained,
         'Average overhead from building constraints': time_constraint,
         'Average constrained iteration time': time_constrained,
         'Number of constrained iterations': steps_constrained}
    df = pd.DataFrame(data=d)

    print(df.to_markdown())
