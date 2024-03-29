#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import scipy.sparse as sps
import pandas as pd
from time import time
try:
    import pyamg
except:
    import os
    input('pyamg required for preconditioning, press enter to pip install')
    os.system('pip install pyamg')
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
    start_pre = time()
    ml = pyamg.ruge_stuben_solver(params['A'])
    pre = ml.aspreconditioner(cycle='V')
    end_pre = time()

    #Get initial conditions
    Z = prob.function_space(prob.mesh)
    z0 = Function(Z)
    x, y = SpatialCoordinate(prob.mesh)
    z0.assign(project(prob.ic(x,y),Z))
    
    #Run a regular (more optimised GMRES solve)
    start_gmres = time()
    gmres_x, solvedict = ls.gmresWrapper(params,
                                         x0=np.zeros_like(params['b']),
                                         k=k,
                                         tol=tol,
                                         pre=pre)
    end_gmres = time()

    start_optimal = time()
    # pak_x, _ = pyamg.krylov.fgmres(params['A'], params['b'],
    #                               x0=np.zeros_like(params['b']),
    #                               maxiter=k,
    #                               tol=tol,
    #                               M=pre)
    end_optimal = time()
    
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
    gmres_mass = abs(gmres_inv['mass']-params['m0'])
    gmres_energy = abs(gmres_inv['energy']-params['e0'])
    
    cgmres_inv = heat.compute_invariants(prob,cgmres_x,z0.dat.data)
    cgmres_mass = abs(cgmres_inv['mass']-params['m0'])
    cgmres_energy = abs(cgmres_inv['energy']-params['e0'])
    
    if not (cgmres_mass < 0.5 * gmres_mass):
        warning('CGMRES does not lead to a significant improvement in mass '
                + 'with M=%s and tol=%s' % (M, tol))
    if not (cgmres_energy < 0.5 * gmres_energy):
        warning('CGMRES does not lead to a significant improvement in energy '
                + 'with M=%s and tol=%s' % (M, tol))
    #Extract timings
    out = geodict['timings']
    #Append additional information
    out['unconstrained_steps'] = geodict['steps'] - out['constrained_steps']
    out['time_pre'] = end_pre - start_pre
    out['time_gmres'] = end_gmres - start_gmres
    out['time_optimal'] = end_optimal - start_optimal
    out['conservation'] = {'gmres_mass': gmres_mass,
                           'gmres_energy': gmres_energy,
                           'cgmres_mass': cgmres_mass,
                           'cgmres_energy': cgmres_energy}


    return out


if __name__=="__main__":

    #Initialise tables
    M = []
    runtime = []
    time_unconstrained = []
    time_constrained = []
    time_pre = []
    time_gmres = []
    time_optimal = []
    time_constraint = []
    steps_unconstrained = []
    steps_constrained = []
    mass_gain = []
    energy_gain = []
    
    #Compute timings
    for i in range(4,12):
        M.append(2**i)
        out = time_cgmres(M=M[-1])
        runtime.append(out['runtime'])
        time_unconstrained.append(out['iter_time_unconstrained'])
        time_constrained.append(out['iter_time_constrained'])
        time_constraint.append(out['constraint_building'])
        time_pre.append(out['time_pre'])
        time_gmres.append(out['time_gmres'])
        time_optimal.append(out['time_optimal'])
        steps_constrained.append(out['constrained_steps'])
        steps_unconstrained.append(out['unconstrained_steps'])
        con = out['conservation']
        mass_gain.append(max(con['gmres_mass'],1e-16)/max(con['cgmres_mass'],1e-16))
        energy_gain.append(max(con['gmres_energy'],1e-16)/max(con['cgmres_energy'],1e-16))

        
    #Tabulate
    d = {'M': M,
         'Preconditioning time': time_pre,
         'GMRES run time': time_gmres,
         'CGMRES run time': runtime,
         'Average unconstrained iteration time': time_unconstrained,
         'Number of unconstrained iterations': steps_unconstrained,
         'Average overhead from building constraints': time_constraint,
         'Average constrained iteration time': time_constrained,
         'Number of constrained iterations': steps_constrained,
         'Gain in mass conservation': mass_gain,
         'Gain in energy conservation': energy_gain}
    df = pd.DataFrame(data=d)

    #Improve formatting (need to switch to_markdown -> to_latex for
    #this to work)
    tf = '{:.2e}'
    cf = '{:.1e}'
    format_mapping = {#
        'Preconditioning time': tf,
        'GMRES run time': tf,
        'CGMRES run time': tf,
        'Average unconstrained iteration time': tf,
        'Average overhead from building constraints': tf,
        'Average constrained iteration time': tf,
        'Gain in mass conservation': cf,
        'Gain in energy conservation': cf,
    }
    for key, value in format_mapping.items():
        df[key] = df[key].apply(value.format)

    print(df.to_markdown(index=None))
