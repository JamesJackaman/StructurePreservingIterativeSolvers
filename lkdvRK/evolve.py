"""
Study solution evolution
"""
#global
from firedrake import *
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse.linalg as spsla
import os

#local
import lkdvRK
import refd
import LinearSolver as ls

'''
Solve over all time steps
'''
def evolve(N=10,M=50,degree=1,tstages=2,T=1,
           k=50,tol=1e-6,ctol=1e-12,contol=10,
           solver=ls.cgmresWrapper,
           counter=None,solver_str=None):
    
    #Get linear forms and Firedrake problem class
    forms, prob = lkdvRK.linforms(N=N,M=M,degree=degree,tstages=tstages,T=T)
    mesh = prob.mesh
    U = prob.function_space(mesh)
    #Get bilin function
    n = FacetNormal(mesh)
    h = prob.mlength/M
    sigma = 10/h

    #Initialise solution array
    sol = [refd.nptofd(prob,forms['z0'])]

    #Initalise solver initial guess by repeating initial conditions
    z = np.tile(forms['z0'],prob.ns)

    #Initialise conserved quantities
    time = [0]
    mass = [forms['m0']]
    momentum = [forms['mo0']]
    energy = [forms['e0']]
    #And error
    err = [0]

    #Initialise FD solution function
    z0 = Function(U)

    #precondition
    pre = spsla.spilu(forms['A'], drop_tol=1e-4,
                      fill_factor=10)
    
    #Iteratively solve the remaining steps
    for i in range(1,N+1):
        #Update forms
        forms, _ = lkdvRK.linforms(N=N,M=M,T=T,
                                   degree=degree,tstages=tstages,
                                   zinit=sol[-1])
        #Solve
        z, sdict = solver(forms, x0=z, prob = prob, pre = pre,
                          k=k, tol=tol, ctol=ctol, contol=contol)
        if counter==True:
            #Save number of iterations required
            steps.append(sdict['steps'])
            esteps.append(sdict['esteps'])
        #Convert back to FD
        z_ = refd.flatten(sol[-1].dat.data)
        z0.assign(refd.nptofd(prob,lkdvRK.z1calc(prob,z,z_)))
        #Compute conserved quantities
        u0 = z0.sub(0)
        w0 = z0.sub(2)
        u0.function_space().mesh()
        mass.append(assemble(u0*dx))
        momentum.append(assemble(0.5*u0**2*dx))
        energy.append(assemble(0.5 * w0**2*dx - 0.5 * u0**2*dx))
        #Append
        sol.append(z0)
        time.append(forms['T']/N * i)

        #compute error
        err_ = lkdvRK.compute_error(forms,prob,z,t=time[-1])
        err.append(err_)

        
    out = {'sol': sol,
           'time': time,
           'err': err,
           'dm': np.abs(mass-mass[0]),
           'dmo': np.abs(momentum-momentum[0]),
           'de': np.abs(energy-energy[0])}            
        
    return out


'''
Plot deviations of conserved quantities over time with GMRES and CGMRES
at a user-specified tolerance
'''
def DeviationPlotter(tol=1e-6,
                     filename=os.path.dirname(os.getcwd())+'/plots/lkdvRKEvolve.pdf'):
    #Standard solve
    standard = evolve(tol=tol,solver=ls.gmresWrapper)
    del standard['sol'] #Printing sol not meaningful, it's a Firedrake
                        #object
    print('standard =', standard)
    conserved = evolve(tol=tol,solver=ls.cgmresWrapper)
    del conserved['sol']
    print('conserved =', conserved)    
    
    lw = 2
    
    #Standard plots
    plt.semilogy(standard['time'],standard['dm'],
                 color='r',linestyle='dotted', linewidth=lw,
                 label='GMRES: Mass deviation')
    plt.semilogy(standard['time'],standard['dmo'],
                 color='r',linestyle='dashed', linewidth=lw,
                 label='GMRES: Momentum deviation')
    plt.semilogy(standard['time'],standard['de'],
                 color='r',linestyle='dashed', linewidth=lw,
                 label='GMRES: Energy deviation')

    #Conservative plots
    plt.semilogy(conserved['time'],conserved['dm'],
                 color='b',linestyle='dotted', linewidth=lw,
                 label='CGMRES: Mass deviation')
    plt.semilogy(conserved['time'],conserved['dmo'],
                 color='b',linestyle='dotted', linewidth=lw,
                 label='CGMRES: Momentum deviation')
    plt.semilogy(conserved['time'],conserved['de'],
                 color='b',linestyle='dashed', linewidth=lw,
                 label='CGMRES: Energy deviation')

    plt.grid(which='major', linestyle='--',axis='y')
    plt.xlabel(r'$t$')

    plt.legend(loc='lower left',ncol=2, bbox_to_anchor=(-0.03,-0.5), borderaxespad=0)

    plt.tight_layout()
    
    plt.savefig(filename)

    return None


if __name__=="__main__":
    DeviationPlotter()
    
    
    
