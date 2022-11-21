"""
Study solution evolution
"""
#global
from firedrake import *
import numpy as np
import matplotlib.pylab as plt
import os

#local
import swe
import refd
import LinearSolver as ls

'''
Solve over all time steps
'''
def evolve(N=100,M=50,degree=1,k=50,tol=1e-6,ctol=1e-14,
           solver=ls.cgmresWrapper):
    #Get linear forms and Firedrake problem class
    forms, prob = swe.linforms(N=N,M=M,degree=degree)
    mesh = prob.mesh
    U = prob.function_space(mesh)
    
    #Initialise solution array
    sol = []
    
    #save initial condition
    z0 = refd.nptofd(prob,forms['z0'])
    sol.append(z0)

    #Initialise conserved quantities
    time = [0]
    mass = [forms['m0']]
    energy = [forms['e0']]
    
    #Iteratively solve the remaining steps
    for i in range(1,N):
        #Update forms
        forms, _ = swe.linforms(N=N,M=M,degree=degree,zinit=sol[-1])
        #Initial guess at previous solution
        x0 = refd.flatten(sol[-1].dat.data)
        #Solve
        z, _ = solver(forms, x0=np.zeros_like(forms['b']), k=k, tol=tol, ctol=ctol)
        #Convert back to FD
        z0.assign(refd.nptofd(prob,z))
        #Compute conserved quantities
        inv = swe.compute_invariants(prob,z)
        mass.append(inv['mass'])
        energy.append(inv['energy'])
        #Append
        sol.append(z0)
        time.append(forms['T']/N * i)

        out = {'sol': sol,
               'time': time,
               'dm': np.abs(mass-mass[0]),
               'de': np.abs(energy-energy[0])}
        
    return out

#Plot deviations of conserved quantities over time with GMRES and
#CGMRES at a user-specified tolerance
def DeviationPlotter(tol=1e-6,
                     filename=os.path.dirname(os.getcwd())+'/plots/sweEvolve.pdf'):
    #Standard solve
    standard = evolve(tol=tol,solver=ls.gmresWrapper)
    conserved = evolve(tol=tol,solver=ls.cgmresWrapper)

    lw = 2

    #Standard plots
    plt.semilogy(standard['time'],standard['dm'],
                 color='r',linestyle='dotted',linewidth=2,
                 label='GMRES: Mass deviation')
    plt.semilogy(standard['time'],standard['de'],
                 color='r',linestyle='dashed',linewidth=2,
                 label='GMRES: Energy deviation')

    #Conservative plots
    plt.semilogy(conserved['time'],conserved['dm'],
                 color='b',linestyle='dotted',linewidth=2,
                 label='CGMRES: Mass deviation')
    plt.semilogy(conserved['time'],conserved['de'],
                 color='b',linestyle='dashed',linewidth=2,
                 label='CGMRES: Energy deviation')

    plt.grid(which='major', linestyle='--',axis='y')
    plt.xlabel(r'$t$')

    plt.legend(loc='lower left',ncol=2, bbox_to_anchor=(0.05,-0.3), borderaxespad=0)

    plt.tight_layout()
    
    plt.savefig(filename)

    return None
    



if __name__=="__main__":
    DeviationPlotter()
    
    
    
