'''
Linear system set up and other heat equation specific functions
'''
#Global imports
from firedrake import *
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import sys

#local
import refd

#Define class containing problem parameters and other globally useful
#objects
class problem(object):
    def __init__(self,N,M,degree,T):
        self.degree = degree #Polynomial degree
        self.N = N #number of temporal nodes
        self.M = M #number of spatial elements
        self.T = T #end time
        self.dt = float(T)/N #time step
        self.mesh = UnitSquareMesh(self.M,self.M)

    def function_space(self,mesh):
        CG = FunctionSpace(mesh,'CG',self.degree)
        return CG

    #Initial condition
    def ic(self,x,y):
        u = 1e3 * ( (x * (x-1))**5 + (y * (y - 1))**6 ) #a big bi-polynomial
        return u
        

def linforms(N=100,M=50,degree=1,T=10,zinit=None):
    #set up problem class
    prob = problem(N=N,M=M,degree=degree,T=T)
    #Set up finite element stuff
    mesh = prob.mesh
    Z = prob.function_space(mesh)
    
    #set up DG stuff
    n = FacetNormal(mesh)
    
    #Set up initial conditions
    z0 = Function(Z)

    t = 0.
    x = SpatialCoordinate(Z.mesh())
    
    if zinit==None:
        z0.assign(project(prob.ic(x[0],x[1]),Z))
    else:
        z0.assign(zinit)

    # tripcolor(z0)
    # plt.show()

    #Define timestep
    dt = prob.dt
    
    #Build weak form
    phi = TestFunction(Z)
    z1 = Function(Z)
    z_trial = TrialFunction(Z)
    z1.assign(z0)

    zt = (z_trial - z0) / dt
    zmid = 0.5 * (z_trial + z0)

    F = zt * phi * dx \
        + inner(grad(zmid), grad(phi)) * dx

    
    #Read out A and b
    A = assemble(lhs(F),mat_type='aij').M.handle.getValuesCSR()
    A = sp.sparse.csr_matrix((A[2],A[1],A[0]))
    b = assemble(rhs(F)).dat.data


    #And for M
    M_form = inner(z_trial, phi) * dx
    M = assemble(lhs(M_form),mat_type='aij').M.handle.getValuesCSR()
    M = sp.sparse.csr_matrix((M[2],M[1],M[0]))
    
    #And for L
    L_form = inner(grad(z_trial), grad(phi)) * dx 
    L = assemble(lhs(L_form),mat_type='aij').M.handle.getValuesCSR()
    L = sp.sparse.csr_matrix((L[2],L[1],L[0]))

    #And for the vector Lz0
    z_ = z0.dat.data
    Lz0 = L @ z_

    #And for the old 'energy'
    old_energy = 0.5 * z_ @ M @ z_ - 0.25 * prob.dt * z_ @ L @ z_
    
    #And don't forget mass
    omega = assemble(phi * dx).dat.data
    
    #Get the initial values for the invariants
    m0 = assemble(z0*dx)
    e0 = 0
    

    #Generate output dictionary
    out = {
        'A': A,
        'b': b,
        'M': M,
        'Lz0': Lz0,
        'old_energy': old_energy,
        'omega': omega,
        'L': L,
        'm0': m0,
        'e0': e0,
        'z0': z_, #initial vector
        'dt': prob.dt,
    }
        
    return out, prob


def compute_invariants(prob,uvec,uold):

    #set up DG stuff
    n = FacetNormal(prob.mesh)
        
    z = refd.nptofd(prob,uvec)
    try:
        z_ = refd.nptofd(prob,uold)
    except:
        z_ = uold

    zmid = 0.5 * (z + z_)
    mass = assemble(z*dx)
    energy = assemble(0.5 * z**2*dx - 0.5 * z_**2*dx
                      +prob.dt * inner(grad(zmid),grad(zmid))*dx)
    
    inv_dict = {'mass' : mass,
                'energy' : energy}

    return inv_dict


if __name__=="__main__":
    dict, prob = linforms()
    print(dict)


