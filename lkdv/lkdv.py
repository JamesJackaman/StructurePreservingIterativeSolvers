'''
Linear system set up and other lkdv specific functions
'''
#Global imports
from firedrake import *
import numpy as np
import matplotlib.pylab as plt

#local
import refd

#Define class containing problem parameters and other globally useful
#objects
class problem(object):
    def __init__(self,N,M,degree,space):
        self.mlength = 40 #length of spatial mesh
        self.degree = degree #spatial polynomial degree
        self.dim = 3 #Save the dimension of the problem
        self.N = N #number of temporal nodes
        self.M = M #number of spatial elements
        self.space = space #finite element space, can be CG or dG.
        self.mesh = PeriodicIntervalMesh(self.M,self.mlength)

    def function_space(self,mesh):
        U = FunctionSpace(mesh,self.space,self.degree)
        return MixedFunctionSpace((U,U,U))

    def exact(self,x,t):
        """
        An "exact" initial condition for the linear KdV equation
        """
        alpha = 4
        period = 2*pi/self.mlength
        beta = alpha*period
        u = sin(beta*(x-(1-beta**2)*t)) + 1
        return u
        
'''

Build linear system and vectors/matrices needed for constraints
zinit - the initial condition in terms of the linear system, if None
project exact solution at initial time into function space

'''
def linforms(N=100,M=50,degree=1,T=1,space='DG',zinit=None):
    #set up problem class
    prob = problem(N=N,M=M,degree=degree,space=space)
    #Set up finite element stuff
    mesh = prob.mesh
    Z = prob.function_space(mesh)

    #set up DG stuff (if needed)
    n = FacetNormal(mesh)
    h = prob.mlength/M
    sigma = 10/h
    
    #include first spatial derivative and best approximation of it
    def gfunc(uh,vh):
        g = uh.dx(0)*vh*dx - jump(uh,n[0])*avg(vh)*dS 
        return g
    def gfuncproject(u,space):
        p_func = Function(space)
        p_test = TestFunction(space)
        p_form = p_func*p_test*dx - gfunc(u,p_test)
        solve(p_form==0,p_func,
          solver_parameters={'ksp_type': 'preonly',
                             'pc_type': 'lu'})
        return p_func
    
    #Set up initial conditions
    z0 = Function(Z)
    u0,v0,w0 = z0.split()

    t = 0.
    x = SpatialCoordinate(Z.mesh())
    if zinit==None:
        u0.assign(project(prob.exact(x[0],t),Z.sub(0)))
        w0.assign(gfuncproject(u0,Z.sub(2)))
    else:
        u0.assign(zinit.sub(0))
        w0.assign(zinit.sub(2))

    #Define timestep
    dt = float(T)/N
    
    #Build weak form
    phi, psi, chi = TestFunctions(Z)
    z1 = Function(Z)
    z_trial = TrialFunction(Z)
    u_trial, v_trial, w_trial = split(z_trial)
    z1.assign(z0)

    u0, v0, w0 = split(z0)

    ut = (u_trial - u0) / dt
    umid = 0.5 * (u_trial + u0)
    wmid = 0.5 * (w_trial + w0)

    F1 = (ut) * phi * dx + gfunc(v_trial,phi)
    F2 = (v_trial - umid) * psi * dx \
        - gfunc(wmid, psi)
    F3 = w_trial*chi*dx - gfunc(u_trial,chi)
    
    F = F1 + F2 + F3

    
    #Read out A and b
    A = assemble(lhs(F),mat_type='aij').M.values
    b = np.asarray(assemble(rhs(F)).dat.data).reshape(-1)

    #Get form for mass matrix
    M_form = u_trial * phi * dx
    M = assemble(lhs(M_form),mat_type='aij').M.values
    #And for L
    L_form = w_trial * chi * dx
    L = assemble(lhs(L_form),mat_type='aij').M.values
    #And the vector needed for finding the mass
    omega = np.asarray(assemble(phi * dx).dat.data).reshape(-1)

    #Get the initial values for the invariants
    m0 = assemble(u0*dx)
    mo0 = assemble(0.5*u0**2*dx)
    e0 = assemble((0.5 * w0**2 - 0.5 * u0**2)*dx)

    #Get initial vector
    u0, v0, w0 = z0.split()
    z_vec = np.asarray(assemble(z0).dat.data).reshape(-1)


    #Generate output dictionary
    out = {
        'A': A,
        'b': b,
        'z0': z_vec,
        'M': M,
        'L': L,
        'omega': omega,
        'm0': m0,
        'mo0': mo0,
        'e0': e0,
        'T': T,
    }
        
    return out, prob


'''
Given the solution vector, compute value of conserved quantity.
'''
def compute_invariants(prob,uvec):

    z = refd.nptofd(prob,uvec)
    u,v, w = z.split()
    mass = assemble(u*dx)
    momentum = assemble(0.5*u**2*dx)
    energy = assemble((0.5*w**2 - 0.5 * u**2)*dx)

    inv_dict = {'mass' : mass,
                'momentum' : momentum,
                'energy' : energy}

    return inv_dict


if __name__=="__main__":
    dict, prob = linforms()
    print(dict)

    refd.nptofd(prob,dict['b'])
