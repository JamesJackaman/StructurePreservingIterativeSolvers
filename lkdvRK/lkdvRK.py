'''
Linear system set up and other functions requiring Firedrake
'''
#Global imports
from firedrake import *
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import irksome as irk
import copy

#local
import refd

#Initialise problem, with customisable 
class problem(object):
    def __init__(self,N,M,degree,tstages,space='CG',T=1):
        self.mlength = 40
        self.degree = degree
        self.tstages = tstages
        self.dim = 3
        self.N = N
        self.M = M
        self.dt = float(T)/N
        self.space = space
        self.mesh = PeriodicIntervalMesh(self.M,self.mlength)
        #higher order stuff
        self.butcher_tableau = irk.GaussLegendre(tstages)
        self.ns = self.butcher_tableau.num_stages
        self.nf = self.dim #hard coded for problem

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
        

#Build linear system and vectors/matrices needed for constraints
def linforms(N=100,M=50,degree=1,tstages=2,T=10,t=Constant(0),zinit=None):
    #set up problem class
    prob = problem(N=N,M=M,T=T,degree=degree,tstages=tstages)
    #Set up finite element stuff
    mesh = prob.mesh
    Z = prob.function_space(mesh)

    #set up DG stuff
    n = FacetNormal(mesh)
    h = prob.mlength/M
    sigma = 10/h
    #including some definitions
    def gfunc(uh,vh):
        g = uh.dx(0)*vh*dx - jump(uh,n[0])*avg(vh)*dS 
        return g

    #Find static v
    def v_finder(uh):
        wh = w_finder(uh)
        vh = Function(Z.sub(0))
        phi = TestFunction(Z.sub(0))
        form = vh * phi * dx - uh * phi * dx - gfunc(wh,phi)
        solve(form==0,vh,
              solver_parameters={'ksp_type': 'preonly',
                                 'pc_type': 'lu'})
        return vh
    
    #Find static w
    def w_finder(uh):
        wh = Function(Z.sub(0))
        phi = TestFunction(Z.sub(0))
        form = wh * phi * dx - gfunc(uh,phi)
        solve(form==0,wh,
              solver_parameters={'ksp_type': 'preonly',
                                 'pc_type': 'lu'})
        return wh
    
    #Set up initial conditions
    z0 = Function(Z)
    u0, v0, w0 = z0.split()

    x = SpatialCoordinate(Z.mesh())
    if zinit==None:
        u0.assign(project(prob.exact(x[0],t),Z.sub(0)))
        v0.assign(v_finder(u0)) #Assign v
        w0.assign(w_finder(u0)) #Assign w
    else:
        z0.sub(0).assign(zinit.sub(0))
        z0.sub(2).assign(zinit.sub(2))


    #Build weak form
    phi, psi, chi = TestFunctions(Z)
    z_trial = TrialFunction(Z)
    z1 = Function(Z)
    u, v, w = split(z1)
    z1.assign(z0)
    u_trial, v_trial, w_trial = split(z_trial)

    F1 = irk.Dt(u) * phi * dx + gfunc(v,phi)
    F2 = (v - u) * psi * dx \
        - gfunc(w,psi)
    F3 = w*chi*dx - gfunc(u,chi)
    F = F1 + F2 + F3

    Fbig, zbig, _, _, _ = irk.getForm(F, prob.butcher_tableau,
                                      Constant(t), Constant(prob.dt),
                                      z1,
                                      bc_type="ODE")
    
    G = derivative(Fbig,zbig)

    
    #Read out A and b
    A = assemble(lhs(G),mat_type='aij').M.handle.getValuesCSR()
    A = sp.sparse.csr_matrix((A[2],A[1],A[0]))
    b = refd.combine(assemble(rhs(Fbig)).dat.data)

    #Get form for mass matrix
    M_form = u_trial * phi * dx
    M = assemble(lhs(M_form),mat_type='aij').M.handle.getValuesCSR()
    M = sp.sparse.csr_matrix((M[2],M[1],M[0]))
    #And for L
    L_form = w_trial * chi * dx
    L = assemble(lhs(L_form),mat_type='aij').M.handle.getValuesCSR()
    L = sp.sparse.csr_matrix((L[2],L[1],L[0]))
    #And the vector needed for finding the mass
    omega = np.asarray(assemble(phi * dx).dat.data).reshape(-1)

    #Get the initial values for the invariants
    u0, v0, w0 = split(z0)
    m0 = assemble(u0*dx)
    mo0 = assemble(0.5*u0**2*dx)
    e0 = assemble(0.5*(w0**2 - u0**2)*dx)

    #Generate output dictionary
    out = {
        'A': A,
        'b': b,
        'M': M,
        'L': L,
        'omega': omega,
        'm0': m0,
        'mo0': mo0,
        'e0': e0,
        'T': T,
        'z0': refd.flatten(z0.dat.data),
    }

    return out, prob


#A function to reconstruct the solution z1 using RK formula
#from the stage values (zbig) at z0
def z1calc(prob,zbig,z0):
    dt = prob.dt #timestep
    b = prob.butcher_tableau.b
    ns = prob.ns #num of stages
    dof = len(z0) #dof of space

    z1 = copy.deepcopy(z0) #Initialise output vector

    #Reconstruct solution vector at z1
    for s in range(ns):
        z1 += dt * b[s] * zbig[s*dof:(s+1)*dof]

    return z1

#Computes values of invariants given a vector zbig
def compute_invariants(params,prob,zbig):

    #set up DG stuff
    n = FacetNormal(prob.mesh)
    h = prob.mlength/prob.M
    sigma = 10/h
    #including some definitions
    def bilin(uh,vh):
        a = uh.dx(0)*vh.dx(0)*dx
        b = - (jump(uh,n[0])*avg(vh.dx(0)) 
               +jump(vh,n[0])*avg(uh.dx(0)) ) *dS
        d = (sigma) * jump(uh,n[0])*jump(vh,n[0]) *dS
        return a + b + d

    #map zs (zbig) to z1
    z1vec = z1calc(prob,zbig,params['z0'])
    
    z = refd.nptofd(prob,z1vec)
    u,v,w = z.split()
    mass = assemble(u*dx)
    momentum = assemble(0.5*u**2*dx)
    energy = assemble( 0.5 * (w**2 - u**2)*dx)

    inv_dict = {'mass' : mass,
                'momentum': momentum,
                'energy' : energy}

    return inv_dict

#Compute the error of z1 at time t given the solution
#at stage values zbig
def compute_error(params,prob,zbig,t):
    #Construct solution
    z = refd.nptofd(prob,z1calc(prob,zbig,params['z0']))

    #Get mesh coordinates
    Z = z.function_space()
    mesh = Z.mesh()
    x, = SpatialCoordinate(mesh)

    #Get exact solution
    ex_ = prob.exact(x,t)
    ex = Function(Z)
    ex.sub(0).assign(interpolate(ex_,Z.sub(0)))

    #Compute error in u
    err = assemble((ex[0]-z[0])**2*dx)**0.5
    
    return err


if __name__=="__main__":
    dict, prob = linforms()
    print(dict)
