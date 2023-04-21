'''
main solve routines
'''
#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import scipy.sparse as sps
import scipy.optimize as spo
import warnings
from time import time

#FGMRES implementation, hand implemented for a fair comparison
def gmres(A, b, x0, k, tol = 1e-50, pre = None, timing=None):
    #Start timing
    if timing:
        t_start = time()
        
    #If not using preconditioner, set up identity as placeholder
    if pre is None:
        pre = sps.identity(len(b))

    if hasattr(pre, 'solve'):#Check if spsla.LinearOperator object
        def prefunc(vec):
            return pre.solve(vec)
    else:
        def prefunc(vec):
            try:
                out = pre @ vec
            except:
                raise ValueError('Preconditioner not supported')
            return out
        
    x = []
    residual = []
    
    r = (b - A.dot(x0)) #define r0
    
    x.append(r)
    residual.append(np.linalg.norm(r))
         
    q = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    h = np.zeros((k+1,k))

    if timing:
        t_iter_start = time() 
    for j in range(k):
        steps = j+1
        y = np.asarray(A @ prefunc(q[j]))
        
        for i in range(j+1):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0):
            q[j+1] =  y / h[j+1,j]
        else:
            warnings.warn('GMRES broke down, either initial guess is exact or' +
                          ' , more likely, something has gone wrong.')
            break

        res = np.zeros(j+2)
        res[0] = beta
        
        yk = np.linalg.lstsq(h[:j+2,:j+1], res, rcond=None)[0]
        
        x.append(prefunc(np.transpose(q[:j+1,:])) @ yk + x0)
        residual.append(np.linalg.norm(A.dot(x[-1]) - b))
        if timing:
            t_iter = (time() - t_iter_start) / steps
        if residual[-1] < tol:
            break

    #Report timings
    if timing:
        t_end = time()
        timings = {'runtime': t_end - t_start,
                   'iter_time': t_iter}
    else:
        timings = {}

    #Build output dictionary
    dict = {'name': 'gmres',
            'x':x,
            'res':residual[1:],
            'steps': steps,
            'timings': timings}
    
    return x[-1], dict


#Standard CGMRES implementation
def cgmres(A, b ,x0, k,
           tol=1e-8,
           contol=10,#contol*tol is when constraints are first enforced
           conlist=[],
           pre = None,
           timing = None):

    if timing:
        t_start = time()
    
    ctol = 1e-12 #specify the tolerance to which constraints *must* be
                 #enforced
    
    #If not using preconditioner, set up identity as placeholder
    if pre is None:
        pre = sps.identity(len(b))

    if hasattr(pre, 'solve'):#Check if spsla.LinearOperator object
        def prefunc(vec):
            return pre.solve(vec)
    else:
        def prefunc(vec):
            try:
                out = pre @ vec
            except:
                raise ValueError('Preconditioner not supported')
            return out

    safety = None #Set up switch to make sure constraints are enforced
        
    x = []
    residual = []
    r = (b - A.dot(x0)) #define r0

    x.append(r)
    residual.append(np.linalg.norm(r))
         
    q = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    #convert type
    
    h = np.zeros((k+1,k))

    if timing:
        t_iter_start = time()
        constrained_steps = 0
    for j in range(k):
        steps = j+1
        y = np.asarray(A @ prefunc(q[j]))
        
        for i in range(j+1):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0):
            q[j+1] =  y / h[j+1,j]
        else:
            warnings.warn('GMRES broke down, either initial guess is exact or' +
                          ' , more likely, something has gone wrong.')
            break

        res = np.zeros(j+2)
        res[0] = beta

        Q = np.transpose(q[:j+1,:]) #Allocate current size of Q
        Qt = np.transpose(Q)

        #Set up function
        def func(z):
            F = res - h[:j+2,:j+1] @ z
            out = np.inner(F,F)
            return out

        def jac(z):
            out = np.zeros_like(z)

            #original term
            F = res - h[:j+2,:j+1] @ z
            #Component wise differentiation of F
            for m in range(len(z)):
                ej = np.zeros_like(z)
                ej[m] = 1 
                dF = - h[:j+2,:j+1] @ ej

                #assemble j-th component of jac
                out[m] = 2 * np.inner(dF,F)
                
            return out

        def hess(z):
            dim = len(z)
            out = np.zeros((dim,dim))
            for n in range(dim):
                for m in range(dim):
                    e1 = np.zeros_like(z)
                    e1[n] = 1
                    e2 = np.zeros_like(z)
                    e2[m] = 1
                    #assemble n,m-th component of hessian
                    out[n,m] = 2 * np.inner( h[:j+2,:j+1] @ e1, h[:j+2,:j+1] @ e2)

            return out
        
        #Add constraints
        clist = []
        for const in conlist:
            clist.append({"type": "eq",
                          "fun": const,
                          "args": (x0,prefunc(Q))})


        #Initialise guess
        y0 = np.zeros((j+1,))
        if j!=0:
            y0[:-1] = yk
            
        #For the first iterations use gmres
        if residual[-1]>contol*tol and j<k:
            solve = spo.minimize(func,y0,tol=None,jac=jac,hess=hess,
                                 constraints=[],
                                 method='trust-constr',
                                 options={'xtol': 1e-12,
                                          'gtol': 1e-12,
                                          'barrier_tol': 1e-12,
                                          'maxiter': 1e3})

        else:
            try:
                constrained_steps += 1 #Add a constrained step
                solve = spo.minimize(func,y0,tol=None,jac=jac,hess=hess,
                                     constraints=clist[:],
                                     method='trust-constr',
                                     options={'xtol': ctol,
                                              'gtol': ctol,
                                              'barrier_tol': ctol,
                                              'maxiter': 1e4})
                safety = True
                #If constraints are violated, turn off safety to avoid
                #possible termination on next iteration
                if solve.constr_violation>ctol:
                    safety = False
                    warnings.warn("Iteration %d failed to preserve constraints with deviation of %e" % (j,solve.constr_violation),
                                  RuntimeWarning)
            except:
                warning('Constrained solve failed, defaulted to standard solve for iteration %d.' % j \
                        + ' Problem likely overconstrained, a smaller solver tolerance may be required.')
                solve = spo.minimize(func,y0,tol=None,jac=jac,hess=hess,
                                     constraints=[],
                                     method='trust-constr',
                                     options={'xtol': 1e-12,
                                              'gtol': 1e-12,
                                              'barrier_tol': 1e-12,
                                              'maxiter': 1e3})
                
        if solve.message!='Optimization terminated successfully':
            if solve.message!='`xtol` termination condition is satisfied.':
                if solve.message!='`gtol` termination condition is satisfied.':
                    warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                                  RuntimeWarning)
        yk = solve.x
        
        x.append(prefunc(Q) @ yk + x0)

        #Compute residual
        residual.append(np.linalg.norm(A.dot(x[-1]) - b))

        #Measure iteration time
        if timing:
            #Unconstrained timing
            if constrained_steps==0:
                t_iter_unconstrained = (time() - t_iter_start) / steps
                t_iter_start_constrained = time()
            else:
                t_iter_constrained = (time() - t_iter_start_constrained) / constrained_steps
                
            

        if residual[-1] < tol and safety==True:
            break

    #Report timings
    if timing:
        t_end = time()
        timings = {'runtime': t_end-t_start,
                   'iter_time_unconstrained': t_iter_unconstrained,
                   'iter_time_constrained': t_iter_constrained,
                   'constrained_steps': constrained_steps}
    else:
        timings = {}

    #build output dictionary
    dict = {'name':'cgmres',
            'x': x,
            'res': residual[1:],
            'steps': steps,
            'timings': timings}

    return x[-1], dict


#Prototypical CGMRES implementation, here constraints are
#enforced one-by-one as the number of iterations increases
def cgmres_p(A, b ,x0, k,
            conlist=[],
            pre = None):

    #Set tolerance
    tol=1e-15
    
    #If not using preconditioner, set up identity as placeholder
    if pre is None:
        pre = sps.identity(len(b))

    if hasattr(pre, 'solve'):#Check if spsla.LinearOperator object
        def prefunc(vec):
            return pre.solve(vec)
    else:
        def prefunc(vec):
            try:
                out = pre @ vec
            except:
                raise ValueError('Preconditioner not supported')
            return out
        
    x = []
    residual = []
    r = (b - A.dot(x0)) #define r0

    x.append(r)

         
    q = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    #convert type
    
    h = np.zeros((k+1,k))
    
    for j in range(k):
        y = np.asarray(A @ prefunc(q[j]))
        
        for i in range(j+1):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0):
            q[j+1] =  y / h[j+1,j]

        res = np.zeros(j+2)
        res[0] = beta

        Q = np.transpose(q[:j+1,:]) #Allocate current size of Q
        Qt = np.transpose(Q)

        #Set up function
        def func(z):
            F = res - h[:j+2,:j+1] @ z
            out = np.inner(F,F)
            return out

        def jac(z):
            out = np.zeros_like(z)

            #original term
            F = res - h[:j+2,:j+1] @ z
            #Component wise differentiation of F
            for m in range(len(z)):
                ej = np.zeros_like(z)
                ej[m] = 1 
                dF = - h[:j+2,:j+1] @ ej

                #assemble j-th component of jac
                out[m] = 2 * np.inner(dF,F)
                
            return out

        def hess(z):
            dim = len(z)
            out = np.zeros((dim,dim))
            for n in range(dim):
                for m in range(dim):
                    e1 = np.zeros_like(z)
                    e1[n] = 1
                    e2 = np.zeros_like(z)
                    e2[m] = 1
                    #assemble n,m-th component of hessian
                    out[n,m] = 2 * np.inner( h[:j+2,:j+1] @ e1, h[:j+2,:j+1] @ e2)

            return out
        
        #Add constraints
        clist = []
        for const in conlist:
            clist.append({"type": "eq",
                          "fun": const,
                          "args": (x0,prefunc(Q))})


        #Initialise guess
        y0 = np.zeros((j+1,))
        if j!=0:
            y0[:-1] = yk


        #For the first iteration just use gmres
        solve = spo.minimize(func,y0,tol=tol,jac=jac,hess=hess,
                             constraints=clist[:j],
                             method='trust-constr',
                             options={'xtol': 1e-12,
                                      'gtol': 1e-12,
                                      'barrier_tol': 1e-12,
                                      'maxiter': 1e3})


        if solve.message!='Optimization terminated successfully':
            if solve.message!='`xtol` termination condition is satisfied.':
                if solve.message!='`gtol` termination condition is satisfied.':
                    warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                                  RuntimeWarning)
        yk = solve.x
        
        x.append(prefunc(Q) @ yk + x0)

        #Compute residual
        residual.append(np.linalg.norm(A.dot(x[-1]) - b))


    #build output dictionary
    dict = {'name':'geosolve',
            'x': x,
            'res': residual}

    return x[-1], dict
