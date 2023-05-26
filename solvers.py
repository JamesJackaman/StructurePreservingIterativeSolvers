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

def constraint_checker(x,x0,Z,const_list):
    dev = 0
    for const in const_list:
        dev = max(dev, const['fun'](x,x0,Z))
    return dev

#A wrapper to avoid late-binding issues
class constraint_container:
    def __init__(self,const,x0,Z):
        #Check constraint type
        if hasattr(const, '__dict__'):#If constraint is in the form of
                                      #a class, optimise for speed
            self.optimise = True
        elif type(const) is dict:#If given as a dictionary do not attempt to optimise
            self.optimise = None
        else:
            raise NotImplementedError('Constraints must be either dictionaries or classes')
        
        if self.optimise:
            self.MZ = const.M @ Z
            self.term0 = 0.5 * x0 @ const.M @ x0 + const.c + const.v @ x0
            self.term1 = const.v @ Z + x0 @ self.MZ
            self.term2 = 0.5 * Z.T @ self.MZ
        else:
            self.const = const
            self.x0 = x0
            self.Z = Z

    def constraint_func(self,z):
        if self.optimise:
            out = self.term0 + self.term1 @ z + z @ self.term2 @ z
        else:
            out = self.const['func'](z,self.x0,self.Z)
        return out
    def constraint_jac(self,z):
        if self.optimise:
            out = self.term1 + 2 * z @ self.term2
        else:
            out = self.const['jac'](z,self.x0,self.Z)
        return out

        

#FGMRES implementation, hand implemented for a fair comparison
def gmres(A, b, x0, k, tol = 1e-50, pre = None):
        
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
    z = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    h = np.zeros((k+1,k))

    for j in range(k):
        steps = j+1
        z[j] = np.asarray(prefunc(q[j]))
        y = np.asarray(A @ z[j])
        
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

        Z = np.transpose(z[:j+1,:]) #Allocate current size of z

        yk = np.linalg.lstsq(h[:j+2,:j+1], res, rcond=None)[0]
        
        x.append(Z @ yk + x0)
        residual.append(np.linalg.norm(A.dot(x[-1]) - b))

        if residual[-1] < tol:
            break

    #Build output dictionary
    dict = {'name': 'gmres',
            'x':x,
            'res':residual[1:],
            'steps': steps}
    
    return x[-1], dict


#Standard CGMRES implementation
def cgmres(A, b ,x0, k,
           tol=1e-8,
           contol=10,#contol*tol is when constraints are first enforced
           conlist=[],
           pre = None,
           timing = None):

    ctol = 1e-12 #tolerance of minimisation problem
    
    if timing:
        t_start = time()
        jit = {'start': time(),
                       'start_iter': [],
                       'end_iter': [],
                       'start_constraints': [],
                       'end_constraints': []}
    
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
    z = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    #convert type
    
    h = np.zeros((k+1,k))

    if timing:
        t_iter_start = time()
        constrained_steps = 0
    for j in range(k):
        if timing:
            jit['start_iter'].append(time())
        steps = j+1
        z[j] = np.asarray(prefunc(q[j]))
        y = np.asarray(A @ z[j])
        
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
        
        Z = np.transpose(z[:j+1,:]) #Allocate current size of Z
        

        #Set up function and jacobian (of problem)
        def func(z):
            F = res - h[:j+2,:j+1] @ z
            out = np.inner(F,F)
            return out

        def jac(z):
            F = res - h[:j+2,:j+1] @ z
            Ht = np.transpose(h[:j+2,:j+1])
            return -2 * Ht @ F
        
        #Initialise constraint list
        clist = []

        #Initialise guess
        y0 = np.zeros((j+1,))
        if j!=0:
            y0[:-1] = yk
            
        #For the first iterations use gmres
        if residual[-1]>contol*tol and j<k-1:
            solve = spo.minimize(func,y0,tol=None,jac=jac,
                                 constraints=[],
                                 method='SLSQP',
                                 options={'ftol': ctol**2,
                                          'maxiter': 1e3})
        else:
            try:
                if timing:
                    constrained_steps += 1 #Add a constrained step
                    jit['start_constraints'].append(time())

                for const in conlist:
                    #Build constraint functions
                    cc = constraint_container(const,x0,Z)
                    clist.append({"type": "eq",
                                  "fun": cc.constraint_func,
                                  "jac": cc.constraint_jac})
                if timing:
                    jit['end_constraints'].append(time())

                solve = spo.minimize(func,y0,tol=None,jac=jac,
                                     constraints=clist[:],
                                     method='SLSQP',
                                     options={'ftol': ctol**2,
                                              'maxiter': 1e3})

                #check if constrained solver has silently failed
                if not timing:
                    if np.isnan(max(solve.x)):
                        raise ValueError
                
                safety = True

                # If constraints are violated, turn off safety to avoid
                # possible termination on next iteration (slow so avoid when timing)
                if not timing:
                    if constraint_checker(solve.x,x0,Z,clist)>ctol:
                        safety = False
                        warnings.warn("Iteration %d failed to preserve constraints with deviation of %e" % (j,solve.constr_violation),
                                      RuntimeWarning)
            except:
                warning('Constrained solve failed, defaulted to standard solve for iteration %d.' % j \
                        + ' Problem likely overconstrained, a smaller solver tolerance may be required.')
                solve = spo.minimize(func,y0,tol=None,jac=jac,
                                     constraints=[],
                                     method='SLSQP',
                                     options={'ftol': ctol**2,
                                              'maxiter': 1e3})
                
        if solve.message!='Optimization terminated successfully':
            if solve.message!='`xtol` termination condition is satisfied.':
                if solve.message!='`gtol` termination condition is satisfied.':
                    warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                                  RuntimeWarning)
        yk = solve.x
        
        x.append(Z @ yk + x0)

        #Compute residual
        residual.append(np.linalg.norm(A.dot(x[-1]) - b))

        #Measure iteration time
        if timing:
            jit['end_iter'].append(time())

        if residual[-1] < tol and safety==True:
            break

    #Report timings
    if timing:
        jit['end'] = time()
        iter_time = np.asarray(jit['end_iter'])-np.asarray(jit['start_iter'])
        iter_unconstrained = iter_time[:-constrained_steps]
        iter_constrained = iter_time[len(iter_unconstrained):]
        timings = {'runtime': jit['end'] - jit['start'],
                   'iter_time_unconstrained': np.mean(iter_unconstrained),
                   'iter_time_constrained': np.mean(iter_constrained),
                   'constraint_building': np.mean(np.asarray(jit['end_constraints'])
                                                  - np.asarray(jit['start_constraints'])),
                   'constrained_steps': constrained_steps}
    else:
        timings = None

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
    z = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    #convert type
    
    h = np.zeros((k+1,k))
    
    for j in range(k):
        z[j] = np.asarray(prefunc(q[j]))
        y = np.asarray(A @ z[j])
        
        for i in range(j+1):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0):
            q[j+1] =  y / h[j+1,j]

        res = np.zeros(j+2)
        res[0] = beta

        Z = np.transpose(z[:j+1,:]) #Allocate current size of Z

        #Set up function and jacobian
        def func(z):
            F = res - h[:j+2,:j+1] @ z
            return np.inner(F,F)

        def jac(z):
            F = res - h[:j+2,:j+1] @ z
            Ht = np.transpose(h[:j+2,:j+1])
            return -2 * Ht @ F

        
        #Add constraints
        clist = []
        for const in conlist:
            cc = constraint_container(const,x0,Z)
            clist.append({"type": "eq",
                          "fun": cc.constraint_func,
                          "jac": cc.constraint_jac})


        #Initialise guess
        y0 = np.zeros((j+1,))
        if j!=0:
            y0[:-1] = yk


        #Try prototypical solver
        solve = spo.minimize(func,y0,tol=1e-15,jac=jac,
                             constraints=clist[:j],
                             method='SLSQP',
                             options={'ftol': 1e-20,
                                      'maxiter': 1e3})

        #If solve failed, default to an uncontrained solve
        if np.isnan(max(solve.x)):
            warning('Constrained solve silently failed on iteration %d' % j)
            solve = spo.minimize(func,y0,tol=None,jac=jac,
                                 constraints=[],
                                 method='SLSQP',
                                 options={'ftol': 1e-20,
                                          'maxiter': 1e3})
            

        if solve.message!='Optimization terminated successfully':
            if solve.message!='`xtol` termination condition is satisfied.':
                if solve.message!='`gtol` termination condition is satisfied.':
                    warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                                  RuntimeWarning)
        yk = solve.x
        
        x.append(Z @ yk + x0)

        #Compute residual
        residual.append(np.linalg.norm(A.dot(x[-1]) - b))


    #build output dictionary
    dict = {'name':'geosolve',
            'x': x,
            'res': residual}

    return x[-1], dict
