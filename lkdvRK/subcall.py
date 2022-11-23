'''
Call evolve.py from command line with argparser and
saves the output to file. This file is used for
parallel generation of the errors
'''
#Global imports
import argparse
import pickle
import os

#Local imports
import LinearSolver as ls
import evolve

#Convert string to python function
def choose_solver(string):
    if string=='Exact':
        s = ls.exact
    elif string=='GMRES':
        s = ls.gmresWrapper
    elif string=='CGMRES':
        s = ls.cgmresWrapper
    else:
        raise ValueError('Invalid choice of solver')
    return s

if __name__=="__main__":
    #Make sure tmp directory exists
    if os.path.isdir('tmp')==False:
        os.mkdir('tmp')
    #argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default='CGMRES')
    parser.add_argument('--tstages', type=int, default=3)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--N', type=int,default=10)
    parser.add_argument('--M', type=int,default=50)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--k', type=int, default=50)
    args, _ = parser.parse_known_args()
    
    #Get solver
    solver = choose_solver(args.solver)

    print('solver = ', args.solver)
    print('tstages = ', args.tstages)
    print('degree = ', args.degree)
    
    #Compute error
    err = evolve.evolve(N=args.N,M=args.M,
                        degree=args.degree,tstages=args.tstages,
                        k=args.k,tol=args.tol,
                        solver=solver)
    
    #Postprocess keys
    err['degree'] = args.degree
    err['tstages'] = args.tstages
    err['solver'] = args.solver
    del err['sol'] #Firedrake function cannot be saved to file in this way

    #Save output
    filename = 'tmp/error%s%s%s%s%s%s.pickle' % (args.solver, args.tstages,
                                                 args.degree, args.tol,
                                                 args.N, args.M)
    file = open(filename, "wb")
    pickle.dump(err,file)
    file.close()
