'''
Generate errors, in parallel, for RK formulation of lkdv and
save them to file. After generating run ErrorPlotter.py to
visualise results.
'''
#Global imports
import pickle
import subprocess
import time

if __name__=="__main__":

    MaxProcesses = 12 #Maximal number of parallel processes
    Processes = []

    N = 10
    M = 400
    k = 50 #Max iterations
    ctol = 1e-12 #Constraint enforcement tolerance
    
    def checkrunning():
        for p in reversed(range(len(Processes))):
            if Processes[p].poll() is not None:
                del Processes[p]
        return len(Processes)
    
    #Specify tstages, degrees and solvers
    solvers = ['Exact','GMRES','CGMRES']

    #In the current implementation, the current arrays must have the
    #same length
    tstages = [2,3,4]
    degrees = [3,4,5]
    tols = [1e-3,1e-5,1e-7]

    #generate the data
    for degree in degrees:
        tol = tols[degrees.index(degree)]
        tstage = tstages[degrees.index(degree)]
        for solver in solvers:
            print('solver = ', solver)
            print('tstages = ', tstage)
            print('degree = ', degree)
            
            process = subprocess.Popen('python subcall.py --solver %s --tstages %s --degree %s --N %s --M %s --tol %s --ctol %s --k %s' % (
                solver,tstage,degree,N,M,tol,ctol,k),
                                       shell=True, stdout=subprocess.PIPE)
            Processes.append(process)
            
            while checkrunning()==MaxProcesses:
                time.sleep(1)
                    
    while checkrunning()!=0:
        time.sleep(1)

    #Save the data globally
    out = []
    for solver in solvers:
        for degree in degrees:
            tstage = tstages[degrees.index(degree)]
            tol = tols[degrees.index(degree)]
            try:
                #load data
                filename = 'tmp/error%s%s%s%s%s%s.pickle' % (solver, tstage,
                                                             degree, tol,
                                                             N, M)
                with open(filename, 'rb') as file:
                    out_ = pickle.load(file)
                    #Append to dictionary
                    out.append(out_)
                    print(out)
            except Exception as e:
                print('Loading %s failed with' % filename)
                print("Error : "+str(e))    
                

    #Write to file
    file = open("tmp/error.pickle", "wb")
    pickle.dump(out,file)
    file.close()
