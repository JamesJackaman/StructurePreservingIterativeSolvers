#global
from firedrake import *
import numpy as np
import os
import pandas as pd
import matplotlib.pylab as plt

# local
import swe

#initialise global lists
## here we will tabulate data
def tabulator(params,prob,dict_list,filename=None):
    #initalise tables
    df = pd.DataFrame()
    names = []
    #loop over dictionaries (each is a method)
    for data in dict_list:
        name = data['name']
        names.append(name)
        df[name + ' residual norm'] = data['res']

        #compute invariants
        dev1 = []
        dev2 = []
        for j in range(1,np.shape(data['x'])[0]):
            inv = swe.compute_invariants(prob,data['x'][j])
            dev1.append(inv['mass'] - params['m0'])
            dev2.append(inv['energy'] - params['e0'])

        df[name + ' mass deviation'] = dev1
        df[name + ' energy deviation'] = dev2


    #write to file, if desired
    if filename!=None:
        df.to_csv(filename + '.csv', index=False)
        
    print(df)

    out = {'df': df,
           'names': names}
    
    return out


def convergence_plot(vis_out,
                     filename=os.path.dirname(os.getcwd())+'/plots/sweConvergence.pdf'):
    #Extract relevant data
    names = vis_out['names']
    df = vis_out['df']
    res1 = df[names[0] + ' residual norm']
    m1 = np.abs(df[names[0] + ' mass deviation']) + 1e-16
    e1 = np.abs(df[names[0] + ' energy deviation']) + 1e-16
    res2 = df[names[1] + ' residual norm']
    m2 = np.abs(df[names[1] + ' mass deviation']) + 1e-16
    e2 = np.abs(df[names[1] + ' energy deviation']) + 1e-16
    
    lw = 2

    #gmres plots
    plt.plot(res1, linewidth=lw, color='r',linestyle='solid',label='GMRES: Residual')
    plt.plot(m1, linewidth=lw, color='r',linestyle='dotted',label='GMRES: Deviation in mass')
    plt.plot(e1, linewidth=lw, color='r',linestyle='dashed',label='GMRES: Deviation in energy')

    #cgmres plots
    plt.plot(res2, linewidth=lw, color='b',linestyle='solid',label='CGMRES: Residual')
    plt.plot(m2, linewidth=lw, color='b',linestyle='dotted',label='CGMRES: Deviation in mass')
    plt.plot(e2, linewidth=lw, color='b',linestyle='dashed',label='CGMRES: Deviation in energy')

    plt.grid(which='both', linestyle='--',axis='y')

    plt.yscale('log')
    
    #Add some labels
    plt.xlabel(r'Iteration number')

    #Make the legend look half decent
    plt.legend(loc='lower left',ncol=2, bbox_to_anchor=(0.0,-0.4))
    

    plt.tight_layout()

    plt.savefig(filename)

    return -1

