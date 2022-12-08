#global
from firedrake import *
import numpy as np
import os
import pandas as pd
import matplotlib.pylab as plt

# local
import lkdv

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
        new = pd.DataFrame({name + ' residual norm': data['res']})
        df = pd.concat([df,new], axis=1)
        
        #compute invariants
        dev1 = []
        dev2 = []
        dev3 = []
        for j in range(1,np.shape(data['x'])[0]):
            inv = lkdv.compute_invariants(prob,data['x'][j])
            dev1.append(inv['mass'] - params['m0'])
            dev2.append(inv['momentum'] - params['mo0'])
            dev3.append(inv['energy'] - params['e0'])

        new = pd.DataFrame({name + ' mass deviation': dev1})
        df = pd.concat([df,new], axis=1)
        new = pd.DataFrame({name + ' momentum deviation': dev2})
        df = pd.concat([df,new], axis=1)
        new = pd.DataFrame({name + ' energy deviation': dev3})
        df = pd.concat([df,new], axis=1)

    #write to file, if desired
    if filename!=None:
        texfile = open(filename + '.tex', 'w')
        texfile.write(df.to_latex(index=False))
        texfile.close()
        df.to_csv(filename + '.csv', index=False)

    #Print table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print(df)

    out = {'df': df,
           'names': names}
    

    return out



def convergence_plot(vis_out,
                     filename=os.path.dirname(os.getcwd())+'/plots/lkdvConvergence.pdf'):
    #Extract relevant data
    names = vis_out['names']
    df = vis_out['df']
    res1 = df[names[0] + ' residual norm']
    m1 = np.abs(df[names[0] + ' mass deviation']) + 1e-16
    mo1 = np.abs(df[names[0] + ' momentum deviation']) + 1e-16
    e1 = np.abs(df[names[0] + ' energy deviation']) + 1e-16
    res2 = df[names[1] + ' residual norm']
    m2 = np.abs(df[names[1] + ' mass deviation']) + 1e-16
    mo2 = np.abs(df[names[1] + ' momentum deviation']) + 1e-16
    e2 = np.abs(df[names[1] + ' energy deviation']) + 1e-16

    #Some global custom options for plots
    # font = {'size' : 15}
    # plt.rc('font', **font)
    lw = 2

    #'standard' gmres plots
    plt.plot(res1, linewidth=lw, color='r',linestyle='solid',label='GMRES: Residual')
    plt.plot(m1, linewidth=lw, color='r',linestyle='dotted',label='GMRES: Deviation in mass')
    plt.plot(m1, linewidth=lw, color='r',linestyle='dashdot',label='GMRES: Deviation in momentum')
    plt.plot(e1, linewidth=lw, color='r',linestyle='dashed',label='GMRES: Deviation from energy')

    #'Conservative' gmres plots
    plt.plot(res2, linewidth=lw, color='b',linestyle='solid',label='CGMRES: Residual')
    plt.plot(m2, linewidth=lw, color='b',linestyle='dotted',label='CGMRES: Deviation in mass')
    plt.plot(mo2, linewidth=lw, color='b',linestyle='dashdot',label='CGMRES: Deviation in momemtum')
    plt.plot(e2, linewidth=lw, color='b',linestyle='dashed',label='CGMRES: Deviation from energy')

    plt.grid(which='both', linestyle='--',axis='y')

    plt.yscale('log')
    
    #Add some labels
    plt.xlabel(r'Iteration number')


    #Make the legend look half decent
    plt.legend(loc='lower left',ncol=2, bbox_to_anchor=(-0.1,-0.6))
    

    plt.tight_layout()

    os.chdir('../') #Move to parent directory before saving
    plt.savefig(filename)
    print('Figure saved as %s' % filename)

    return -1


