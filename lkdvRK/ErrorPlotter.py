'''
Process and plot the data generated in ErrorGenerator.py
'''
import numpy as np
import matplotlib.pylab as plt
import pickle
import os

lw = 2

#Import data
with open('tmp/error.pickle','rb') as file:
    data = pickle.load(file)


#Manually specify the number of different types of experiments
num = 3

#Initialise output plots
plt.figure(1)
color = plt.cm.rainbow(np.linspace(0, 1, int(len(data)/num)))
color = ['r','b','k']
colors = color
for i in range(num):
    colors = np.concatenate((colors,color),axis=0)

#Loop over data
for i in range(len(data)):
    dic = data[i]
    if dic['solver']=='Exact':
        plt.semilogy(dic['time'],dic['err'], linewidth=lw, linestyle='solid', color=colors[i], label='Exact: q=%s, s=%s' % (dic['degree'], dic['tstages']))
    elif dic['solver']=='GMRES':
        plt.semilogy(dic['time'],dic['err'], linewidth=lw, linestyle='dotted', color=colors[i], label='GMRES: q=%s, s=%s' % (dic['degree'], dic['tstages']))
    elif dic['solver']=='CGMRES':
        plt.semilogy(dic['time'],dic['err'], linewidth=lw, linestyle='dashed', color=colors[i], label='CGMRES: q=%s, s=%s' % (dic['degree'], dic['tstages']))


plt.xlabel(r'$t$')
plt.ylabel(r'$L_2$ error')
        
plt.legend(loc='lower left', bbox_to_anchor=(0.,-0.3),
           fontsize='small',ncol=3)
parent = os.path.dirname(os.getcwd())
filename = parent + '/plots/lkdvRKError.pdf'
plt.savefig(filename,bbox_inches='tight')
print('Figure saved as %s' % filename)
