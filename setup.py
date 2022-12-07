'''
Installs additional required packages and creates
required subdirectories
'''
import os

#Create directories
if os.path.isdir("plots")==False:
    os.system("mkdir plots")

#Install extra modules (only inside Firedrake venv)
try:
    import firedrake
except:
    raise AssertionError('Users must run this script within a Firedrake virtual environment')
try:
    import pandas
except:
    os.system('pip3 install pandas')
try:
    import pickle
except:
    os.system('pip3 install pickle')
