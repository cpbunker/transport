'''
Christian Bunker
M^2QM at UF
October 2021

Steady state transport of a single electron through a one dimensional wire
Part of the wire is scattering region, where the electron spin degrees of
freedom can interact with impurity spin degrees of freedom

Impurity hamiltonians calculated from dft, Jie-Xiang's Co dimer manuscript
'''

import ml

import numpy as np
import matplotlib.pyplot as plt
import itertools

import sys

#### setup

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);
verbose = 4;
sourcei = int(sys.argv[1]);
numEvals = 5;

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets = np.array([xi for xi in itertools.product(*tuple(states))]); # product states
dets = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]]

# initialize source vector
assert(sourcei >= 0 and sourcei < len(dets));
source = np.zeros(5);
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
print("\nSource:\n"+source_str);

# load data
fname = "dat/"+str(source_str[1:-1])+".npy";
print("Loading data from "+fname);
features = np.load(fname);

# recall features are energy, Jx, Jy, Jz, DO, DT, An, JK1, JK2
# after that is source/entangled state info
# two targets: Peak of + state, peak of - state
feature_strs = ["E", "Jx", "Jy", "Jz", "DO", "DT", "An", "JK1", "JK2"];
included_features = ["Jx", "Jz", "DO", "DT", "An", "JK1"];

# process important features into X
nsamples, nfeatures = len(features[:,0]), len(included_features);
print("nsamples, nfeatures = ", nsamples, nfeatures);
X = np.empty((nsamples, nfeatures));
Xcoli = 0;
for coli in range(len(feature_strs)):
    if feature_strs[coli] in included_features:
        X[:,Xcoli] = features[:,coli];
        Xcoli += 1;

ys = features[:,-2:];
X = ml.Scale(X);

# machine learning!
for y in [ys[:,0]]:
    
    # find principal component
    ml.PlotPolyFit(X, y, 1, included_features);
    Xpca = ml.PCA(X, 1, inverse = True);
    ml.PlotPolyFit(Xpca, y, 3, included_features);

    # use random forest regression
    #ml.PlotForestFit(X, y, 50, included_features, numEvals)


    




