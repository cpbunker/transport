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
feature_strs = np.array(["E", "Jx", "Jy", "Jz", "DO", "DT", "An", "JK1", "JK2"]);
phys_features = np.array(["Jx", "Jz", "DO", "DT", "An", "JK1"]);

# target
ys = features[:,-2:];
y = np.mean(ys, axis = 1);

# process physical features into X
nsamples, nfeatures = len(features[:,0]), len(phys_features);
print("nsamples, nfeatures = ", nsamples, nfeatures);
X = np.empty((nsamples, nfeatures));
Xcoli = 0;
for coli in range(len(feature_strs)):
    if feature_strs[coli] in phys_features:
        X[:,Xcoli] = features[:,coli];
        Xcoli += 1;

# now have 6 dim data
ml.PlotPolyFit(X, y, 1, phys_features, mytitle = "Raw data");
X = ml.Scale(X);
ml.PlotPolyFit(X, y, 1, phys_features, mytitle = "Scaled data");

if False:
    # filter out by percentile
    ypercentile = 99;
    ycutoff = np.percentile(y, ypercentile);
    Xf, yf = [], []; # only samples above cutoff
    for si in range(len(y)):
        if(y[si] > ycutoff): 
            Xf.append(X[si]);
            yf.append(y[si]);
    Xf, yf = np.array(Xf), np.array(yf);
    print("Data kept: ",len(yf)/len(y));
    ml.PlotPolyFit(Xf, yf, 1, phys_features, mytitle = "Filtered data");

    # dim reduction
    pcais = ml.PCA(Xf, nfeatures, components = True);
    Xpca = ml.PCA(Xf, 4, inverse = True);
    ml.PlotPolyFit(Xpca, yf, 1, phys_features, mytitle = "Filtered 4D data");

    # explained variance of dim reduction
    plt.plot(range(1, nfeatures+1), np.cumsum(ml.PCA(Xf, nfeatures, explained = True)));
    plt.ylabel("Explained variance");
    plt.xlabel("PCA components");
    plt.show();

    # filter + lowest PCA = most important dims --> 2 dim representation (binned)
    if sourcei == 0:
        imp_features = ["DO","Jz"];
        imp_indices = [2,1];
    elif sourcei == 1:
        imp_features = ["Jx", "JK1"];
        imp_indices = [0,5];
    X = X[:,imp_indices];
    ml.PlotPolyFit(X, y, 1, imp_features, mytitle = "2d data");

    # violin plot of binned data
    ml.PlotBinsViolin(X, y, imp_features);

    # average out unimportant dims to make purely 2d
    ml.PlotTarget2d(X, y, imp_features);

if True:

    # filter for density estimation
    ypercentile = 99;
    ycutoff = np.percentile(y, ypercentile);
    Xf, yf = [], []; # only samples above cutoff
    for si in range(len(y)):
        if(y[si] > ycutoff): 
            Xf.append(X[si]);
            yf.append(y[si]);
    Xf, yf = np.array(Xf), np.array(yf);
    print("Data kept: ",len(yf)/len(y));
    ml.PlotPolyFit(Xf, yf, 1, phys_features, mytitle = "Filtered data");

    # find optimal # of GMM components
    Ns = range(5,40);
    scores = [];
    for N in Ns:
        scores.append(ml.GMM(Xf, yf, N, score = True));
    plt.plot(Ns, scores);
    plt.xlabel("GMM components");
    plt.ylabel("BIC score");
    plt.show();
    ncomps = Ns[np.argmin(scores)];
    print("Optimal # components = ", ncomps);

    # find high density area
    densest = ml.GMM(Xf, yf, ncomps, density = True);
    print(densest);
    ml.PlotPolyFit([densest], [0], 1, phys_features, mytitle = "Densest point");
    

    





