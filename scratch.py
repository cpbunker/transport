
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn import linear_model

import sys

# top level
plt.style.use("seaborn-dark-palette");
np.set_printoptions(precision = 4, suppress = True);

# def particles and their single particle states
species = np.array([1,1,1]); # num of each species, which are one e, elec, spin-3/2, spin-3/2
spec_strs = ["e","1","2"];
states = [[0,1],[2,3,4,5],[6,7,8,9]]; # e up, down, spin 1 mz, spin 2 mz
state_strs = ["0.5_","-0.5_","1.5_","0.5_","-0.5_","-1.5_","1.5_","0.5_","-0.5_","-1.5_"];
dets = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]]

# initialize source vector
sourcei = int(sys.argv[1]);
assert(sourcei >= 0 and sourcei < len(dets));
source = np.zeros(5);
source[sourcei] = 1;
source_str = "|";
for si in dets[sourcei]: source_str += state_strs[si];
source_str += ">";
print("\nSource:\n"+source_str);

# load data
fname = "dat/ml/incidentE/"+str(source_str[1:-1])+".npy";
print("Loading data from "+fname);
features = np.load(fname);
X = features[:,0];
X = X[:, np.newaxis];

# target
ys = features[:,-2:];
y = np.mean(ys, axis = 1) - (0/5)*abs(ys[:,1]-ys[:,0]); # mean + regularization

# linear fit
model = sklearn.linear_model.LinearRegression(fit_intercept = True);
model.fit(X, y);

# plot data
colors = seaborn.color_palette("dark");
fig, axes = plt.subplots();
axes.scatter(X[:,0], y, marker = 's', color = colors[0]);
axes.plot(X[:,0], model.intercept_+X[:,0]*model.coef_, color = colors[0]);
axes.set_xlabel("Incident energy");
axes.set_ylabel("Entanglement prob.");
axes.set_title("Linear fit of incident energy");
plt.show();
