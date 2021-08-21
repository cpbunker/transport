'''
plot results of SIAM data, somewhere in dat folder
give data folders and splots at command line
'''

import plot

import sys

# read off data files from sys.argv
files = sys.argv[1:];
datafs = [];
labs = []
for f in files:
    if(f[-4:] == ".npy"): # screen out all non .npys
        datafs.append(f);
        for ci in range(len(f)): # screen filename for Vg val
            if f[ci]=='V' and f[ci+1]=='g': # next is Vg val
                labs.append("Vg = "+f[ci+2:-4]);

plot.CompObservables(datafs, (2,2), "", labs, mytitle = "$V_g$ sweep at $\mu = 10.0$, $U=100.0$");
