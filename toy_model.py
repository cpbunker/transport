'''
'''

from transport import fci_mod

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

############################################################################
#### toy model for Oppenheimer calculation

# 2 site Hamiltonian with spin mixing
n_loc_dof = 2;
n_spatial_dof = 2;
t = 1.0*np.eye(n_loc_dof);
J = 0.5*np.array([[0,1],[1,0]]);
Hsys = np.zeros((n_spatial_dof, n_spatial_dof, n_loc_dof, n_loc_dof));
Hsys[0,1] = -t;
Hsys[1,0] = -t;
Hsys[0,0] = J;
Hsys[1,1] = J;
print("Hsys =\n",Hsys);
Hsys_2d = fci_mod.mat_4d_to_2d(Hsys);

# construct HL by cutting hopping
HL = np.copy(Hsys);
HL[0,1] = np.zeros_like(t);
HL[1,0] = np.zeros_like(t);
print("HL =\n",HL);

# eigenstates of HL
HL_2d = fci_mod.mat_4d_to_2d(HL);
Es_arr, ks_arr = np.linalg.eigh(HL_2d);
print("Eigenstates of HL_2d =");
for m in range(len(Es_arr)):
    print(m, Es_arr[m], ks_arr.T[m]);

# separate as initial and final
inds = [0,3,1,2]; # initial, initial, final, final
inds_i, inds_f = inds[:len(inds)//2], inds[len(inds)//2:];
Emas = np.array([Es_arr[i] for i in inds_i]);
kmas = np.array([ks_arr.T[i] for i in inds_i]);
Enbs = np.array([Es_arr[i] for i in inds_f]);
knbs = np.array([ks_arr.T[i] for i in inds_f]);
print("Initial states=");
for m in range(len(Emas)):
    print(m, Emas[m], kmas[m]);
print("Final states=");
for n in range(len(Enbs)):
    print(n, Enbs[n], knbs[n]);

Es_all = np.append(Emas, Enbs)
ks_all = np.append(kmas, knbs).reshape(len(Es_all), len(kmas[0]));

# initial -> final matrix elements
Melements = np.empty((len(ks_all), len(ks_all)));
for m in range(len(ks_all)):
    for n in range(len(ks_all)):
        Melements[n,m] = np.dot(np.conj(ks_all[n]), np.dot(Hsys_2d-HL_2d,ks_all[m]));
print("Melements =\n",Melements);

# spin polarized init and final states
psis = np.array([[1,0,0,0], # up inititial
                 [0,1,0,0], # down initial
                 [0,0,1,0], # up final
                 [0,0,0,1]]); # down final
# get D coefs which transform from k states to psi states
Dcoefs = np.empty((len(psis),len(ks_all)));
for f in range(len(psis)):
    for m in range(len(ks_all)):
        Dcoefs[f,m] = np.dot( np.conj(ks_all[m]), psis[f]);
print("Dcoefs =\n",Dcoefs);

print(np.matmul( Dcoefs.T, np.matmul(Melements, Dcoefs)))



