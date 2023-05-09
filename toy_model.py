'''
'''

from transport import fci_mod, bardeen

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

def print_H_alpha(H):
    assert(len(np.shape(H)) == 4);
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",H[i,j,:,:]);

############################################################################
#### toy model for Oppenheimer calculation

n_loc_dof = 2;
t = 1.0*np.eye(n_loc_dof);
if(n_loc_dof==2):
    J = -0.25*np.array([[0,1],[1,0]]);
else:
    raise NotImplementedError;

if True: # typical 1D well with spin mixing
         # VRprime is perturbation

    # set up the 1D well physics
    # explicit LR symmetry
    tLR = 1.0*np.eye(n_loc_dof);
    tinfty = 1.0*tLR;
    tLRprime = 1.0*tLR;
    VLR = 0.0*tLR;
    Vinfty = 0.5*tLR;
    VLRprime = 0.5*tLR;
    NLR = 200;
    Ninfty = 20;

    # central region physics: barrier with spin mixing
    tC = 1.0*tLR;
    VC = 0.5*tLR;
    NC = 5;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
    for NCi in range(NC):
        for NCj in range(NC):
            if(NCi == NCj): # exchange on diagonal
                HC[NCi,NCj] = VC+J;
            elif(abs(NCi -NCj) == 1): # nn hopping
                HC[NCi,NCj] += -tC;
    HCprime = np.copy(HC);

    # print
    print("HC =");
    print_H_alpha(HC);
    print("HC - HCprime =");
    print_H_alpha(HC-HCprime);

    # bardeen results for spin flip scattering
    #
    #### Notes:
    #
    # bardeen.kernel syntax:
    # tinfty, tL, tLprime, tR, tRprime,
    # Vinfty, VL, VLprime, VR, VRprime,
    # Ninfty, NL, NR, HC,HCprime,
    # I am setting VLprime = VRprime = Vinfty for best results according
    # tests performed in run_barrier_bardeen 
    Evals, Mvals, Sxvals = bardeen.kernel_mixed(tinfty,tLR,tLRprime, tLR, tLRprime,
                              Vinfty, VLR, VLRprime, VLR, VLRprime,
                              Ninfty, NLR, NLR, HC, HCprime,
                              E_cutoff=0.1,verbose=10);
    

if False: # 2 site Hamiltonian with spin mixing
          # hopping cutting is perturbation

    # Hsys
    n_spatial_dof = 2;
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



