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
    Jnumber = -0.5
    J = (Jnumber/2)*np.array([[0,1],[1,0]]);
else:
    raise NotImplementedError;

if True: # typical 1D well with spin mixing
         # VRprime is perturbation

    # alpha -> beta
    alphas = [1,2];
    alpha_strs = ["\\uparrow \\uparrow","\\uparrow \downarrow","\downarrow \\uparrow","\downarrow \downarrow"];    # plotting
    nplots_x = len(alphas);
    nplots_y = len(alphas);
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True, sharey=True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);

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
    VC = 0.4*tLR; # compare 0.4 vs 0.3
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
    if(len(Evals)%2!=0): Evals, Mvals, Sxvals = Evals[2:-1], Mvals[2:-1], Sxvals[2:-1];

    # effective matrix elements
    E_plus = Evals[Sxvals>0];
    E_minus = Evals[Sxvals<0];
    E_ab = np.array([ (E_plus+E_minus)/2, (E_plus+E_minus)/2]);
    M_plus = Mvals[Sxvals>0];
    M_minus = Mvals[Sxvals<0];
    M2_nsf = (1/4)*np.real(np.conj(M_plus)*M_plus + np.conj(M_minus)*M_minus);
    M2_nsf += (1/4)*np.real(np.conj(M_plus)*M_minus + np.conj(M_minus)*M_plus);
    M2_sf = (1/4)*np.real(np.conj(M_plus)*M_plus + np.conj(M_minus)*M_minus);
    M2_sf += (-1/4)*np.real(np.conj(M_plus)*M_minus + np.conj(M_minus)*M_plus);
    M2_ab = np.empty((n_loc_dof,np.shape(E_ab)[1], n_loc_dof));
    M2_ab[0,:,0] = M2_nsf;
    M2_ab[1,:,1] = M2_nsf;
    M2_ab[0,:,1] = M2_sf;
    M2_ab[1,:,0] = M2_sf;
    if False:
        Mfig, Max = plt.subplots();
        Max.plot(np.real(np.conj(M_plus)*M_plus),label="a");
        Max.plot(np.real(np.conj(M_minus)*M_minus),label="b");
        Max.plot(np.real(np.conj(M_plus)*M_minus),label="c");
        Max.plot(np.real(np.conj(M_minus)*M_plus),label="d");
        Max.plot(M2_nsf,label="M2_nsf");
        Max.plot(M2_sf,label="M2_sf");
        plt.legend();
        plt.show();
        assert False;
    Evals, Mvals = E_ab, M2_ab;

    # bardeen Ts
    Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                               tLR, tLR, VLR, VLR, NLR, NLR,verbose=1);

    # benchmark
    Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
    print("Output shapes:");
    for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

  # initial and final states
    for alphai in range(len(alphas)):
        for betai in range(len(alphas)):
            alpha, beta = alphas[alphai], alphas[betai];

            # plot based on initial state
            xvals = np.real(Evals[alphai])+2*tLR[alphai,alphai];
            axes[alphai,betai].scatter(xvals, Tvals[betai,:,alphai], marker=mymarkers[0]);
            
            # % error
            axright = axes[alphai,betai].twinx();
            axes[alphai,betai].scatter(xvals, Tvals_bench[betai,:,alphai], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[betai,:,alphai]-Tvals_bench[betai,:,alphai])/Tvals_bench[betai,:,alphai]),color=accentcolors[1]); 
            
            #format
            if(betai==len(alphas)-1): axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
            axes[alphai,betai].set_title("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");
            axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
            axes[-1,betai].set_xscale('log', subs = []);
            axes[alphai,0].set_ylabel("$T$");
            
    # show
    plt.tight_layout();
    plt.show();
    
    

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



