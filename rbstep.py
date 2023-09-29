'''
Christian Bunker
M^2QM at UF
November 2022

Spin independent scattering from a rectangular potential barrier, with
different well heights on either side (so that scattering is inelastic)
solved in time-dependent QM using bardeen theory method in transport/bardeen
'''

from transport import bardeen

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 3;
save_figs = False;
save_data = False;

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
error_lims = ()# (0,50);

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

#################################################################
#### spinless 1D IETS

# tight binding params
n_loc_dof = 1; 
tLR = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tLR;
VL = 0.0*tLR;
Vinfty = 0.5*tLR;
Ninfty = 20;
NC = 11;

#### hyper parameters ####
Ecut = 0.1;
defines_Sz = np.copy(tLR);

# T vs VR
if False:

    # variables
    NLR = 200;
    VRvals = np.array([-0.05*tLR,-0.005*tLR,0.005*tLR,0.05*tLR]);
    #VRvals = np.array([0.05*tLR]);
    int_power = -2;

    # plotting
    numplots = len(VRvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(11/2,3*numplots/2);

    # central region
    tC = 1.0*tLR;
    VC = 0.4*tLR;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print("HC =");
    print_H_j(HC);

    # HC, except Sz is a good quantum number
    HCobs = np.copy(HC);

    # bardeen results for heights of barrier covering well
    for VRvali in range(len(VRvals)):
        VLprime = VRvals[VRvali]+Vinfty; # left cover needs to be higher from view of right well
        VRprime = 0+Vinfty;
        assert(not np.any(np.ones((len(VC)),)[np.diagonal(VC) < np.diagonal(VRvals[VRvali])]));
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC, HCprime, matrix that defines alpha (non-observable) basis
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR,
                                      Vinfty, VL, VLprime, VRvals[VRvali], VRprime,
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
                                      E_cutoff=Ecut*np.eye(n_loc_dof),
                                      interval = 10**(int_power), verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VL, VRvals[VRvali], NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VL, VRvals[VRvali], HC, Evals,verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

         # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[VRvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[VRvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright = axes[VRvali].twinx();
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 
            if(error_lims): axright.set_ylim(*error_lims);
            
        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axes[VRvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[VRvali].set_title("$V_R = "+str(VRvals[VRvali][0,0])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

        # save
        folder = "data/rbstep/nosup/vsVR/"
        if(save_data):
            print("Saving data to "+folder);
            np.save(folder+"VR_{:.3f}_Tvals.npy".format(VRvals[VRvali][0,0]), Tvals);
            np.save(folder+"VR_{:.3f}_Tvals_bench.npy".format(VRvals[VRvali][0,0]), Tvals_bench);
            np.savetxt(folder+"VR_{:.3f}_info.txt".format(VRvals[VRvali][0,0]),
                       np.array([tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VRvals[VRvali][0,0], VRprime[0,0], Ninfty, NLR, NLR]),
                       header = "tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VRvals[VRvali][0,0], VRprime[0,0], Ninfty, NLR, NLR");
            np.savetxt(folder+"VR_{:.3f}_HC.txt".format(VRvals[VRvali][0,0]),HC[:,:,0,0], fmt="%.4f");

        #### end loop over VRvals

    # format and show
    axes[-1].set_xscale("log", subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $",fontsize=myfontsize);
    plt.tight_layout();
    plt.suptitle("$N_C = "+str(NC)+", V_C = "+str(VC[0,0])+", N_{LR} = "+str(NLR)+", \Delta \\varepsilon = 10^{"+str(int_power)+"} $");
    fname = "figs/rbstep/sup/step_VR.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs NLR
if False:

    # variables
    NLRvals = [50,200,1000];
    VR = -0.05*tLR;
    int_power = -2;

    # plotting
    numplots = len(NLRvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(11/2,3*numplots/2);

    # central region
    tC = 1.0*tLR;
    VC = 0.4*tLR;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print("HC =");
    print_H_j(HC);

    # HC, except Sz is a good quantum number
    HCobs = np.copy(HC);

    # primes
    VLprime = VR+Vinfty; # left cover needs to be higher from view of right well
    VRprime = 0+Vinfty;
    assert(not np.any(np.ones((len(VC)),)[np.diagonal(VC) < np.diagonal(VR)]));

    # bardeen results for heights of barrier covering well
    for NLRvali in range(len(NLRvals)):

        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        # returns two arrays of size (n_loc_dof, n_left_bound)
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NLRvals[NLRvali], NLRvals[NLRvali], HC, HCobs, defines_Sz,
                                      E_cutoff=Ecut*np.eye(n_loc_dof),
                                      interval = 10**(int_power), verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VL, VR, NLRvals[NLRvali], NLRvals[NLRvali], verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VL, VR, HC, Evals,verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

         # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[NLRvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[NLRvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright = axes[NLRvali].twinx();
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 
            if(error_lims): axright.set_ylim(*error_lims);
            
        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axes[NLRvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[NLRvali].set_title("$N_{LR} = "+str(NLRvals[NLRvali])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

        # save
        folder = "data/rbstep/nosup/vsNLR/"
        if(save_data):
            print("Saving data to "+folder);
            np.save(folder+"NLR_{:.0f}_Tvals.npy".format(NLRvals[NLRvali]), Tvals);
            np.save(folder+"NLR_{:.0f}_Tvals_bench.npy".format(NLRvals[NLRvali]), Tvals_bench);
            np.savetxt(folder+"NLR_{:.0f}_info.txt".format(NLRvals[NLRvali]),
                       np.array([tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VR[0,0], VRprime[0,0], Ninfty, NLRvals[NLRvali], NLRvals[NLRvali]]),
                       header = "tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VR[0,0], VRprime[0,0], Ninfty, NLR, NLR");
            np.savetxt(folder+"NLR_{:.0f}_HC.txt".format(NLRvals[NLRvali]),HC[:,:,0,0], fmt="%.4f");

        #### end loop over NLRvals

    # format and show
    axes[-1].set_xscale("log", subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $",fontsize=myfontsize);
    plt.tight_layout();
    plt.suptitle("$N_C = "+str(NC)+", V_C = "+str(VC[0,0])+", V_R = "+str(VR[0,0])+", \Delta \\varepsilon = 10^{"+str(int_power)+"} $");
    fname = "figs/rbstep/sup/step_NLR.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs \Delta \varepsilon
if True:

    # variables
    NLR = 200;
    VR = -0.05*tLR;
    powervals = [-6,-3,-2,np.nan];

    # plotting
    numplots = len(powervals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(11/2,3*numplots/2);

    # central region
    tC = 1.0*tLR;
    VC = 0.4*tLR;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print("HC =");
    print_H_j(HC);

    # HC, except Sz is a good quantum number
    HCobs = np.copy(HC);

    # primes
    VLprime = VR+Vinfty; # left cover needs to be higher from view of right well
    VRprime = 0+Vinfty;
    assert(not np.any(np.ones((len(VC)),)[np.diagonal(VC) < np.diagonal(VR)]));

    # bardeen results for heights of barrier covering well
    for powervali in range(len(powervals)):

        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        # returns two arrays of size (n_loc_dof, n_left_bound)
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
                                      E_cutoff=Ecut*np.eye(n_loc_dof),
                                      interval = 10**(powervals[powervali]), verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VL, VR, NLR, NLR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VL, VR, HC, Evals,verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

         # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[powervali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[powervali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright = axes[powervali].twinx();
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 
            if(error_lims): axright.set_ylim(*error_lims);
            
        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axes[powervali].set_ylabel('$T$',fontsize=myfontsize);
        axes[powervali].set_title("$\Delta \\varepsilon = 10^{"+str(powervals[powervali])+"}$", x=0.2, y = 0.7, fontsize=myfontsize);

        # save
        folder = "data/rbstep/nosup/vsdeps/"
        if(save_data):
            print("Saving data to "+folder);
            np.save(folder+"deps_{:.0f}_Tvals.npy".format(powervals[powervali]), Tvals);
            np.save(folder+"deps_{:.0f}_Tvals_bench.npy".format(powervals[powervali]), Tvals_bench);
            np.savetxt(folder+"deps_{:.0f}_info.txt".format(powervals[powervali]),
                       np.array([tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VR[0,0], VRprime[0,0], Ninfty, NLR, NLR]),
                       header = "tinfty[0,0], tLR[0,0], tLR[0,0], Vinfty[0,0], VL[0,0], VLprime[0,0], VR[0,0], VRprime[0,0], Ninfty, NLR, NLR");
            np.savetxt(folder+"deps_{:.0f}_HC.txt".format(powervals[powervali]),HC[:,:,0,0], fmt="%.4f");

        #### end loop over powervals

    # format and show
    axes[-1].set_xscale("log", subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $",fontsize=myfontsize);
    plt.tight_layout();
    plt.suptitle("$N_C = "+str(NC)+", V_C = "+str(VC[0,0])+", V_R = "+str(VR[0,0])+", N_{LR} = "+str(NLR)+" $");
    fname = "figs/rbstep/sup/step_deps.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();






