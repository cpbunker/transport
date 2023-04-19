'''
Christian Bunker
M^2QM at UF
November 2022

Spin independent scattering from a rectangular potential barrier
benchmarked to exact solution
solved in time-dependent QM using bardeen theory method in transport/bardeen
'''

from transport import bardeen

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

#################################################################
#### benchmarking T in spinless 1D case

# tight binding params
n_loc_dof = 1; 
tL = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tL;
tR = 1.0*tL;
Vinfty = 0.5*tL;
VL = 0.0*tL;
VR = 0.0*tL;

# T vs NL
if False:

    NLvals = [50,100,500];
    numplots = len(NLvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tL;
    VC = 0.1*tL;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] = VC;
    for j in range(NC-1):
        HC[j,j+1] = -tC;
        HC[j+1,j] = -tC;
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);

    # bardeen results for different well widths
    for NLi in range(len(NLvals)):
        Ninfty = 20;
        NL = NLvals[NLi];
        NR = 1*NL;
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, Vinfty, VR, Vinfty,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=VC[0,0], verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tL, tR, VL, VR, NL, NR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm(tL, tR, VL, VR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[NLi].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[NLi].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[NLi].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axright.set_ylim(0,50);
        axes[NLi].set_ylabel('$T$',fontsize=myfontsize);
        axes[NLi].set_title('$N_L = '+str(NLvals[NLi])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();
    #plt.savefig("figs/bard_barrier/NL.pdf");

# T vs VLR prime
if False:

    Vprimevals = [Vinfty/10,Vinfty/5,Vinfty];
    numplots = len(Vprimevals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tL;
    VC = 0.1*tL;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);

    # bardeen results for heights of barrier covering well
    for Vprimei in range(len(Vprimevals)):
        Ninfty = 20;
        NL = 200;
        NR = 1*NL;
        VLprime = Vprimevals[Vprimei];
        VRprime = Vprimevals[Vprimei];
        
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=VC[0,0],verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tL, tR, VL, VR, NL, NR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm(tL, tR, VL, VR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[Vprimei].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[Vprimei].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[Vprimei].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axright.set_ylim(0,50);
        axes[Vprimei].set_ylabel('$T$',fontsize=myfontsize);
        axes[Vprimei].set_title("$V_L' = "+str(Vprimevals[Vprimei][0,0])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();
    #plt.savefig("figs/bard_barrier/VLprime.pdf");

# worst case vs best case
if False:

    numplots = 2;
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tL;
    VC = 0.1*tL;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] = VC;
    for j in range(NC-1):
        HC[j,j+1] = -tC;
        HC[j+1,j] = -tC;
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);
    print_H_j(HCprime);

    # bardeen results for worst case
    axno = 0;
    Ninfty = 20;
    NL = 50;
    NR = 1*NL;
    VLprime = Vinfty/10;
    VRprime = Vinfty/10;
    # bardeen.kernel syntax:
    # tinfty, tL, tLprime, tR, tRprime,
    # Vinfty, VL, VLprime, VR, VRprime,
    # Ninfty, NL, NR, HC,HCprime,
    Evals, Mvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                  Vinfty, VL, VLprime, VR, VRprime,
                                  Ninfty, NL, NR, HC, HCprime,
                                  E_cutoff=VC[0,0],verbose=1);
    Tvals = bardeen.Ts_bardeen(Evals, Mvals, tL, tR, VL, VR, NL, NR, verbose=1);

    # benchmark
    Tvals_bench = bardeen.Ts_wfm(tL, tR, VL, VR, HC, Evals, verbose=0);
    print("Output shapes:");
    for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

    # only one loc dof, and transmission is diagonal
    for alpha in range(n_loc_dof):
        axright = axes[axno].twinx();

        # truncate to bound states and plot
        xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
        axes[axno].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

        # % error
        axes[axno].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
        axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

    # format
    axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
    axright.set_ylim(0,50);
    axes[axno].set_ylabel('$T$',fontsize=myfontsize);
    axes[axno].set_title("Worst", x=0.2, y = 0.7, fontsize=myfontsize);

    # bardeen results for best case
    axno = 1;
    Ninfty = 20;
    NL = 500;
    NR = 1*NL;
    VLprime = Vinfty;
    VRprime = Vinfty;
    # bardeen.kernel syntax:
    # tinfty, tL, tLprime, tR, tRprime,
    # Vinfty, VL, VLprime, VR, VRprime,
    # Ninfty, NL, NR, HC,HCprime,
    Evals, Mvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                  Vinfty, VL, VLprime, VR, VRprime,
                                  Ninfty, NL, NR, HC, HCprime,
                                  E_cutoff=VC[0,0],verbose=1);
    Tvals = bardeen.Ts_bardeen(Evals, Mvals, tL, tR, VL, VR, NL, NR, verbose=1);
    
    # benchmark
    Tvals_bench = bardeen.Ts_wfm(tL, tR, VL, VR, HC, Evals, verbose=0);
    print("Output shapes:");
    for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

    # only one loc dof, and transmission is diagonal
    for alpha in range(n_loc_dof):
        axright = axes[axno].twinx();

        # truncate to bound states and plot
        xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
        axes[axno].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

        # % error
        axes[axno].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
        axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

    # format
    axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
    axright.set_ylim(0,50);
    axes[axno].set_ylabel('$T$',fontsize=myfontsize);
    axes[axno].set_title("Best", x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();


# island in barrier for 2nd order processes
if True:

    # central region
    tC = 1*tL;
    VC = 0.1*tL;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);

    VIvals = np.array([0.01,0.05,0.1]);
    numplots = len(VIvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # V primes
    VLprime = 1*Vinfty;
    VRprime = 1*Vinfty;
    Ninfty = 20;
    NL = 500;
    NR = 1*NL;

    # bardeen results vs heights of island
    for VIi in range(len(VIvals)):

        # island
        VIval = VIvals[VIi]*tL;
        HC[NC//2,NC//2] = VIval;
        
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=0.1*VC[0,0],verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tL, tR, VL, VR, NL, NR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm(tL, tR, VL, VR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[VIi].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[VIi].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[VIi].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axright.set_ylim(0,50);
        axes[VIi].set_ylabel('$T$',fontsize=myfontsize);
        axes[VIi].set_title("$V_I' = "+str(VIvals[VIi])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();
    #plt.savefig("figs/bard_barrier/VLprime.pdf");



