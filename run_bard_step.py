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

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","^","s","d","*","X","P"];
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
VLprime = Vinfty;

# T vs VLR prime
if True:

    VRvals = np.array([0.1*Vinfty,0.2*Vinfty]);
    numplots = len(VRvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tL;
    VC = 0.3*tL;
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
    for VRi in range(len(VRvals)):
        Ninfty = 20;
        NL = 100;
        NR = 1*NL;
        VR = VRvals[VRi];
        VRprime = VRvals[VRi]+Vinfty;
        assert(not np.any(np.ones((len(VC)),)[np.diagonal(VC) < np.diagonal(VR)]));
        
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Tvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=VC[0,0],verbose=10);

        # % error
        axright = axes[VRi].twinx();
        Tvals_bench = bardeen.benchmark(tL, tR, VL, VR, HC, Evals, verbose=verbose);

        # for each dof
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            yvals = np.diagonal(Tvals[alpha,:,alpha,:]);
            yvals_bench = np.diagonal(Tvals_bench[alpha,:,alpha,:]);
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[Vprimei].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[Vprimei].plot(xvals, yvals_bench, color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((yvals-yvals_bench)/yvals_bench),color=accentcolors[1]);

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize);
        axright.set_ylim(0,50);
        axes[Vprimei].set_ylabel('$T$',fontsize=myfontsize);
        axes[Vprimei].set_title("$V_R' = "+str(VRvals[VRi][0,0])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();

