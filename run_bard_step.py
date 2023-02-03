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
mymarkers = ["o","+","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

def get_T_exact(Es,mytL,myVL,mytC,myVC,myNC):
    '''
    Get analytical T for square barrier scattering, landau & lifshitz pg 79
    '''
    kavals = np.arccos((Es-2*mytL-myVL)/(-2*mytL));
    kappavals = np.arccosh((Es-2*mytC-myVC)/(-2*mytC));
    print("Es:\n", Es[:8]);
    print("kavals:\n", kavals[:8]);
    print("kappavals:\n", kappavals[:8]);
    exact_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
    exact_exp = np.exp(-2*myNC*kappavals);
    exact_T = exact_prefactor*exact_exp;
    exact_correction = np.power(1+(exact_prefactor-2)*exact_exp+exact_exp*exact_exp,-1);
    exact_T *= exact_correction;
    return np.array([np.real(exact_T)]);

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

# T vs VR
if False:

    VRvals = np.array([0.1*Vinfty,0.2*Vinfty]);
    numplots = len(VRvals);
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
        # returns two arrays of size (n_loc_dof, n_left_bound)
        Evals, Tvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=VC[0,0],verbose=10);

        # benchmark
        axright = axes[Vprimei].twinx();
        Tvals_bench = bardeen.benchmark(tL, tR, VL, VR, HC, Evals, verbose=0);

        # for each dof
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[Vprimei].scatter(xvals, Tvals[alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[Vprimei].scatter(xvals, Tvals_bench[alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha]-Tvals_bench[alpha])/Tvals_bench[alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axright.set_ylim(0,50);
        axes[VRi].set_ylabel('$T$',fontsize=myfontsize);
        axes[VRi].set_title("$V_L' = "+str(Vprimevals[VRi][0,0])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();




