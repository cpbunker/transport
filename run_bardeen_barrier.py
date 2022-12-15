'''
Christian Bunker
M^2QM at UF
November 2022

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper) solved in time-dependent QM
using bardeen theory method in transport/bardeen
'''

from transport import bardeen

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

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

# tight binding params
n_loc_dof = 1; # spin up and down
tL = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tL;
tR = 1.0*tL;
ts = (tinfty, tL, tinfty, tR, tinfty);
Vinfty = 1.0*tL;
VL = 0.0*tL;
VR = 0.0*tL;
Vs = (Vinfty, VL, Vinfty, VR, Vinfty);

def get_ideal_T(Es,mytL,myVL,mytC,myVC,myNC):
    '''
    Get analytical T for square barrier scattering, landau & lifshitz pg 79
    '''

    kavals = np.arccos((Es-2*mytL-myVL)/(-2*mytL));
    kappavals = np.arccosh((Es-2*mytC-myVC)/(-2*mytC));
    print(Es[:8]-2*mytC-myVC);
    print(kavals[:8]);
    print(kappavals[:8]);

    ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
    ideal_exp = np.exp(-2*myNC*kappavals);
    ideal_T = ideal_prefactor*ideal_exp;
    ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
    ideal_T *= ideal_correction;
    return np.real(ideal_T);

#################################################################
#### benchmarking T in spinless 1D case

# T vs NL
if True:

    NLvals = [50,100];
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
    print("HC =\n",HC);

    # central region prime
    tCprime = tC;
    VCprime = VC;
    HCprime = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HCprime[j,j] = VCprime;
    for j in range(NC-1):
        HCprime[j,j+1] = -tCprime;
        HCprime[j+1,j] = -tCprime;
    print("HCprime =\n",HCprime);

    # bardeen results for different well widths
    for NLi in range(len(NLvals)):
        Ninfty = 100;
        NL = NLvals[NLi];
        NR = 1*NL;
        Evals, Tvals = bardeen.kernel(*ts, *Vs, Ninfty, NL, NR, HC, HCprime,verbose=0);
        Evals = np.real(Evals+2*tL);
        Tvals = np.real(Tvals);
        for alpha in range(n_loc_dof): 

            # truncate to bound states
            #Tvals[alpha] = Tvals[alpha,Evals[alpha] <= VC[alpha,alpha]];
            #Evals[alpha] = Evals[alpha,Evals[alpha] <= VC[alpha,alpha]];

            # plot
            #axes[NLi].scatter(Evals[alpha], Tvals[alpha], marker=mymarkers[0], color=mycolors[0]);

            # compare
            ideal_Tvals_alpha = get_ideal_T(Evals[alpha],tL[alpha,alpha],VL[alpha,alpha],tC[alpha,alpha],VC[alpha,alpha],NC);
            assert(np.shape(Evals[alpha]) == np.shape(ideal_Tvals_alpha));
            print(ideal_Tvals_alpha);
            axes[NLi].plot(Evals[alpha],np.real(ideal_Tvals_alpha), color=accentcolors[0], linewidth=mylinewidth);
            #axes[NLi].set_ylim(0,1.1*max(Tvals[alpha]));

        axes[NLi].set_ylabel('$T$',fontsize=myfontsize);
        axes[NLi].set_title('$N_L = '+str(NLvals[NLi])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

        # % error
        axright = axes[NLi].twinx();
        #axright.plot(Evals,100*abs((Tvals-np.real(ideal_Tvals_alpha))/ideal_Tvals),color=accentcolors[1]);
        axright.set_ylabel("$\%$ error",fontsize=myfontsize);
        axright.set_ylim(0,10)

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();





