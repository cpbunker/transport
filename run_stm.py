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
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mymarkevery = 50;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

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

#################################################################
#### benchmarking T in spinless 1D case

# T vs NL
if True:

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
        Evals, Tvals = bardeen.kernel(*ts, *Vs, Ninfty, NL, NR, HC, HCprime);
        Evals = np.real(Evals+2*mytL);
        Tvals = np.real(Tvals);
        for alpha in range(np.shape(Evals)[-1]):  # bound states only
            Tvals[:,alpha] = Tvals[Evals <= myVC,alpha];
            Evals[:,alpha] = Evals[Evals <= myVC,alpha];
        axes[NRi].scatter(Evals, Tvals, marker=mymarkers[0], color=mycolors[0]);
        axes[NRi].set_ylim(0,1.1*max(Tvals));

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
    for axright in axrights: axright.set_ylim(0,0.1;
    plt.tight_layout();
    plt.show();





