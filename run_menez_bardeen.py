'''
Christian Bunker
M^2QM at UF
November 2022

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper) solved in time-dependent QM
using bardeen theory method in transport/bardeen
'''

from transport import bardeen, wfm

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

def get_ideal_T(Es,mytL,myVL,mytC,myVC,myNC):
    '''
    Get analytical T for spin-spin scattering, Menezes paper
    '''

    return False

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

def print_H_alpha(H):
    assert(len(np.shape(H)) == 4);
    for j in range(np.shape(H)[0]):
        print("H["+str(j)+","+str(j)+"] =\n",H[j,j,:,:]);

#################################################################
#### all possible T_{\alpha -> \beta}

if True:

    # range of energies
    logElims = -4,0
    Evals = np.logspace(*logElims,myxvals, dtype = complex);

    # R and T matrices
    alphas = [0,1,2,3];
    alpha_strs = ["\\uparrow \\uparrow","\\uparrow \downarrow","\downarrow \\uparrow","\downarrow \downarrow"];
    hspacesize = len(alphas);
    Rvals = np.empty((hspacesize,hspacesize,len(Evals)), dtype = float);
    Tvals = np.empty((hspacesize,hspacesize,len(Evals)), dtype = float);

    # plotting
    nplots_x = len(alphas);
    nplots_y = len(alphas);
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);

    # tight binding params
    n_loc_dof = len(alphas); # spin up and down for each
    tL = 1.0*np.eye(n_loc_dof);
    tinfty = 1.0*tL;
    tR = 1.0*tL;
    ts = (tinfty, tL, tinfty, tR, tinfty);
    Vinfty = 0.5*tL;
    VL = 0.0*tL;
    VR = 0.0*tL;
    Vs = (Vinfty, VL, Vinfty, VR, Vinfty);
    Jval = -0.5;

    # central region
    NC = 1;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
    for NCi in range(NC):
        HC[NCi,NCi] = wfm.h_kondo(Jval,0.5);
    print_H_alpha(HC);

    # central region prime
    HCprime = np.zeros_like(HC);
    for NCi in range(NC):
        HCprime[NCi,NCi] = np.diagflat(np.diagonal(HC[NCi,NCi]));
    print_H_alpha(HCprime);

    # bardeen results 
    Ninfty = 100;
    NL = 1*Ninfty
    NR = 1*NL;
    Evals, Tvals = bardeen.kernel(*ts, *Vs, Ninfty, NL, NR, HC, HCprime,verbose=verbose);
    print(np.shape(Tvals),np.shape(Evals)); assert False
    # initial states
    for alpha in alphas:
        Evals[alpha] = np.real(Evals[alpha]+2*tL[alpha,alpha]);
        Tvals[alpha] = np.real(Tvals[alpha]);

        # final states
        for beta in alphas:

            # truncate to bound states and plot
            yvals = Tvals[alpha,beta];
            xvals = Evals[alpha,alpha]
            #yvals = Tvals[alpha,Evals[alpha] <= VC[alpha,alpha]];
            #xvals = Evals[alpha,Evals[alpha] <= VC[alpha,alpha]];
            axes[alpha,beta].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

            # compare
            #ideal_Tvals_alpha = get_ideal_T(Evals[alpha],tL[alpha,alpha],VL[alpha,alpha],tC[alpha,alpha],VC[alpha,alpha],NC);
            #axes[NLi].plot(Evals[alpha],np.real(ideal_Tvals_alpha), color=accentcolors[0], linewidth=mylinewidth);
            #axes[NLi].set_ylim(0,1.1*max(Tvals[alpha]));

        #format
        #axes[alpha,-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);

    # format and show
    axes[-1,-1].set_xscale('log', subs = []);
    plt.tight_layout();
    plt.show();





