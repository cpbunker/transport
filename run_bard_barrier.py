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

# tight binding params
n_loc_dof = 1; 
tL = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tL;
tR = 1.0*tL;
# ts = (tinfty, tL, tinfty, tR, tinfty);
Vinfty = 0.5*tL;
VL = 0.0*tL;
VR = 0.0*tL;
# Vs = (Vinfty, VL, Vinfty, VR, Vinfty);

def get_ideal_T(Es,mytL,myVL,mytC,myVC,myNC):
    '''
    Get analytical T for square barrier scattering, landau & lifshitz pg 79
    '''

    kavals = np.arccos((Es-2*mytL-myVL)/(-2*mytL));
    kappavals = np.arccosh((Es-2*mytC-myVC)/(-2*mytC));
    print("Es:\n", Es[:8]);
    print("kavals:\n", kavals[:8]);
    print("kappavals:\n", kappavals[:8]);

    ideal_prefactor = np.power(4*kavals*kappavals/(kavals*kavals+kappavals*kappavals),2);
    ideal_exp = np.exp(-2*myNC*kappavals);
    ideal_T = ideal_prefactor*ideal_exp;
    ideal_correction = np.power(1+(ideal_prefactor-2)*ideal_exp+ideal_exp*ideal_exp,-1);
    ideal_T *= ideal_correction;
    return np.real(ideal_T);

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

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
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);
    print_H_j(HCprime);

    # bardeen results for different well widths
    for NLi in range(len(NLvals)):
        Ninfty = 50;
        NL = NLvals[NLi];
        NR = 1*NL;
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Tvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, Vinfty, VR, Vinfty,
                                      Ninfty, NL, NR, HC, HCprime,cutoff=VC[0,0],verbose=verbose);

        # % error
        axright = axes[NLi].twinx();

        # for each dof
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            yvals = np.diagonal(Tvals[alpha,:,alpha,:]);
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[NLi].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

            # compare
            ideal_Tvals_alpha = get_ideal_T(xvals,tL[alpha,alpha],VL[alpha,alpha],tC[alpha,alpha],VC[alpha,alpha],NC);
            axes[NLi].plot(xvals,np.real(ideal_Tvals_alpha), color=accentcolors[0], linewidth=mylinewidth);
            #axes[NLi].set_ylim(0,1.1*max(Tvals[alpha]));
            axright.plot(xvals,100*abs((yvals-np.real(ideal_Tvals_alpha))/ideal_Tvals_alpha),color=accentcolors[1]);

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize);
        axright.set_ylim(0,50);
        axes[NLi].set_ylabel('$T$',fontsize=myfontsize);
        axes[NLi].set_title('$N_L = '+str(NLvals[NLi])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();

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
        HC[j,j] = VC;
    for j in range(NC-1):
        HC[j,j+1] = -tC;
        HC[j+1,j] = -tC;
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);
    print_H_j(HCprime);

    # bardeen results for heights of barrier covering well
    for Vprimei in range(len(Vprimevals)):
        Ninfty = 50;
        NL = 100;
        NR = 1*NL;
        VLprime = Vprimevals[Vprimei];
        VRprime = Vprimevals[Vprimei];
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Tvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,cutoff=VC[0,0],verbose=verbose);

        # % error
        axright = axes[Vprimei].twinx();

        # for each dof
        for alpha in range(n_loc_dof): 

            # truncate to bound states and plot
            yvals = np.diagonal(Tvals[alpha,:,alpha,:]);
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[Vprimei].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

            # compare
            ideal_Tvals_alpha = get_ideal_T(xvals,tL[alpha,alpha],VL[alpha,alpha],tC[alpha,alpha],VC[alpha,alpha],NC);
            axes[Vprimei].plot(xvals,np.real(ideal_Tvals_alpha), color=accentcolors[0], linewidth=mylinewidth);
            #axes[Vprimei].set_ylim(0,1.1*max(Tvals[alpha]));
            axright.plot(xvals,100*abs((yvals-np.real(ideal_Tvals_alpha))/ideal_Tvals_alpha),color=accentcolors[1]);

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize);
        axright.set_ylim(0,50);
        axes[Vprimei].set_ylabel('$T$',fontsize=myfontsize);
        axes[Vprimei].set_title("$V_L' = "+str(Vprimevals[Vprimei])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();

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
    Ninfty = 50;
    NL = 50;
    NR = 1*NL;
    VLprime = Vinfty/10;
    VRprime = Vinfty/10;
    # bardeen.kernel syntax:
    # tinfty, tL, tLprime, tR, tRprime,
    # Vinfty, VL, VLprime, VR, VRprime,
    # Ninfty, NL, NR, HC,HCprime,
    Evals, Tvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                  Vinfty, VL, VLprime, VR, VRprime,
                                  Ninfty, NL, NR, HC, HCprime,cutoff=VC[0,0],verbose=verbose);

    # % error
    axright = axes[axno].twinx();

    # for each dof
    for alpha in range(n_loc_dof): 

        # truncate to bound states and plot
        yvals = np.diagonal(Tvals[alpha,:,alpha,:]);
        xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
        axes[axno].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

        # compare
        ideal_Tvals_alpha = get_ideal_T(xvals,tL[alpha,alpha],VL[alpha,alpha],tC[alpha,alpha],VC[alpha,alpha],NC);
        axes[axno].plot(xvals,np.real(ideal_Tvals_alpha), color=accentcolors[0], linewidth=mylinewidth);
        #axes[axno].set_ylim(0,1.1*max(Tvals[alpha]));
        axright.plot(xvals,100*abs((yvals-np.real(ideal_Tvals_alpha))/ideal_Tvals_alpha),color=accentcolors[1]);

    # format
    axright.set_ylabel("$\%$ error",fontsize=myfontsize);
    axright.set_ylim(0,50);
    axes[axno].set_ylabel('$T$',fontsize=myfontsize);
    axes[axno].set_title("Worst", x=0.2, y = 0.7, fontsize=myfontsize);

    # bardeen results for best case
    axno = 1;
    Ninfty = 50;
    NL = 500;
    NR = 1*NL;
    VLprime = Vinfty;
    VRprime = Vinfty;
    # bardeen.kernel syntax:
    # tinfty, tL, tLprime, tR, tRprime,
    # Vinfty, VL, VLprime, VR, VRprime,
    # Ninfty, NL, NR, HC,HCprime,
    Evals, Tvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                  Vinfty, VL, VLprime, VR, VRprime,
                                  Ninfty, NL, NR, HC, HCprime,cutoff=VC[0,0],verbose=verbose);

    # % error
    axright = axes[axno].twinx();

    # for each dof
    for alpha in range(n_loc_dof): 

        # truncate to bound states and plot
        yvals = np.diagonal(Tvals[alpha,:,alpha,:]);
        xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
        axes[axno].scatter(xvals, yvals, marker=mymarkers[0], color=mycolors[0]);

        # compare
        ideal_Tvals_alpha = get_ideal_T(xvals,tL[alpha,alpha],VL[alpha,alpha],tC[alpha,alpha],VC[alpha,alpha],NC);
        axes[axno].plot(xvals,np.real(ideal_Tvals_alpha), color=accentcolors[0], linewidth=mylinewidth);
        #axes[axno].set_ylim(0,1.1*max(Tvals[alpha]));
        axright.plot(xvals,100*abs((yvals-np.real(ideal_Tvals_alpha))/ideal_Tvals_alpha),color=accentcolors[1]);

    # format
    axright.set_ylabel("$\%$ error",fontsize=myfontsize);
    axright.set_ylim(0,50);
    axes[axno].set_ylabel('$T$',fontsize=myfontsize);
    axes[axno].set_title("Best", x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();





