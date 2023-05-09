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
    print("HC =");
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);
    print("HC - HCprime =");
    print_H_j(HC-HCprime);

    # bardeen results for different well widths
    for NLi in range(len(NLvals)):
        Ninfty = 20;
        NL = NLvals[NLi];
        NR = 1*NL;
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, Vinfty, VR, Vinfty,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=VC[0,0], verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tL, tR, VL, VR, NL, NR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tL, tR, VL, VR, HC, Evals, verbose=0);
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
    print("HC =");
    print_H_j(HC);

    # central region prime
    HCprime = np.copy(HC);
    print("HC - HCprime =");
    print_H_j(HC-HCprime);

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
        Evals, Mvals = bardeen.kernel_well(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=VC[0,0],verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tL, tR, VL, VR, NL, NR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tL, tR, VL, VR, HC, Evals, verbose=0);
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

# perturbation is hopping from central region to right lead ONLY
if True:

    # sizes
    NC = 5;
    Ninfty = 20;
    NL = 200;
    NR = 1*NL;

    # central region
    tC = 1.0*tL;
    VC = 0.5*tL; #### do for 0.5, 2.5, 5.0, 50.0 ####

    hopvals = np.array([0.0*tC,1.0*tC]);
    numplots = len(hopvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # compare perturb being thyb vs VRprime
    for hopvali in range(len(hopvals)):
        #hopvali=1;
        hopval = hopvals[hopvali];

        # V primes
        if(hopvali == 0): # thyb is perturbation
            VLprime = 1*VL;
            VRprime = 1*VR;
            HT_perturb = True;
            myinterval = 1e-3;
        else: # VRprime is perturbation
            VLprime = 1*Vinfty;
            VRprime = 1*Vinfty;
            HT_perturb = False;
            myinterval = 1e-9;
        thyb = 1.0*tL;
        Vhyb = 1.0*VLprime;

        # construct ham
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
        for j in range(NC):
            HC[j,j] += VC;
        for j in range(NC-1):
            HC[j,j+1] += -tC;
            HC[j+1,j] += -tC;
        # overwrite hyb
        HC[0,0] = Vhyb;
        HC[-1,-1] = Vhyb;
        HC[0,1] = -thyb;
        HC[1,0] = -thyb;
        HC[-2,-1] = -thyb;
        HC[-1,-2] = -thyb;
        print("HC =");
        print_H_j(HC);

        # central region prime
        HCprime = np.copy(HC);
        HCprime[0,0] = Vhyb;
        HCprime[-1,-1] = Vhyb;
        #HCprime[0,1] = -hopval;
        #HCprime[1,0] = -hopval;
        HCprime[-2,-1] = -hopval;
        HCprime[-1,-2] = -hopval;
        print("HC - HCprime =");
        print_H_j(HC-HCprime);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=0.1,interval=myinterval,
                                      HT_perturb=HT_perturb,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tL, tR, VL, VR, NL, NR, verbose=1);
        
        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tL, tR, VL, VR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):

            # sort
            sortis = np.argsort(Evals[alpha]);
            Evals[alpha] = Evals[alpha][sortis];
            Tvals[alpha,:,alpha] = Tvals[alpha,:,alpha][sortis];
            Tvals_bench[alpha,:,alpha] = Tvals_bench[alpha,:,alpha][sortis];

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[hopvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);
            axes[hopvali].set_xlim(min(xvals),0.1);
            for ax in axes: ax.set_ylim(0.0,np.real(max(Tvals[alpha,:,alpha])).round(4) );
            
            # % error
            axright = axes[hopvali].twinx();
            axes[hopvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axrighttops = [100,50,50,20]; VCoptions = [0.5,2.5,5.0,50];
        axright.set_ylim(0,axrighttops[VCoptions.index(VC)]);
        axes[hopvali].set_ylabel("$T (t_{DR} = "+str(hopvals[hopvali][0,0])+")$",fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();

# perturbation is HT
if False:

    # sizes
    NC = 3;
    Ninfty = 20;
    NL = 200;
    NR = 1*NL;

    # central region
    tC = 1.0*tL;
    VC = 0.5*tL;
    thyb = 1.0*tL;
    Vhyb = 1.0*VL;
    Ueff = 0.0;
    Weff = (2*VC+Ueff)*thyb*np.conj(thyb)/( 2*(VC+Ueff)*(-VC));
    #Weff = 0.0;

    # construct central region ham
    assert(Ueff == 0.0);
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    # overwrite hyb
    HC[0,0] = Vhyb;
    HC[-1,-1] = Vhyb;
    HC[0,1] = -thyb;
    HC[1,0] = -thyb;
    HC[-2,-1] = -thyb;
    HC[-1,-2] = -thyb;
    assert(Ueff == 0.0 and NC == 3);
    HC[0,2] = Weff;
    HC[2,0] = Weff;
    print("HC =");
    print_H_j(HC);

    hopvals = np.array([0.0*tC,1.0*tC]);
    numplots = len(hopvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # compare perturb being thyb vs VRprime
    for hopvali in range(len(hopvals)):
        hopval = hopvals[hopvali];

        # central region prime
        HCprime = np.copy(HC);
        HCprime[0,0] = Vhyb;
        HCprime[-1,-1] = Vhyb;
        HCprime[0,1] = -hopval;
        HCprime[1,0] = -hopval;
        HCprime[-2,-1] = -hopval;
        HCprime[-1,-2] = -hopval;
        HCprime[0,2] = 0.0;
        HCprime[2,0] = 0.0;

        # V primes
        if(hopvali == 0):
            VLprime = 1*VL;
            VRprime = 1*VR;
            HT_perturb = True;
        else:
            VLprime = 1*Vinfty;
            VRprime = 1*Vinfty;
            HT_perturb = False;
            HC[0,2] = 0.0;
            HC[2,0] = 0.0;
        print("HC - HCprime =");
        print_H_j(HC-HCprime);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tLprime, tR, tRprime,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel(tinfty, tL, tinfty, tR, tinfty,
                                      Vinfty, VL, VLprime, VR, VRprime,
                                      Ninfty, NL, NR, HC, HCprime,
                                      E_cutoff=0.1,HT_perturb=HT_perturb,verbose=10);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tL, tR, VL, VR, NL, NR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm(tL, tR, VL, VR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):

            # sort
            sortis = np.argsort(Evals[alpha]);
            Evals[alpha] = Evals[alpha][sortis];
            Tvals[alpha,:,alpha] = Tvals[alpha,:,alpha][sortis];
            Tvals_bench[alpha,:,alpha] = Tvals_bench[alpha,:,alpha][sortis];

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tL[alpha,alpha];
            axes[hopvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axright = axes[hopvali].twinx();
            axes[hopvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axright.set_ylim(0,100);
        axes[hopvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[hopvali].set_title("$t_{hyb} = "+str(hopvals[hopvali][0,0])+'$', x=0.2, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();



