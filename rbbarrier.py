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
save_figs = False;

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
error_lims = ();

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

#################################################################
#### benchmarking T in spinless 1D case

# tight binding params
n_loc_dof = 1; 
tLR = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tLR;
VLR = 0.0*tLR;
Vinfty = 0.5*tLR;
Ninfty = 20;

#### hyper parameters ####
#defines_alpha = np.copy(tLR);

# T vs NL
if False:

    NLvals = [50,200,500];
    numplots = len(NLvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1.0*tLR;
    VC = 0.4*tLR;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] = VC;
    for j in range(NC-1):
        HC[j,j+1] = -tC;
        HC[j+1,j] = -tC;
    print("HC =");
    print_H_j(HC);

    # bardeen results for different well widths
    for NLi in range(len(NLvals)):
        NLR = NLvals[NLi];
        # bardeen.kernel _wellsyntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR,
                                      Vinfty, VLR, Vinfty, VLR, Vinfty,
                                      Ninfty, NLR, NLR, HC, HC,
                                      E_cutoff=VC, verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[NLi].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[NLi].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[NLi].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        if(error_lims): axright.set_ylim(*error_lims);
        axes[NLi].set_ylabel('$T$',fontsize=myfontsize);
        axes[NLi].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[NLi].set_title('$N_L = N_R = '+str(NLvals[NLi])+'$', x=0.4, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $",fontsize=myfontsize);
    fig.suptitle("$N_C = "+str(NC)+", V_C = "+str(VC[0,0])+", t_{vac} = "+str(tC[0,0])+"$");
    plt.tight_layout();
    fname = "figs/rbbarrier/barrier_NL.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs NC
if True:

    NCvals = [1,3,5,51];
    numplots = len(NCvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # bardeen results for heights of barrier covering well
    for NCvali in range(len(NCvals)):
        NCval = NCvals[NCvali];
        NLR = 200;
        VLRprime = Vinfty;

        # central region
        tC = 1*tLR;
        VC = 0.4*tLR;
        HC = np.zeros((NCval,NCval,n_loc_dof,n_loc_dof));
        for j in range(NCval):
            HC[j,j] += VC;
        for j in range(NCval-1):
            HC[j,j+1] += -tC;
            HC[j+1,j] += -tC;
        print("HC =");
        print_H_j(HC);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VLR, VLRprime,
                                      Ninfty, NLR, NLR, HC, HC,
                                      E_cutoff=VC,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[NCvali].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[NCvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[NCvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            errorvals = 100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]);
            axright.plot(xvals,errorvals,color=accentcolors[1]);

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        if(error_lims): axright.set_ylim(*error_lims);
        axes[NCvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[NCvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[NCvali].set_title("$N_C = "+str(NCval)+'$', x=0.4, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale("log", subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $",fontsize=myfontsize);
    fig.suptitle("$N_L = N_R = "+str(NLR)+", V_C = "+str(VC[0,0])+", t_{vac} = "+str(tC[0,0])+"$");
    plt.tight_layout();
    fname = "figs/rbbarrier/barrier_NC.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs VC
if False:

    Vinfty = 2.5*tLR;
    VCvals = np.array([0.1*tLR,0.4*tLR,0.8*tLR,2.0*tLR]);
    numplots = len(VCvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # bardeen results for heights of barrier covering well
    for VCvali in range(len(VCvals)):
        NC = 11;
        NLR = 200;
        VLRprime = Vinfty;
        VCval = VCvals[VCvali];

        # central region
        tC = 1*tLR;
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
        for j in range(NC):
            HC[j,j] += VCval;
        for j in range(NC-1):
            HC[j,j+1] += -tC;
            HC[j+1,j] += -tC;
        print("HC =");
        print_H_j(HC);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VLR, VLRprime,
                                      Ninfty, NLR, NLR, HC, HC,
                                      E_cutoff=VCval,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[VCvali].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[VCvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[VCvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        if(error_lims): axright.set_ylim(*error_lims);
        axes[VCvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[VCvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[VCvali].set_title("$V_C = "+str(VCval[0,0])+'$', x=0.4, y = 0.7, fontsize=myfontsize);
        axes[VCvali].axvline(VCval[0,0], color = "gray", linestyle = "dashed");

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L $',fontsize=myfontsize);
    fig.suptitle("$N_L = N_R = "+str(NLR)+", N_C = "+str(NC)+", V_{\infty} = "+str(Vinfty[0,0])+", t_{vac} = "+str(tC[0,0])+"$");
    plt.tight_layout();
    fname = "figs/rbbarrier/barrier_VC.pdf";
    if(NC != 11): fname = "figs/rbbarrier/barrier_VC_marg.pdf"
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs tC (reduced-tC insulating region of width NC=3)
if False:

    tCvals = [1.0*tLR,0.8*tLR,0.2*tLR];
    numplots = len(tCvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # bardeen results for heights of barrier covering well
    for tCvali in range(len(tCvals)):
        NLR = 200;
        VLRprime = Vinfty;

        # central region
        NC = 3;
        VC = 0.1*tLR;
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
        for j in range(NC):
            if(j == NC//2): # middle to keep LR symmetry !!!
                HC[j,j] += VC; 
            else:
                HC[j,j] += VLR; #raise Exception("Change so that there is no gap between VC and VRprime");
        for j in range(NC-1):
            HC[j,j+1] += -tCvals[tCvali];
            HC[j+1,j] += -tCvals[tCvali];
        print("HC =");
        print_H_j(HC);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VLR, VLRprime,
                                      Ninfty, NLR, NLR, HC, HC,
                                      E_cutoff=VC,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR, verbose=0);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[tCvali].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[tCvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[tCvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        if(error_lims): axright.set_ylim(*error_lims);
        axes[tCvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[tCvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[tCvali].set_title("$t_{vac} = "+str(tCvals[tCvali][0,0])+'$', x=0.4, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale("log", subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $",fontsize=myfontsize);
    fig.suptitle("$N_L = N_R = "+str(NLR)+", N_C = "+str(NC)+", V_C = "+str(VC[0,0])+"$");
    plt.tight_layout();
    fname = "figs/rbbarrier/barrier_tC.pdf";
    if(VC[0,0] != 0.4): fname = "figs/rbbarrier/barrier_tC_marg.pdf"
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs VLR prime
if False:

    Vprimevals = [Vinfty/5,Vinfty,10*Vinfty];
    numplots = len(Vprimevals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tLR;
    VC = 0.4*tLR;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print("HC =");
    print_H_j(HC);

    # bardeen results for heights of barrier covering well
    for Vprimei in range(len(Vprimevals)):
        NLR = 200;
        VLRprime = Vprimevals[Vprimei];
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VLR, VLRprime,
                                      Ninfty, NLR, NLR, HC, HC,
                                      E_cutoff=VC,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[Vprimei].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[Vprimei].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[Vprimei].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axright.set_ylim(*error_lims);
        axes[Vprimei].set_ylabel('$T$',fontsize=myfontsize);
        axes[Vprimei].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[Vprimei].set_title("$V_L' = V_R' = "+str(Vprimevals[Vprimei][0,0])+'$', x=0.4, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale("log", subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_L)/t_L $",fontsize=myfontsize);
    fig.suptitle("$N_L = N_R = "+str(NLR)+", N_C = "+str(NC)+", V_C = "+str(VC[0,0])+", t_{vac} = "+str(tC[0,0])+"$");
    plt.tight_layout();
    fname = "figs/rbbarrier/barrier_VLprime.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();








###############################################################################
#### left right symmetry breaking

# T vs VR
if False:

    VRvals = [0.0*tLR, 0.001*tLR, 0.01*tLR];
    numplots = len(VRvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # central region
    tC = 1*tLR;
    VC = 0.4*tLR;
    NC = 11;
    HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
    for j in range(NC):
        HC[j,j] += VC;
    for j in range(NC-1):
        HC[j,j+1] += -tC;
        HC[j+1,j] += -tC;
    print("HC =");
    print_H_j(HC);

    # bardeen results for heights of barrier covering well
    for VRvali in range(len(VRvals)):
        NLR = 200; # try increasing
        VLRprime = Vinfty;
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VRvals[VRvali], VLRprime,
                                      Ninfty, NLR, NLR, HC, HC,
                                      E_cutoff=VC[0,0],interval=1e-2,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VRvals[VRvali], NLR, NLR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VRvals[VRvali], HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):
            axright = axes[VRvali].twinx();

            # truncate to bound states and plot
            xvals = np.real(Evals[alpha])+2*tLR[alpha,alpha];
            axes[VRvali].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);

            # % error
            axes[VRvali].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        #axright.set_ylim(*error_lims);
        axes[VRvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[VRvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[VRvali].set_title("$V_R = "+str(VRvals[VRvali][0,0])+'$', x=0.4, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    plt.show();
    fname = "figs/bard_barrier/bard_barrier_VLprime.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

###############################################################################
#### messing with hopping

# perturbation is hopping from central region to right lead ONLY
if False:

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



