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

# units
kelvin2eV =  8.617e-5; # units eV/K

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 3;
save_figs = True;

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
Ecut = 0.1;
defines_Sz = np.array([[1]]);

#T vs NL
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

    # HC, except Sz is a good quantum number
    HCobs = np.copy(HC);

    # bardeen results for different well widths
    for NLi in range(len(NLvals)):
        NLR = NLvals[NLi];
        # bardeen.kernel _wellsyntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR,
                                      Vinfty, VLR, Vinfty, VLR, Vinfty,
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
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
if False:

    # params
    tC = 1*tLR;
    VC = 0.4*tLR;
    VC = 5.0*tLR; Vinfty = 5.0*tLR; #!!!

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
        HC = np.zeros((NCval,NCval,n_loc_dof,n_loc_dof));
        for j in range(NCval):
            HC[j,j] += VC;
        for j in range(NCval-1):
            HC[j,j+1] += -tC;
            HC[j+1,j] += -tC;
        print("HC =");
        print_H_j(HC);

        # HC, except Sz is a good quantum number
        HCobs = np.copy(HC);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VLR, VLRprime,
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
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

        # HC, except Sz is a good quantum number
        HCobs = np.copy(HC);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VLR, VLRprime,
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
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

        # HC, except Sz is a good quantum number
        HCobs = np.copy(HC);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, VLRprime, VLR, VLRprime,
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
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

    # HC, except Sz is a good quantum number
    HCobs = np.copy(HC);

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
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
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
#### current through the barrier

if True:

    # experimental params
    Vbmax = 0.05;
    chempot_R = 0.0; # middle of the band
    kBT = 0.0001;
    Vbvals = np.linspace(-Vbmax, Vbmax, 99);

    # params
    tC = 1*tLR;
    VC = 0.4*tLR;
    VC = 5.0*tLR; Vinfty = 5.0*tLR; #!!!

    # set up barrier and get matrix elements
    NCvals = [5,9];
    numplots = len(NCvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);

    # bardeen results for heights of barrier covering well
    for NCvali in range(len(NCvals)):
        NCval = NCvals[NCvali];
        NLR = 200;

        # central region
        HC = np.zeros((NCval,NCval,n_loc_dof,n_loc_dof));
        for j in range(NCval):
            HC[j,j] += VC;
        for j in range(NCval-1):
            HC[j,j+1] += -tC;
            HC[j+1,j] += -tC;
        print("HC =");
        print_H_j(HC);

        # HC, except Sz is a good quantum number
        HCobs = np.copy(HC);
        
        # bardeen.kernel syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well(tinfty, tLR, tLR, 
                                      Vinfty, VLR, Vinfty, VLR, Vinfty,
                                      Ninfty, NLR, NLR, HC, HCobs, defines_Sz,
                                      E_cutoff=VC,verbose=1);

        # get Bardeen expression for current
        current = bardeen.current(Evals, Mvals, Vbvals,
                                  tLR, chempot_R, kBT, verbose = 1);

        print("Output shapes:");
        for arr in [Evals, Mvals, current, Evals[abs(Evals-chempot_R)<Vbmax]]: print(np.shape(arr));

        # just plot total current
        total_current = np.sum( np.sum(current, axis=0), axis=0);
        if(NCvali==0): I0 = np.max(total_current); # normalize
        axes[NCvali].scatter(Vbvals, total_current/I0, marker=mymarkers[0], color=mycolors[0]);

        # format
        axes[NCvali].set_ylabel('$I/I_0$',fontsize=myfontsize);
        axes[NCvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[NCvali].set_title("$N_C = {:.0f}$".format(NCval), x=0.4, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xlabel("$V_b$",fontsize=myfontsize);
    fig.suptitle("$ \mu_R = {:.2f}, k_B T = {:.2f}, N_L = {:.0f}, V_C = {:.2f}$".format(chempot_R, kBT/kelvin2eV, NLR, VC[0,0]));
    plt.tight_layout();
    fname = "figs/rbbarrier/current.pdf"
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();
