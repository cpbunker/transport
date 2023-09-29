'''
Christian Bunker
M^2QM at UF
August 2023

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper) 
benchmarked to exact solution
solved in time-dependent QM using bardeen theory method in transport/bardeen

"No superposition method" : the Kondo term is included in Hsys only, so we can
work directly in the eigenbasis of Sz, without need for superposition.
Therefore Kondo shows up in H_sys - HL, causing nonzero overlap between
initial, final states of different spin

INELASTIC PROCESSES are now considered, via an impurity spin-flip cost Delta
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
error_lims = (0,100);

def print_H_j(H):
    assert(len(np.shape(H)) == 4);
    for alpha in range(np.shape(H)[-1]):
        print("H["+str(alpha)+","+str(alpha)+"] =\n",H[:,:,alpha,alpha]);

def print_H_alpha(H):
    assert(len(np.shape(H)) == 4);
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",H[i,j,:,:]);

# tight binding params
n_loc_dof = 2; 
tLR = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tLR;
VLR = 0.0*tLR; # NB explicit LR symmetry
Vinfty = 0.5*tLR;
NLR = 200;
Ninfty = 20;

#### hyper parameters ####
Ecut = 0.1;
interval = 1e-2;
eigval_tol = 1e-5;

###############################################
# the Kondo term is included in Hsys only, so can work in eigenbasis of Sz
# therefore Kondo shows up in H_sys - HL, causing nonzero overlap between
# initial, final states of different spin

# inelastic processes

# T vs Delta at fixed J
if True:
    
    # alpha -> beta
    alphas = [0,1];
    alpha_strs = ["\\uparrow","\downarrow"];

    # plotting
    plot_alpha = False;
    if(plot_alpha):
        indvals = np.array([0.02]);
        nplots_x = len(alphas);
        nplots_y = len(alphas);
    else:
        indvals = np.array([2e-6,2e-5,2e-4]);
        nplots_x = 1
        nplots_y = len(indvals);
        alpha_initial, alpha_final = 0,0;
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);
        
    # iter over J
    for indvali in range(len(indvals)):

        #### inelastic
        Deltaval = indvals[indvali]*np.array([[0,0],[0,1]]);  # <------- !!!!

        # central region
        Jval = -0.05;
        NC = 11;
        VC = 0.4*tLR;
        tC = 1.0*tLR;
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # diagonal in space
                    HC[NCi,NCj] += VC+Deltaval+(Jval/2)*np.array([[0,1],[1,0]]); #spin mix
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HC[NCi,NCj] += -tC;

        # central region prime
        tCprime = tC;
        HCprime = np.zeros_like(HC);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # diagonal in space
                    HCprime[NCi,NCj] += VC+Deltaval; # no spin mix
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HCprime[NCi,NCj] += -tCprime;

        # print
        print("HC =");
        print_H_alpha(HC);
        print("HC - HCprime =");
        print_H_alpha(HC-HCprime);

        # bardeen.kernel syntax:
        # tinfty, tL, tR,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC, HCprime, matrix that defines alpha (non-observable) basis
        Evals, Mvals = bardeen.kernel_well_prime(tinfty,tLR, tLR, 
                                  Vinfty+Deltaval, VLR+Deltaval, Vinfty+Deltaval, VLR+Deltaval, Vinfty+Deltaval,
                                  Ninfty, NLR, NLR, HC, HCprime,                                                
                                  E_cutoff=np.eye(n_loc_dof)*Ecut+Deltaval,
                                  interval=interval,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR+Deltaval, VLR+Deltaval, NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR+Deltaval, VLR+Deltaval, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        if plot_alpha: # iter over initial and final states
            for alphai in range(len(alphas)):
                for betai in range(len(alphas)):
                    alpha, beta = alphas[alphai], alphas[betai];
                    # plot based on initial state
                    xvals = np.real(Evals[alphai])+2*tLR[alphai,alphai];
                    axes[alphai,betai].scatter(xvals, Tvals[betai,:,alphai], marker=mymarkers[0],color=mycolors[0]);
                    # % error
                    axright = axes[alphai,betai].twinx();
                    axes[alphai,betai].scatter(xvals, Tvals_bench[betai,:,alphai], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
                    axright.plot(xvals,100*abs((Tvals[betai,:,alphai]-Tvals_bench[betai,:,alphai])/Tvals_bench[betai,:,alphai]),color=accentcolors[1]);            
                    #format
                    if(betai==len(alphas)-1): axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
                    if(error_lims): axright.set_ylim(*error_lims);
                    axes[alphai,betai].set_ylabel("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");
                    axes[alphai, betai].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
                    axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
                    axes[-1,betai].set_xscale('log', subs = []);
                    fig.suptitle("$ J = "+str(Jval)+", \Delta = "+str(indvals[indvali])+", V_C = "+str(VC[0,0])+", N_C = "+str(NC)+", t_{vac} = "+str(tC[0,0])+", \Delta \\varepsilon = "+str(interval)+"$");

        else: # plot for fixed initial spin
            xvals = np.real(Evals[alpha_initial])+2*tLR[alpha_initial,alpha_initial];
            axes[indvali].scatter(xvals, Tvals[alpha_final,:,alpha_initial], marker=mymarkers[0],color=mycolors[0]);
            # % error
            axright = axes[indvali].twinx();
            axes[indvali].scatter(xvals, Tvals_bench[alpha_final,:,alpha_initial], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha_final,:,alpha_initial]-Tvals_bench[alpha_final,:,alpha_initial])/Tvals_bench[alpha_final,:,alpha_initial]),color=accentcolors[1]);            
            #format
            axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
            if(error_lims): axright.set_ylim(*error_lims);
            axes[indvali].set_title("$\Delta = "+str(indvals[indvali])+"$", x=0.4, y = 0.7, fontsize=myfontsize);
            axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
            axes[indvali].set_ylabel("$T("+alpha_strs[alpha_initial]+"\\rightarrow"+alpha_strs[alpha_final]+")$");
            axes[indvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
            fig.suptitle("$J = "+str(Jval)+", V_C = "+str(VC[0,0])+", N_C = "+str(NC)+", t_{vac} = "+str(tC[0,0])+", \Delta \\varepsilon = "+str(interval)+"$");

    # show
    plt.tight_layout();
    fname = "figs/rbmenez/nosup_inel/menez_Delta";
    if(not plot_alpha):
        if( (alpha_initial, alpha_final) == (0,0) ): fname +="_nsf.pdf";
        elif( (alpha_initial, alpha_final) == (0,1) ): fname +="_sf.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs Jval
# fixed nonzero Delta (energy cost of impurity spin-flip)
# compare with PRA paper, Fig 10
if False:
    
    # alpha -> beta
    alphas = [0,1];
    alpha_strs = ["\\uparrow","\downarrow"];

    # plotting
    plot_alpha = False;
    if(plot_alpha):
        indvals = np.array([-0.05]);
        nplots_x = len(alphas);
        nplots_y = len(alphas);
    else:
        indvals = np.array([-0.005,-0.05,-0.5]);
        nplots_x = 1
        nplots_y = len(indvals);
        alpha_initial, alpha_final = 0,1;
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);

    #### inelastic
    Deltaval = 0.02*np.array([[0,0],[0,1]]);  # <------- !!!!
        
    # iter over J
    for indvali in range(len(indvals)):

        # central region
        Jval = indvals[indvali];
        NC = 11;
        VC = 0.4*tLR;
        tC = 1.0*tLR;
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # diagonal in space
                    HC[NCi,NCj] += VC+Deltaval+(Jval/2)*np.array([[0,1],[1,0]]); #spin mix
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HC[NCi,NCj] += -tC;

        # central region prime
        tCprime = tC;
        HCprime = np.zeros_like(HC);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # diagonal in space
                    HCprime[NCi,NCj] += VC+Deltaval; # no spin mix
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HCprime[NCi,NCj] += -tCprime;

        # print
        print("HC =");
        print_H_alpha(HC);
        print("HC - HCprime =");
        print_H_alpha(HC-HCprime);

        # bardeen.kernel syntax:
        # tinfty, tL, tR,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC, HCprime, matrix that defines alpha (non-observable) basis
        Evals, Mvals = bardeen.kernel_well_prime(tinfty,tLR, tLR, 
                                  Vinfty+Deltaval, VLR+Deltaval, Vinfty+Deltaval, VLR+Deltaval, Vinfty+Deltaval,
                                  Ninfty, NLR, NLR, HC, HCprime,                                                
                                  E_cutoff=np.eye(n_loc_dof)*Ecut+Deltaval,
                                  interval=interval,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR+Deltaval, VLR+Deltaval, NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR+Deltaval, VLR+Deltaval, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        if plot_alpha: # iter over initial and final states
            for alphai in range(len(alphas)):
                for betai in range(len(alphas)):
                    alpha, beta = alphas[alphai], alphas[betai];
                    # plot based on initial state
                    xvals = np.real(Evals[alphai])+2*tLR[alphai,alphai];
                    axes[alphai,betai].scatter(xvals, Tvals[betai,:,alphai], marker=mymarkers[0],color=mycolors[0]);
                    # % error
                    axright = axes[alphai,betai].twinx();
                    axes[alphai,betai].scatter(xvals, Tvals_bench[betai,:,alphai], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
                    axright.plot(xvals,100*abs((Tvals[betai,:,alphai]-Tvals_bench[betai,:,alphai])/Tvals_bench[betai,:,alphai]),color=accentcolors[1]);            
                    #format
                    if(betai==len(alphas)-1): axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
                    if(error_lims): axright.set_ylim(*error_lims);
                    axes[alphai,betai].set_ylabel("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");
                    axes[alphai, betai].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
                    axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
                    axes[-1,betai].set_xscale('log', subs = []);
                    fig.suptitle("$ J = "+str(Jval)+", \Delta = "+str(Deltaval[-1,-1])+", V_C = "+str(VC[0,0])+", N_C = "+str(NC)+", t_{vac} = "+str(tC[0,0])+", \Delta \\varepsilon = "+str(interval)+"$");

        else: # plot for fixed initial spin
            xvals = np.real(Evals[alpha_initial])+2*tLR[alpha_initial,alpha_initial];
            axes[indvali].scatter(xvals, Tvals[alpha_final,:,alpha_initial], marker=mymarkers[0],color=mycolors[0]);
            # % error
            axright = axes[indvali].twinx();
            axes[indvali].scatter(xvals, Tvals_bench[alpha_final,:,alpha_initial], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha_final,:,alpha_initial]-Tvals_bench[alpha_final,:,alpha_initial])/Tvals_bench[alpha_final,:,alpha_initial]),color=accentcolors[1]);            
            #format
            axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
            if(error_lims): axright.set_ylim(*error_lims);
            axes[indvali].set_title("$J = "+str(Jval)+"$", x=0.4, y = 0.7, fontsize=myfontsize);
            axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
            axes[indvali].set_ylabel("$T("+alpha_strs[alpha_initial]+"\\rightarrow"+alpha_strs[alpha_final]+")$");
            axes[indvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
            fig.suptitle("$\Delta = "+str(Deltaval[-1,-1])+", V_C = "+str(VC[0,0])+", N_C = "+str(NC)+", t_{vac} = "+str(tC[0,0])+", \Delta \\varepsilon = "+str(interval)+"$");

    # show
    plt.tight_layout();
    fname = "figs/rbmenez/nosup_inel/menez_Delta";
    if(not plot_alpha):
        if( (alpha_initial, alpha_final) == (0,0) ): fname +="_nsf.pdf";
        elif( (alpha_initial, alpha_final) == (0,1) ): fname +="_sf.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

