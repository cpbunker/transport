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

#################################################################
#### matrix elements are the right barrier being removed and Kondo term
#### being added to the central region

# tight binding params
n_loc_dof = 2; 
tLR = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tLR;
VLR = 0.0*tLR; # NB explicit LR symmetry
Vinfty = 0.5*tLR;
NLR = 200;
Ninfty = 20;

# cutoffs
Ecut = 0.4;
error_lims = (0,20);

###############################################
# the Kondo term is included in Hsys only, so can work in eigenbasis of Sz
# therefore Kondo shows up in H_sys - HL, causing nonzero overlap between
# initial, final states of different spin

# T vs Jval
if True:
    
    # alpha -> beta
    alphas = [0,1];
    alpha_strs = ["\\uparrow","\downarrow"];

    # plotting
    plot_alpha = True;
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
    
    # iter over J vals
    for indvali in range(len(indvals)):

        # central region
        Jval = indvals[indvali];
        tC = 1.0*tLR;
        VC = 0.4*tLR;
        NC = 11;
        my_kondo = VC + (Jval/2)*np.array([[0,1],[1,0]]);
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): 
                    HC[NCi,NCj] += my_kondo; # spin mix
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HC[NCi,NCj] += -tC;

        # central region prime
        tCprime = tC;
        HCprime = np.zeros_like(HC);
        kondo_replace = np.diagflat(np.diagonal(my_kondo));
        assert(bardeen.is_alpha_conserving(kondo_replace, n_loc_dof));
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): 
                    HCprime[NCi,NCj] += kondo_replace; # no spin mix
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HCprime[NCi,NCj] += -tC;

        # print
        print("HC =");
        print_H_alpha(HC);
        print("HC - HCprime =");
        print_H_alpha(HC-HCprime);

        # bardeen results for spin flip scattering
        # bardeen.kernel syntax:
        # tinfty, tL, tR,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well_prime(tinfty, tLR, tLR, 
                                  Vinfty, VLR, Vinfty, VLR, Vinfty,
                                  Ninfty, NLR, NLR, HC, HCprime,
                                  E_cutoff=Ecut*np.eye(n_loc_dof),
                                  verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        if plot_alpha:
            # iter over initial and final states
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
                    #axright.set_ylim(*error_lims);
                    axes[alphai,betai].set_title("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");
                    axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
                    axes[-1,betai].set_xscale('log', subs = []);
                    axes[alphai,0].set_ylabel("$T$");

        else:
            # plot based on initial state
            xvals = np.real(Evals[alpha_initial])+2*tLR[alpha_initial,alpha_initial];
            axes[indvali].scatter(xvals, Tvals[alpha_final,:,alpha_initial], marker=mymarkers[0],color=mycolors[0]);
            # % error
            axright = axes[indvali].twinx();
            axes[indvali].scatter(xvals, Tvals_bench[alpha_final,:,alpha_initial], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha_final,:,alpha_initial]-Tvals_bench[alpha_final,:,alpha_initial])/Tvals_bench[alpha_final,:,alpha_initial]),color=accentcolors[1]);            
            #format
            axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
            #axright.set_ylim(*error_lims);
            axes[indvali].set_title("$J = "+str(Jval)+"$", x=0.4, y = 0.7, fontsize=myfontsize);
            axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
            axes[indvali].set_ylabel("$T("+alpha_strs[alpha_initial]+"\\rightarrow"+alpha_strs[alpha_final]+")$");
            axes[indvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    # show
    plt.tight_layout();
    fig.suptitle("$N_C = "+str(NC)+"$");
    fname = "figs/rbmenez/nosup_el/menez_Jval";
    if(not plot_alpha):
        if( (alpha_initial, alpha_final) == (0,0) ): fname +="_nsf.pdf";
        elif( (alpha_initial, alpha_final) == (0,1) ): fname +="_sf.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs NJ
if False:
    
    # alpha -> beta
    alphas = [0,1];
    alpha_strs = ["\\uparrow","\downarrow"];

    # plotting
    plot_alpha = False;
    if(plot_alpha):
        indvals = np.array([1]);
        nplots_x = len(alphas);
        nplots_y = len(alphas);
    else:
        indvals = np.array([1,5,11]);
        nplots_x = 1
        nplots_y = len(indvals);
        alpha_initial, alpha_final = 0,1;
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);
    
    # iter over NJ
    for indvali in range(len(indvals)):

        # central region
        Jval = -0.05;
        tC = 1.0*tLR;
        VC = 0.4*tLR;
        NC = 11;
        NJ = indvals[indvali]; assert(NJ <= NC);
        my_kondo = VC + (Jval/2)*np.array([[0,1],[1,0]]);
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): 
                    HC[NCi,NCj] += VC;
                    if( abs(NCi - NC//2) <= NJ//2 ): # spin flip also
                        HC[NCi,NCj] += (Jval/2)*np.array([[0,1],[1,0]]);
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HC[NCi,NCj] += -tC;

        # central region prime
        tCprime = tC;
        HCprime = np.zeros_like(HC);
        kondo_replace = np.diagflat(np.diagonal(my_kondo));
        assert(bardeen.is_alpha_conserving(kondo_replace, n_loc_dof));
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): 
                    HCprime[NCi,NCj] += kondo_replace; # no spin flip
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HCprime[NCi,NCj] += -tC;

        # print
        print("HC =");
        print_H_alpha(HC);
        print("HC - HCprime =");
        print_H_alpha(HC-HCprime);

        # bardeen results for spin flip scattering
        # bardeen.kernel syntax:
        # tinfty, tL, tR,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well_prime(tinfty, tLR, tLR, 
                                  Vinfty, VLR, Vinfty, VLR, Vinfty,
                                  Ninfty, NLR, NLR, HC, HCprime,
                                  E_cutoff=Ecut*np.eye(n_loc_dof),
                                  verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        if plot_alpha:
            # iter over initial and final states
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
                    #axright.set_ylim(*error_lims);
                    axes[alphai,betai].set_title("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");
                    axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
                    axes[-1,betai].set_xscale('log', subs = []);
                    axes[alphai,0].set_ylabel("$T$");

        else:
            # plot based on initial state
            xvals = np.real(Evals[alpha_initial])+2*tLR[alpha_initial,alpha_initial];
            axes[indvali].scatter(xvals, Tvals[alpha_final,:,alpha_initial], marker=mymarkers[0],color=mycolors[0]);
            # % error
            axright = axes[indvali].twinx();
            axes[indvali].scatter(xvals, Tvals_bench[alpha_final,:,alpha_initial], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha_final,:,alpha_initial]-Tvals_bench[alpha_final,:,alpha_initial])/Tvals_bench[alpha_final,:,alpha_initial]),color=accentcolors[1]);            
            #format
            axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
            #axright.set_ylim(*error_lims);
            axes[indvali].set_title("$N_J = "+str(NJ)+"$", x=0.4, y = 0.7, fontsize=myfontsize);
            axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
            axes[indvali].set_ylabel("$T("+alpha_strs[alpha_initial]+"\\rightarrow"+alpha_strs[alpha_final]+")$");
            axes[indvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    # show
    plt.tight_layout();
    fig.suptitle("$N_C = "+str(NC)+",\, J = "+str(Jval)+"$");
    fname = "figs/rbmenez/nosup_el/menez_NJ";
    if(not plot_alpha):
        if( (alpha_initial, alpha_final) == (0,0) ): fname +="_nsf.pdf";
        elif( (alpha_initial, alpha_final) == (0,1) ): fname +="_sf.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();

# T vs tC (reduced-tC insulating region of width NC=3)
if False:
    
    # alpha -> beta
    alphas = [0,1];
    alpha_strs = ["\\uparrow","\downarrow"];

    # plotting
    plot_alpha = False;
    if(plot_alpha):
        indvals = np.array([1.0*tLR]);
        nplots_x = len(alphas);
        nplots_y = len(alphas);
    else:
        indvals = np.array([1.0*tLR,0.8*tLR,0.2*tLR]);
        nplots_x = 1
        nplots_y = len(indvals);
        alpha_initial, alpha_final = 0,1;
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);
    
    # iter over NJ
    for indvali in range(len(indvals)):

        # central region
        Jval = -0.05;
        tC = indvals[indvali];
        VC = 0.4*tLR;
        NC = 3;
        my_kondo = VC + (Jval/2)*np.array([[0,1],[1,0]]);
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof),dtype=complex);
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): # diagonal in space
                    if(NCi < NC//2): # left insulator
                        HC[NCi,NCj] += VLR;
                    elif(NCi == NC//2): # middle to keep LR symmetry !!!
                        HC[NCi,NCj] += my_kondo; #spin mix
                    elif(NCi > NC//2): # right insulator
                        HC[NCi,NCj] += VLR;
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HC[NCi,NCj] += -tC;

        # central region prime
        tCprime = tC;
        HCprime = np.zeros_like(HC);
        kondo_replace = np.diagflat(np.diagonal(my_kondo));
        assert(bardeen.is_alpha_conserving(kondo_replace, n_loc_dof));
        for NCi in range(NC):
            for NCj in range(NC):
                if(NCi == NCj): 
                    if(NCi < NC//2): # left insulator
                        HCprime[NCi,NCj] += VLR;
                    elif(NCi == NC//2): # middle to keep LR symmetry !!!
                        HCprime[NCi,NCj] += kondo_replace; # no spin mix
                    elif(NCi > NC//2): # right insulator
                        HCprime[NCi,NCj] += VLR;
                elif(abs(NCi -NCj) == 1): # nn hopping
                    HCprime[NCi,NCj] += -tC;

        # print
        print("HC =");
        print_H_alpha(HC);
        print("HC - HCprime =");
        print_H_alpha(HC-HCprime);

        # bardeen results for spin flip scattering
        # bardeen.kernel syntax:
        # tinfty, tL, tR,
        # Vinfty, VL, VLprime, VR, VRprime,
        # Ninfty, NL, NR, HC,HCprime,
        Evals, Mvals = bardeen.kernel_well_prime(tinfty, tLR, tLR, 
                                  Vinfty, VLR, Vinfty, VLR, Vinfty,
                                  Ninfty, NLR, NLR, HC, HCprime,
                                  E_cutoff=Ecut*np.eye(n_loc_dof),
                                  verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR,verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm_well(tLR, tLR, VLR, VLR, HC, Evals, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        if plot_alpha:
            # iter over initial and final states
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
                    #axright.set_ylim(*error_lims);
                    axes[alphai,betai].set_title("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");
                    axes[-1,betai].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
                    axes[-1,betai].set_xscale('log', subs = []);
                    axes[alphai,0].set_ylabel("$T$");

        else:
            # plot based on initial state
            xvals = np.real(Evals[alpha_initial])+2*tLR[alpha_initial,alpha_initial];
            axes[indvali].scatter(xvals, Tvals[alpha_final,:,alpha_initial], marker=mymarkers[0],color=mycolors[0]);
            # % error
            axright = axes[indvali].twinx();
            axes[indvali].scatter(xvals, Tvals_bench[alpha_final,:,alpha_initial], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha_final,:,alpha_initial]-Tvals_bench[alpha_final,:,alpha_initial])/Tvals_bench[alpha_final,:,alpha_initial]),color=accentcolors[1]);            
            #format
            axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
            #axright.set_ylim(*error_lims);
            axes[indvali].set_title("$t_{vac} = "+str(tC[0,0])+"$", x=0.4, y = 0.7, fontsize=myfontsize);
            axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
            axes[indvali].set_ylabel("$T("+alpha_strs[alpha_initial]+"\\rightarrow"+alpha_strs[alpha_final]+")$");
            axes[indvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    # show
    plt.tight_layout();
    fig.suptitle("$N_C = "+str(NC)+",\, J = "+str(Jval)+"$");
    fname = "figs/rbmenez/nosup_el/menez_tC";
    if(not plot_alpha):
        if( (alpha_initial, alpha_final) == (0,0) ): fname +="_nsf.pdf";
        elif( (alpha_initial, alpha_final) == (0,1) ): fname +="_sf.pdf";
    if(save_figs): plt.savefig(fname); print("Saving data as",fname);
    else: plt.show();


