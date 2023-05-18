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
#### use bardeen.kernel, where perturb is central-right lead hopping

# tight binding params
n_loc_dof = 1; 
tLR = 1.0*np.eye(n_loc_dof);
tinfty = 1.0*tLR;
VLR = 0.0*tLR;
Vinfty = 0.5*tLR;
Ninfty = 20;
error_lims = (0,20);

# T vs NC
if True:

    tCvals = np.array([1.0*tLR,0.8*tLR,0.2*tLR]);
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
        VC = 0.4*tLR;
        HC = np.zeros((NC,NC,n_loc_dof,n_loc_dof));
        for j in range(NC):
            if(j==NC//2): # preserves LR symmetry
                HC[j,j] += VC;
        for j in range(NC-1):
            HC[j,j+1] += -tCvals[tCvali];
            HC[j+1,j] += -tCvals[tCvali];
        print("HC =");
        print_H_j(HC);

        # entire system Hamiltonian. syntax:
        # tinfty, tL, tR, 
        # Vinfty, VL, VR, 
        # Ninfty, NL, NR, HC
        Hsys, offset = bardeen.Hsysmat(tinfty, tLR, tLR,
                               Vinfty, VLR, VLR,
                               Ninfty, NLR, NLR, HC);
        
        # bardeen.kernel gets matrix elements by cutting the hopping
        jvals = np.array(range(len(Hsys))) + offset;
        cuti = len(jvals)//2;
        print(">>>", len(jvals), cuti, cuti+1);
        Evals, Mvals = bardeen.kernel(Hsys, tLR[0,0], cuti, cuti+1, 
                                      E_cutoff=VC[0,0],interval=1e-3,verbose=1);
        Tvals = bardeen.Ts_bardeen(Evals, Mvals,
                                   tLR, tLR, VLR, VLR, NLR, NLR, verbose=1);

        # benchmark
        Tvals_bench = bardeen.Ts_wfm(Hsys[Ninfty:-Ninfty,Ninfty:-Ninfty], Evals, tLR[0,0], verbose=0);
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
            axright.scatter(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),marker="_",color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        #axright.set_ylim(*error_lims);
        axes[tCvali].set_ylabel('$T$',fontsize=myfontsize);
        axes[tCvali].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
        axes[tCvali].set_title("$t_C = "+str(tCvals[tCvali][0,0])+'$', x=0.4, y = 0.7, fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel('$(\\varepsilon_m + 2t_L)/t_L \,\,|\,\, V_C = '+str(VC[0,0])+'$',fontsize=myfontsize);
    plt.tight_layout();
    fname = "figs/bard_chain/bard_chain.pdf";
    plt.savefig(fname); print("Saving data to "+fname);
    #plt.show();

