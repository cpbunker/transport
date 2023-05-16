'''
Christian Bunker
M^2QM at UF
November 2022

Spin independent scattering in a 1d monatomic chain
solved in time-dependent QM using bardeen theory method in transport/bardeen
then benchmarked to exact solution
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

# perturbation is cutting hopping
if True:
    E_cutoff = 2.0;
    
    # iter over vaccuum hopping
    tvacvals = [0.016772,0.000671];
    numplots = len(tvacvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);
    for tvaci in range(len(tvacvals)):

        # load Hsys
        fname = "transport/bardeen/models/Au_chain_"+str(tvacvals[tvaci])+".npy";
        print("Loading data from "+fname);
        Hsys = np.load(fname);
        Ntotal, n_loc_dof = np.shape(Hsys)[0], np.shape(Hsys)[-1];
        tbulk = -Hsys[0,1,0,0];
        print(">>> tbulk = ",tbulk," Ha"); assert False

        # visualize Hsys
        figH, axesH = plt.subplots(n_loc_dof, sharex = True);
        if(n_loc_dof == 1): axesH = [axesH];
        mid = len(Hsys)//2;
        jvals = np.linspace(-mid, -mid +len(Hsys)-1,len(Hsys), dtype=int);
        for alpha in range(n_loc_dof):
            axesH[alpha].plot(jvals, np.diagonal(Hsys[:,:,alpha,alpha]), label = "V", color=accentcolors[0]);
            axesH[alpha].plot(jvals[:-1], np.diagonal(Hsys[:,:,alpha,alpha], offset=1), label = "t", color=mycolors[0]);
        axesH[0].legend();
        axesH[0].set_title("$H_{sys} "+str(np.shape(Hsys))+"$");
        figH.show();

        # bardeen.kernel generates initial energies and corresponding matrix elements
        tcuti = mid;
        Evals, Mvals = bardeen.kernel(Hsys, tbulk, tcuti, tcuti,
                                      E_cutoff=E_cutoff, verbose=1);

        # bardeen.Ts_bardeen generates continuum limit transmission probs
        tL, tR = Hsys[0,1], Hsys[-2,-1];
        VL, VR = Hsys[0,0], Hsys[-1,-1];
        NL, NR = mid, mid;
        Tvals = bardeen.Ts_bardeen(Evals, Mvals, tL, tR, VL, VR, NL, NR, verbose=1);
        
        # benchmark
        Tvals_bench = bardeen.Ts_wfm(Hsys, Evals, tbulk, verbose=0);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):

            # plot
            xvals = (np.real(Evals[alpha])+2*tbulk)/(tbulk);
            axes[tvaci].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);
            axes[tvaci].set_xlim(min(xvals),E_cutoff);
            axes[tvaci].set_ylim(0.0,np.real(max(Tvals_bench[alpha,:,alpha])).round(4) );
            
            # % error
            axright = axes[tvaci].twinx();
            axes[tvaci].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.scatter(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),marker="_",color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        #axright.set_ylim(0,50);
        axes[tvaci].set_ylabel("$T (t_{vac} = "+str(tvacvals[tvaci])+")$",fontsize=myfontsize);
        axes[tvaci].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_{bulk})/t_{bulk}$",fontsize=myfontsize);
    axes[0].set_title("$t_{bulk} = "+"{:.4f}".format(tbulk)+"$");
    plt.tight_layout();
    plt.show();



