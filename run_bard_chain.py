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
    tcuti = 20;
    E_cutoff = 1.0;
    
    # iter over vaccuum hopping
    tvacvals = [1.0];
    numplots = len(tvacvals);
    fig, axes = plt.subplots(numplots, sharex = True);
    if numplots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*numplots/2);
    for tvaci in range(len(tvacvals)):

        # load Hsys
        fname = "transport/bardeen/models/Au_chain.npy";
        print("Loading data from "+fname);
        Hsys = np.load(fname);
        print(Hsys[:,:,0,0]);
        n_loc_dof = np.shape(Hsys)[-1];
        tbulk = Hsys[0,1,0,0];

        # bardeen.kernel generate initial energies and corresponding matrix elements
        Evals, Mvals = bardeen.kernel(Hsys, tbulk, tcuti, tcuti,
                                      E_cutoff=E_cutoff, verbose=10);
        #Tvals = bardeen.Ts_bardeen(Evals, Mvals, verbose=1);
        
        # benchmark
        Tvals_bench = bardeen.Ts_wfm(Hsys, Evals, tbulk, verbose=0);
        Tvals = np.copy(Tvals_bench);
        print("Output shapes:");
        for arr in [Evals, Tvals, Tvals_bench]: print(np.shape(arr));

        # only one loc dof, and transmission is diagonal
        for alpha in range(n_loc_dof):

            # sort
            sortis = np.argsort(Evals[alpha]);
            Evals[alpha] = Evals[alpha][sortis];
            Tvals[alpha,:,alpha] = Tvals[alpha,:,alpha][sortis];
            Tvals_bench[alpha,:,alpha] = Tvals_bench[alpha,:,alpha][sortis];

            # plot
            xvals = (np.real(Evals[alpha])+2*tbulk)/(2*tbulk);
            print(xvals)
            axes[tvaci].scatter(xvals, Tvals[alpha,:,alpha], marker=mymarkers[0], color=mycolors[0]);
            axes[tvaci].set_xlim(min(xvals),E_cutoff);
            for ax in axes: ax.set_ylim(0.0,np.real(max(Tvals[alpha,:,alpha])).round(4) );
            
            # % error
            axright = axes[tvaci].twinx();
            axes[tvaci].scatter(xvals, Tvals_bench[alpha,:,alpha], marker=mymarkers[1], color=accentcolors[0], linewidth=mylinewidth);
            axright.plot(xvals,100*abs((Tvals[alpha,:,alpha]-Tvals_bench[alpha,:,alpha])/Tvals_bench[alpha,:,alpha]),color=accentcolors[1]); 

        # format
        axright.set_ylabel("$\%$ error",fontsize=myfontsize,color=accentcolors[1]);
        axright.set_ylim(0,50);
        axes[tvaci].set_ylabel("$T (t_{vac} = "+str(tvacvals[tvaci])+")$",fontsize=myfontsize);

    # format and show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlabel("$(\\varepsilon_m + 2t_{bulk})/t_{bulk}$",fontsize=myfontsize);
    plt.title("$t_{bulk} = "+"{:.2f}".format(tbulk)+", t_{cut} = "+"{:.2f}".format(Hsys[tcuti, tcuti+1,0,0])+"$");
    plt.tight_layout();
    plt.show();



