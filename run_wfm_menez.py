'''
Christian Bunker
M^2QM at UF
November 2022

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper)
benchmarked to exact solution 
solved in time-independent QM using wfm method in transport/wfm

'''

from transport import wfm, fci_mod

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;

# fig standardizing
myxvals = 49;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

def h_kondo(J,s2):
    '''
    Kondo interaction between spin 1/2 and spin s2
    '''
    n_loc_dof = int(2*(2*s2+1));
    h = np.zeros((n_loc_dof,n_loc_dof),dtype=complex);
    if(s2 == 0.5):
        h[0,0] = 1;
        h[1,1] = -1;
        h[2,2] = -1;
        h[3,3] = 1;
        h[1,2] = 2;
        h[2,1] = 2;
        h *= J/4;
    else: raise NotImplementedError;
    return h;

#################################################################
#### all possible T_{\alpha -> \beta}

if True:

    # tight binding params
    tl = 1.0;
    Msites = 1;
    Jval = -0.5;

    # range of energies
    logElims = -4,0
    Evals = np.logspace(*logElims,myxvals, dtype = complex);

    # R and T matrices
    alphas = [1,2];
    alpha_strs = ["\\uparrow \\uparrow","\\uparrow \downarrow","\downarrow \\uparrow","\downarrow \downarrow"];
    hspacesize = len(alphas);
    Rvals = np.empty((hspacesize,hspacesize,len(Evals)), dtype = float);
    Tvals = np.empty((hspacesize,hspacesize,len(Evals)), dtype = float);

    # plotting
    nplots_x = len(alphas);
    nplots_y = len(alphas);
    fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
    fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);

    # 2nd qu'd operator for S dot s
    hSR = h_kondo(Jval,0.5)[alphas[0]:alphas[-1]+1,alphas[0]:alphas[-1]+1];
    hLL = np.zeros_like(hSR);
    hRL = np.zeros_like(hSR);

    # package together hamiltonian blocks
    hblocks = [hLL];
    for _ in range(Msites): hblocks.append(np.copy(hSR));
    hblocks.append(hRL);
    hblocks = np.array(hblocks);
    tnn = [];
    for _ in range(len(hblocks)-1): tnn.append(-tl*np.eye(*np.shape(hSR)));
    tnn = np.array(tnn);
    tnnn = np.zeros_like(tnn)[:-1];
    if(verbose): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

    # iter over initial states (alpha)
    for alphai in range(len(alphas)):
        alpha = alphas[alphai];

        # source = up electron, down impurity
        source = np.zeros(hspacesize);
        source[alphai] = 1;

        # sweep over range of energies
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc

            # R and T
            # kernel returns R[beta], T[beta] at given E
            Rbeta, Tbeta = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False);
            Rvals[alphai,:,Evali] = Rbeta;
            Tvals[alphai,:,Evali] = Tbeta;

    # plot results
    for alphai in range(len(alphas)):
        for betai in range(len(alphas)):
            alpha, beta = alphas[alphai], alphas[betai];
            axes[alphai,betai].plot(np.real(Evals), Tvals[alphai,betai], color=mycolors[0], marker=mymarkers[1], markevery=mymarkevery, linewidth = mylinewidth);
            axes[alphai,betai].plot(np.real(Evals), Rvals[alphai,betai], color=mycolors[1], marker=mymarkers[2], markevery=mymarkevery, linewidth = mylinewidth);
            axes[alphai,betai].set_title("$T("+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+")$");

            # format
            axes[-1,betai].set_xlabel('$K_i/t$',fontsize=myfontsize);
            axes[-1,betai].set_xscale('log', subs = []);

        #check R+T=1
        axes[alphai,-1].plot(np.real(Evals), Tvals[alphai,0]+Tvals[alphai,1]+Rvals[alphai,0]+Rvals[alphai,1], color=accentcolors[1]);

    # show
    plt.tight_layout();
    plt.show();   


