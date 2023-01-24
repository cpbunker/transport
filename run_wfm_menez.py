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
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mymarkers = ["o","^","s","d","X","P","*"];
mymarkevery = 50;
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# tight binding params
tl = 1.0;
Msites = 1;
Jval = -0.5;


def h_kondo_2e(J,s2):
    '''
    Kondo interaction between spin 1/2 and spin s2, 2nd-quantized form
    '''

    # m2 states
    ms = [];
    m2 = s2;
    while(m2 >= -s2):
        ms.append(m2);
        m2 -= 1;

    assert( len(ms) == 2*s2+1);
    Nstates = 2 + len(ms);
    h = np.zeros((Nstates,Nstates,Nstates,Nstates));

    if(s2 == 0.5):

        # S \pm parts
        h[0,1,3,2] = 2;
        h[3,2,0,1] = 2;
        h[1,0,2,3] = 2;
        h[2,3,1,0] = 2;

        # Sz parts
        h[0,0,2,2] = 1;
        h[2,2,0,0] = 1;
        h[0,0,3,3] = -1;
        h[3,3,0,0] = -1;
        h[1,1,2,2] = -1;
        h[2,2,1,1] = -1;
        h[1,1,3,3] = 1;
        h[3,3,1,1] = 1;

        # scale with J
        h = (J/4.0)*h;

    elif(s2 == 1.0):

        # S \pm parts
        h[2,3,1,0] = 1/np.sqrt(2);
        h[1,0,2,3] = 1/np.sqrt(2);
        h[3,2,0,1] = 1/np.sqrt(2);
        h[0,1,3,2] = 1/np.sqrt(2);
        h[3,4,1,0] = 1/np.sqrt(2);
        h[1,0,3,4] = 1/np.sqrt(2);
        h[4,3,0,1] = 1/np.sqrt(2);
        h[0,1,4,3] = 1/np.sqrt(2);

        # Sz parts
        h[2,2,0,0] = 1/2;
        h[0,0,2,2] = 1/2;
        h[2,2,1,1] = -1/2;
        h[1,1,2,2] = -1/2;
        h[4,4,0,0] = -1/2;
        h[0,0,4,4] = -1/2;
        h[4,4,1,1] = 1/2;
        h[1,1,4,4] = 1/2;

        # scale with J
        h = J*h;

    else: raise ValueError("Unsupported s2");

    return h;

#################################################################
#### all possible T_{\alpha -> \beta}

if True:

    # range of energies
    logElims = -4,0
    Evals = np.logspace(*logElims,myxvals, dtype = complex);

    # R and T matrices
    alphas = [0,1,2,3];
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
    h1e = np.zeros((hspacesize,hspacesize))
    g2e = h_kondo_2e(Jval, 0.5); # J, spin
    states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
    hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form
    hLL = np.zeros_like(hSR);
    hRL = np.zeros_like(hSR);
    #print(hSR)
    #print(wfm.h_kondo(Jval,0.5));
    #assert False

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
    for alpha in alphas:

        # source = up electron, down impurity
        source = np.zeros(hspacesize);
        source[alpha] = 1;

        # sweep over range of energies
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            # R and T
            # kernel returns R[beta], T[beta] at given E
            Rbeta, Tbeta = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False);
            Rvals[alpha,:,Evali] = Rbeta;
            Tvals[alpha,:,Evali] = Tbeta;

    # plot results
    for alpha in alphas:
        for beta in alphas:
            axes[alpha,beta].plot(np.real(Evals), Tvals[alpha,beta]);
            axes[alpha,beta].set_title("$"+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+"$");
    # format and show
    axes[-1,-1].set_xscale('log', subs = []);
    plt.tight_layout();
    plt.show();   


