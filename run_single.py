'''
Christian Bunker
M^2QM at UF
November 2022

Scattering of a single electron from a spin-1/2 impurity w/ Kondo-like
interaction strength J (e.g. menezes paper) solved in time-independent QM
using Green's function method in transport/wfm
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

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

#################################################################
#### T \alpha -> \beta

if True:

    # tb params
    Msites = 1;
    Jval = -0.5;

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

    # iter over initial states (alpha)
    for alpha in alphas:

        # source = up electron, down impurity
        source = np.zeros(hspacesize);
        source[alpha] = 1;

        # 2nd qu'd operator for S dot s
        h1e = np.zeros((hspacesize,hspacesize))
        g2e = wfm.utils.h_kondo_2e(Jval, 0.5); # J, spin
        states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
        hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

        # leads
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
        if(verbose and alpha == 0): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

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


#################################################################
#### replication of continuum solution

if False:

    # inelastic ?
    Delta = 0.001; # inelastic splitting
    inelastic = False
    if Delta > 0.0: inelastic = True;
    num_plots = 4;
    if inelastic: num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);
    
    # non contact interaction
    Msites = 1; 

    # iter over effective J
    Jvals = np.array([-0.005,-0.05,-0.5,-5.0,]);
    for Jvali in range(len(Jvals)):
        Jval = Jvals[Jvali];
        
        # 2nd qu'd operator for S dot s
        h1e = np.zeros((4,4))
        g2e = wfm.utils.h_kondo_2e(Jval, 0.5); # J, spin
        states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
        hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

        # zeeman splitting
        hzeeman = np.array([[0, 0, 0, 0],
                        [0,Delta, 0, 0],
                        [0, 0, 0, 0], # spin flip gains PE delta
                        [0, 0, 0, Delta]]);
        hSR += hzeeman;

        # truncate to coupled channels
        hSR = hSR[1:3,1:3];
        hzeeman = hzeeman[1:3,1:3];

        # leads
        hLL = np.copy(hzeeman);
        hRL = np.copy(hzeeman)

        # source = up electron, down impurity
        sourcei, flipi = 1,0
        source = np.zeros(np.shape(hSR)[0]);
        source[sourcei] = 1;

        # package together hamiltonian blocks
        hblocks = [hLL];
        for _ in range(Msites): hblocks.append(np.copy(hSR));
        hblocks.append(hRL);
        hblocks = np.array(hblocks);

        # hopping
        tnn = [];
        for _ in range(len(hblocks)-1): tnn.append(-tl*np.eye(*np.shape(hSR)));
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];
        if(verbose and Jvali == 0): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

        # sweep over range of energies
        # def range
        logElims = -6,0
        Evals = np.logspace(*logElims,myxvals);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float); 
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            if(Evali < 1): # verbose
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False, verbose = verbose);
            else: # not verbose
                 Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot tight binding results
        ax0, ax1, ax2, ax3 = 0,1,2,3;
        if inelastic: ax0, ax2 = 0,1
        axes[ax0].plot(Evals,Tvals[:,flipi], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        axes[ax2].plot(Evals,Tvals[:,sourcei], color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[ax2].plot(Evals, totals, color="red", label = "total ");
        
        # continuum results
        if inelastic:
            #axes[ax0].axvline(0.025, color = "gray");
            axes[ax0].plot(Evals, menez_Tf, color = mycolors[Jvali],linestyle = "dashed", marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth); 
            axes[ax2].plot(Evals, menez_Tnf, color = mycolors[Jvali],linestyle = "dashed", marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax0].set_ylim(0,0.4)
            axes[ax0].set_ylabel('$T_{f}$', fontsize = myfontsize );
            axes[ax2].set_ylim(0,1);
            axes[ax2].set_ylabel('$T_{nf}$', fontsize = myfontsize );
            
        # differences
        if not inelastic:
            axes[ax1].plot(Evals,abs(Tvals[:,flipi]-menez_Tf)/menez_Tf,color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax3].plot(Evals,abs(Tvals[:,sourcei]-menez_Tnf)/menez_Tnf,color = mycolors[Jvali], marker = mymarkers[Jvali], markevery = mymarkevery, linewidth = mylinewidth);
            axes[ax0].set_ylim(0,0.4)
            axes[ax0].set_ylabel('$T_{f}$', fontsize = myfontsize );
            axes[ax1].set_ylim(0,0.1);
            axes[ax1].set_ylabel('$|T_{f}-T_{f,c}|/T_{f,c}$', fontsize = myfontsize );
            axes[ax2].set_ylim(0,1);
            axes[ax2].set_ylabel('$T_{nf}$', fontsize = myfontsize );
            axes[ax3].set_ylim(0,0.1);
            axes[ax3].set_ylabel('|$T_{nf}-T_{nf,c}|/T_{nf,c}$', fontsize = myfontsize );
    
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize); 
    plt.tight_layout();
    plt.show();


#################################################################
#### physical origin

if False:
    num_plots = 2;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # tight binding parameters
    th = 1.0;
    U1 = 0.0;
    U2 = 100.0;

    # iter over effective J by changing epsilon
    Jvals = np.array([0.1,0.5,1.0,5.0]);
    epsvals = (U1-U2)/2 + np.sqrt(U1*U2 + np.power((U1-U2)/2,2) - 2*th*th*(U1+U2)/Jvals);
    for epsi in range(len(epsvals)):
        epsilon = epsvals[epsi];
        Jval = Jvals[epsi];
        print("Jval = ",Jval);
        print("U1 - epsilon = ",U1 - epsvals[epsi]);
        print("U2+epsilon = ",U2+epsvals[epsi]);

        # SR physics: site 1 is in chain, site 2 is imp with large U
        hSR = np.array([[U1,-th,th,0], # up down, -
                        [-th,epsilon, 0,-th], # up, down (source)
                        [th, 0, epsilon, th], # down, up (flip)
                        [0,-th,th,U2+2*epsilon]]); # -, up down
        hSR += (Jvals[epsi]/4)*np.eye(4);
        
        # source = up electron, down impurity
        source = np.zeros(np.shape(hSR)[0]);
        sourcei, flipi = 1,2;
        source[sourcei] = 1;

        # lead physics
        hLL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);
        hRL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);

        # package together hamiltonian blocks
        hblocks = np.array([hLL, hSR, hRL]);
        for hb in hblocks: hb += -epsilon*np.eye(len(source));  # constant shift so source is at zero
        tnn_mat = -tl*np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]);
        tnn = np.array([np.copy(tnn_mat), np.copy(tnn_mat)]);
        tnnn = np.zeros_like(tnn)[:-1];
        #if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn, "\ntnnn:", tnnn)

        if True: # do the downfolding explicitly
            matA = np.array([[0, 0],[0,0]]);
            matB = np.array([[-th,-th],[th,th]]);
            matC = np.array([[-th,th],[-th,th]]);
            matD = np.array([[U1-epsilon, 0],[0,U2+epsilon]]);
            mat_downfolded = matA - np.dot(matB, np.dot(np.linalg.inv(matD), matC))  
            print("Downfolded J = ",2*abs(mat_downfolded[0,0]) );
        
        # sweep over range of energies
        # def range
        logElims = -3,0
        Evals = np.logspace(*logElims,myxvals);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
        jprimevals = Jval/(4*tl*kavals);
        menez_Tf = jprimevals*jprimevals/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        menez_Tnf = (1+jprimevals*jprimevals/4)/(1+(5/2)*jprimevals*jprimevals+(9/16)*np.power(jprimevals,4));
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            Rdum, Tdum =wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot Tvals vs E
        axes[0].plot(Evals,Tvals[:,flipi], color = mycolors[epsi], marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        axes[1].plot(Evals,Tvals[:,sourcei], color = mycolors[epsi], marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        totals = Tvals[:,sourcei] + Tvals[:,flipi] + Rvals[:,sourcei] + Rvals[:,flipi];
        #axes[1].plot(Evals, totals, color="red");

        # menezes prediction in the continuous case
        axes[0].plot(Evals, menez_Tf, color = mycolors[epsi],linestyle = "dashed", marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth); 
        axes[1].plot(Evals, menez_Tnf, color = mycolors[epsi],linestyle = "dashed", marker = mymarkers[epsi], markevery = mymarkevery, linewidth = mylinewidth);
        axes[0].set_ylim(0,0.4)
        axes[0].set_ylabel('$T_{f}$', fontsize = myfontsize );
        axes[1].set_ylim(0,1);
        axes[1].set_ylabel('$T_{nf}$', fontsize = myfontsize );

    # format
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.show();






