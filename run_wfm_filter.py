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
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

def h_kondo(se,s2,J) -> np.ndarray:
    '''
    Kondo interaction between electron and spin s2
    '''
    if(se != 0.5 or s2 != 0.5): raise NotImplementedError;
    n_loc_dof = int((2*se+1)*(2*s2+1));
    h = np.zeros((n_loc_dof,n_loc_dof),dtype=complex);
    h[0,0] = 1;
    h[1,1] = -1;
    h[2,2] = -1;
    h[3,3] = 1;
    h[1,2] = 2;
    h[2,1] = 2;
    h *= J/4;
    return h;

def epsilon(sigma, m, alpha, delta, B, Vb) -> float:
    '''
    Channel potential energy, depending on
    - electron z spin s
    - molecule z spin m
    - channel a = left, right
    Physical params:
    - ferromagnetic splitting delat
    - Magnetic field applied to molecule B
    - bias applied to right leab Vb
    '''
    if(alpha=='L'): ai=0;
    elif(alpha=='R'): ai=1;
    elif(alpha=='C'): ai=2;
    else: raise ValueError;
    return (sigma+1/2)*delta*np.eye(3)[ai,0] + (m-1/2)*B + Vb*np.eye(3)[ai,1];

def h_alpha(se, s2, alpha, delta, B, Vb) -> np.ndarray:
    '''
    get hamiltonian for lead alpha = L, R
        Physical params:
    - ferromagnetic splitting delat
    - Magnetic field applied to molecule B
    - bias applied to right leab Vb
    '''
    if(se != 0.5 or s2 != 0.5): raise NotImplementedError;
    n_loc_dof = int((2*se+1)*(2*s2+1));
    h = np.zeros((n_loc_dof,n_loc_dof),dtype=complex);
    for me in [0,1]:
        for ms in [0,1]:
            loci = 2*me+ms;
            h[loci,loci] += epsilon(-(me-1/2),-(ms-1/2), alpha, delta, B, Vb);

    return h;

#################################################################
#### filter = suppress T_nf by making DeltaE < 0 and shifting
#### chem pot in RL

if True:

    # tight binding params
    tl = 1.0;
    Msites = 1;
    Jval = -0.5;
    Deltaval = 0.1;
    Vbval = Deltaval/2;
    Bvals = np.array([0.0,Vbval/2,Vbval]);
    for Bval in Bvals:

        # range of energies
        logElims = -4,0
        Evals = np.logspace(*logElims,myxvals, dtype = complex);

        # R and T matrices
        alphas = [1,2];
        alpha_strs = ["\\uparrow \\uparrow","\\uparrow \downarrow","\downarrow \\uparrow","\downarrow \downarrow"];
        n_loc_dof = len(alphas);
        Rvals = np.empty((n_loc_dof,n_loc_dof,len(Evals)), dtype = float);
        Tvals = np.empty((n_loc_dof,n_loc_dof,len(Evals)), dtype = float);

        # plotting
        nplots_x = len(alphas);
        nplots_y = len(alphas);
        fig, axes = plt.subplots(nrows = nplots_y, ncols = nplots_x, sharex = True);
        fig.set_size_inches(nplots_x*7/2,nplots_y*3/2);

        # 2nd qu'd operator for S dot s
        hSR = h_kondo(0.5,0.5,Jval)[alphas[0]:alphas[-1]+1,alphas[0]:alphas[-1]+1];
        hSR += h_alpha(0.5,0.5,'C',Deltaval, Bval, Vbval)[alphas[0]:alphas[-1]+1,alphas[0]:alphas[-1]+1];
        hLL = h_alpha(0.5,0.5,'L',Deltaval, Bval, Vbval)[alphas[0]:alphas[-1]+1,alphas[0]:alphas[-1]+1];
        hRL = h_alpha(0.5,0.5,'R',Deltaval, Bval, Vbval)[alphas[0]:alphas[-1]+1,alphas[0]:alphas[-1]+1];

        # package together hamiltonian blocks
        hblocks = [hLL];
        for _ in range(Msites): hblocks.append(np.copy(hSR));
        hblocks.append(hRL);
        hblocks = np.array(hblocks);
        tnn = [];
        for _ in range(len(hblocks)-1): tnn.append(-tl*np.eye(*np.shape(hSR)));
        tnn = np.array(tnn);
        tnnn = np.zeros_like(tnn)[:-1];

        # iter over initial states (alpha)
        for alphai in range(n_loc_dof):
            alpha = alphas[alphai];

            # source = up electron, down impurity
            source = np.zeros(n_loc_dof);
            source[alphai] = 1;

            E_shift = hblocks[0,alphai,alphai]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks: hb += -E_shift*np.eye(n_loc_dof);
            if(alpha==2):
                if(verbose): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

            # sweep over range of energies
            for Evali in range(len(Evals)):

                # energy
                Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
                Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

                # R and T
                # kernel returns R[beta], T[beta] at given E
                Rbeta, Tbeta = wfm.kernel(hblocks, tnn, tnnn, tl, Energy, source, all_debug = False);
                Rvals[alphai,:,Evali] = Rbeta;
                Tvals[alphai,:,Evali] = Tbeta;

        # plot results
        for alphai in range(n_loc_dof):
            for betai in range(n_loc_dof):
                alpha, beta = alphas[alphai], alphas[betai];
                axes[alphai,betai].plot(np.real(Evals), Tvals[alphai,betai], color=mycolors[0], marker=mymarkers[1], markevery=mymarkevery, linewidth = mylinewidth);
                axes[alphai,betai].plot(np.real(Evals), Rvals[alphai,betai], color=mycolors[1], marker=mymarkers[2], markevery=mymarkevery, linewidth = mylinewidth);
                axes[alphai,betai].set_title("$"+alpha_strs[alpha]+"\\rightarrow"+alpha_strs[beta]+"$");
                
                # format
                axes[-1,betai].set_xlabel('$K_i/t$',fontsize=myfontsize);
                axes[-1,betai].set_xscale('log', subs = []);
                
            #check R+T=1
            #axes[alphai,-1].plot(np.real(Evals), Tvals[alphai,0]+Tvals[alphai,1]+Rvals[alphai,0]+Rvals[alphai,1], color=accentcolors[1]);
            axes[alphai,-1].plot(np.real(Evals), np.sqrt(Tvals[alphai,0]*Tvals[alphai,1]), color=accentcolors[1]);

        # show
        plt.tight_layout();
        plt.show();
        print("We want spin-flip T (lower left blue) and no-flip R (lower right green) only nonzero");


