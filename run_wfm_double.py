'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 2:
Scattering of single electron off of two localized spin-1/2 impurities
Following cicc, imp spins are confined to single sites, separated by x0
    imp spins can flip
    e-imp interactions treated by (effective) J Se dot Si
    look for resonances in transmission as function of kx0 for fixed E, k
    ie as impurities are pulled further away from each other
    since this is discrete, separate by x0 = N0 a lattice spacings
'''

from transport import wfm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import sys

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
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# constructing the hamiltonian
def reduced_ham(params, S) -> np.ndarray:
    D1, D2, J12, JK1, JK2 = params;
    assert(D1 == D2);
    h = np.array([[S*S*D1+(S-1)*(S-1)*D2+S*(S-1)*J12+(JK1/2)*S+(JK2/2)*(S-1), S*J12, np.sqrt(2*S)*(JK2/2) ], # up, s, s-1
                    [S*J12, (S-1)*(S-1)*D1+S*S*D2+S*(S-1)*J12+(JK1/2)*S + (JK2/2)*(S-1), np.sqrt(2*S)*(JK1/2) ], # up, s-1, s
                    [np.sqrt(2*S)*(JK2/2), np.sqrt(2*S)*(JK1/2),S*S*D1+S*S*D2+S*S*J12+(-JK1/2)*S +(-JK2/2)*S]], # down, s, s
                   dtype = complex);

    return h;

def entangle(H,bi,bj) -> np.ndarray:
    '''
    Perform a change of basis on a matrix such that basis vectors bi, bj become entangled (unentangled)
    in new ham, index bi -> + entangled state, bj -> - entangled state
    '''
    assert(bi < bj);
    assert(bj < max(np.shape(H)));

    # rotation matrix
    R = np.zeros_like(H);
    for i in range(np.shape(H)[0]):
        for j in range(np.shape(H)[1]):
            if( i != bi and i != bj):
                if(i == j):
                    R[i,j] = 1; # identity
            elif( i == bi and j == bi):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j == bj):
                R[i,j] = -1/np.sqrt(2);
            elif( i == bi and j == bj):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j== bi):
                R[i,j] = 1/np.sqrt(2);

    return np.matmul(np.matmul(R.T,H),R);

##################################################################################
#### effects of spatial separation

if True: # check similarity to menezes prediction at diff N
    num_plots = 3;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # iter over spatial separation
    tl = 1.0;
    JK = -0.05;
    Dval = 0.0;
    J12 = 0.0;
    n_loc_dof = 3;
    source = np.zeros(n_loc_dof);
    pair = (0,1); # pair[0] is the + state after entanglement
    sourcei = 2;
    source[sourcei] = 1;
    myspinS = 0.5;
    Nvals = [2,5,20,50];
    for Nvali in range(len(Nvals)):
        Nval = Nvals[Nvali];

        # construct hblocks
        hblocks, tnn = [], [];
        impis = [1,Nval];
        for j in range(2+Nval): # LL, imp 1... imp 2, RL
            # define all physical params
            JK1, JK2 = 0, 0;
            if(j == impis[0]): JK1 = JK;
            elif(j == impis[1]): JK2 = JK;
            params = Dval, Dval, J12, JK1, JK2;
            # construct h_SR (determinant basis)
            hSR = reduced_ham(params,myspinS);           
            # transform to eigenbasis
            hSR_diag = entangle(hSR, 0,1);
            hblocks.append(np.copy(hSR_diag));
            tnn.append(-tl*np.eye(n_loc_dof));
            if(verbose > 3 ):
                print("\nJK1, JK2 = ",JK1, JK2);
                print(" - ham:\n", hSR);
                print(" - transformed ham:\n", np.real(hSR_diag));

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(n_loc_dof);
        if(verbose > 3 ): print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

        # hopping
        tnn = np.array(tnn)[:-1];
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # sweep over range of energies
        # def range
        logElims = -6, -2;
        Evals = np.logspace(*logElims,myxvals);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
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
        axes[0].plot(Evals,Tvals[:,sourcei], color = mycolors[Nvali], marker = mymarkers[1+Nvali], markevery = mymarkevery, linewidth = mylinewidth);
        axes[1].plot(Evals,Tvals[:,pair[0]], color = mycolors[Nvali], marker = mymarkers[1+Nvali], markevery = mymarkevery, linewidth = mylinewidth);
        axes[2].plot(Evals,Tvals[:,pair[1]], color = mycolors[Nvali], marker = mymarkers[1+Nvali], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[2].plot(Evals, totals, color="red", label = "total ");

    #### end loop over Nvals
        
    # format
    axes[0].set_ylim(-0.05*1.0,1.0+0.05*1.0)
    axes[0].set_ylabel('$T_{i}$', fontsize = myfontsize );
    axes[1].set_ylim(-0.05*0.4,0.4);
    axes[1].set_ylabel('$T_{+}$', fontsize = myfontsize );
    #axes[2].set_ylim(0,1.0);
    axes[2].set_ylabel('$T_{-}$', fontsize = myfontsize );
    axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
    
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    #for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/double/Nlimit.pdf');
    #plt.show();

if False: # compare T- vs N to see how T- is suppressed at small N
    num_plots = 1
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over energy
    Jval = 0.1;
    Evals = [10**(-4),10**(-3),10**(-2),10**(-1)];
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

        # sweep over N
        Nvals = np.linspace(2,21,20,dtype = int);
        Rvals = np.empty((len(Nvals),len(source)), dtype = float);
        Tvals = np.empty((len(Nvals),len(source)), dtype = float);
        for Nvali in range(len(Nvals)):
        
            # location of impurities
            N0 = Nvals[Nvali] - 1;
            print(">>> N0 = ",N0);

            # construct hams
            # since t=tl everywhere, can use h_cicc_eff to get LL, RL blocks directly
            i1, i2 = [1], [1+N0];
            hblocks, tnn = wfm.utils.h_cicc_eff(Jval, tl, i1, i2, pair);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Nvali] = Rdum;
            Tvals[Nvali] = Tdum;

        # plot T_- vs N
        axes[0].plot(Nvals,Tvals[:,pair[1]], color = mycolors[Evali], marker = mymarkers[Evali], markevery = 5, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[1].plot(Nvals, totals, color="red");

    # format
    axes[-1].set_xlabel('$N$',fontsize = myfontsize);
    axes[0].set_ylabel('$T_{-}$', fontsize = myfontsize );
    plt.tight_layout();
    plt.savefig('figs/Nlimit2.pdf');
    plt.show();

##################################################################################
#### effects of J

if True: # compare T+ vs E at different J
    num_plots = 3;
    fig, axes = plt.subplots(num_plots, sharex = True);
    if num_plots == 1: axes = [axes];
    fig.set_size_inches(7/2,3*num_plots/2);

    # sweep over J
    Nval = 2;
    tl = 1.0;
    Dval = 0.0;
    J12 = 0.0;
    n_loc_dof = 3;
    source = np.zeros(n_loc_dof);
    pair = (0,1); # pair[0] is the + state after entanglement
    sourcei = 2;
    source[sourcei] = 1;
    myspinS = 0.5;
    Jvals = np.array([-0.005,-0.01,-0.05,-0.1]);
    for Jvali in range(len(Jvals)):
        JK = Jvals[Jvali];

        # construct hblocks
        hblocks, tnn = [], [];
        impis = [1,Nval];
        for j in range(2+Nval): # LL, imp 1... imp 2, RL
            # define all physical params
            JK1, JK2 = 0, 0;
            if(j == impis[0]): JK1 = JK;
            elif(j == impis[1]): JK2 = JK;
            params = Dval, Dval, J12, JK1, JK2;
            # construct h_SR (determinant basis)
            hSR = reduced_ham(params,myspinS);           
            # transform to eigenbasis
            hSR_diag = entangle(hSR, 0,1);
            hblocks.append(np.copy(hSR_diag));
            tnn.append(-tl*np.eye(n_loc_dof));
            if(verbose > 3 ):
                print("\nJK1, JK2 = ",JK1, JK2);
                print(" - ham:\n", hSR);
                print(" - transformed ham:\n", np.real(hSR_diag));

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(n_loc_dof);
        if(verbose > 3 ): print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

        # hopping
        tnn = np.array(tnn)[:-1];
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
       

        # sweep over energy
        logElims = -6,-2;
        Evals = np.logspace(*logElims,myxvals);
        kavals = np.arccos((Evals-2*tl)/(-2*tl));
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot T_+ vs E
        axes[0].plot(Evals,Tvals[:,sourcei], color = mycolors[Jvali], marker = mymarkers[1+Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        axes[1].plot(Evals,Tvals[:,pair[0]], color = mycolors[Jvali], marker = mymarkers[1+Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        axes[2].plot(Evals,Tvals[:,pair[1]], color = mycolors[Jvali], marker = mymarkers[1+Jvali], markevery = mymarkevery, linewidth = mylinewidth);
        totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
        #axes[2].plot(Evals, totals, color="red");


    # format
    axes[0].set_ylim(-0.05*1.0,1.0+0.05*1.0);
    axes[0].set_ylabel('$T_{i}$', fontsize = myfontsize );
    axes[1].set_ylim(-0.05*0.4,0.4);
    axes[1].set_ylabel('$T_{+}$', fontsize = myfontsize );
    #axes[2].set_ylim(0,0.4);
    axes[2].set_ylabel('$T_{-}$', fontsize = myfontsize );
    axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    
    # show
    axes[-1].set_xscale('log', subs = []);
    axes[-1].set_xlim(10**(logElims[0]), 10**(logElims[1]));
    axes[-1].set_xticks([10**(logElims[0]), 10**(logElims[1])]);
    axes[-1].set_xlabel('$K_i/t$',fontsize = myfontsize);
    #for axi in range(len(axes)): axes[axi].set_title(mypanels[axi], x=0.07, y = 0.7, fontsize = myfontsize);
    plt.tight_layout();
    plt.savefig('figs/double/Jlimit.pdf');
    #plt.show();

