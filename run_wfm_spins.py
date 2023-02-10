'''
Christian Bunker
M^2QM at UF
October 2021

Quasi 1 body transmission through spin impurities project, part 4:
Cobalt dimer modeled as two spin-3/2 impurities mo
Spin interaction parameters calculated from dft, Jie-Xiang's Co dimer manuscript
'''

from transport import wfm

import numpy as np
import matplotlib.pyplot as plt
import sys

#### top level
#np.set_printoptions(precision = 4, suppress = True);
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
            
#########################################################
#### effects of Ki and Delta E

if True: # T+ at different Delta E by changing D

    # axes
    ylabels = ["+","-","i"];
    nplots = 4;
    fig, axes = plt.subplots(nplots, sharex=True);
    if(nplots==1): axes = [axes];
    fig.set_size_inches(7/2,nplots*3/2);

    # tight binding params
    tl = 1.0;
    tp = 1.0;
    JK = -0.5*tl/100;
    J12 = tl/100;
    myspinS = 1;
    n_loc_dof = 3;
    source = np.zeros((n_loc_dof,));
    sourcei=n_loc_dof-1;
    source[sourcei] = 1.0;
    
    # Evals should be order of D (0.1 meV for Mn to 1 meV for MnPc)
    Esplitvals = (1)*np.array([0.0,0.001,0.002,0.003,0.004]);
    Dvals = Esplitvals/(1-2*myspinS);
    for Dvali in range(len(Dvals)):
        Dval = Dvals[Dvali];

        # optical distances, N = 2 fixed
        N0 = 1; # N0 = N - 1

        # construct hblocks
        hblocks = [];
        impis = [1,2];
        for j in range(4): # LL, imp 1, imp 2, RL
            # define all physical params
            JK1, JK2 = 0, 0;
            if(j == impis[0]): JK1 = JK;
            elif(j == impis[1]): JK2 = JK;
            params = Dval, Dval, J12, JK1, JK2;
            # construct h_SR (determinant basis)
            hSR = reduced_ham(params,S=myspinS);           
            # transform to eigenbasis
            hSR_diag = entangle(hSR, 0,1);
            hblocks.append(np.copy(hSR_diag));
            if(verbose > 3 ):
                print("\nJK1, JK2 = ",JK1, JK2);
                print(" - ham:\n", hSR);
                print(" - transformed ham:\n", np.real(hSR_diag));
                print(" - DeltaE = ",Esplitvals[Dvali]);

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(np.shape(hblocks[0])[0]);
        if(verbose > 3 ): print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

        # hopping
        tnn = np.array([-tl*np.eye(len(source)),-tp*np.eye(len(source)),-tl*np.eye(len(source))]);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # iter over E, getting T
        logElims = -6,-2
        Evals = np.logspace(*logElims,myxvals, dtype = complex);
        Rvals = np.empty((len(Evals),len(source)), dtype = float);
        Tvals = np.empty((len(Evals),len(source)), dtype = float);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
            Rvals[Evali] = Rdum;
            Tvals[Evali] = Tdum;

        # plot
        for axi in range(nplots):
            # plot T_\sigma
            if(axi in np.array(range(n_loc_dof))):
                axes[axi].plot(np.real(Evals), Tvals[:,axi], color=mycolors[Dvali], marker=mymarkers[1+Dvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[axi].set_ylabel("$T_"+str(ylabels[axi])+"$");
            # plot \overline{p^2}
            else: 
                axes[axi].plot(np.real(Evals), np.sqrt(Tvals[:,0]*Tvals[:,2]), color=mycolors[Dvali], marker=mymarkers[1+Dvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[axi].set_ylabel("$\overline{p^2}$");
    # format
    axes[-1].set_xlabel('$K_i/t$',fontsize=myfontsize);
    axes[-1].set_xscale('log', subs = []);
                
    # show
    plt.tight_layout();
    plt.show();
