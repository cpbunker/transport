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
def diag_ham(params, S) -> np.ndarray:
    '''
    '''
    for el in params:
        if( not isinstance(el, float)): raise TypeError;
        
    D1, D2, J12, JK1, JK2 = params;
    D = (D1+D2)/2;
    DeltaD = D1-D2;
    h = np.eye(3)*(2*S*S*D + (S*S-S)*J12);
    h += np.array([[ (1-2*S)*D + S*J12, (S-1/2)*DeltaD, 0],
                   [ (S-1/2)*DeltaD, (1-2*S)*D - S*J12, 0],
                   [ 0, 0, S*J12]]);
    h += (JK1/2)*np.array([[S-1/2,1/2, np.sqrt(S)],
                           [1/2,S-1/2,-np.sqrt(S)],
                           [np.sqrt(S),-np.sqrt(S),-S]]);
    h += (JK2/2)*np.array([[S-1/2,-1/2,np.sqrt(S)],
                           [-1/2,S-1/2,np.sqrt(S)],
                           [np.sqrt(S),np.sqrt(S),-S]]);
    return h;
            
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
    sourcei = n_loc_dof-1;
    source[sourcei] = 1.0;
    
    # Evals should be order of D (0.1 meV for Mn to 1 meV for MnPc)
    Esplitvals = (1)*np.array([0.0,0.001,0.002,0.003,0.004]);
    Dvals = Esplitvals/(1-2*myspinS);
    for Dvali in range(len(Dvals)):
        Dval = Dvals[Dvali];

        # construct hblocks
        hblocks = [];
        impis = [1,2];
        for j in range(4): # LL, imp 1, imp 2, RL
            # define all physical params
            JK1, JK2 = 0.0, 0.0;
            if(j == impis[0]): JK1 = JK;
            elif(j == impis[1]): JK2 = JK;
            params = Dval, Dval, J12, JK1, JK2;

            # construct h_SR in |+>, |->, |i> basis
            hSR_diag = diag_ham(params,myspinS);           
            hblocks.append(np.copy(hSR_diag));
            if(verbose > 3 ):
                print("\nJK1, JK2 = ",JK1, JK2);
                print(" - ham:\n", hSR_diag);
                print(" - DeltaE = ",Esplitvals[Dvali]);

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(n_loc_dof);
        if(verbose > 3 ): print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

        # hopping
        tnn = np.array([-tl*np.eye(n_loc_dof),-tp*np.eye(n_loc_dof),-tl*np.eye(n_loc_dof)]);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # iter over E, getting T
        logElims = -6,-2
        Evals = np.logspace(*logElims,myxvals, dtype = complex);
        Rvals = np.empty((len(Evals),n_loc_dof), dtype = float);
        Tvals = np.empty((len(Evals),n_loc_dof), dtype = float);
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
