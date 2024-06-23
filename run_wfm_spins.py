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
    ylabels = ["+","-","i"][:1];
    nplots = len(ylabels)+1;
    fig, axes = plt.subplots(nplots, sharex=True);
    if(nplots==1): axes = [axes];
    fig.set_size_inches(7/2,nplots*3/2);

    # tight binding params
    tl = 1.0;
    Nval = 201;
    JK = 6*np.pi*tl/(Nval-1); 
    J12 = tl/100;
    myspinS = 1;
    n_loc_dof = 3;
    source = np.zeros((n_loc_dof,));
    sourcei = n_loc_dof-1;
    source[sourcei] = 1.0;

    # energy of the incident electron
    K_indep = False; # what to put on x axis
    if(K_indep):               
        logKlims = -4,-1
        Kvals = np.logspace(*logKlims,myxvals, dtype = complex); # K > 0 always
        knumbers = np.arccos((Kvals-2*tl)/(-2*tl));
    else:
        knumberlims = 0.1*(np.pi/(Nval-1)), 6.0*(np.pi/(Nval-1));
        knumbers = np.linspace(knumberlims[0], knumberlims[1], myxvals, dtype=complex);
        Kvals = 2*tl - 2*tl*np.cos(knumbers);
    print(knumbers);
    
    # Evals should be order of D (0.1 meV for Mn to 1 meV for MnPc)
    Esplitvals = (-1)*np.array([0.0]) #,0.001,0.002,0.003,0.004]);
    Dvals = Esplitvals/(1-2*myspinS);
    for Dvali in range(len(Dvals)):
        Dval = Dvals[Dvali];

        # construct hblocks
        hblocks, tnn = [], [];
        impis = [1,Nval];
        for j in range(2+impis[1]): # LL, imp 1, ... imp 2, RL
            # define all physical params
            JK1, JK2 = 0.0, 0.0;
            if(j == impis[0]): JK1 = JK;
            elif(j == impis[1]): JK2 = JK;
            params = Dval, Dval, J12, JK1, JK2;

            # construct h_SR in |+>, |->, |i> basis
            hSR_diag = diag_ham(params,myspinS);           
            hblocks.append(np.copy(hSR_diag));
            tnn.append( -tl*np.eye(n_loc_dof));
            if(verbose > 3 and (j in impis)):
                print("\nj = {:.0f}, JK1 = {:.4f}, JK2 = {:.4f}".format(j, JK1, JK2));
                print(" - ham:\n", hSR_diag);
                print(" - DeltaE = ",Esplitvals[Dvali]);

        # finish hblocks
        hblocks = np.array(hblocks);
        E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
        for hb in hblocks:
            hb += -E_shift*np.eye(n_loc_dof);
        if(verbose > 3 ): print("Delta E / t = ", (hblocks[0][0,0] - hblocks[0][2,2])/tl);

        # hopping
        tnn = np.array(tnn[:-1]);
        tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

        # iter over K, getting T
        Rvals = np.empty((len(Kvals),n_loc_dof), dtype = float);
        Tvals = np.empty((len(Kvals),n_loc_dof), dtype = float);
        for Kvali in range(len(Kvals)):

            # energy
            Kval = Kvals[Kvali]; # Eval > 0 always, what I call K in paper
            Energy = Kval - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source,
                                is_psi_jsigma = False, is_Rhat = False, all_debug = False);
            Rvals[Kvali] = Rdum;
            Tvals[Kvali] = Tdum;

        # plot
        if(K_indep): indep_vals = np.real(Kvals);
        else: indep_vals = np.real(knumbers)/(np.pi/(Nval-1));
        for axi in range(nplots):
            # plot T_\sigma
            if(axi in [0]): #np.array(range(n_loc_dof))):
                axes[axi].plot(indep_vals, Tvals[:,axi], label="$\Delta E=${:.4f}".format(Esplitvals[Dvali]), color=mycolors[Dvali], marker=mymarkers[1+Dvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[axi].set_ylabel("$T_"+str(ylabels[axi])+"$");
                #axes[axi].set_ylim(0,0.2);
            # plot \overline{p^2}
            else: 
                axes[axi].plot(indep_vals, np.sqrt(Tvals[:,0]*Tvals[:,2]), color=mycolors[Dvali], marker=mymarkers[1+Dvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[axi].set_ylabel("$\overline{p^2}$");
                #axes[axi].set_ylim(0,0.3);
    # format
    if(K_indep):
        axes[-1].set_xlabel("$K_i/t$",fontsize=myfontsize);
        axes[-1].set_xscale("log", subs = []);
    else:
        axes[-1].set_xlabel("$k_i (N-1)/\pi$", fontsize=myfontsize);
                
    # show
    axes[0].set_title("$s =${:.1f}, $N =${:.0f}, $J =${:.4f}".format(
        myspinS, Nval, JK)+", $J_{12} =$"+"{:.4f}".format(J12));
                     
    plt.tight_layout();
    plt.show();
