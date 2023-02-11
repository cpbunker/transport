'''
Christian Bunker
M^2QM at UF
November 2022

Scattering from two spin-s MSQs
Want to make a SWAP gate
solved in time-independent QM using wfm method in transport/wfm
'''

from transport import wfm

import numpy as np
import matplotlib.pyplot as plt

#### top level
#np.set_printoptions(precision = 4, suppress = True);
verbose = 0;

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

# constructing the hamiltonian
def reduced_ham(params, S) -> np.ndarray:
    D1, D2, J12, JK1, JK2 = params;
    assert(D1 == D2);
    h = np.array([[S*S*D1+(S-1)*(S-1)*D2+S*(S-1)*J12+(JK1/2)*S+(JK2/2)*(S-1), S*J12, np.sqrt(2*S)*(JK2/2) ], # up, s, s-1
                    [S*J12, (S-1)*(S-1)*D1+S*S*D2+S*(S-1)*J12+(JK1/2)*S + (JK2/2)*(S-1), np.sqrt(2*S)*(JK1/2) ], # up, s-1, s
                    [np.sqrt(2*S)*(JK2/2), np.sqrt(2*S)*(JK1/2),S*S*D1+S*S*D2+S*S*J12+(-JK1/2)*S +(-JK2/2)*S]], # down, s, s
                   dtype = complex);

    return h;
            
#########################################################
#### barrier in right lead for total reflection

if True: # T+ at different Delta E by changing D

    # axes
    ylabels = ["{\\uparrow \\uparrow \downarrow}","{\\uparrow \downarrow \\uparrow}","{\downarrow \\uparrow \\uparrow}"];
    nplots = 4;
    fig, axes = plt.subplots(nplots, sharex=True);
    if(nplots==1): axes = [axes];
    fig.set_size_inches(7/2,nplots*3/2);

    # tight binding params
    tl = 1.0;
    tp = 1.0;
    JK = -0.5*tl/100;
    JK = -0.1;
    J12 = 0*tl/100;
    Dval = 0.0/100; # order of D: 0.1 meV for Mn to 1 meV for MnPc
    myspinS = 0.5;
    n_loc_dof = 3;
    source = np.zeros((n_loc_dof,));
    sourcei = 0;
    source[sourcei] = 1.0;

    # iter over E 
    rhoJvals = np.array([0.5,1.0,2.0,5.0]);
    Evals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed \rho J a

        # iter over barrier distance
        kNBmax = 3*np.pi/4;
        NBmax = int(kNBmax/k_rho);
        print(NBmax); #assert False;
        NBvals = np.linspace(0,NBmax,myxvals,dtype=int);
        kNBvals = k_rho*NBvals;
        Rvals = np.empty((myxvals,n_loc_dof), dtype = float);
        Tvals = np.empty((myxvals,n_loc_dof), dtype = float);
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # construct hblocks from spin ham
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
                # don't entangle since SWAP is performed in logical basis
                hblocks.append(np.copy(hSR));
                if(verbose > 3 ):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", hSR);

            # add large barrier at end
            tnn = [-tl*np.eye(n_loc_dof),-tp*np.eye(n_loc_dof),-tl*np.eye(n_loc_dof)];
            for _ in range(NBval):
                hblocks.append(np.zeros_like(hblocks[0]));
                tnn.append(-tl*np.eye(n_loc_dof));
            hblocks[-1] += 5*tl*np.eye(len(source));

            # finish hblocks
            hblocks = np.array(hblocks);
            E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
            for hb in hblocks:
                hb += -E_shift*np.eye(n_loc_dof);
                
            # hopping
            tnn = np.array(tnn);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # get R, T coefs
            Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
            Rvals[NBvali] = Rdum;
            Tvals[NBvali] = Tdum;
             
        # plot Rs vs NBvals
        for axi in range(nplots):
            # plot R_\sigma
            if(axi in np.array(range(n_loc_dof))):
                axes[axi].plot(kNBvals/np.pi, Rvals[:,axi], color=mycolors[Evali], marker=mymarkers[1+Evali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[axi].set_ylabel("$R_"+str(ylabels[axi])+"$");
                axes[axi].set_ylim(0.0,1.0);
            # plot \overline{p^2}
            else: 
                axes[axi].set_ylabel("$\overline{p^2}$");
                axes[axi].plot(kNBvals/np.pi, Rvals[:,0]+Rvals[:,1]+Rvals[:,2],color=mycolors[Evali]);
    # format
    axes[-1].set_xlabel('$kaN_B /\pi$',fontsize=myfontsize);
                
    # show
    plt.tight_layout();
    plt.show();


