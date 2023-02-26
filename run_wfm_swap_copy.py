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
verbose = 1;

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

if True: # distance of the barrier NB on the x axis

    # tight binding params
    tl = 1.0;
    JK = -0.5*tl/100;
    J12 = 0*tl/100;
    Dval = 1.0/100; # order of D: 0.1 meV for Mn to 1 meV for MnPc
    myspinS = 0.5;
    n_loc_dof = 3;

    if True:
        myspinS = 0.5;
        Dval=0.0;
        JK=-0.1;

    # axes
    ylabels = ["\\uparrow \\uparrow \downarrow","\\uparrow \downarrow \\uparrow","\downarrow \\uparrow \\uparrow"];
    nrows, ncols = n_loc_dof, n_loc_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);

    # iter over E (colors)
    rhoJvals = np.array([0.5,1.0,2.0]);
    Evals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    Rvals = np.empty((n_loc_dof,n_loc_dof,len(Evals),myxvals)); # by  init spin, final spin, energy, NB
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed \rho J a
        
        # iter over barrier distance (x axis)
        kNBmax = 3*np.pi/4;
        NBmax = int(kNBmax/k_rho);
        if(verbose): print(NBmax); #assert False;
        NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
        kNBvals = k_rho*NBvals;
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # construct hblocks from spin ham
            hblocks = [];
            impis = [1,2];
            for j in range(3): # LL, imp 1, imp 2, RL
                # define all physical params
                JK1, JK2 = 0, 0;
                if(j == impis[0]): JK1 = JK;
                elif(j == impis[1]): JK2 = JK;
                params = Dval, Dval, J12, JK1, JK2;
                # construct h_SR (determinant basis)
                hSR = reduced_ham(params,myspinS);
                # don't entangle since SWAP is performed in logical basis
                hblocks.append(np.copy(hSR));
                if(verbose > 3 ):
                    print("\nJK1, JK2 = ",JK1, JK2);
                    print(" - ham:\n", hSR);

            # add large barrier at end
            tnn = [-tl*np.eye(n_loc_dof),-tl*np.eye(n_loc_dof)];
            for _ in range(NBval):
                hblocks.append(np.zeros_like(hblocks[0]));
                tnn.append(-tl*np.eye(n_loc_dof));
            hblocks[-1] += 5*tl*np.eye(n_loc_dof);
                        
            # hopping
            tnn = np.array(tnn);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # iter over sources
            for sourcei in range(n_loc_dof):
                source = np.zeros((n_loc_dof,));
                source[sourcei] = 1.0;

                # finish hblocks
                hblocks = np.array(hblocks);
                E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
                for hb in hblocks:
                    hb += -E_shift*np.eye(n_loc_dof);
                    
                # get R, T coefs
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
                Rvals[sourcei,:,Evali,NBvali] = Rdum;
            #### end loop over sourcei
        #### end loop over NB

        # determine kNB*
        kNBstar = kNBvals[np.argmin(Rvals[0,0,Evali])];
         
        # plot R_\sigma vs NBvals
        for sourcei in range(n_loc_dof):
            for sigmai in range(n_loc_dof):
                axes[sourcei,sigmai].plot(kNBvals/np.pi, Rvals[sourcei,sigmai,Evali], color=mycolors[Evali], marker=mymarkers[1+Evali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[sourcei,sigmai].axvline(kNBstar/np.pi, color=mycolors[Evali]);
                axes[sourcei,sigmai].set_title("$"+str(ylabels[sourcei])+"\\rightarrow"+str(ylabels[sigmai])+"$");
                axes[sourcei,sigmai].set_ylim(0.0,1.0);
                axes[-1,sigmai].set_xlabel('$k_i aN_B /\pi$',fontsize=myfontsize);
                    
    # show
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([tl,JK,J12,Dval,myspinS,n_loc_dof]);
    fname = "data/wfm_swap/NB/"+str(rhoJvals.round(1));
    #np.savetxt(fname+".txt", Evals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nEvals =");
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, Rvals);

if False: # incident kinetic energy on the x axis

    # tight binding params
    tl = 1.0;
    tp = 1.0;
    JK = -0.5*tl/100;
    J12 = 0*tl/100;
    Dval = 1.0/100; # order of D: 0.1 meV for Mn to 1 meV for MnPc
    myspinS = 1.0;
    n_loc_dof = 3;

    if True:
        myspinS = 0.5;
        Dval=0;
        JK=-0.1;

    # axes
    ylabels = ["\\uparrow \\uparrow \downarrow","\\uparrow \downarrow \\uparrow","\downarrow \\uparrow \\uparrow"];
    nrows, ncols = n_loc_dof, n_loc_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);

    # iter over kNBvals
    kNBvals = np.pi*np.array([0.3,0.4,0.5]);
    Rvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(kNBvals))); # by  init spin, final spin, energy, NB
    for NBvali in range(len(kNBvals)):

        # iter over E 
        rhoJvals = np.linspace(0.5,2.0,myxvals);
        Evals = JK*JK/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed \rho J a

            # set barrier distance
            NBval = int(kNBvals[NBvali]/k_rho);
            if(verbose): print(NBval); #assert False;

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
                hSR = reduced_ham(params,myspinS);
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
            hblocks[-1] += 5*tl*np.eye(n_loc_dof);

            # hopping
            tnn = np.array(tnn);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping

            # iter over sources
            for sourcei in range(n_loc_dof):
                source = np.zeros((n_loc_dof,));
                source[sourcei] = 1;

                # finish hblocks
                hblocks = np.array(hblocks);
                E_shift = hblocks[0,sourcei,sourcei]; # const shift st hLL[sourcei,sourcei] = 0
                for hb in hblocks:
                    hb += -E_shift*np.eye(n_loc_dof);
                
                # get R, T coefs
                Rdum, Tdum = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, all_debug = False);
                Rvals[sourcei,:,Evali,NBvali] = Rdum;
            #### end loop over sourcei
        #### end loop over E

        # determine rhoJa*
        rhoJstar = rhoJvals[np.argmin(Rvals[0,0,:,NBvali])];
             
        # plot R_\sigma vs NBvals
        for sourcei in range(n_loc_dof):
            for sigmai in range(n_loc_dof):
                axes[sourcei,sigmai].plot(rhoJvals, Rvals[sourcei,sigmai,:,NBvali], color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[sourcei,sigmai].axvline(rhoJstar, color=mycolors[NBvali]);
                axes[sourcei,sigmai].set_title("$R("+str(ylabels[sourcei])+"\\rightarrow"+str(ylabels[sigmai])+")$");
                axes[sourcei,sigmai].set_ylim(0.0,1.0);
                axes[-1,sigmai].set_xlabel('$\\rho Ja$',fontsize=myfontsize);
                    
    # show
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([tl,JK,J12,Dval,myspinS,n_loc_dof]);
    #fname = "data/wfm_swap/E/"+str((kNBvals/np.pi).round(1));
    #np.savetxt(fname+".txt", kNBvals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nkNBvals =");
    #np.save(fname+"_x", rhoJvals);
    #np.save(fname, Rvals);




