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
myxvals = 99;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# constructing the hamiltonian
def h_cicc(J, i1, i2) -> np.ndarray: 
    '''
    TB matrices for ciccarrello system (1 electron, 2 spin-1/2s)
    Args:
    - J, float, eff heisenberg coupling
    - i1, list of sites for first spin-1/2
    - i2, list of sites for second spin-1/2
    '''
    if(not isinstance(i1, list) or not isinstance(i2, list)): raise TypeError;
    assert(i1[0] == 1);
    if(not i1[-1] < i2[0]): raise Exception("i1 and i2 cannot overlap");
    NC = i2[-1]; # num sites in the central region
    
    # heisenberg interaction matrices
    Se_dot_S1 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to first spin impurity
                        [0,1,0,0,0,0,0,0],
                        [0,0,-1,0,2,0,0,0],
                        [0,0,0,-1,0,2,0,0],
                        [0,0,2,0,-1,0,0,0],
                        [0,0,0,2,0,-1,0,0],
                        [0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,1] ]);
    Se_dot_S2 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to second spin impurity
                        [0,-1,0,0,2,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,-1,0,0,2,0],
                        [0,2,0,0,-1,0,0,0],
                        [0,0,0,0,0,1,0,0],
                        [0,0,0,2,0,0,-1,0],
                        [0,0,0,0,0,0,0,1] ]);

    # insert these local interactions
    h_cicc =[];
    Nsites = NC+1; # N sites in SR + 1 for LL
    for sitei in range(Nsites): # iter over all sites
        if(sitei in i1 and sitei not in i2):
            h_cicc.append(Se_dot_S1);
        elif(sitei in i2 and sitei not in i1):
            h_cicc.append(Se_dot_S2);
        elif(sitei not in i1 and sitei not in i2):
            h_cicc.append(np.zeros_like(Se_dot_S1) );
        else:
            raise Exception("i1 and i2 cannot overlap");
    return np.array(h_cicc, dtype=complex);
            
#########################################################
#### barrier in right lead for total reflection

# tight binding params
tl = 1.0;
myspinS = 0.5;
n_loc_dof = int((2*myspinS+1)**3);
n_mol_dof = int((2*myspinS+1)**2);
Jval = -0.1;
VB = 5.0*tl;

if False: # distance of the barrier NB on the x axis

    # axes
    ylabels = ["\downarrow \\uparrow \\uparrow","\downarrow \\uparrow \downarrow","\downarrow \downarrow \\uparrow","\downarrow \downarrow \downarrow"];
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex='col', sharey = 'row');
    fig.set_size_inches(ncols*7/2,nrows*3/2);

    # iter over E (colors)
    rhoJvals = np.array([0.5,1.0,5.0]);
    Evals = Jval*Jval/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
    Rvals = np.empty((n_loc_dof,n_loc_dof,len(Evals),myxvals)); # by  init spin, final spin, energy, NB
    for Evali in range(len(Evals)):

        # energy
        Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
        Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed \rho J a
        
        # iter over barrier distance (x axis)
        kNBmax = 0.5*np.pi;
        NBmax = int(kNBmax/k_rho);
        if(verbose): print("NBmax = ",NBmax); 
        NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
        kNBvals = k_rho*NBvals;
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # construct hblocks from spin ham
            hblocks_cicc = h_cicc(Jval, [1],[2]);

            # add large barrier at end
            NC = len(hblocks_cicc); assert(NC==3); # num sites in central region
            hblocks, tnn = [], []; # new empty array all the way to barrier, will add cicc later
            for _ in range(NC+NBval):
                hblocks.append(0.0*np.eye(n_loc_dof));
                tnn.append(-tl*np.eye(n_loc_dof));
            hblocks, tnn = np.array(hblocks,dtype=complex), np.array(tnn[:-1]);
            hblocks[0:NC] += hblocks_cicc;
            hblocks[-1] += VB*np.eye(n_loc_dof);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
            if(Evali == 0 and NBvali == 0): print("\nhblocks = \n",np.real(hblocks));

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
        kNBstar = kNBvals[np.argmin(Rvals[1,1,Evali])];
         
        # plot R_\sigma vs NBvals
        elecspin = 1;
        for sourcei in range(n_mol_dof):
            for sigmai in range(n_mol_dof):
                axes[sourcei,sigmai].plot(kNBvals/np.pi, Rvals[4*elecspin+sourcei,4*elecspin+sigmai,Evali], color=mycolors[Evali], marker=mymarkers[1+Evali], markevery=mymarkevery, linewidth=mylinewidth);
                #if(sourcei == sigmai): axes[sourcei,sigmai].axvline(kNBstar/np.pi, color=mycolors[Evali]);
                axes[sourcei,sigmai].set_title("$R("+str(ylabels[sourcei])+"\\rightarrow"+str(ylabels[sigmai])+")$");
                axes[sourcei,-1].set_ylim(-0.1,1.1);
                axes[sourcei,-1].set_yticks([0,1.0]);
                axes[sourcei,sigmai].axhline(0.0,color='lightgray',linestyle='dashed');
                axes[sourcei,sigmai].axhline(1.0,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlim(0,0.5);
                axes[-1,sigmai].set_xlabel('$k_i aN_B /\pi$',fontsize=myfontsize);
                #axes[sourcei,-1].plot(kNBvals/np.pi, np.sum(Rvals[4*elecspin+sourcei,:,Evali],axis=0), color='red');
                    
    # show
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([myspinS,tl,Jval]);
    fname = "data/wfm_swap/NB/"+str(rhoJvals.round(1));
    #np.savetxt(fname+".txt", Evals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nEvals =");
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, Rvals);

if True: # incident kinetic energy on the x axis

    # axes
    ylabels = ["\downarrow \\uparrow \\uparrow","\downarrow \\uparrow \downarrow","\downarrow \downarrow \\uparrow","\downarrow \downarrow \downarrow"];
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);

    # iter over kNBvals
    kNBvals = np.pi*np.array([0.3]); # 0.4, 0.5
    myxvals = 29
    Rvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(kNBvals))); # by  init spin, final spin, energy, NB
    for NBvali in range(len(kNBvals)):

        # iter over E 
        rhoJvals = np.linspace(0.5,5.0,myxvals);
        Evals = Jval*Jval/(rhoJvals*rhoJvals*np.pi*np.pi*tl);
        print(Evals)
        print("longest NB = ",int((kNBvals[-1]/np.arccos( (Evals-2*tl)/(-2*tl)))[-1]));
        assert False
        for Evali in range(len(Evals)):

            # energy
            Eval = Evals[Evali]; # Eval > 0 always, what I call K in paper
            Energy = Eval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed \rho J a

            # set barrier distance
            NBval = int(kNBvals[NBvali]/k_rho);
            if(verbose): print("NB = ",NBval); 

            # construct hblocks from spin ham
            hblocks_cicc = h_cicc(Jval, [1],[2]);

            # add large barrier at end
            NC = len(hblocks_cicc); assert(NC==3); # num sites in central region
            hblocks, tnn = [], []; # new empty array all the way to barrier, will add cicc later
            for _ in range(NC+NBval):
                hblocks.append(0.0*np.eye(n_loc_dof));
                tnn.append(-tl*np.eye(n_loc_dof));
            hblocks, tnn = np.array(hblocks,dtype=complex), np.array(tnn[:-1]);
            hblocks[0:NC] += hblocks_cicc;
            hblocks[-1] += VB*np.eye(n_loc_dof);
            tnnn = np.zeros_like(tnn)[:-1]; # no next nearest neighbor hopping
            if(Evali == 0 and NBvali == 0): print("\nhblocks = \n",np.real(hblocks));

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
        rhoJstar = rhoJvals[np.argmin(Rvals[1,1,:,NBvali])];
             
        # plot R_\sigma vs NBvals
        elecspin = 1;
        for sourcei in range(n_mol_dof):
            for sigmai in range(n_mol_dof):
                axes[sourcei,sigmai].plot(rhoJvals, Rvals[4*elecspin+sourcei,4*elecspin+sigmai,:,NBvali], color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                #if(sourcei == sigmai): axes[sourcei,sigmai].axvline(rhoJstar, color=mycolors[NBvali]);
                axes[sourcei,sigmai].set_title("$R("+str(ylabels[sourcei])+"\\rightarrow"+str(ylabels[sigmai])+")$");
                axes[sourcei,sigmai].set_ylim(-0.1,1.1);
                axes[sourcei,sigmai].axhline(0.0,color='lightgray',linestyle='dashed');
                axes[sourcei,sigmai].axhline(1.0,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlim(0.5,5.0);
                axes[-1,sigmai].set_xlabel('$\\rho Ja$',fontsize=myfontsize);
                #axes[sourcei,-1].plot(kNBvals/np.pi, np.sum(Rvals[4*elecspin+sourcei,4*elecspin+sigmai,:,NBvali],axis=0), color='red');
                    
    # show
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([myspinS,tl,Jval]);
    #fname = "data/wfm_swap/E/"+str((kNBvals/np.pi).round(1));
    #np.savetxt(fname+".txt", kNBvals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nkNBvals =");
    #np.save(fname+"_x", rhoJvals);
    #np.save(fname, Rvals);




