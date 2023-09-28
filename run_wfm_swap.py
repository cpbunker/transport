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

elecspin = 1; # initial electron is spin down
assert(elecspin==1); # else add uparrow labels below
ylabels = ["\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"];
            
#########################################################
#### barrier in right lead for total reflection

# tight binding params
tl = 1.0;
myspinS = 0.5;
n_mol_dof = int((2*myspinS+1)**2);
n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
Jval = -0.2*tl;
VB = 5.0*tl;


if False: # distance of the barrier NB on the x axis

    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex='col', sharey = 'row');
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    vlines = True; # whether to highlight certain x vals with vertical dashed lines

    # iter over incident kinetic energy (colors)
    Kpowers = np.array([-2,-3,-4]); # incident kinetic energy/t = 10^Kpower
    Kvals = np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers));
    Rvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals)); # by  init spin, final spin, energy, NB
    Tvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals)); # by  init spin, final spin, energy, NB
    for Kvali in range(len(Kvals)):

        # energy
        Kval = Kvals[Kvali]; # Kval > 0 always, what I call K_i in paper
        Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed energy
        
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
            if(Kvali == 0 and NBvali == 0): print("\nhblocks = \n",np.real(hblocks));

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
                Rvals[sourcei,:,Kvali,NBvali] = Rdum;
                Tvals[sourcei,:,Kvali,NBvali] = Tdum;
            #### end loop over sourcei
        #### end loop over NB

        # determine fidelity and kNB*, ie x val where the SWAP happens
        fidelity_list = np.array([np.max(Rvals[n_mol_dof*elecspin+1,n_mol_dof*elecspin+2,Kvali]),
                                 np.max(Rvals[n_mol_dof*elecspin+2,n_mol_dof*elecspin+1,Kvali]),
                                 np.max(1-Rvals[n_mol_dof*elecspin+1,n_mol_dof*elecspin+1,Kvali]),
                                 np.max(1-Rvals[n_mol_dof*elecspin+2,n_mol_dof*elecspin+2,Kvali])]);
        kNBstar = kNBvals[np.argmin(Rvals[n_mol_dof*elecspin+1,n_mol_dof*elecspin+1,Kvali])];
        print("kNBstar/pi, fidelity(kNBstar) = ",kNBstar/np.pi, np.mean(fidelity_list));
         
        # plot R_\sigma vs NBvals
        Rvals_up = Rvals[:,np.array(range(n_loc_dof))<n_mol_dof];
        Rvals_down = Rvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                axes[sourcei,sigmai].plot(kNBvals/np.pi, Rvals[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,Kvali], label = "$K_i/t= 10^{"+str(Kpowers[Kvali])+"}$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);
                
                ##### additional plotting
                # starred SWAP locations
                if(vlines and sourcei == sigmai): axes[sourcei,sigmai].axvline(kNBstar/np.pi, color=mycolors[Kvali], linestyle="dotted"); 
                # reflection summed over final states (columns)
                if(sourcei<n_mol_dof-1):
                    if(Kvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    axes[sourcei,-1].plot(kNBvals/np.pi, np.sum(Rvals_down[n_mol_dof*elecspin+sourcei,:,Kvali],axis=0), linestyle="solid", label=showlegstring+"Total $R(\\rightarrow \downarrow_e)$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[sourcei,-1].plot(kNBvals/np.pi, np.sum(Rvals_up[n_mol_dof*elecspin+sourcei,:,Kvali],axis=0), linestyle="dashed", label=showlegstring+"Total $R(\\rightarrow \\uparrow_e)$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);
                    #axes[sourcei,-1].plot(kNBvals/np.pi, 0.5-np.sum(Tvals[n_mol_dof*elecspin+sourcei,:,Kvali],axis=0), linestyle="dashdot", label=showlegstring+"0.5-Total $T$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[0,-1].legend();

                # formatting
                axes[sourcei,sigmai].set_title("$R("+str(ylabels[sourcei])+"\\rightarrow"+str(ylabels[sigmai])+")$");
                axes[sourcei,-1].set_ylim(-0.1,1.1);
                axes[sourcei,-1].set_yticks([0,1.0]);
                axes[sourcei,sigmai].axhline(0.0,color='lightgray',linestyle='dashed');
                axes[sourcei,sigmai].axhline(1.0,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlim(0,0.5);
                axes[-1,sigmai].set_xlabel('$k_i aN_B /\pi$',fontsize=myfontsize);
                axes[1,0].legend();
                    
    # show
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([tl,myspinS,Jval,VB]);
    fname = "data/wfm_swap/NB/";
    #np.savetxt(fname+".txt", Kvals, header="[tl,myspinS,Jval,VB] =\n"+str(param_vals)+"\nKvals =");
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, Rvals);

if False: # incident kinetic energy on the x axis
         # NB you STILL have to change NB
         # this is where another voltage might be useful!!!

    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    vlines = True; # whether to highlight certain x vals with vertical dashed lines

    # iter over kNBvals (colors)
    kNBvals = np.pi*np.array([0.2,0.3]);
    Rvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(kNBvals))); # by  init spin, final spin, energy, NB
    Tvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(kNBvals))); # by  init spin, final spin, energy, NB
    for NBvali in range(len(kNBvals)):

        # iter over incident kinetic energy (x axis)
        Kpowers = np.array([-2,-3,-4]); # incident kinetic energy/t = 10^Kpower
        Kvals = np.logspace(Kpowers[-1],Kpowers[0],num=myxvals);
        print("longest NB = ",int((kNBvals[-1]/np.arccos( (Kvals-2*tl)/(-2*tl)))[0]));
        for Kvali in range(len(Kvals)):

            # energy
            Kval = Kvals[Kvali]; # Kval > 0 always, what I call K in paper
            Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper
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
            if(Kvali == 0 and NBvali == 0): print("\nhblocks = \n",np.real(hblocks));

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
                Rvals[sourcei,:,Kvali,NBvali] = Rdum;
                Tvals[sourcei,:,Kvali,NBvali] = Tdum;
            #### end loop over sourcei
        #### end loop over E

        # determine fidelity and K*, ie x val where the SWAP happens
        fidelity_list = np.array([np.max(Rvals[n_mol_dof*elecspin+1,n_mol_dof*elecspin+2,:,NBvali]),
                                 np.max(Rvals[n_mol_dof*elecspin+2,n_mol_dof*elecspin+1,:,NBvali]),
                                 np.max(1-Rvals[n_mol_dof*elecspin+1,n_mol_dof*elecspin+1,:,NBvali]),
                                 np.max(1-Rvals[n_mol_dof*elecspin+2,n_mol_dof*elecspin+2,:,NBvali])]);
        Kstar = Kvals[np.argmin(Rvals[1,1,:,NBvali])];
        print("Kstar/t, fidelity(kNBstar) = ",Kstar, np.mean(fidelity_list));
             
        # plot R_\sigma vs NBvals
        Rvals_up = Rvals[:,np.array(range(n_loc_dof))<n_mol_dof];
        Rvals_down = Rvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                axes[sourcei,sigmai].plot(Kvals, Rvals[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], label = "$k_i a N_B/\pi$ = {:.2f}".format(kNBvals[NBvali]/np.pi), color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);

                #### other plotting
                # starred SWAP locations
                if(vlines and sourcei == sigmai): axes[sourcei,sigmai].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
                # reflection summed over final states (columns)
                if(sourcei<n_mol_dof-1):
                    if(NBvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    axes[sourcei,-1].plot(Kvals, np.sum(Rvals_down[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="solid", label=showlegstring+"Total $R(\\rightarrow \downarrow_e)$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[sourcei,-1].plot(Kvals, np.sum(Rvals_up[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="dashed", label=showlegstring+"Total $R(\\rightarrow \\uparrow_e)$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    #axes[sourcei,-1].plot(Kvals, 0.5-np.sum(Tvals[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="dashdot", label=showlegstring+"0.5-Total $T$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[0,-1].legend();
                
                # formatting
                axes[sourcei,sigmai].set_title("$R("+str(ylabels[sourcei])+"\\rightarrow"+str(ylabels[sigmai])+")$");
                axes[sourcei,sigmai].set_ylim(-0.1,1.1);
                axes[sourcei,sigmai].axhline(0.0,color='lightgray',linestyle='dashed');
                axes[sourcei,sigmai].axhline(1.0,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlabel('$K_i/t$',fontsize=myfontsize);
                axes[-1,sigmai].set_xscale('log', subs = []);
                axes[1,0].legend();
                  
    # show
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([myspinS,tl,Jval]);
    #fname = "data/wfm_swap/E/"+str((kNBvals/np.pi).round(1));
    #np.savetxt(fname+".txt", kNBvals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nkNBvals =");
    #np.save(fname+"_x", rhoJvals);
    #np.save(fname, Rvals);

if True: # incident kinetic energy on the x axis
         # NB is now fixed !!!!

    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    vlines = True; # whether to highlight certain x vals with vertical dashed lines

    # iter over fixed NB (colors)
    NBvals = np.array([50,75,94,100]);
    Rvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals))); # by  init spin, final spin, energy, NB
    Tvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals))); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):

        # iter over incident kinetic energy (x axis)
        Kpowers = np.array([-2,-3,-4]); # incident kinetic energy/t = 10^Kpower
                                              # note that at the right NB, R(SWAP) approaches 1 asymptotically at
                                              # lower Ki. But diminishing returns kick in around 10^-4
        Kvals = np.logspace(Kpowers[-1],Kpowers[0],num=myxvals);
        for Kvali in range(len(Kvals)):

            # energy
            Kval = Kvals[Kvali]; # Kval > 0 always, what I call K in paper
            Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper
            k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed \rho J a

            # set barrier distance
            NBval = int(NBvals[NBvali])
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
            if(Kvali == 0 and NBvali == 0): print("\nhblocks = \n",np.real(hblocks));

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
                Rvals[sourcei,:,Kvali,NBvali] = Rdum;
                Tvals[sourcei,:,Kvali,NBvali] = Tdum;
            #### end loop over sourcei
        #### end loop over E

        # determine fidelity and K*, ie x val where the SWAP happens
        fidelity_list = np.array([np.max(Rvals[n_mol_dof*elecspin+1,n_mol_dof*elecspin+2,:,NBvali]),
                                 np.max(Rvals[n_mol_dof*elecspin+2,n_mol_dof*elecspin+1,:,NBvali]),
                                 np.max(1-Rvals[n_mol_dof*elecspin+1,n_mol_dof*elecspin+1,:,NBvali]),
                                 np.max(1-Rvals[n_mol_dof*elecspin+2,n_mol_dof*elecspin+2,:,NBvali])]);
        Kstar = Kvals[np.argmin(Rvals[1,1,:,NBvali])];
        print("Kstar/t, fidelity(kNBstar) = ",Kstar, np.mean(fidelity_list));
             
        # plot R_\sigma vs NBvals
        Rvals_up = Rvals[:,np.array(range(n_loc_dof))<n_mol_dof];
        Rvals_down = Rvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                axes[sourcei,sigmai].plot(Kvals, Rvals[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]), color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);

                #### other plotting
                # starred SWAP locations
                if(vlines and sourcei == sigmai): axes[sourcei,sigmai].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
                # reflection summed over final states (columns)
                if(sourcei<n_mol_dof-1):
                    if(NBvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    axes[sourcei,-1].plot(Kvals, np.sum(Rvals_down[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="solid", label=showlegstring+"Total $R(\\rightarrow \downarrow_e)$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[sourcei,-1].plot(Kvals, np.sum(Rvals_up[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="dashed", label=showlegstring+"Total $R(\\rightarrow \\uparrow_e)$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    #axes[sourcei,-1].plot(Kvals, 0.5-np.sum(Tvals[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="dashdot", label=showlegstring+"0.5-Total $T$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[0,-1].legend();
                
                # formatting
                axes[sourcei,sigmai].set_title("$R("+str(ylabels[sourcei])+"\\rightarrow"+str(ylabels[sigmai])+")$");
                axes[sourcei,sigmai].set_ylim(-0.1,1.1);
                axes[sourcei,sigmai].axhline(0.0,color='lightgray',linestyle='dashed');
                axes[sourcei,sigmai].axhline(1.0,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlabel('$K_i/t$',fontsize=myfontsize);
                axes[-1,sigmai].set_xscale('log', subs = []);
                axes[1,0].legend();
                  
    # show
    fig.suptitle("$s=${:.1f}, $J/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, VB/tl))
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([myspinS,tl,Jval]);
    #fname = "data/wfm_swap/E/"+str((kNBvals/np.pi).round(1));
    #np.savetxt(fname+".txt", kNBvals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nkNBvals =");
    #np.save(fname+"_x", rhoJvals);
    #np.save(fname, Rvals);

