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

import sys

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
ylabels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
    "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"];
            
#########################################################
#### barrier in right lead for total reflection

#### top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 1;

# fig standardizing
myxvals = 99;
myfontsize = 14;
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X","+"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
vlines = True; # whether to highlight certain x vals with vertical dashed lines
summed_columns = False;
case = int(sys.argv[2]);

# tight binding params
tl = 1.0;
myspinS = 0.5;
n_mol_dof = int((2*myspinS+1)**2);
n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
Jval = -0.2*tl;
VB = 5.0*tl;

# gate to measure against
which_gate = sys.argv[1];

if(which_gate=="SQRT"):
    the_ticks = [-1.0,0.0,1.0];
    U_q = np.array([[1,0,0,0],
                       [0,complex(0.5,0.5),complex(0.5,-0.5),0],
                       [0,complex(0.5,-0.5),complex(0.5,0.5),0],
                       [0,0,0,1]], dtype=complex); #  SWAP^1/2 gate
elif(which_gate=="SWAP"):
    the_ticks = [0.0,1.0];
    U_q = np.array([[1,0,0,0],
                   [0,0,1,0],
                   [0,1,0,0],
                   [0,0,0,1]], dtype=complex); # SWAP gate
elif(which_gate=="I"):
    the_ticks = [0.0,1.0];
    U_q = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]], dtype=complex); # Identity gate
else:
    raise NotImplementedError("which_gate not supported");

# from Uq to Ugate
U_gate = np.zeros( (2*len(U_q),2*len(U_q)), dtype=complex);
U_gate[:n_mol_dof,:n_mol_dof] = U_q[:n_mol_dof,:n_mol_dof];
U_gate[n_mol_dof:,n_mol_dof:] = U_q[:n_mol_dof,:n_mol_dof];
print("U_gate =\n",U_gate)

# input state to measure fidelity for
chi_state = np.array([0,0,0,0,1,1,-1,1]);
chi_state = chi_state/np.sqrt(np.dot(np.conj(chi_state), chi_state)); # normalize
print("chi_state =\n",chi_state);

if(case in [1,2]): # distance of the barrier NB on the x axis
    
    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex='col', sharey = 'row');
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    if(case==1): NB_indep = False;
    elif(case==2): NB_indep = True # whether to put NB, alternatively wavenumber*NB

    # iter over incident kinetic energy (colors)
    Kpowers = np.array([-2,-3,-4]); # incident kinetic energy/t = 10^Kpower
    Kvals = np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers));
    rhatvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals),dtype=complex); # by  init spin, final spin, energy, NB
    Fvals_Uchi = np.empty((len(Kvals), myxvals),dtype=float);
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

            # since we don't iter over sources, ALL sources must have 0 chem potential
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            for sourcei in range(n_loc_dof):
                assert(hblocks[0][sourcei,sourcei]==0.0);
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, rhat = True, all_debug = False);

            # fidelity w/r/t U, chi
            #print(np.matmul(rhatvals[:,:,Kvali,NBvali], chi_state)[n_mol_dof:])
            F_element = np.dot( np.conj(np.matmul(U_gate, chi_state)), np.matmul(rhatvals[:,:,Kvali,NBvali], chi_state));
            Fvals_Uchi[Kvali, NBvali] = np.sqrt( np.real( np.conj(F_element)*F_element ));
            print(Kvali, NBvals[NBvali], kNBvals[NBvali]/np.pi, Fvals_Uchi[Kvali, NBvali]);
            
        #### end loop over NB

        # determine fidelity and kNB*, ie x val where the SWAP happens
        yvals = np.copy(rhatvals);
        if(which_gate != "SQRT"): yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals));
        yvals_up = yvals[:,np.array(range(n_loc_dof))<n_mol_dof];
        yvals_down = yvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        kNBstar = kNBvals[np.argmax(Fvals_Uchi[Kvali])];
        print("kNBstar/pi, fidelity(kNBstar) = ",kNBstar/np.pi, np.max(Fvals_Uchi[Kvali]));
         
        # plot as a function of NBvals
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                # plot rhat
                axes[sourcei,sigmai].plot(kNBvals/np.pi, np.real(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,Kvali], label = "$K_i/t= 10^{"+str(Kpowers[Kvali])+"}$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[sourcei,sigmai].plot(kNBvals/np.pi, np.imag(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,Kvali], linestyle="dashed", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);
                
                # plot fidelity, starred SWAP locations
                if(sourcei==2 and sigmai==1):
                    axes[sigmai,sourcei].plot(kNBvals/np.pi,Fvals_Uchi[Kvali], label = "$K_i/t= 10^{"+str(Kpowers[Kvali])+"}$",color=mycolors[Kvali]);
                    axes[sigmai,sourcei].axvline(kNBstar/np.pi, color=mycolors[Kvali], linestyle="dotted");
                    axes[sigmai,sourcei].set_title("$F(\hat{U},|\chi \\rangle)$");
                    axes[1,0].legend();
                if(vlines): axes[sourcei,sigmai].axvline(kNBstar/np.pi, color=mycolors[Kvali], linestyle="dotted");
                    
                # plot with final states (columns) summed together
                if(sourcei<n_mol_dof-1 and summed_columns):
                    if(Kvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    axes[sourcei,-1].plot(kNBvals/np.pi, np.sum(yvals_down[n_mol_dof*elecspin+sourcei,:,Kvali],axis=0), linestyle="solid", label=showlegstring+"$\sum_{\sigma_1 \sigma_2} \langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \hat{r} |\downarrow_e \sigma_1 \sigma_2 \\rangle$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[sourcei,-1].plot(kNBvals/np.pi, np.sum(yvals_up[n_mol_dof*elecspin+sourcei,:,Kvali],axis=0), linestyle="dashed", label=showlegstring+"$\sum_{\sigma_1 \sigma_2} \langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \hat{r} |\\uparrow_e \sigma_1 \sigma_2 \\rangle$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery, linewidth=mylinewidth);

                # formatting
                axes[sourcei,sigmai].set_title("$\langle"+str(ylabels[n_mol_dof*elecspin+sourcei])+"| \hat{r} |"+str(ylabels[n_mol_dof*elecspin+sigmai])+"\\rangle$");
                axes[sourcei,-1].set_yticks(the_ticks);
                axes[sourcei,-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks:
                    axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlim(0,kNBmax/np.pi);
                axes[-1,sigmai].set_xlabel('$k_i aN_B /\pi$',fontsize=myfontsize);
                   
    # show
    fig.suptitle("$s=${:.1f}, $J/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, VB/tl));
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([tl,myspinS,Jval,VB]);
    fname = "data/wfm_gate/NB/";
    #np.savetxt(fname+".txt", Kvals, header="[tl,myspinS,Jval,VB] =\n"+str(param_vals)+"\nKvals =");
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, yvals);

elif(case in [3,4]): # incident kinetic energy on the x axis
         # NB you STILL have to change NB
         # this is where another voltage might be useful!!!

    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);

    # iter over kNBvals (colors)
    kNBvals = np.pi*np.array([0.2,0.3,0.4,0.5]);
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(kNBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    Fvals_Uchi = np.empty((myxvals, len(kNBvals)),dtype=float);    
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

            # since we don't iter over sources, ALL sources must have 0 chem potential
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            for sourcei in range(n_loc_dof):
                assert(hblocks[0][sourcei,sourcei]==0.0);
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, rhat = True, all_debug = False);

            # fidelity w/r/t U, chi
            F_element = np.dot( np.conj(np.matmul(U_gate, chi_state)), np.matmul(rhatvals[:,:,Kvali,NBvali], chi_state));
            Fvals_Uchi[Kvali, NBvali] = np.sqrt( np.real( np.conj(F_element)*F_element ));
           
            #### end loop over sourcei
        #### end loop over E

        # determine fidelity and K*, ie x val where the SWAP happens
        yvals = np.copy(rhatvals);
        if(which_gate != "SQRT"): yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals));
        yvals_up = yvals[:,np.array(range(n_loc_dof))<n_mol_dof];
        yvals_down = yvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        Kstar = Kvals[np.argmax(Fvals_Uchi[:,NBvali])];
        print("Kstar/t, fidelity(kNBstar) = ",Kstar, np.max(Fvals_Uchi[:,NBvali]));
             
        # plot as a function of K
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                # plot rhat
                axes[sourcei,sigmai].plot(Kvals, np.real(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], label = "$k_i a N_B/\pi$ = {:.2f}".format(kNBvals[NBvali]/np.pi), color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[sourcei,sigmai].plot(Kvals, np.imag(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], linestyle="dashed", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                
                # plot fidelity, starred SWAP locations
                if(sourcei==2 and sigmai==1):
                    axes[sigmai,sourcei].plot(Kvals,Fvals_Uchi[:,NBvali], label = "$k_i a N_B/\pi$ = {:.2f}".format(kNBvals[NBvali]/np.pi),color=mycolors[NBvali]);
                    axes[sigmai,sourcei].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
                    axes[sigmai,sourcei].set_title("$F(\hat{U},|\chi \\rangle)$");
                    axes[1,0].legend();
                if(vlines): axes[sourcei,sigmai].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
                    
                # plot with final states (columns) summed together
                if(sourcei<n_mol_dof-1 and summed_columns):
                    if(Kvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    axes[sourcei,-1].plot(Kvals, np.sum(yvals_down[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="solid", label=showlegstring+"$\sum_{\sigma_1 \sigma_2} \langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \hat{r} |\downarrow_e \sigma_1 \sigma_2 \\rangle$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[sourcei,-1].plot(Kvals, np.sum(yvals_up[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="dashed", label=showlegstring+"$\sum_{\sigma_1 \sigma_2} \langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \hat{r} |\\uparrow_e \sigma_1 \sigma_2 \\rangle$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);

                # formatting
                axes[sourcei,sigmai].set_title("$\langle"+str(ylabels[n_mol_dof*elecspin+sourcei])+"| \hat{r} |"+str(ylabels[n_mol_dof*elecspin+sigmai])+"\\rangle$");
                axes[sourcei,-1].set_yticks(the_ticks);
                axes[sourcei,-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks:
                    axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlabel('$K_i/t$',fontsize=myfontsize);
                axes[-1,sigmai].set_xscale('log', subs = []);
                
    # show
    fig.suptitle("$s=${:.1f}, $J/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, VB/tl));
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([myspinS,tl,Jval,VB]);
    #fname = "data/wfm_swap/E/"+str((kNBvals/np.pi).round(1));
    #np.savetxt(fname+".txt", kNBvals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nkNBvals =");
    #np.save(fname+"_x", rhoJvals);
    #np.save(fname, yvals);

elif(case in[5,6]): # incident kinetic energy on the x axis
         # NB is now fixed !!!!

    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    if(case==5): K_indep = True;
    elif(case==6): K_indep = False; # whether to put Ki/t on x axis, alternatively wavenumber

    # iter over fixed NB (colors)
    NBvals = np.array([50,75,94,100]);
    Fvals_Uchi = np.empty((myxvals, len(NBvals)),dtype=float); 
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):

        # iter over incident kinetic energy (x axis)
        Kpowers = np.array([-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
                                              # note that at the right NB, R(SWAP) approaches 1 asymptotically at
                                              # lower Ki. But diminishing returns kick in around 10^-4
        Kvals = np.logspace(Kpowers[-1],Kpowers[0],num=myxvals);
        Energies = Kvals - 2*tl; # -2t < Energy < 2t, what I call E in paper
        knumbers = np.arccos(Energies/(-2*tl)); # wavenumbers
        for Kvali in range(len(Kvals)):

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

            # since we don't iter over sources, ALL sources must have 0 chem potential
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            for sourcei in range(n_loc_dof):
                assert(hblocks[0][sourcei,sourcei]==0.0);
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, rhat = True, all_debug = False);

            # fidelity w/r/t U, chi
            F_element = np.dot( np.conj(np.matmul(U_gate, chi_state)), np.matmul(rhatvals[:,:,Kvali,NBvali], chi_state));
            Fvals_Uchi[Kvali, NBvali] = np.sqrt( np.real( np.conj(F_element)*F_element ));
           
        #### end loop over Kvals

        # determine fidelity and K*, ie x val where the SWAP happens
        yvals = np.copy(rhatvals);
        if(which_gate != "SQRT"): yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals));
        yvals_up = yvals[:,np.array(range(n_loc_dof))<n_mol_dof];
        yvals_down = yvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        Kstar = Kvals[np.argmax(Fvals_Uchi[:,NBvali])];
        print("Kstar/t, fidelity(kNBstar) = ",Kstar, np.max(Fvals_Uchi[:,NBvali]));
                    
        # plot as a function of K
        if(K_indep): indep_var = Kvals; # what to put on x axis
        else: indep_var = knumbers*NBvals[NBvali]/np.pi;
        if(not K_indep): vlines = False;
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                # plot rhat
                axes[sourcei,sigmai].plot(indep_var, np.real(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]), color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                axes[sourcei,sigmai].plot(indep_var, np.imag(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], linestyle="dashed", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);

                # plot fidelity, starred SWAP locations
                if(sourcei==2 and sigmai==1):
                    axes[sigmai,sourcei].plot(Kvals,Fvals_Uchi[:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali]);
                    axes[sigmai,sourcei].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
                    axes[sigmai,sourcei].set_title("$F(\hat{U},|\chi \\rangle)$");
                    axes[1,0].legend();
                if(vlines): axes[sourcei,sigmai].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
 
                # plot reflection summed over final states (columns)
                if(sourcei<n_mol_dof-1 and summed_columns):
                    Rvals_up = np.real(np.conj(rhatvals)*rhatvals)[:,np.array(range(n_loc_dof))<n_mol_dof];
                    Rvals_down = np.real(np.conj(rhatvals)*rhatvals)[:,np.array(range(n_loc_dof))>=n_mol_dof];
                    if(NBvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    axes[sourcei,-1].plot(indep_var, np.sum(Rvals_down[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="solid", label=showlegstring+"Total $R(\\rightarrow \downarrow_e)$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[sourcei,-1].plot(indep_var, np.sum(Rvals_up[n_mol_dof*elecspin+sourcei,:,:,NBvali],axis=0), linestyle="dashed", label=showlegstring+"Total $R(\\rightarrow \\uparrow_e)$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery, linewidth=mylinewidth);
                    axes[0,-1].legend();
                
                # formatting
                axes[sourcei,sigmai].set_title("$\langle"+str(ylabels[n_mol_dof*elecspin+sourcei])+"| \hat{r} |"+str(ylabels[n_mol_dof*elecspin+sigmai])+"\\rangle$");
                axes[sourcei,-1].set_yticks(the_ticks);
                axes[sourcei,-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks:
                    axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                if(K_indep): 
                    axes[-1,sigmai].set_xlabel('$K_i/t$',fontsize=myfontsize);
                    axes[-1,sigmai].set_xscale('log', subs = []);
                else:
                    axes[-1,sigmai].set_xlabel('$k_i a N_B/\pi$',fontsize=myfontsize);
                  
    # show
    fig.suptitle("$s=${:.1f}, $J/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, VB/tl));
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([myspinS,tl,Jval]);
    #fname = "data/wfm_swap/E/"+str((kNBvals/np.pi).round(1));
    #np.savetxt(fname+".txt", kNBvals, header="[tl, tp, JK, J12, Dval, myspinS, n_loc_dof] =\n"+str(param_vals)+"\nkNBvals =");
    #np.save(fname+"_x", rhoJvals);
    #np.save(fname, Rvals);

