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
           
#########################################################
#### barrier in right lead for total reflection

#### top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 1;
final_plots=False;
if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
vlines = not final_plots; # whether to highlight certain x vals with vertical dashed lines
summed_columns = True;
case = sys.argv[2];
elecspin = 0; # itinerant e is spin up

# fig standardizing
myxvals = 29;
if(final_plots): myxvals = 99;
myfontsize = 14;
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X","+"];
mymarkevery = (myxvals//3, myxvals//3);
mypanels = ["(a)","(b)","(c)","(d)"];
ylabels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
    "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"];
 

# tight binding params
tl = 1.0;
myspinS = 0.5;
n_mol_dof = int((2*myspinS+1)**2);
n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
Jval = -0.2*tl;
VB = 5.0*tl;
V0 = 0.0*tl; # just affects title, not implemented physically

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
print("U_gate =\n",U_gate);

if(case in ["NB","kNB"]): # distance of the barrier NB on the x axis
    
    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex='col', sharey = 'row');
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    if(case=="NB"): NB_indep = True;
    elif(case=="kNB"): NB_indep = False # whether to put NB, alternatively wavenumber*NB

    # iter over incident kinetic energy (colors)
    Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
    Kvals = np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers));
    rhatvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals),dtype=complex); # by  init spin, final spin, energy, NB
    Fvals_Uchi = np.empty((len(Kvals), myxvals),dtype=float);
    for Kvali in range(len(Kvals)):

        # energy
        Kval = Kvals[Kvali]; # Kval > 0 always, what I call K_i in paper
        Energy = Kval - 2*tl; # -2t < Energy < 2t, what I call E in paper
        k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed energy
        
        # iter over barrier distance (x axis)
        kNBmax = 0.75*np.pi;
        NBmax = int(kNBmax/k_rho);
        if(NB_indep): NBmax = 150;
        if(verbose): print("2*NBmax = ",2*NBmax); 
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
            M_matrix = np.matmul(np.conj(U_gate.T), rhatvals[:,:,Kvali,NBvali]);
            trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
            if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
            Fvals_Uchi[Kvali, NBvali] = np.real(trace_fidelity);
            if(verbose>4): print(Kvali, 2*NBvals[NBvali], 2*kNBvals[NBvali]/np.pi, Fvals_Uchi[Kvali, NBvali]);
            
        #### end loop over NB

        # determine fidelity and kNB*, ie x val where the SWAP happens
        rhatvals_up = rhatvals[:,np.array(range(n_loc_dof))<n_mol_dof]; # ref'd is e up
        rhatvals_down = rhatvals[:,np.array(range(n_loc_dof))>=n_mol_dof]; # ref'd is e down
        yvals = np.copy(rhatvals); rbracket = "";
        if(which_gate != "SQRT"): # make everything real
            rbracket = "|"
            yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals)); 
        if(NB_indep): indep_var = 2*NBvals; # what to put on x axis
        else: indep_var = 2*kNBvals/np.pi;
        indep_star = indep_var[np.argmax(Fvals_Uchi[Kvali])];
        if(verbose): print("xstar, fidelity(xstar) = "+str(indep_star)+", {:.4f}".format(np.max(Fvals_Uchi[Kvali])));
        if(True and Kvali==1):
            the_sourceindex, the_NBindex = 1, np.argmax(Fvals_Uchi[Kvali]);
            the_y, the_x = np.imag(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]), np.real(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]);
            comp_phase = 2*np.arctan( the_y/(the_x+np.sqrt(the_x*the_x+the_y*the_y)));
            print(yvals[:,:,Kvali,the_NBindex])
            print(rhatvals[:,:,Kvali,the_NBindex])
            print(np.exp(complex(0,-comp_phase))*rhatvals[:,:,Kvali,the_NBindex])
            assert False;
            
        # plot as a function of NBvals
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                # formatting
                axes[sourcei,sigmai].set_title("$"+rbracket+"\langle"+str(ylabels[n_mol_dof*elecspin+sourcei])+"| \mathbf{r} |"+str(ylabels[n_mol_dof*elecspin+sigmai])+"\\rangle"+rbracket+"$");
                axes[sourcei,-1].set_yticks(the_ticks);
                axes[sourcei,-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks: axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlim(0,np.max(indep_var));
                if(NB_indep):
                    axes[-1,sigmai].set_xlabel('$2N_B$',fontsize=myfontsize);
                else:
                    axes[-1,sigmai].set_xlabel('$2k_i aN_B /\pi$',fontsize=myfontsize);
 
                # plot rhat
                axes[sourcei,sigmai].plot(indep_var, np.real(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,Kvali], label = "$K_i/t= 10^{"+str(Kpowers[Kvali])+"}$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery);
                axes[sourcei,sigmai].plot(indep_var, np.imag(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,Kvali], linestyle="dashed", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery);
                
                # plot fidelity, starred SWAP locations
                if(sourcei==2 and sigmai==1):
                    axes[sigmai,sourcei].plot(indep_var,Fvals_Uchi[Kvali], label = "$K_i/t= 10^{"+str(Kpowers[Kvali])+"}$",color=mycolors[Kvali],marker=mymarkers[1+Kvali],markevery=mymarkevery);
                    axes[sigmai,sourcei].axvline(indep_star, color=mycolors[Kvali], linestyle="dotted");
                    for tick in the_ticks: axes[sigmai,sourcei].axhline(tick,color='lightgray',linestyle='dashed');
                    axes[sigmai,sourcei].set_title("$F_{avg}(\mathbf{r}, \mathbf{U}_{"+which_gate+"})$");
                    axes[1,0].legend();
                if(vlines): axes[sourcei,sigmai].axvline(indep_star, color=mycolors[Kvali], linestyle="dotted");
                    
                # plot with final states (columns) summed together
                if((sourcei<n_mol_dof-1 and sigmai==sourcei) and (which_gate=="SWAP" and summed_columns)):
                    if(Kvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    for tick in the_ticks: axes[sourcei,-1].axhline(tick,color='lightgray',linestyle='dashed');
                    # for this row (incoming state), sum over all columns (outgoing states) with given \sigma_e
                    #axes[sourcei,-1].plot(indep_var, np.sum(np.real(np.conj(rhatvals_up[n_mol_dof*elecspin+sourcei,:,Kvali])*rhatvals_up[n_mol_dof*elecspin+sourcei,:,Kvali]),axis=0), linestyle="solid", label=showlegstring+"$\sum_{\sigma_1 \sigma_2} |\langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \mathbf{r} |\\uparrow_e \sigma_1 \sigma_2 \\rangle |^2$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery);
                    axes[sourcei,-1].plot(indep_var, np.sum(np.real(np.conj(rhatvals_down[n_mol_dof*elecspin+sourcei,:,Kvali])*rhatvals_down[n_mol_dof*elecspin+sourcei,:,Kvali]),axis=0), linestyle="dashed", label=showlegstring+"$\sum_{\sigma_1 \sigma_2}| \langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \mathbf{r} |\downarrow_e \sigma_1 \sigma_2 \\rangle |^2$", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery);     
                    axes[0,-1].legend();
                    
    # show
    suptitle = which_gate+" gate, $s=${:.1f}, $J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle);
    plt.tight_layout();
    plt.show();

    # save data
    fname = "data/wfm_gate/"+which_gate+"/"+case+"/";
    #np.savetxt(fname+".txt", Kvals, header=suptitle);
    #np.save(fname+"_x", indep_vars);
    #np.save(fname+"_y", yvals);

elif(case in["ki","K"]): # incident kinetic energy on the x axis
         # NB is now fixed !!!!

    # axes
    nrows, ncols = n_mol_dof, n_mol_dof;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    if(case=="ki"): K_indep = False;
    elif(case=="K"): K_indep = True; # whether to put Ki/t on x axis, alternatively wavenumber

    # iter over fixed NB (colors)
    NBvals = np.array([50,100,131,150]);
    Fvals_Uchi = np.empty((myxvals, len(NBvals)),dtype=float); 
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):

        # set barrier distance
        NBval = int(NBvals[NBvali])
        if(verbose): print("NB = ",NBval); 

        # iter over incident kinetic energy (x axis)
        Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
                                              # note that at the right NB, R(SWAP) approaches 1 asymptotically at
                                              # lower Ki. But diminishing returns kick in around 10^-4
        Kvals = np.logspace(Kpowers[-1],Kpowers[0],num=myxvals);
        Energies = Kvals - 2*tl; # -2t < Energy < 2t, what I call E in paper
        knumbers = np.arccos(Energies/(-2*tl)); # wavenumbers
        for Kvali in range(len(Kvals)):

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
            M_matrix = np.matmul(np.conj(U_gate.T), rhatvals[:,:,Kvali,NBvali]);
            trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
            if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
            Fvals_Uchi[Kvali, NBvali] = np.real(trace_fidelity);

        #### end loop over Kvals

        # determine fidelity and K*, ie x val where the SWAP happens
        rhatvals_up = rhatvals[:,np.array(range(n_loc_dof))<n_mol_dof]; # ref'd is e up
        rhatvals_down = rhatvals[:,np.array(range(n_loc_dof))>=n_mol_dof]; # ref'd is e down
        yvals = np.copy(rhatvals); rbracket = "";
        if(which_gate != "SQRT"): # make everything real
            rbracket = "|"
            yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals)); 
        if(K_indep): indep_var = Kvals; # what to put on x axis
        else: indep_var = 2*knumbers*NBvals[NBvali]/np.pi;
        Kstar = indep_var[np.argmax(Fvals_Uchi[:,NBvali])];
        if(verbose): print("Kstar/t, fidelity(kNBstar) = ",Kstar, ", {:.4f}".format(np.max(Fvals_Uchi[:,NBvali])));
        if(False and Kvali==1):
            the_sourceindex, the_NBindex = 1, np.argmax(Fvals_Uchi[Kvali]);
            the_y, the_x = np.imag(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]), np.real(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]);
            comp_phase = 2*np.arctan( the_y/(the_x+np.sqrt(the_x*the_x+the_y*the_y)));
            print(yvals[:,:,Kvali,the_NBindex])
            print(rhatvals[:,:,Kvali,the_NBindex])
            print(np.exp(complex(0,-comp_phase))*rhatvals[:,:,Kvali,the_NBindex])
            assert False;
            
        # plot as a function of K
        for sourcei in range(n_mol_dof):
            for sigmai in range(sourcei+1):
                # formatting
                axes[sourcei,sigmai].set_title("$"+rbracket+"\langle"+str(ylabels[n_mol_dof*elecspin+sourcei])+"| \mathbf{r} |"+str(ylabels[n_mol_dof*elecspin+sigmai])+"\\rangle"+rbracket+"$");
                axes[sourcei,-1].set_yticks(the_ticks);
                axes[sourcei,-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks: axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                if(K_indep): 
                    axes[-1,sigmai].set_xlabel('$K_i/t$',fontsize=myfontsize);
                    axes[-1,sigmai].set_xscale('log', subs = []);
                else:
                    axes[-1,sigmai].set_xlabel('$2k_i a N_B/\pi$',fontsize=myfontsize);
 
                # plot rhat
                axes[sourcei,sigmai].plot(indep_var, np.real(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], label = "$2N_B$ = {:.0f}".format(2*NBvals[NBvali]), color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery);
                axes[sourcei,sigmai].plot(indep_var, np.imag(yvals)[n_mol_dof*elecspin+sourcei,n_mol_dof*elecspin+sigmai,:,NBvali], linestyle="dashed", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery);

                # plot fidelity, starred SWAP locations
                if(sourcei==2 and sigmai==1):
                    axes[sigmai,sourcei].plot(indep_var,Fvals_Uchi[:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali]);
                    axes[sigmai,sourcei].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
                    for tick in the_ticks: axes[sigmai,sourcei].axhline(tick,color='lightgray',linestyle='dashed');
                    axes[sigmai,sourcei].set_title("$F_{avg}(\mathbf{r},\mathbf{U}_{"+which_gate+"})$");
                    axes[1,0].legend();
                if(vlines): axes[sourcei,sigmai].axvline(Kstar, color=mycolors[NBvali], linestyle="dotted");
 
                # plot reflection summed over final states (columns)
                if(sourcei<n_mol_dof-1 and summed_columns):
                    if(NBvali==0): showlegstring = ""; 
                    else: showlegstring = "_"; # hides duplicate labels
                    for tick in the_ticks: axes[sourcei,-1].axhline(tick,color='lightgray',linestyle='dashed');
                    # for this row (incoming state), sum over all columns (outgoing states) with given \sigma_e
                    #axes[sourcei,-1].plot(indep_var, np.sum(np.real(np.conj(rhatvals_up[n_mol_dof*elecspin+sourcei,:,:,NBvali])*rhatvals_up[n_mol_dof*elecspin+sourcei,:,:,NBvali]),axis=0), linestyle="solid", label=showlegstring+"$\sum_{\sigma_1 \sigma_2} |\langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \mathbf{r} |\\uparrow_e \sigma_1 \sigma_2 \\rangle |^2$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery);
                    axes[sourcei,-1].plot(indep_var, np.sum(np.real(np.conj(rhatvals_down[n_mol_dof*elecspin+sourcei,:,:,NBvali])*rhatvals_down[n_mol_dof*elecspin+sourcei,:,:,NBvali]),axis=0), linestyle="dashed", label=showlegstring+"$\sum_{\sigma_1 \sigma_2}| \langle"+str(ylabels[n_mol_dof*elecspin+sigmai])+"| \mathbf{r} |\downarrow_e \sigma_1 \sigma_2 \\rangle |^2$", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery);
                    axes[0,-1].legend();
                         
    # show
    suptitle = "$s=${:.1f}, $J/t=${:.2f}, $V_0/tl={:.2f}$, $V_B/t=${:.2f}".format(myspinS, Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle);
    plt.tight_layout();
    plt.show();

    # save data
    fname = "data/wfm_gate/"+which_gate+"/"+case+"/";
    #np.savetxt(fname+".txt", Kvals, header=suptitle)
    #np.save(fname+"_x", indep_vars);
    #np.save(fname+"_y", yvals);



























