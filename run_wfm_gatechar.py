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


def get_U_gate(which_gate):
    if(which_gate=="SQRT"):
        U_q = np.array([[1,0,0,0],
                           [0,complex(0.5,0.5),complex(0.5,-0.5),0],
                           [0,complex(0.5,-0.5),complex(0.5,0.5),0],
                           [0,0,0,1]], dtype=complex); #  SWAP^1/2 gate
    elif(which_gate=="SWAP"):
        U_q = np.array([[1,0,0,0],
                       [0,0,1,0],
                       [0,1,0,0],
                       [0,0,0,1]], dtype=complex); # SWAP gate
    elif(which_gate=="I"):
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
    return U_gate;
          
#########################################################
#### barrier in right lead for total reflection

#### top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 5;
final_plots = False;
if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
vlines = False;
case = sys.argv[2];
elecspin = 0; # itinerant e is spin up

# fig standardizing
myxvals = 29; myfigsize = (4,2); myfontsize = 14;
if(final_plots): myxvals = 99; myfigsize = (8,4); myfontsize = 18;
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X","+"];
mymarkevery = (myxvals//3, myxvals//3);
mypanels = ["(a)","(b)","(c)","(d)"];
the_ticks = [0.0,1.0];
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
n_chi_states = (3,4*8);

if(case in ["NB","kNB"]): # at fixed Ki, as a function of NB,
         # minimize over a set of states \chi_k
         # for each gate of interest

    # axes
    gates = ["SQRT","SWAP","I"];
    gate_strs = ["$\mathbf{U} = \mathbf{U}_{SQRT}$","$\mathbf{U} = \mathbf{U}_{SWAP}$","$\mathbf{U} = \mathbf{I}$"];
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="NB"): NB_indep = True # whether to put NB, alternatively wavenumber*NB
    elif(case=="kNB"): NB_indep = False;

    # iter over gates
    for gatevali in range(len(gates)):
        U_gate = get_U_gate(gates[gatevali]);

        # iter over incident kinetic energy (colors)
        Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
        Kvals = np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers)); # Kval > 0 always, what I call K_i in paper
        kvals = np.arccos((Kvals-2*tl)/(-2*tl)); # k corresponding to fixed energy
        Fvals_min = np.empty((len(Kvals), myxvals),dtype=float); # fidelity min'd over chi states
        for Kvali in range(len(Kvals)):

            # energy
            Energy = Kvals[Kvali] - 2*tl; # -2t < Energy < 2t, what I call E in paper
            k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed energy
                   
            # iter over barrier distance (x axis)
            kNBmax = 0.75*np.pi;
            NBmax = int(kNBmax/k_rho);
            if(NB_indep): NBmax = 150;
            NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
            if(verbose): print("2*NBmax = ",2*NBmax); 
            
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
                rhat = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, rhat = True, all_debug = False);

                # fidelity minimized over chi states
                M_matrix = np.matmul(np.conj(U_gate.T), rhat);
                trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
                if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
                Fvals_min[Kvali, NBvali] = np.real(trace_fidelity);
                if(verbose>4): print(Kvali, NBvals[NBvali], kvals[Kvali]*NBvals[NBvali]/np.pi, Fvals_min[Kvali, NBvali]);
                
            #### end loop over NB

            # determine fidelity and kNB*, ie x val where the SWAP happens
            if(NB_indep): xvals = 2*NBvals;
            else: xvals = 2*kvals[Kvali]*NBvals/np.pi;
            xstar = xvals[np.argmax(Fvals_min[Kvali])];
            if(verbose): print("NBstar, fidelity(NBstar) = ",xstar, np.max(Fvals_min[Kvali]));

            # plot formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.2+the_ticks[0],0.2+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[gatevali].set_xlim(0,np.max(xvals));
            if(NB_indep): axes[-1].set_xlabel('$2N_B $',fontsize=myfontsize);
            else: axes[-1].set_xlabel('$2 k_i a N_B/\pi $',fontsize=myfontsize);
            axes[gatevali].annotate(gate_strs[gatevali], (xvals[0],1.01),fontsize=myfontsize);
            if(gates[gatevali] == "I"): axes[gatevali].set_title("$\mathbf{U} = \mathbf{I}$");
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{r}, \mathbf{U})$");
 
            # plot fidelity, starred SWAP locations, as a function of NB
            axes[gatevali].plot(xvals,Fvals_min[Kvali], label = "$K_i/t= 10^{"+str(Kpowers[Kvali])+"}$",color=mycolors[Kvali],marker=mymarkers[1+Kvali],markevery=mymarkevery);
            if(vlines): axes[gatevali].axvline(xstar, color=mycolors[Kvali], linestyle="dotted");           

        #### end loop over Ki

    #### end loop over gates
            
    # show
    suptitle = "$s=${:.1f}, $J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, V0/tl, VB/tl);
    if(final_plots): suptitle = "$J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle);
    plt.tight_layout();
    if(final_plots): # save legend
        fname = "figs/gate/F_vs_NB";
        plt.savefig(fname+".pdf")
        fig_leg = plt.figure()
        fig_leg.set_size_inches(3/2,3/2)
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        ax_leg.legend(*axes[-1].get_legend_handles_labels(), loc='center')
        # hide the axes frame and the x/y labels
        ax_leg.axis('off')
        plt.savefig(fname+"_legend.pdf");
    else:
        axes[-1].legend();
        plt.show();

    # save data
    #np.savetxt(fname+".txt", Kvals, header=suptitle)
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, yvals);

if(case in ["ki", "K"]): # at fixed NB, as a function of Ki,
         # minimize over a set of states \chi_k
         # for each gate of interest

    # axes
    gates = ["SQRT","SWAP","I"];
    gate_strs = ["$\mathbf{U} = \mathbf{U}_{SQRT}$","$\mathbf{U} = \mathbf{U}_{SWAP}$","$\mathbf{U} = \mathbf{I}$"];  
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = np.array([axes]);
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="ki"): K_indep = False;
    elif(case=="K"): K_indep = True; # whether to put Ki/t on x axis, alternatively wavenumber
   
    # iter over gates
    for gatevali in range(len(gates)):
        U_gate = get_U_gate(gates[gatevali]);

        # iter over barrier distance (colors)
        NBvals = np.array([50,100,150]);
        Fvals_min = np.empty((myxvals, len(NBvals)),dtype=float); # fidelity min'd over chi states
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # iter over incident kinetic energy (x axis)
            Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
            Kvals = np.logspace(Kpowers[0],Kpowers[-1],num=myxvals); # Kval > 0 always, what I call K_i in paper
            kvals = np.arccos((Kvals-2*tl)/(-2*tl)); # k corresponding to fixed energy
            for Kvali in range(len(Kvals)):

                # energy
                Energy = Kvals[Kvali] - 2*tl; # -2t < Energy < 2t, what I call E in paper

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
                rhat = wfm.kernel(hblocks, tnn, tnnn, tl, Energy , source, rhat = True, all_debug = False);

                # fidelity minimized over chi states
                M_matrix = np.matmul(np.conj(U_gate.T), rhat);
                trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
                if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
                Fvals_min[Kvali, NBvali] = np.real(trace_fidelity);
                if(verbose>4): print(Kvali, NBvals[NBvali], kvals[Kvali]*NBvals[NBvali]/np.pi, Fvals_min[Kvali, NBvali]);
                
            #### end loop over Ki

            # determine fidelity and kNB*, ie x val where the SWAP happens
            if(K_indep): xvals = Kvals;
            xstar = xvals[np.argmax(Fvals_min[:,NBvali])];
            print("Kstar, fidelity(Kstar) = ",xstar, np.max(Fvals_min[:,NBvali]));

            # plot formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            if(K_indep):
                axes[-1].set_xlabel('$K_i/t$',fontsize=myfontsize);
                axes[-1].set_xscale('log', subs = []);
            else:
                axes[-1].set_xlabel('$2k_i a N_B/\pi$',fontsize=myfontsize);
            axes[gatevali].annotate(gate_strs[gatevali], (xvals[-1],1.01),fontsize=myfontsize);
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{r},\mathbf{U})$",fontsize=myfontsize);
                
            # plot fidelity, starred SWAP locations, as a function of NB
            axes[gatevali].plot(xvals,Fvals_min[:,NBvali], label = "$2N_B = ${:.0f}".format(2*NBvals[NBvali]),color=mycolors[NBvali],marker=mymarkers[1+NBvali],markevery=mymarkevery);
            if(vlines): axes[gatevali].axvline(xstar, color=mycolors[NBvali], linestyle="dotted");

        #### end loop over NB

    #### end loop over gates
            
    # show
    suptitle = "$s=${:.1f}, $J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, V0/tl, VB/tl);
    if(final_plots): suptitle = "$J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle, fontsize=myfontsize);
    plt.tight_layout();
    if(final_plots): # save legend
        fname = "figs/gate/F_vs_K";
        plt.savefig(fname+".pdf")
        fig_leg = plt.figure()
        fig_leg.set_size_inches(3/2,3/2)
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        ax_leg.legend(*axes[-1].get_legend_handles_labels(), loc='center')
        # hide the axes frame and the x/y labels
        ax_leg.axis('off')
        plt.savefig(fname+"_legend.pdf");
    else:
        axes[-1].legend();
        plt.show();

    # save data
    #np.savetxt(fname+".txt", NBvals, header=suptitle)
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, yvals);
    
if(case in ["chik_alpha", "chik_beta"]): # at fixed NB and fixed Ki
                      # iter over all chi_k

    def sample_chik_space(a_samples, b_samples):
        for samples in [a_samples, b_samples]:
            if( not isinstance(samples, int)): raise TypeError;
        # mesh values
        a1_vals = np.linspace(0,1,a_samples);
        b2_vals = np.linspace(0,2*np.pi, 1+b_samples)[:-1];
        # return var
        chis = np.zeros((len(a1_vals),len(b2_vals), 8),dtype=complex);
        # iter over alpha1
        for a1vali in range(len(a1_vals)):
            # iter over beta2
            for b2vali in range(len(b2_vals)):
                # input state to measure fidelity for
                a1val = np.sqrt(a1_vals[a1vali]);
                a2val = np.sqrt(1-a1val*a1val);
                # itinerant e is spin up
                chi = np.array([a1val,a2val,a2val*np.exp(complex(0,b2_vals[b2vali])),a1val,0,0,0,0]);
                chis[a1vali,b2vali] = chi/np.sqrt(np.dot(np.conj(chi), chi)); # normalize
        return a1_vals, b2_vals, chis;           
 
    # axes
    gates = ["SQRT","SWAP","I"];
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
     

    # NB and Ki should be best for that gate, but not change within gate
    delta_NB = -0; # detuning
    NBvals = delta_NB + np.array([100,100,100]);
    Kpowers = np.array([-5,-4,-3]);
    #Kpowers = np.array([-3,-3,-3]);
    Kvals = np.power(10.0,Kpowers);
    n_chi_states = (3,4*8);
    if(case == "chik_alpha"): n_chi_states = (n_chi_states[-1]+1, n_chi_states[0]+1);
    
    # iter over gates
    for gatevali in range(len(gates)):
        U_gate = get_U_gate(gates[gatevali]);

        # fixed barrier distance and kinetic energy
        NBval = NBvals[gatevali];

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
        print("\nhblocks = \n",np.real(hblocks));

        # since we don't iter over sources, ALL sources must have 0 chem potential
        source = np.zeros((n_loc_dof,));
        source[-1] = 1;
        for sourcei in range(n_loc_dof):
            assert(hblocks[0][sourcei,sourcei]==0.0);
                
        # get reflection operator
        rhat = wfm.kernel(hblocks, tnn, tnnn, tl, Kvals[gatevali] - 2*tl, source, rhat = True, all_debug = False);

        # store fidelity results
        Fvals_alphabeta = np.zeros(n_chi_states, dtype=float);

        # iter over alpha1 values 
        alpha1_vals, beta2_vals, chi_states = sample_chik_space(*n_chi_states);
        for alpha1_vali in range(len(alpha1_vals)):

            # iter over beta2 values 
            for beta2_vali in range(len(beta2_vals)):

                # fidelity
                F_element = np.dot( np.conj(np.matmul(U_gate, chi_states[alpha1_vali,beta2_vali])), np.matmul(rhat, chi_states[alpha1_vali,beta2_vali]));
                if( np.imag(np.conj(F_element)*F_element)>1e-10): print(np.conj(F_element)*F_element); assert False;
                Fvals_alphabeta[alpha1_vali,beta2_vali] = np.sqrt( np.real( np.conj(F_element)*F_element));

        #### end loop over alpha1, beta2

        # plot as colors and x axis
        if(case=="chik_beta"):
            colorvals, xvals, yvals = np.sqrt(alpha1_vals), beta2_vals/np.pi, Fvals_alphabeta;
            xlabel, plotlabel = "$\\beta_2 / \pi$", "$\\alpha = ${:.2f}";
        else:
            colorvals, xvals, yvals = beta2_vals/np.pi, np.sqrt(alpha1_vals), Fvals_alphabeta.T;
            xlabel, plotlabel = "$\\alpha$", "$\\beta_2 / \pi =$ {:.2f}";
        for colorvali in range(len(colorvals)):
                
            # plot formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[-1].set_xlabel(xlabel,fontsize=myfontsize);
            axes[gatevali].set_title("$\mathbf{U} = \mathbf{U}_{"+gates[gatevali]+"}$"+", $N_B =${:.0f}".format(NBval)+", $K_i/t = 10^{"+str(Kpowers[gatevali])+"}$");
            if(gates[gatevali] == "I"): axes[gatevali].set_title("$\mathbf{U} = \mathbf{I}$"+", $N_B =${:.0f}".format(NBval)+", $K_i/t = 10^{"+str(Kpowers[gatevali])+"}$");
            axes[gatevali].set_ylabel("$F_{min}(\mathbf{r},\mathbf{U})$");
                
            # plot fidelity vs chi_k angles
            axes[gatevali].plot(xvals,yvals[colorvali], label = plotlabel.format(colorvals[colorvali]),color=mycolors[colorvali],marker=mymarkers[1+colorvali],markevery=mymarkevery);

    #### end loop over gates

    # show
    if(final_plots): # do something else with legend
        legfig, legax = plt.subplots()
    else: axes[-1].legend();
    suptitle = "$s=${:.1f}, $J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, V0/tl, VB/tl);
    if(final_plots): suptitle = "$J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle);
    plt.tight_layout();
    plt.show();
