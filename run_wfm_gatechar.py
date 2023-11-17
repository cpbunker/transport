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

def sample_chi_space(space_size, n_samples):
    if( not isinstance(space_size, int)): raise TypeError;
    if( not isinstance(n_samples, int)): raise TypeError;

    # fill randomly
    rand_generator = np.random.default_rng(seed=11152023);
    ret = np.zeros((n_samples, space_size), dtype=complex);
    for n in range(n_samples):
        rpart = rand_generator.random((space_size,));
        ipart = rand_generator.random((space_size,));
        bothparts = rpart+complex(0,1)*ipart;
        ret[n] = bothparts/np.sqrt(np.dot(np.conj(bothparts),bothparts));

    return ret;

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
the_ticks = [0.0,1.0];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
vlines = True; # whether to highlight certain x vals with vertical dashed lines
case = int(sys.argv[2]);

# tight binding params
tl = 1.0;
myspinS = 0.5;
n_mol_dof = int((2*myspinS+1)**2);
n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
Jval = -0.2*tl;
VB = 5.0*tl;

# construct chi states
n_udef_states, n_rand_states = 6, 4; # user defined and randomly sampled states
chi_states_udef = np.zeros((n_udef_states, n_mol_dof), dtype=complex);
chi_states_udef[:n_mol_dof] = np.eye(n_mol_dof);
chi_states_udef[n_mol_dof] = np.array([1,1,1,1]);
chi_states_udef[n_mol_dof+1] = np.array([1,1,-1,1]);
for chivali in range(n_udef_states): # normalize
    chi_states_udef[chivali] = chi_states_udef[chivali]/np.sqrt( np.dot(np.conj(chi_states_udef[chivali]),chi_states_udef[chivali]));
chi_states = np.append(chi_states_udef, sample_chi_space(n_mol_dof, n_rand_states),axis=0);
chi_states = chi_states[n_mol_dof:n_mol_dof+2];
print("chi states =\n", chi_states);
# broadcast 2 qubit chi states to 2 qubit + electron space
new_chi_states = np.zeros( (np.shape(chi_states)[0]*2, np.shape(chi_states)[1]*2), dtype=complex);
for chivali in range(len(chi_states)):
    for spin in [0,1]:
        new_chi_states[2*chivali+spin][spin*n_mol_dof:(1+spin)*n_mol_dof] = chi_states[chivali];
chi_states = new_chi_states;

if(case in [1,2]): # at fixed Ki, as a function of NB,
         # minimize over a set of states \chi_k
         # for each gate of interest

    # axes
    gates = ["SQRT","SWAP","I"];
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    if(case==1): NB_indep = False;
    elif(case==2): NB_indep = True # whether to put NB, alternatively wavenumber*NB

    # iter over gates
    for gatevali in range(len(gates)):
        U_gate = get_U_gate(gates[gatevali]);

        # iter over incident kinetic energy (colors)
        Kpowers = np.array([-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
        Kvals = np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers)); # Kval > 0 always, what I call K_i in paper
        kvals = np.arccos((Kvals-2*tl)/(-2*tl)); # k corresponding to fixed energy
        Fvals_min = np.empty((len(Kvals), myxvals),dtype=float); # fidelity min'd over chi states
        which_chi_min = np.empty((len(Kvals), myxvals),dtype=int); # index of chi states which min'd the fidelity
        for Kvali in range(len(Kvals)):

            # energy
            Energy = Kvals[Kvali] - 2*tl; # -2t < Energy < 2t, what I call E in paper
            k_rho = np.arccos(Energy/(-2*tl)); # k corresponding to fixed energy
                   
            # iter over barrier distance (x axis)
            if(NB_indep):
                NBmax = 100;
            else:
                kNBmax = 0.5*np.pi;
                NBmax = int(kNBmax/k_rho);
            NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
            if(verbose): print("NBmax = ",NBmax); 
            
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
                Fvals_chi = np.zeros((len(chi_states),), dtype=float);
                for chivali in range(len(chi_states)):
                    F_element = np.dot( np.conj(np.matmul(U_gate, chi_states[chivali])), np.matmul(rhat, chi_states[chivali]));
                    Fvals_chi[chivali] = np.sqrt( np.real( np.conj(F_element)*F_element ));
                Fvals_min[Kvali, NBvali] = np.min(Fvals_chi);
                which_chi_min[Kvali, NBvali] = np.argmin(Fvals_chi);
                print(Kvali, NBvals[NBvali], kvals[Kvali]*NBvals[NBvali]/np.pi, Fvals_min[Kvali, NBvali], which_chi_min[Kvali, NBvali]);
                
            #### end loop over NB

            # determine fidelity and kNB*, ie x val where the SWAP happens
            if(NB_indep): xvals = NBvals;
            else: xvals = kvals[Kvali]*NBvals/np.pi;
            xstar = xvals[np.argmax(Fvals_min[Kvali])];
            print("NBstar, fidelity(NBstar) = ",xstar, np.max(Fvals_min[Kvali]));
                  
            # plot fidelity, starred SWAP locations, as a function of NB
            axes[gatevali].plot(xvals,Fvals_min[Kvali], label = "$K_i/t= 10^{"+str(Kpowers[Kvali])+"}$",color=mycolors[Kvali]);
            axes[gatevali].set_title("$F("+gates[gatevali]+")$");
            if(vlines): axes[gatevali].axvline(xstar, color=mycolors[Kvali], linestyle="dotted");

            # formatting
            axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[gatevali].set_xlim(0,np.max(xvals));
            axes[-1].set_xlabel('$N_B $',fontsize=myfontsize);
            #axes[0].legend();

        #### end loop over Ki
            
    # show
    fig.suptitle("$s=${:.1f}, $J/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, VB/tl));
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([tl,myspinS,Jval,VB]);
    fname = "data/wfm_gatechar/NB/";
    #np.savetxt(fname+".txt", Kvals, header="[tl,myspinS,Jval,VB] =\n"+str(param_vals)+"\nKvals =");
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, yvals);

if(case in [3,4]): # at fixed Ki, as a function of NB,
         # minimize over a set of states \chi_k
         # for each gate of interest

    # axes
    gates = ["SQRT","SWAP","I"];
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);

    # iter over gates
    for gatevali in range(len(gates)):
        U_gate = get_U_gate(gates[gatevali]);

        # iter over barrier distance (colors)
        NBvals = np.array([50,75,94,100]);
        #NBvals = np.array([80,85,90,95,100]);
        Fvals_min = np.empty((myxvals, len(NBvals)),dtype=float); # fidelity min'd over chi states
        which_chi_min = np.empty((myxvals, len(NBvals)),dtype=int); # index of chi states which min'd the fidelity
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # iter over incident kinetic energy (x axis)
            Kpowers = np.array([-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
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
                Fvals_chi = np.zeros((len(chi_states),), dtype=float);
                for chivali in range(len(chi_states)):
                    F_element = np.dot( np.conj(np.matmul(U_gate, chi_states[chivali])), np.matmul(rhat, chi_states[chivali]));
                    Fvals_chi[chivali] = np.sqrt( np.real( np.conj(F_element)*F_element ));
                Fvals_min[Kvali, NBvali] = np.min(Fvals_chi);
                which_chi_min[Kvali, NBvali] = np.argmin(Fvals_chi);
                print(Kvali, NBvals[NBvali], kvals[Kvali]*NBvals[NBvali]/np.pi, Fvals_min[Kvali, NBvali], which_chi_min[Kvali, NBvali]);
                
            #### end loop over Ki

            # determine fidelity and kNB*, ie x val where the SWAP happens
            xvals = Kvals;
            xstar = xvals[np.argmax(Fvals_min[:,NBvali])];
            print("NBstar, fidelity(NBstar) = ",xstar, np.max(Fvals_min[:,NBvali]));
                  
            # plot fidelity, starred SWAP locations, as a function of NB
            axes[gatevali].plot(xvals,Fvals_min[:,NBvali], label = "$N_B = ${:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali]);
            axes[gatevali].set_title("$F("+gates[gatevali]+")$");
            if(vlines): axes[gatevali].axvline(xstar, color=mycolors[NBvali], linestyle="dotted");

            # formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[-1].set_xlabel('$K_i/t$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
            #axes[0].legend();

        #### end loop over NB
            
    # show
    fig.suptitle("$s=${:.1f}, $J/t=${:.2f}, $V_B/t=${:.2f}".format(myspinS, Jval/tl, VB/tl));
    plt.tight_layout();
    plt.show();

    # save data
    param_vals = np.array([tl,myspinS,Jval,VB]);
    fname = "data/wfm_gatechar/Ki/";
    #np.savetxt(fname+".txt", Kvals, header="[tl,myspinS,Jval,VB] =\n"+str(param_vals)+"\nKvals =");
    #np.save(fname+"_x", kNBvals/np.pi);
    #np.save(fname, yvals);

