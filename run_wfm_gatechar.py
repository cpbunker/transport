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

from run_wfm_gate import h_cicc, get_U_gate;
          
#########################################################
#### barrier in right lead for total reflection

#### top level
np.set_printoptions(precision = 8, suppress = True);
verbose = 1;
which_gate = sys.argv[1]; # not used
case = sys.argv[2];
final_plots = int(sys.argv[3]);
#if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
vlines = not final_plots;
elecspin = 0; # itinerant e is spin up

# fig standardizing
myxvals = 29; myfigsize = (5/1.2,3/1.2); myfontsize = 14;
if(final_plots): myxvals = 99; myfigsize = (8,4); myfontsize = 18;
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X","+"];
mymarkevery = (myxvals//3, myxvals//3);
mypanels = ["(a)","(b)","(c)","(d)"];
ylabels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
    "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"];
 
# tight binding params
tl = 1.0;
myTwoS = 1;
n_mol_dof = (myTwoS+1)*(myTwoS+1); 
n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
Jval = -0.02*tl;
VB = 5.0*tl;
V0 = 0.0*tl; # just affects title, not implemented physically
the_ticks = [0.0,1.0]; # always positive since it is fidelity

if(case in ["NB","kNB"]): # at fixed Ki, as a function of NB,
         # minimize over a set of states \chi_k
         # for each gate of interest

    # axes
    gates = ["SQRT","SWAP"]; #,"I"];
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="NB"): NB_indep = True # whether to put NB, alternatively wavenumber*NB
    elif(case=="kNB"): NB_indep = False;

    # iter over gates
    for gatevali in range(len(gates)):
        U_gate, dummy_ticks = get_U_gate(gates[gatevali],myTwoS);

        # iter over incident kinetic energy (colors)
        Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
        knumbers = np.sqrt(np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers)-1));
        Kvals = 2*tl - 2*tl*np.cos(knumbers);
        Energies = Kvals - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
        Fvals_min = np.empty((len(Kvals), myxvals),dtype=float); # fidelity min'd over chi states
        for Kvali in range(len(Kvals)):
                   
            # iter over barrier distance (x axis)
            kNBmax = 0.75*np.pi;
            NBmax = int(kNBmax/knumbers[Kvali]);
            if(NB_indep): NBmax = 150;
            NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
            if(verbose): print("k^2 = {:6f}, NBmax = {:.0f}".format(knumbers[Kvali]**2, NBmax)); 
            
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
                rhat = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, rhat = True, all_debug = False);

                # fidelity minimized over chi states
                M_matrix = np.matmul(np.conj(U_gate.T), rhat);
                trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
                if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
                Fvals_min[Kvali, NBvali] = np.real(trace_fidelity);
                if(verbose>4): print(Kvali, NBvals[NBvali], "{:.6f}, {:.6f}".format(knumbers[Kvali]*NBvals[NBvali]/np.pi, Fvals_min[Kvali, NBvali]));
                
            #### end loop over NB

            # determine fidelity and kNB*, ie x val where the SWAP happens
            if(NB_indep): indep_vals = NBvals; 
            else: indep_vals = 2*knumbers[Kvali]*NBvals/np.pi;
            indep_star = indep_vals[np.argmax(Fvals_min[Kvali])];
            if(verbose): print("NBstar, fidelity(NBstar) = {:.6f}. {:.4f}".format(indep_star, np.max(Fvals_min[Kvali])));

            # plot formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.2+the_ticks[0],0.2+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[gatevali].set_xlim(0,np.max(indep_vals));
            if(NB_indep): axes[-1].set_xlabel('$N_B $',fontsize=myfontsize);
            else: axes[-1].set_xlabel('$2 k_i a N_B/\pi $',fontsize=myfontsize);
            axes[gatevali].annotate(gate_strs[gatevali], (indep_vals[0],1.01),fontsize=myfontsize);
            if(gates[gatevali] == "I"): axes[gatevali].set_title("$\mathbf{U} = \mathbf{I}$");
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{r}, \mathbf{U})$");
 
            # plot fidelity, starred SWAP locations, as a function of NB
            if(abs(knumbers[1] - np.sqrt(10.0**Kpowers[1])) < 1e-10):
                mylabel = "$k_i^2 a^2 = 10^{"+str(Kpowers[Kvali])+"}$"
            else: 
                mylabel = "$k_i^2 a^2 = {:.6f} $".format(knumbers[Kvali]**2);
            axes[gatevali].plot(indep_vals,Fvals_min[Kvali], label = mylabel,color=mycolors[Kvali],marker=mymarkers[1+Kvali],markevery=mymarkevery);
            if(vlines): axes[gatevali].axvline(indep_star, color=mycolors[Kvali], linestyle="dotted");           

        #### end loop over Ki

    #### end loop over gates
            
    # show
    suptitle = "$s=${:.1f}, $J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(0.5*myTwoS, Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle);
    plt.tight_layout();
    if(final_plots): # save fig and legend
        fname = "figs/gate/F_vs_"+case;
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

if(case in ["Ki", "ki"]): # at fixed NB, as a function of Ki,
         # minimize over a set of states \chi_k
         # for each gate of interest

    # axes
    gates = ["SQRT","SWAP"]; #,"I"]; 
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = np.array([axes]);
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="ki"): K_indep = False;
    elif(case=="Ki"): K_indep = True; # whether to put ki^2 on x axis, alternatively ki a Nb/pi 
   
    # iter over gates
    for gatevali in range(len(gates)):
        U_gate, dummy_ticks = get_U_gate(gates[gatevali],myTwoS);

        # iter over barrier distance (colors)
        NBvals = np.array([50,100,150]);
        NBvals = np.array([700,900]); assert(Jval==-0.02);
        Fvals_min = np.empty((myxvals, len(NBvals)),dtype=float); # fidelity min'd over chi states
        rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];
            if(verbose): print("NB = ",NBval); 

            # iter over incident kinetic energy (x axis)
            kNBmax = 2.0*np.pi
            Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
            if(K_indep):
                knumbers = np.sqrt(np.logspace(Kpowers[0],Kpowers[-1],num=myxvals)); del kNBmax;
                print(knumbers**2); 
            else:
                kNB_times2overpi = np.linspace(0.001, 2*kNBmax/np.pi - 0.001, myxvals);
                knumbers = kNB_times2overpi *np.pi/(2*NBval)
                print(2*knumbers*NBvals[NBvali]/np.pi)
            Kvals = 2*tl - 2*tl*np.cos(knumbers);
            Energies = Kvals - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
            for Kvali in range(len(Kvals)):

                # construct hblocks from spin ham
                hblocks_cicc = h_cicc(myTwoS, Jval, [1],[2]);

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
                if((Kvali == 0 and NBvali == 0) and (gatevali==0 and verbose>4)): print("\nhblocks = \n",np.real(hblocks));

                # since we don't iter over sources, ALL sources must have 0 chem potential
                source = np.zeros((n_loc_dof,));
                source[-1] = 1;
                for sourcei in range(n_loc_dof):
                    assert(hblocks[0][sourcei,sourcei]==0.0);
                        
                # get reflection operator
                rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, rhat = True, all_debug = False);

                # fidelity minimized over chi states
                M_matrix = np.matmul(np.conj(U_gate.T), rhatvals[:,:,Kvali,NBvali]);
                trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
                if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
                Fvals_min[Kvali, NBvali] = np.real(trace_fidelity);
                if(verbose>4): print(Kvali, NBvals[NBvali], "{:.6f}. {:.6f}".format(knumbers[Kvali]*knumbers[Kvali], Fvals_min[Kvali, NBvali]));
                
            #### end loop over Ki

            # determine fidelity and kNB*, ie x val where the SWAP is best
            if(K_indep): indep_vals = knumbers*knumbers;
            else: indep_vals = 2*knumbers*NBvals[NBvali]/np.pi;
            indep_argmax = np.argmax(Fvals_min[:,NBvali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[:,NBvali]));
                print(indep_comment,"\n",rhatvals[:,:,indep_argmax,NBvali]);
                if(gates[gatevali] == "SWAP"): np.savetxt("data/wfm_swap/rhat_{:.0f}_{:.0f}.txt".format(indep_argmax,NBvali),rhatvals[:,:,indep_argmax,NBvali], fmt="%.4f", header=indep_comment);

            # plot formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            if(K_indep):
                axes[-1].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
                axes[-1].set_xscale('log', subs = []);
            else:
                axes[-1].set_xlabel('$2k_i a N_B/\pi$',fontsize=myfontsize);
            axes[gatevali].annotate("$\mathbf{U}_{"+gates[gatevali]+"}$", (indep_vals[int(3*len(indep_vals)/4)],1.01),fontsize=myfontsize);
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{r},\mathbf{U})$",fontsize=myfontsize);
                
            # plot fidelity, starred SWAP locations
            axes[gatevali].plot(indep_vals,Fvals_min[:,NBvali], label = "$N_B = ${:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali],marker=mymarkers[1+NBvali],markevery=mymarkevery);
            if(vlines): axes[gatevali].axvline(indep_star, color=mycolors[NBvali], linestyle="dotted");

        #### end loop over NB

    #### end loop over gates
            
    # show
    suptitle = "$s=${:.1f}, $J=${:.2f}, $V_0=${:.1f}, $V_B=${:.1f}".format(0.5*myTwoS, Jval/tl, V0/tl, VB/tl);
    if(final_plots): suptitle = None;
    fig.suptitle(suptitle, fontsize=myfontsize);
    plt.tight_layout();
    if(final_plots): # save fig and legend
        fname = "figs/gate/F_vs_"+case;
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
    

