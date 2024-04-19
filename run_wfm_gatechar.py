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

from run_wfm_gate import h_cicc, get_U_gate, get_Fval;
          
#########################################################
#### barrier in right lead for total reflection

#### top level
np.set_printoptions(precision = 8, suppress = True);
verbose = 1;
case = sys.argv[2];
final_plots = int(sys.argv[3]);
#if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
vlines = not final_plots;
elecspin = 0; # itinerant e is spin up

# fig standardizing
myxvals = 49; myfigsize = (5/1.2,3/1.2); myfontsize = 14;
if(final_plots): myxvals = 99; 
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
Jval = float(sys.argv[1]);
VB = 5.0*tl;
V0 = 0.0*tl; # just affects title, not implemented physically
the_ticks = [0.0,1.0]; # always positive since it is fidelity

# axes
gates = ["SeS12","SQRT","SWAP"]#,"I"];
nrows, ncols = len(gates), 1;
fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);

#### plot already existing data
if(final_plots == 10):

    # title and colors
    title_and_colors = "data/gate/"+case+"_ALL_J{:.2f}_title.txt".format(Jval);
    suptitle = open(title_and_colors,"r").read().splitlines()[0][1:];
    colorvals = np.loadtxt(title_and_colors);
    if(case in ["NB","kNB"]):
        which_color = "K";
        which_color_list = np.arange(len(colorvals));
    elif(case in ["K","ki"]):
        which_color = "NB";
        colorvals = colorvals.astype(int);
        which_color_list = 1*colorvals;

    # iter over gates
    for gatevali in range(len(gates)):

        # load and plot Fidelity
        for colori in range(len(colorvals)):

            # load data
            yvals = np.load(("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_"+which_color+"{:.0f}_y.npy").format(Jval, which_color_list[colori]));
            xvals = np.load(("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_"+which_color+"{:.0f}_x.npy").format(Jval, which_color_list[colori]));

            # determine label and plot
            if(case in ["NB","kNB"]):
                correct_Kpower = False;
                for Kpower in range(-9,-1):
                    if(abs(colorvals[colori] - 10.0**Kpower) < 1e-10): correct_Kpower = 1*Kpower;
                if(correct_Kpower is not False): mylabel = "$k_i^2 a^2 = 10^{"+str(correct_Kpower)+"}$";
                else: mylabel = "$k_i^2 a^2 = {:.6f} $".format(colorvals[colori]);
            elif(case in ["K","ki"]):
                mylabel = "$N_B = ${:.0f}".format(colorvals[colori]);
            axes[gatevali].plot(xvals,yvals, label = mylabel, color=mycolors[colori],marker=mymarkers[1+colori],markevery=mymarkevery);

        #### end loop over Ki

        # plot formatting
        #axes[gatevali].set_yticks(the_ticks);
        axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
        for tick in the_ticks:
            axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
        axes[gatevali].set_xlim(0,np.max(xvals));
        if(case=="NB"):
            axes[-1].set_xlabel('$N_B$',fontsize=myfontsize);
        elif(case=="Ki"):
            axes[-1].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
        else:
            axes[-1].set_xlabel('$2 k_i a N_B/\pi $',fontsize=myfontsize);
        axes[gatevali].annotate("$\mathbf{U}_{"+gates[gatevali]+"}$", (xvals[0],1.01),fontsize=myfontsize);
        axes[gatevali].set_ylabel("$F_{avg}(\mathbf{r}, \mathbf{U})$",fontsize=myfontsize);


    #### end loop over gates
            
    # show
    fig.suptitle(suptitle);
    plt.tight_layout();
    axes[-1].legend();
    plt.show();

#### generate data
elif(case in ["NB","kNB"]): # at fixed Ki, as a function of NB,
         # minimize over a set of states \chi_k
         # for each gate of interest
    if(case=="NB"): NB_indep = True; # whether to put NB, alternatively wavenumber*NB
    elif(case=="kNB"): NB_indep = False;

    # physical params
    suptitle = "$s=${:.1f}, $J=${:.2f}, $V_0=${:.1f}, $V_B=${:.1f}".format(0.5*myTwoS, Jval, V0, VB);
    if(final_plots): suptitle += "";

    # iter over incident kinetic energy (colors)
    Kpowers = np.array([-2,-3,-4]); # incident kinetic energy/t = 10^Kpower
    knumbers = np.sqrt(np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers)));
    Kvals = 2*tl - 2*tl*np.cos(knumbers);
    Energies = Kvals - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
    rhatvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals),dtype=complex); # by  init spin, final spin, energy, NB
    Fvals_min = np.empty((len(Kvals), myxvals, len(gates)),dtype=float); # fidelity min'd over chi states
    for Kvali in range(len(Kvals)):
               
        # iter over barrier distance (x axis)
        kNBmax = 1.0*np.pi;
        NBmax = np.ceil(kNBmax/knumbers[Kvali]);
        if(NB_indep): NBmax = 150;
        NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
        if(verbose): print("\nk^2 = {:6f}, NBmax = {:.0f}".format(knumbers[Kvali]**2, NBmax)); 
        
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # construct hblocks from spin ham
            hblocks_cicc = h_cicc(myTwoS,Jval, [1],[2]);

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

            # since we don't iter over sources, ALL sources must have 0 chem potential
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            for sourcei in range(n_loc_dof):
                assert(hblocks[0][sourcei,sourcei]==0.0);
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, rhat = True, all_debug = False);

            # iter over gates to get fidelity for each one
            for gatevali in range(len(gates)):
                U_gate, dummy_ticks = get_U_gate(gates[gatevali],myTwoS);
                Fvals_min[Kvali, NBvali, gatevali] = get_Fval(gates[gatevali], myTwoS, U_gate, rhatvals[:,:,Kvali,NBvali]);                
        #### end loop over NB

        # plotting considerations
        if(NB_indep): indep_vals = 1*NBvals; 
        else: indep_vals = 2*knumbers[Kvali]*NBvals/np.pi;
        if(verbose): print(">>> indep_vals = ",indep_vals);
        for gatevali in range(len(gates)):
            # determine fidelity and kNB*, ie x val where the SWAP happens
            indep_argmax = np.argmax(Fvals_min[Kvali,:,gatevali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, max F = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[Kvali,:,gatevali]));
                print("\nU = "+gates[gatevali]+"\n",indep_comment);
                if(gates[gatevali] == "SWAP" and (final_plots > 1)): np.savetxt("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_K{:.0f}_R.txt".format(Jval, Kvali),rhatvals[:,:,Kvali,indep_argmax], fmt="%.4f", header=indep_comment);

            # formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[gatevali].set_xlim(0,np.max(indep_vals));
            if(NB_indep): axes[-1].set_xlabel('$N_B$',fontsize=myfontsize);
            else: axes[-1].set_xlabel('$2 k_i a N_B/\pi $',fontsize=myfontsize);
            if(Kvali==0): axes[gatevali].annotate("$\mathbf{U}_{"+gates[gatevali]+"}$", (indep_vals[0],1.01),fontsize=myfontsize);
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{r}, \mathbf{U})$",fontsize=myfontsize);

            # get and store label info
            if(abs(knumbers[Kvali] - np.sqrt(10.0**Kpowers[Kvali])) < 1e-10):
                mylabel = "$k_i^2 a^2 = 10^{"+str(Kpowers[Kvali])+"}$"
            else: 
                mylabel = "$k_i^2 a^2 = {:.6f} $".format(knumbers[Kvali]**2);
                
            # plot fidelity, starred SWAP locations, as a function of NB
            axes[gatevali].plot(indep_vals,Fvals_min[Kvali,:,gatevali], label = mylabel,color=mycolors[Kvali],marker=mymarkers[1+Kvali],markevery=mymarkevery);
            if(vlines): axes[gatevali].axvline(indep_star, color=mycolors[Kvali], linestyle="dotted");           

            # save Fvals to .npy
            if(final_plots>1):
                np.save("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_K{:.0f}_y.npy".format(Jval, Kvali),Fvals_min[Kvali,:,gatevali]);
                np.save("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_K{:.0f}_x.npy".format(Jval, Kvali),indep_vals);
        #### end loop over gates
            
    #### end loop over Ki
            
    # show
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
        # title and color values
        np.savetxt("data/gate/"+case+"_ALL_J{:.2f}_title.txt".format(Jval),knumbers*knumbers,header=suptitle);
    else:
        axes[-1].legend();
        plt.show();

elif(case in ["K", "ki"]): # at fixed NB, as a function of Ki,
         # minimize over a set of states \chi_k
         # for each gate of interest
    if(case=="ki"): K_indep = False;
    elif(case=="K"): K_indep = True; # whether to put ki^2 on x axis, alternatively ki a Nb/pi 

    # physical params
    suptitle = "$s=${:.1f}, $J=${:.2f}, $V_0=${:.1f}, $V_B=${:.1f}".format(0.5*myTwoS, Jval, V0, VB);
    if(final_plots): suptitle += "";

    # iter over barrier distance (colors)
    NBvals = np.array([50,100,150]);
    #NBvals = np.array([700,900]); assert(Jval==-0.02);
    Fvals_min = np.empty((myxvals, len(NBvals),len(gates)),dtype=float); # fidelity min'd over chi states
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):
        NBval = NBvals[NBvali];
        if(verbose): print("NB = ",NBval); 

        # iter over incident kinetic energy (x axis)
        kNBmax = 2.0*np.pi
        Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
        if(K_indep):
            knumbers = np.sqrt(np.logspace(Kpowers[0],Kpowers[-1],num=myxvals)); del kNBmax;
        else:
            kNB_times2overpi = np.linspace(0.001, 2*kNBmax/np.pi - 0.001, myxvals);
            knumbers = kNB_times2overpi *np.pi/(2*NBval)
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
            
            # since we don't iter over sources, ALL sources must have 0 chem potential
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            for sourcei in range(n_loc_dof):
                assert(hblocks[0][sourcei,sourcei]==0.0);
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, rhat = True, all_debug = False);

             # iter over gates to get fidelity for each one
            for gatevali in range(len(gates)):
                U_gate, dummy_ticks = get_U_gate(gates[gatevali],myTwoS);
                Fvals_min[Kvali, NBvali, gatevali] = get_Fval(gates[gatevali], myTwoS, U_gate, rhatvals[:,:,Kvali,NBvali]);                
        #### end loop over Ki

        # plotting considerations
        if(K_indep): indep_vals = knumbers*knumbers;
        else: indep_vals = 2*knumbers*NBvals[NBvali]/np.pi;
        if(verbose): print(">>> indep_vals = ",indep_vals);
        for gatevali in range(len(gates)):

            # determine fidelity and kNB*, ie x val where the SWAP is best
            indep_argmax = np.argmax(Fvals_min[:,NBvali,gatevali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[:,NBvali,gatevali]));
                print("\nU = "+gates[gatevali]+"\n",indep_comment);
                if(gates[gatevali] == "SWAP" and (final_plots > 1)): np.savetxt("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_NB{:.0f}_R.txt".format(Jval, NBval),rhatvals[:,:,indep_argmax,NBvali], fmt="%.4f", header=indep_comment);

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
            axes[gatevali].plot(indep_vals,Fvals_min[:,NBvali,gatevali], label = "$N_B = ${:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali],marker=mymarkers[1+NBvali],markevery=mymarkevery);
            if(vlines): axes[gatevali].axvline(indep_star, color=mycolors[NBvali], linestyle="dotted");

            # save Fvals to .npy
            if(final_plots > 1):
                np.save("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_NB{:.0f}_y.npy".format(Jval, NBval),Fvals_min[:,NBvali,gatevali]);
                np.save("data/gate/"+case+"_"+gates[gatevali]+"_J{:.2f}_NB{:.0f}_x.npy".format(Jval, NBval),indep_vals);
        #### end loop over gates

    #### end loop over NB
            
    # show
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
        # title and color values
        np.savetxt("data/gate/"+case+"_ALL_J{:.2f}_title.txt".format(Jval),NBvals,header=suptitle);
    else:
        axes[-1].legend();
        plt.show();

