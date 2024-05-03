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
from scipy.linalg import expm as scipy_expm

import sys

from run_wfm_gate import h_cicc, get_U_gate, get_Fval;
          
#########################################################
#### barrier in right lead for total reflection

#### top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 1;
case = sys.argv[2];
final_plots = int(sys.argv[3]);
#if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
vlines = not final_plots;
elecspin = 0; # itinerant e is spin up

# fig standardizing
myxvals = 29; myfigsize = (5/1.2,3/1.2); myfontsize = 14;
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
gates = ["SeS12","SQRT","SWAP","I"];
nrows, ncols = len(gates), 1;
fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
if(nrows==1): axes = [axes];
fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);

#### plot already existing data
if(final_plots == 10):

    # override fig and axes
    plt.close(); del fig, axes;
    # title and colors
    if(case in ["NB","kNB"]): which_color = "K";
    elif(case in ["K","ki","roots"]): which_color = "NB";
    whichval = int(sys.argv[4]);
    title_and_colors = ("data/gate/"+case+"_ALL_J{:.2f}_"+which_color+"{:.0f}_title.txt").format(Jval,whichval);
    suptitle = open(title_and_colors,"r").read().splitlines()[0][1:];
    colorvals = np.loadtxt(title_and_colors,ndmin=1); 
    if(case in ["NB","kNB"]):
        which_color_list = np.arange(len(colorvals));
    elif(case in ["K","ki","roots"]):
        colorvals = colorvals.astype(int);
        which_color_list = 1*colorvals;

    # iter over gates
    if(case not in ["roots"]): 
        gates = sys.argv[5:];
        nrows, ncols = len(gates), 1;
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
        if(nrows==1): axes = [axes];
        fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
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
            #### end loop over colors (here fixed K or NB vals)

            # plot formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
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

    elif(case in ["roots"]): # data structure is different
        roots = sys.argv[5:]; # these are actually the colors, not colorvals
        if(roots[-1] == "SeS12"): mycolors[len(roots)-1] = "black";
        nrows, ncols = np.shape(colorvals)[0], 1;
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
        if(nrows==1): axes = [axes];
        fig.set_size_inches(ncols*myfigsize[0], nrows*myfigsize[1]);
        for colori in range(len(colorvals)): # this iters over axis rows!

            # iter over SWAP roots, which are colors
            for rootvali in range(len(roots)):
                # load data
                yvals = np.load(("data/gate/"+case+"_"+roots[rootvali]+"_J{:.2f}_"+which_color+"{:.0f}_y.npy").format(Jval, which_color_list[colori]));
                xvals = np.load(("data/gate/"+case+"_"+roots[rootvali]+"_J{:.2f}_"+which_color+"{:.0f}_x.npy").format(Jval, which_color_list[colori]));
                mylabel = "$n = $"+roots[rootvali];
                axes[colori].plot(xvals,yvals, label = mylabel, color=mycolors[rootvali],marker=mymarkers[1+rootvali],markevery=mymarkevery);
                
                # maxima
                indep_star = xvals[np.argmax(yvals)];
                if(verbose):
                    indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(yvals));
                    print("\nU^1/"+roots[rootvali]+"\n",indep_comment);
            #### end loop over colors (root vals)
                
             # plot formatting
            #axes[colori].set_yticks(the_ticks);
            axes[colori].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[colori].axhline(tick,color='lightgray',linestyle='dashed');
            axes[colori].set_xlim(0,np.max(xvals));
            axes[-1].set_xlabel('$2 k_i a N_B/\pi $',fontsize=myfontsize);
            axes[colori].annotate("$N_B = {"+str(colorvals[colori])+"}$", (xvals[int(3*len(xvals)/4)],1.01),fontsize=myfontsize);
            axes[colori].set_ylabel("$F_{avg}[\mathbf{R}, (\mathbf{U}_{SWAP})^{1/n}]$",fontsize=myfontsize);
        #### end loop over fixed NB vals
            
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
    Fvals_min = np.empty((len(Kvals), myxvals, len(gates)),dtype=float); # avg fidelity
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
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{R}, \mathbf{U})$",fontsize=myfontsize);

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
        np.savetxt("data/gate/"+case+"_ALL_J{:.2f}_K{:.0f}_title.txt".format(Jval,len(Kvals)),knumbers*knumbers,header=suptitle);
    else:
        axes[-1].legend();
        plt.show();

elif(case in ["K", "ki"]): # at fixed NB, as a function of Ki,
         # minimize over a set of states \chi_k
         # for each gate of interest
    if(case=="ki"): K_indep = False;
    elif(case=="K"): K_indep = True; # whether to put ki^2 on x axis, alternatively ki a Nb/pi

    myxvals = int(myxvals/3) #!!!!!!!!!!!!!!!!!!!!!!!!!!

    # physical params
    suptitle = "$s=${:.1f}, $J=${:.2f}, $V_0=${:.1f}, $V_B=${:.1f}".format(0.5*myTwoS, Jval, V0, VB);
    if(final_plots): suptitle += "";

    # iter over barrier distance (colors)
    NBvals = np.array([575,600]);
    #NBvals = np.array([1000,1400,1800]); assert(Jval==-0.02);
    Fvals_min = np.empty((myxvals, len(NBvals),len(gates)),dtype=float); # avg fidelity
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):
        NBval = NBvals[NBvali];
        if(verbose): print("NB = ",NBval); 

        # iter over incident kinetic energy (x axis)
        kNBmax = 1.5*np.pi
        Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
        if(K_indep):
            knumbers = np.sqrt(np.logspace(Kpowers[0],Kpowers[-1],num=myxvals)); del kNBmax;
        else:
            kNB_times2overpi = np.linspace(0.001, 2*kNBmax/np.pi - 0.001, myxvals);
            kNB_times2overpi = np.linspace(2*1.0, 2*kNBmax/np.pi - 0.001, myxvals); #!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        if(verbose):
            print(">>> indep_vals = ") #,indep_vals);
            for vali in range(len(indep_vals)): print(indep_vals[vali], (Jval/tl)*(2*NBval)/(np.pi*np.pi*indep_vals[vali]), Fvals_min[vali,NBvali,2]);
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
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{R},\mathbf{U})$",fontsize=myfontsize);
                
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
        np.savetxt("data/gate/"+case+"_ALL_J{:.2f}_NB{:.0f}_title.txt".format(Jval,NBvals[-1]),NBvals,header=suptitle);
    else:
        axes[-1].legend();
        plt.show();

elif(case in ["roots"]): # compare different roots of swap

    # override existing axes
    plt.close();
    del gates, fig, axes;
    NBvals = np.array([50,100,200])
    nrows, ncols = len(NBvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    K_indep = False; # whether to plot (ki*a)^2 or 2ki *a * NB/\pi on the x axis
    extend = True; # more multiples of 2kiaNB/pi
    if(extend):
        myxvals = 5*myxvals;
        mymarkevery = (myxvals//3, myxvals//3);
    # physical params;
    suptitle = "$s=${:.1f}, $J=${:.2f}, $V_0=${:.1f}, $V_B=${:.1f}".format(0.5*myTwoS, Jval, V0, VB);
    if(final_plots): suptitle += "";

    # iter over roots
    # roots are functionally the color (replace NBval) and NBs are axes (replace gates)
    # but still order axes as Kvals, NBvals, roots
    roots = np.array(["1","SeS12"]); 
    if(roots[-1] == "SeS12"): mycolors[len(roots)-1] = "black";
    Fvals_min = np.empty((myxvals, len(NBvals),len(roots)),dtype=float); # avg fidelity 
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):
        NBval = NBvals[NBvali];
        if(verbose): print("NB = ",NBval); 

        # iter over incident kinetic energy (x axis)
        kNBmax = 1.5*np.pi;
        if(extend): kNBmax = 5.0*np.pi;
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
            for rootvali in range(len(roots)):
                gatestr = "RZ"+roots[rootvali];
                U_gate, dummy_ticks = get_U_gate(gatestr,myTwoS);
                Fvals_min[Kvali, NBvali, rootvali] = get_Fval(gatestr, myTwoS, U_gate, rhatvals[:,:,Kvali,NBvali]);             
        #### end loop over Ki

        # plotting considerations
        if(K_indep): indep_vals = knumbers*knumbers;
        else: indep_vals = 2*knumbers*NBvals[NBvali]/np.pi;
        if(verbose):
            print(">>> indep_vals = ") 
            for vali in range(len(indep_vals)): print(indep_vals[vali], (Jval/tl)*(2*NBval)/(np.pi*np.pi*indep_vals[vali]), Fvals_min[vali,NBvali,-1]);
        for rootvali in range(len(roots)):

            # determine fidelity and kNB*, ie x val where the SWAP is best
            indep_argmax = np.argmax(Fvals_min[:,NBvali,rootvali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[:,NBvali,rootvali]));
                print("\nU^1/"+roots[rootvali]+"\n",indep_comment);

            # plot formatting
            #axes[NBvali].set_yticks(the_ticks);
            axes[NBvali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[NBvali].axhline(tick,color='lightgray',linestyle='dashed');
            if(K_indep):
                axes[-1].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
                axes[-1].set_xscale('log', subs = []);
            else:
                axes[-1].set_xlabel('$2k_i a N_B/\pi$',fontsize=myfontsize);
            axes[-1].set_xlim(np.floor(indep_vals[0]), np.ceil(indep_vals[-1]));
            axes[NBvali].annotate("$N_B = {"+str(NBvals[NBvali])+"}$", (indep_vals[1],1.01),fontsize=myfontsize);
            axes[NBvali].set_ylabel("$F_{avg}[\mathbf{R}, (\mathbf{U}_{SWAP})^{1/n}]$",fontsize=myfontsize);
                
            # plot fidelity, starred SWAP locations
            axes[NBvali].plot(indep_vals,Fvals_min[:,NBvali,rootvali], label = "$n = "+roots[rootvali]+"$",color=mycolors[rootvali],marker=mymarkers[1+rootvali],markevery=mymarkevery);
            if(vlines): axes[NBvali].axvline(indep_star, color=mycolors[rootvali], linestyle="dotted");

            # save Fvals to .npy
            if(final_plots > 1):
                np.save("data/gate/"+case+"_"+roots[rootvali]+"_J{:.2f}_NB{:.0f}_y.npy".format(Jval, NBval),Fvals_min[:,NBvali,rootvali]);
                np.save("data/gate/"+case+"_"+roots[rootvali]+"_J{:.2f}_NB{:.0f}_x.npy".format(Jval, NBval),indep_vals);
        #### end loop over roots

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
        np.savetxt("data/gate/"+case+"_ALL_J{:.2f}_NB{:.0f}_title.txt".format(Jval,NBvals[-1]),NBvals,header=suptitle);
    else:
        axes[-1].legend(loc="upper right");
        plt.show();
        
elif(case in ["time","timeJ"]):

    # override existing axes
    plt.close();
    del gates, fig, axes;
    zvals = np.array([0.0,0.21]); # different values of \delta J/J
    nrows, ncols = len(zvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="timeJ"): time_indep=False;
    else: time_indep = True;
    del V0, VB;
    myxvals = 5*myxvals;
    mymarkevery = (myxvals//3, myxvals//3);
    analytical = False;

    # physical params;
    suptitle = "$s=${:.1f}, $J_H =${:.2f}".format(0.5*myTwoS, Jval);
    if(final_plots): suptitle += "";

    # iter over roots
    # roots are functionally the color (replace NBval) and NBs are axes (replace gates)
    # but still order axes as Kvals, NBvals, roots
    roots = np.array(["1", "SeS12"]); 
    if(roots[-1] == "SeS12"): mycolors[len(roots)-1] = "black";
    Fvals_min = np.empty((myxvals, len(zvals),len(roots)),dtype=float); # avg fidelity 
    rhatvals = np.zeros((n_loc_dof,n_loc_dof,myxvals,len(zvals)),dtype=complex); 
    
    for axvali in range(len(zvals)): # perturbation strengths (non-identical J1,J2)

        # iter over time (x axis)
        xmax = 10.0*np.pi;
        assert(Jval<0);
        timevals = np.linspace(0,xmax/(-Jval),myxvals,dtype=float);
        xvals = np.linspace(0,xmax,myxvals,dtype=float);
        # fill in rhat 
        if(analytical): # fill in from analytical J + pert theory deltaJ
            rhatvals[0,0,:,axvali] = np.ones_like(xvals);
            rhatvals[1,1,:,axvali] = (0.5+0.5*np.exp(complex(0,1)*xvals))*np.exp(complex(0,-Jval*zvals[axvali]/2)*timevals);
            rhatvals[1,2,:,axvali] = (0.5-0.5*np.exp(complex(0,1)*xvals))*np.exp(complex(0, Jval*zvals[axvali]/2)*timevals);
            rhatvals[2,1,:,axvali] = (0.5-0.5*np.exp(complex(0,1)*xvals))*np.exp(complex(0,-Jval*zvals[axvali]/2)*timevals);
            rhatvals[2,2,:,axvali] = (0.5+0.5*np.exp(complex(0,1)*xvals))*np.exp(complex(0, Jval*zvals[axvali]/2)*timevals);
            rhatvals[3,3,:,axvali] = np.ones_like(xvals) #np.exp(complex(0,8*Jval*zvals[axvali]*zvals[axvali]/3)*timevals);
            #rhatvals[4,4,:,axvali] = np.ones_like(xvals);
            #rhatvals[5,5,:,axvali] = 0.5+0.5*np.exp(complex(0,1)*xvals);
            #rhatvals[5,6,:,axvali] = 0.5-0.5*np.exp(complex(0,1)*xvals);
            #rhatvals[6,5,:,axvali] = 0.5-0.5*np.exp(complex(0,1)*xvals);
            #rhatvals[6,6,:,axvali] = 0.5+0.5*np.exp(complex(0,1)*xvals);
            #rhatvals[7,7,:,axvali] = np.ones_like(xvals);
        else:
            Hexch = (Jval/4)*np.array([[-1, 0, 0, 0, 0, 0, 0, 0],
                                       [0,1+2*zvals[axvali],  -2,0,-2*zvals[axvali], 0, 0,0], # -J_H S1.S2
                                       [0,-2,1-2*zvals[axvali],0,   2*zvals[axvali], 0, 0,0],
                                       [0,0, 0, -1,        0, 2*zvals[axvali],-2*zvals[axvali],0],
                                       [0,-2*zvals[axvali],2*zvals[axvali],0,-1, 0, 0,0],
                                       [0,0, 0, 2*zvals[axvali], 0, 1-2*zvals[axvali],-2,0],
                                       [0,0, 0, -2*zvals[axvali],     0, -2, 1+2*zvals[axvali],0],
                                       [0, 0, 0, 0, 0, 0, 0, -1]],
                                       dtype=float);
            Hexch =(-Jval/4)*np.array([[2, 0, 0, 0, 0, 0, 0, 0], # -J Se.(S1+S2)
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0,-2, 0, 2, 2, 0],
                                  [0, 2, 2, 0,-2, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 2]
                                  ],dtype=complex);
            Hexch +=(Jval*zvals[axvali]/4)*np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 2, 0, 0,-2, 0, 0, 0], # \delta J Se.(S1-S2)
                                       [0, 0,-2, 0, 2, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 2,-2, 0],
                                       [0,-2, 2, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 2, 0,-2, 0, 0],
                                       [0, 0, 0,-2, 0, 0, 2, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0]],
                                       dtype=float);
            assert( np.all(abs(np.transpose(Hexch) - Hexch) < 1e-10));
            for timevali in range(len(timevals)):
                U_coupled = scipy_expm(complex(0,-timevals[timevali])*Hexch);
                assert(np.all(abs(np.matmul(np.conj(np.transpose(U_coupled)), U_coupled)-np.eye(len(U_coupled))) < 1e-10));
                rhatvals[:,:,timevali,axvali] = 1*U_coupled;
                #rhatvals[0,0,timevali,axvali] = np.exp(complex(0,Jval)*timevals[timevali]/4);
                #rhatvals[7,7,timevali,axvali] = np.exp(complex(0,Jval)*timevals[timevali]/4);
                if(False):
                    print("time = {:.2f}\n".format(timevals[timevali]),U_coupled);
                    if(timevali>50): assert False
        
        # iter over gates to get fidelity for each one
        for rootvali in range(len(roots)):
            gatestr = "RZ"+roots[rootvali];
            U_gate, dummy_ticks = get_U_gate(gatestr,myTwoS);
            for xvali in range(len(xvals)):
                Fvals_min[xvali, axvali, rootvali] = get_Fval(gatestr, myTwoS, 
                           U_gate[:,:], rhatvals[:,:,xvali,axvali]); 
            assert(not analytical); # need to add rhat[i>4] correction terms above
                           
        # plotting considerations
        if(time_indep): indep_vals = timevals;
        else: indep_vals = xvals/np.pi;
        if(verbose):
            print(">>> indep_vals =\n",indep_vals);
        for rootvali in range(len(roots)):
            # determine fidelity and kNB*, ie x val where the SWAP is best
            indep_argmax = np.argmax(Fvals_min[:,axvali,rootvali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[:,axvali,rootvali]));
                print("\nU^1/"+roots[rootvali]+"\n",indep_comment);

            # plot formatting
            #axes[axvali].set_yticks(the_ticks);
            axes[axvali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[axvali].axhline(tick,color='lightgray',linestyle='dashed');
            if(time_indep):
                axes[-1].set_xlabel('$\\tau$ ($\hbar$/Energy)',fontsize=myfontsize);
            else:
                axes[-1].set_xlabel('$|J|\\tau/\pi \hbar$',fontsize=myfontsize);
            axes[-1].set_xlim(np.floor(indep_vals[0]), np.ceil(indep_vals[-1]));
            axes[axvali].annotate("$\delta J/J_H = {:.2f}$".format(zvals[axvali]), (indep_vals[1],1.01),fontsize=myfontsize);
            axes[axvali].set_ylabel("$F_{avg}[\mathbf{U}(\\tau),(\mathbf{U}_{SWAP})^{1/n}]$",fontsize=myfontsize);
                
            # plot fidelity, starred SWAP locations
            axes[axvali].plot(indep_vals,Fvals_min[:,axvali,rootvali], label = "$n = "+roots[rootvali]+"$",color=mycolors[rootvali],marker=mymarkers[1+rootvali],markevery=mymarkevery);

            # save Fvals to .npy

        #### end loop over roots

    #### end loop over axvals
            
    # show
    fig.suptitle(suptitle, fontsize=myfontsize);
    plt.tight_layout();
    axes[0].legend();
    plt.show();  
        
elif(case in ["ctap","ctapJ"]): # just time evolve initial state

    # override existing axes
    plt.close();
    del gates, fig, axes;
    zvals = np.array([0.25]);
    nrows, ncols = 1+len(zvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="ctapJ"): time_indep=False;
    else: time_indep = True;
    myxvals = 5*myxvals;
    mymarkevery = (myxvals//3, myxvals//3);
    
    # define time-dep observables
    which_states = np.array([1,4,2]);
    state_labels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
        "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"]; 
    observables = np.empty((len(which_states), myxvals),dtype=float);
    psi0 = np.zeros((n_loc_dof,),dtype=complex);
    psi0[which_states[0]] = 1.0; # |up up down>
    deltaJval = 0.0;
    
    # iter over roots
    # roots are functionally the color (replace NBval) and NBs are axes (replace gates)
    # but still order axes as Kvals, NBvals, roots
    roots = np.array(["2", "SeS12", "1"]); 
    if(roots[-1] == "SeS12"): mycolors[len(roots)-1] = "black";
    if(roots[-2] == "SeS12"): mycolors[len(roots)-2] = "black";
    Fvals_min = np.empty((myxvals, len(zvals), len(roots)),dtype=float); # avg fidelity 
    rhatvals = np.zeros((n_loc_dof,n_loc_dof,myxvals,len(zvals)),dtype=complex); 
    
    for axvali in range(len(zvals)): # perturbation strengths (direct S1-S2 exchange)


        # Hamiltonian and psi0
        Hexch =(-Jval/4)*np.array([[2, 0, 0, 0, 0, 0, 0, 0], # -J Se.(S1+S2)
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0,-2, 0, 2, 2, 0],
                                  [0, 2, 2, 0,-2, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 2]
                                  ],dtype=complex);
        Hexch+=(-Jval*zvals[axvali]/4)*np.array(
                                 [[1, 0, 0, 0, 0, 0, 0, 0], # -JH S1.S2
                                  [0,-1, 2, 0, 0, 0, 0, 0],
                                  [0, 2,-1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0,-1, 2, 0],
                                  [0, 0, 0, 0, 0, 2,-1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1]
                                  ],dtype=complex);
        assert( np.all(abs(np.transpose(Hexch) - Hexch) < 1e-10));
        assert(deltaJval == 0.0);
                                              
        # time evolution
        xmax = 10.0*np.pi;
        assert(Jval<0);
        timevals = np.linspace(0,xmax/abs(Jval),myxvals,dtype=float);
        xvals = np.linspace(0,xmax,myxvals,dtype=float);
    
        # iter over time
        for timevali in range(len(timevals)):
    
            # U(time)
            U_coupled = scipy_expm(complex(0,-timevals[timevali])*Hexch);
            assert(np.all(abs(np.matmul(np.conj(np.transpose(U_coupled)), U_coupled)-np.eye(len(U_coupled))) < 1e-10));
            rhatvals[:,:,timevali,axvali] = 1*U_coupled;

            # time dependent state
            psit = np.matmul(U_coupled, psi0);
            # get observables
            for whichi in range(len(which_states)):
                statei = which_states[whichi];
                bra = np.zeros_like(psi0);
                bra[statei] = 1.0;
                overlap = np.dot(np.conj(bra), psit);
                if( abs(np.imag(np.conj(overlap)*overlap)) > 1e-10): assert False;
                observables[whichi,timevali] = np.real(np.conj(overlap)*overlap);
            
        # iter over gates to get fidelity for each one
        for rootvali in range(len(roots)):
            gatestr = "RZ"+roots[rootvali];
            U_gate, dummy_ticks = get_U_gate(gatestr,myTwoS);
            for timevali in range(len(timevals)):
                Fvals_min[timevali, axvali, rootvali] = get_Fval(gatestr, myTwoS, 
                           U_gate[:,:], rhatvals[:,:,timevali,axvali]);  
            
        # plotting time evol
        if(time_indep): indep_vals = timevals;
        else: indep_vals = xvals/np.pi;
        if(axvali==0):
            for whichi in range(len(which_states)):
                axes[axvali].plot(indep_vals, observables[whichi], label = "$|\langle\psi(\\tau)|"+state_labels[which_states[whichi]]+"\\rangle|^2$", color=mycolors[whichi], marker = mymarkers[1+whichi], markevery = mymarkevery);
            axes[axvali].plot(indep_vals, np.sum(observables, axis=0), color="black");
            axes[axvali].annotate("$J_H/J = {:.2f}$".format(zvals[axvali]), (indep_vals[1],1.01),fontsize=myfontsize);
            axes[axvali].legend(loc="upper right");
                
        # plot fidelity, starred SWAP locations
        for rootvali in range(len(roots)):
            axes[1+axvali].plot(indep_vals,Fvals_min[:,axvali,rootvali], label = "$n = "+roots[rootvali]+"$",color=mycolors[rootvali],marker=mymarkers[1+rootvali],markevery=mymarkevery);
            # save Fvals to .npy

        #### end loop over roots
        
        # format
        #axes[axvali+1].set_yticks(the_ticks);
        axes[axvali+1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
        for tick in the_ticks: axes[axvali+1].axhline(tick,color='lightgray',linestyle='dashed');
        axes[axvali+1].legend(loc="upper right");
        axes[axvali+1].annotate("$J_H/J = {:.2f}$".format(zvals[axvali]), (indep_vals[0],1.01),fontsize=myfontsize);
        axes[axvali+1].set_ylabel("$F_{avg}[\mathbf{U}(\\tau),(\mathbf{U}_{SWAP})^{1/n}]$",fontsize=myfontsize);
    #### end loop over axvals    
    
    # show
    if(time_indep):
        axes[-1].set_xlabel('$\\tau$ ($\hbar$/Energy)',fontsize=myfontsize);
    else:
        axes[-1].set_xlabel('$|J|\\tau/\pi \hbar$',fontsize=myfontsize);
    axes[-1].set_xlim(np.floor(indep_vals[0]), np.ceil(indep_vals[-1]));
    suptitle = "$s=${:.1f}, $J=${:.2f}, $\delta J =${:.2f}".format(0.5*myTwoS, Jval, deltaJval);
    fig.suptitle(suptitle, fontsize=myfontsize);
    plt.tight_layout();
    plt.show();

elif(case in ["ctap_scat"]): # compare different roots of swap

    # override existing axes
    plt.close();
    del gates, fig, axes;
    NBvals = np.array([40,50,100])
    nrows, ncols = len(NBvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    K_indep = False; # whether to plot (ki*a)^2 or 2ki *a * NB/\pi on the x axis
    myxvals = 5*myxvals;
    mymarkevery = (myxvals//3, myxvals//3);
    # physical params;
    suptitle = "$s=${:.1f}, $J=${:.2f}, $V_0=${:.1f}, $V_B=${:.1f}".format(0.5*myTwoS, Jval, V0, VB);
    
    # define time-dep observables
    which_states = np.array([1,4,2]);
    state_labels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
        "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"]; 
    observables = np.empty((len(which_states), myxvals, len(NBvals)),dtype=float);
    psi0 = np.zeros((n_loc_dof,),dtype=complex);
    psi0[which_states[0]] = 1.0; # |up up down>
    
    # iter over NB
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):
        NBval = NBvals[NBvali];
        if(verbose): print("NB = ",NBval); 

        # iter over incident kinetic energy (x axis)
        kNBmax = 10*np.pi;
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

            # iter state weights
            psit = np.matmul(rhatvals[:,:,Kvali,NBvali], psi0);
            # get observables
            for whichi in range(len(which_states)):
                statei = which_states[whichi];
                bra = np.zeros_like(psi0);
                bra[statei] = 1.0;
                overlap = np.dot(np.conj(bra), psit);
                if( abs(np.imag(np.conj(overlap)*overlap)) > 1e-10): assert False;
                observables[whichi, Kvali, NBvali] = np.real(np.conj(overlap)*overlap);         
        #### end loop over Ki

        # plotting considerations
        if(K_indep): indep_vals = knumbers*knumbers;
        else: indep_vals = 2*knumbers*NBvals[NBvali]/np.pi;

        # plot formatting
        #axes[NBvali].set_yticks(the_ticks);
        axes[NBvali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
        for tick in the_ticks: axes[NBvali].axhline(tick,color='lightgray',linestyle='dashed');
        if(K_indep):
            axes[-1].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
            axes[-1].set_xscale('log', subs = []);
        else:
            axes[-1].set_xlabel('$2k_i a N_B/\pi$',fontsize=myfontsize);
        axes[-1].set_xlim(np.floor(indep_vals[0]), np.ceil(indep_vals[-1]));
        axes[NBvali].annotate("$N_B = {"+str(NBvals[NBvali])+"}$", (indep_vals[int(3*len(indep_vals)/4)],1.01),fontsize=myfontsize);
                
        for whichi in range(len(which_states)):
            axes[NBvali].plot(indep_vals, observables[whichi,:,NBvali], label = "$|\langle\psi(\\tau)|"+state_labels[which_states[whichi]]+"\\rangle|^2$", color=mycolors[whichi], marker = mymarkers[1+whichi], markevery = mymarkevery);
        axes[NBvali].plot(indep_vals, np.sum(observables[:,:,NBvali], axis=0), color="black");

    #### end loop over NB
            
    # show
    fig.suptitle(suptitle, fontsize=myfontsize);
    plt.tight_layout();
    axes[-1].legend();
    plt.show();      
        
        
