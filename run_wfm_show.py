from transport import wfm

from run_wfm_gate import get_hblocks, get_U_gate, get_Fval, get_suptitle, get_indep_vals;

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 4, suppress = True);

import sys

# fig standardizing
myfigsize = (5/1.2,3/1.2); myfontsize = 14;
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X","+"];
mypanels = ["(a)","(b)","(c)","(d)"];
state_labels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
        "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"]; 

# tight binding params
tl = 1.0;
elecspin = 0;
myTwoS = 1;
elems_to_keep=np.array([0,1,myTwoS+1,myTwoS+1+1]); #gets relevant qubit indices from TwoS value
n_mol_dof = (myTwoS+1)*(myTwoS+1); 
n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
Jval = float(sys.argv[1]);
case = sys.argv[2];
Vend = 5.0*tl;
final_plots = int(sys.argv[3]);
 
######################################################################################
#### generate data      
if(case in ["onsite_NB30", "onsite_NB100", "onsite_NB500", "onsite_NB100_show"]):
    Vqbarrier = True; # whether to change onsite energy of localized spins
    show = False;
    if("_show" in case): show = True;
    if("_NB30" in case): NBval = 30;
    elif("_NB100" in case): NBval = 100;
    elif("_NB500" in case): NBval = 500;
    else: raise NotImplementedError;
    gate_strings = ["overlap", "conc", "overlap_sf", "I"];
    gate_labels = ["$P_{swap}$", "$C$", "$P_{sf}$", "$F(\mathbf{I})$"];
    
    # select NBval/lambda points
    if(  NBval == 100): xval = 0.13; 
    elif(NBval == 30): xval = 0.10; assert(Jval == -0.4);
    elif(NBval == 500): xval = 0.63;
    else: raise NotImplementedError;
    lambdaval = NBval/xval;
    VBval = 0.0; # just affects titles, not code

    # iter over onsite energies
    Vqvals = np.linspace(-0.99*abs(Jval), 0.99*abs(Jval), 249);
    if(show): Vqvals = np.array([-0.5*abs(Jval), -0.25*abs(Jval), 0.0*abs(Jval), 0.125*abs(Jval), 0.25*abs(Jval)]);
    Fvals = np.empty((len(Vqvals), len(gate_strings)), dtype=float);
    for Vqvali in range(len(Vqvals)):
        
        # plotting
        fig, axes = plt.subplots(2, sharex = True); # for elec spin up and elec spin down
        
        # construct hblocks from cicc-type spin ham
        hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, Vend, NBval,
                        is_Vq = Vqbarrier, bval=Vqvals[Vqvali], verbose=0);

        # define source vector (boundary condition for incident wave)
        source = np.zeros((n_loc_dof,));
        source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;
        
        if(show): # get real-space scattered wavefunction
            psi = wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(2*np.pi/lambdaval), source, 
                     is_psi_jsigma = True, is_Rhat = False, all_debug = False);
                   
            # probability densities
            jvals = np.arange(len(psi));
            pdf = np.conj(psi)*psi;
            if( np.max(abs(np.imag(pdf))) > 1e-10):
                print(np.max(abs(np.imag(pdf))));
                assert False;
            pdf = np.real(pdf);

        # get Rhat
        Rhat = wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(2*np.pi/lambdaval), source, 
                 is_psi_jsigma = False, is_Rhat = True, all_debug = False);
        if(show): print("Rhat =\n", Rhat[elecspin*n_mol_dof:(elecspin+1)*n_mol_dof, elecspin*n_mol_dof:(elecspin+1)*n_mol_dof]);
        
        # evaluate Rhat by its fidelity w/r/t various gates
        to_annotate = "";
        for gatei in range(len(gate_strings)):
            Uhat, dummy_ticks = get_U_gate(gate_strings[gatei], myTwoS);
            F_gs = get_Fval(gate_strings[gatei], myTwoS, Uhat, Rhat, elecspin);
            Fvals[Vqvali, gatei] = 1*F_gs;
            to_annotate += ","+gate_labels[gatei]+" = "+ "{:.4f}".format(F_gs);

        if(show): # plot probability densities
            axes[-1].annotate(to_annotate, (jvals[1],0.0));
            for elecspin_index in range(len(axes)):
                for sigmai in range(n_mol_dof):
                    axes[elecspin_index].plot(jvals, pdf[:,elecspin_index*n_mol_dof+sigmai],
                           label = "$\sigma = "+state_labels[elecspin_index*n_mol_dof+sigmai]+"$");
                axes[elecspin_index].legend();
                axes[elecspin_index].set_ylabel("$\langle j \sigma | \psi \\rangle$");

            # format
            axes[-1].set_xlabel("$j$");
            axes[-1].set_xlim(0, np.max(jvals));
            axes[-1].set_xticks([0, np.max(jvals)]);
            axes[0].set_title("$V_B =${:.2f}, $N_B =${:.0f}, $N_B a/\lambda_i =${:.2f}, $V_q =${:.6f}".format(VBval, NBval, NBval/lambdaval, Vqvals[Vqvali]));
            plt.tight_layout();
            plt.show();
        else:
            plt.close();

    # plot F vs offset
    vs_fig, vs_ax = plt.subplots();
    for gatei in range(len(gate_strings)):
        vs_ax.plot(Vqvals, Fvals[:,gatei], label="$U_{"+gate_strings[gatei]+"}$");

    # format
    vs_ax.legend();
    vs_ax.set_xlabel("$V_q$");
    vs_ax.set_ylabel("$F_{avg} [ \mathbf{R}, \mathbf{U}] $");
    the_ticks = [0.0,1.0];
    for tick in the_ticks: vs_ax.axhline(tick,color='lightgray',linestyle='dashed');
    vs_ax.set_title("$s =${:.1f}, $J =${:.4f}, $V_B =${:.2f}, $N_B =${:.0f}, $N_B/\lambda_i =${:.2f}".format(myTwoS/2, Jval, VBval, NBval, xval));

    # show
    for row in Fvals: print(row);
    plt.tight_layout();
    plt.show();

# change onsite energy in the barrier region
elif(case in ["VB_NB30", "VB_NB100", "VB_NB500", "VB_NB100_show"]): 

    show = False; # whether to plot individual wavefunctions
    if("_show" in case): show = True;
    if("_NB30" in case): NBval = 30;
    elif("_NB100" in case): NBval = 100;
    elif("_NB500" in case): NBval = 500;
    else: raise NotImplementedError;
    gate_strings = ["overlap", "conc", "overlap_sf", "I"];
    gate_labels = ["$P_{swap}$", "$C$", "$P_{sf}$", "$F(\mathbf{I})$"];
    
    # change onsite energies in barrier region to the right of the qubits
    VBvals = np.linspace(-0.002, 0.000, 99) #399);
    if(NBval == 30): VBvals = np.linspace(-0.02, 0.000, 399);
    if(case=="VB_show"): 
        assert(Jval == -0.2);
        VBvals=(1)*np.array([0.0, -0.0002, -0.0004, -0.0006, -0.0008, -0.00083, -0.00084, -0.0010]); 
    
    # select NB/lambda points
    if(  NBval == 100): xvals = np.array([0.13,0.50]); 
    elif(NBval == 30 ): xvals = np.array([0.10,0.50,1.00]); assert(Jval == -0.4);
    elif(NBval == 500): xvals = np.array([1.00,1.17,1.50]);
    else: raise NotImplementedError;
    Vqval = 0.0; # zero onsite energy on qubits
    
    # figure and axes
    nrows, ncols = len(xvals), 1;
    vs_fig, vs_axes = plt.subplots(nrows, ncols, sharex=True, sharey = True);
    if(nrows==1): vs_axes = [vs_axes];
    vs_fig.set_size_inches(ncols*myfigsize[0], nrows*myfigsize[1]);
    
    # iter over NB/lambda (axes)
    Fvals = np.empty((len(VBvals), len(xvals), len(gate_strings)), dtype=float);
    for xvali in range(len(xvals)):

        # iter over onsite energies VB (x axis) 
        for VBvali in range(len(VBvals)):

            # construct hblocks from cicc-type spin ham
            hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, Vqval, VBvals[VBvali], Vend, NBval,
                        verbose=0);

            # define source vector (boundary condition for incident wave)
            source = np.zeros((n_loc_dof,));
            source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;
            lambdaval = NBval/xvals[xvali];
        
            if(show): # get and plot real-space scattered wavefunction
                fig, axes = plt.subplots(2, sharex = True);
                psi = wfm.kernel(hblocks,tnn,tnnn,tl, -2*tl*np.cos(2*np.pi/lambdaval), source, 
                         is_psi_jsigma = True, is_Rhat = False, all_debug = False);
                   
                # probability densities
                jvals = np.arange(len(psi));
                pdf = np.conj(psi)*psi;
                if( np.max(abs(np.imag(pdf))) > 1e-10):
                    print(np.max(abs(np.imag(pdf)))); assert False;
                pdf = np.real(pdf);

            # get Rhat
            Rhat = wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(2*np.pi/lambdaval), source, 
                     is_psi_jsigma = False, is_Rhat = True, all_debug = False);
            if(show): print("Rhat =\n", Rhat[elecspin*n_mol_dof:(elecspin+1)*n_mol_dof, elecspin*n_mol_dof:(elecspin+1)*n_mol_dof]);
        
            # evaluate Rhat by its fidelity w/r/t various gates (colors)
            to_annotate = "";
            for gatei in range(len(gate_strings)):
                Uhat, dummy_ticks = get_U_gate(gate_strings[gatei], myTwoS);
                F_gs = get_Fval(gate_strings[gatei], myTwoS, Uhat, Rhat, elecspin);
                Fvals[VBvali, xvali, gatei] = 1*F_gs;
                to_annotate += ","+gate_labels[gatei]+" = "+ "{:.4f}".format(F_gs);

            if(show): # plot probability densities
                axes[-1].annotate(to_annotate, (jvals[1],0.0));
                for elecspin_index in range(len(axes)):
                    for sigmai in range(n_mol_dof):
                        axes[elecspin_index].plot(jvals, pdf[:,elecspin_index*n_mol_dof+ sigmai],
                           label = "$\sigma = "+state_labels[elecspin_index*n_mol_dof+sigmai]+"$");
                    axes[elecspin_index].legend();
                    axes[elecspin_index].set_ylabel("$\langle j \sigma | \psi \\rangle$");

                # format
                axes[-1].set_xlabel("$j$");
                axes[-1].set_xlim(0, np.max(jvals));
                axes[-1].set_xticks([0, np.max(jvals)]);
                axes[0].set_title("$V_q =${:.2f}, $V_B=${:.6f}, $N_B =${:.0f}, $N_B a/\lambda_i =${:.2f} ".format(Vqval, VBvals[VBvali], NBval, xvals[xvali]));
                plt.tight_layout();
                plt.show();

        # plot F vs onsite energies VB
        for gatei in range(len(gate_strings)):
            vs_axes[xvali].plot(VBvals, Fvals[:,xvali,gatei], label=gate_labels[gatei], color = mycolors[gatei]);        
            # save Fvals to .npy
            if(final_plots > 1):
                XVAL_INT_CONVERSION = 100;
                xy_savename = "data/gate/"+case+"/"+gate_strings[gatei]+"_J{:.2f}_x{:.0f}".format(Jval, XVAL_INT_CONVERSION*xvals[xvali]);
                np.save(xy_savename + "_y.npy", Fvals[:,xvali,gatei]);
                np.save(xy_savename + "_x.npy",VBvals);
                
        # format the V vs VB plot
        #vs_axes[xvali].set_yticks(dummy_ticks);
        vs_axes[xvali].set_ylim(-0.1+dummy_ticks[0],0.1+dummy_ticks[-1]);
        for tick in dummy_ticks: vs_axes[xvali].axhline(tick,color='lightgray',linestyle='dashed');
        vs_axes[xvali].annotate("$N_B a/\lambda_i =${:.2f}".format(xvals[xvali]), (VBvals[1],1.01), fontsize = myfontsize);
    
        #### end loop over onsite energies
    
    #### end loop over xvals

    # format
    vs_axes[-1].legend();
    vs_axes[-1].set_xlabel("$V_B$");
    suptitle = "$s = ${:.1f}, $J =${:.4f}, $N_B =${:.2f}, $V_q =${:.2f}".format(myTwoS/2, Jval, NBval, Vqval);
    vs_axes[0].set_title(suptitle);

    # show
    if(final_plots > 1): # save fig and legend
        # title and color values
        np.savetxt("data/gate/"+case+"/ALL_J{:.2f}_x{:.0f}_title.txt".format(Jval, XVAL_INT_CONVERSION*xvals[-1]),XVAL_INT_CONVERSION*xvals,header = suptitle);
    else:
        plt.tight_layout();
        plt.show(); 

elif(case in ["vsJ"]):
    ferromagnetic = False;
    SRbarrier = True; # whether to change onsite energy of localized spins
    Jvals = np.linspace(Jval, Jval/10, 11); del Jval;
    del gate_strings;
    
    # select NBval/lambda points
    if(  NBval == 100): xvals = np.array([0.1,0.13,0.63]); 
    elif(NBval == 500): xvals = np.array([0.13, 0.30, 0.50, 0.63, 1.00, 1.17]); 
    else: raise NotImplementedError;

    # iter over axes (NB/lambda values)
    fig, axes = plt.subplots(len(xvals), sharex=True);
    if(len(xvals)==1): axes = [axes];
    for xvali in range(len(xvals)):
        lambdaval = NBval/xvals[xvali];

        # iter over colors (onsite energies)
        bvals = np.array([-0.015, -0.010, -0.005, 0.0]);
        Fvals = np.empty((len(Jvals), len(bvals)), dtype=float);
        for bvali in range(len(bvals)):

            # iter over x axis (J)
            for Jvali in range(len(Jvals)):
               
                # construct hblocks from cicc-type spin ham
                hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jvals[Jvali], VB, NBval,
                        is_SRbarrier = SRbarrier, bval=bvals[bvali], verbose=0);

                # define source vector (boundary condition for incident wave)
                source = np.zeros((n_loc_dof,));
                source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;

                # get Rhat
                Rhat = wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(2*np.pi/lambdaval), source, 
                     is_psi_jsigma = False, is_Rhat = True, all_debug = False);
       
                # evaluate Rhat by its fidelity w/r/t various gates
                gs = sys.argv[4];
                Uhat, dummy_ticks = get_U_gate(gs, myTwoS);
                Fvals[Jvali, bvali] = get_Fval(gs, myTwoS, Uhat, Rhat, elecspin, ferromagnetic);

            # plot F vs J
            axes[xvali].plot(Jvals, Fvals[:,bvali], label="$ \mu =${:.4f}".format(bvals[bvali]));

        # format this axis
        axes[xvali].legend(title="$N_B a/\lambda_i =${:.2f}".format(xvals[xvali]));
        axes[xvali].set_ylabel("$F_{avg} [ \mathbf{R}, \mathbf{U}_{"+gs+"}]$");
        the_ticks = [0.0,1.0];
        for tick in the_ticks: axes[xvali].axhline(tick,color='lightgray',linestyle='dashed');
        
    # format overall
    axes[-1].set_xlabel("$J$");
    axes[0].set_title("$N_B =${:.2f}".format(NBval));

    # show
    plt.tight_layout();
    plt.show(); e
