from transport import wfm

from run_wfm_gate import get_hblocks, get_U_gate, get_Fval, get_suptitle, get_indep_vals;

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 4, suppress = True);

import sys

# plotting
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
NBval = int(sys.argv[2]);
case = sys.argv[3];
Vend = 5.0*tl;
VBar = 0.0*tl; # just affects title, not implemented physically
gate_strings = ["SWAP", "I", "conc"];

if(case in ["xvals"]):
    ferromagnetic = False;
    Lbarrier = False; # whether to put small barrier at LL-SR junction
    Lbarrier_val = 0.1;
    Rbarrier = False; # spin dep terms immediately to left of VB
    
    # select NBval/lambda points
    if(  NBval == 100): xvals = [0.13, 0.50, 0.70,1.00]; cicc_offset=0;
    elif(NBval == 200): xvals = [0.26, 1.00]; cicc_offset=NBval//2;
    elif(NBval == 500): xvals = [0.13, 0.30, 0.50, 0.63, 1.00, 1.17]; cicc_offset=0;
    else: raise NotImplementedError;
    for xval in xvals:
        lambdaval = NBval/xval;
        
        # plotting
        fig, axes = plt.subplots(2, sharex = True); # for elec spin up and elec spin down
        
        # construct hblocks from cicc-type spin ham
        hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VB, NBval,
                        is_Lbarrier=Lbarrier, is_Rbarrier=Rbarrier, bval = Lbarrier_val, the_offset=cicc_offset, verbose=1);

        # define source vector (boundary condition for incident wave)
        source = np.zeros((n_loc_dof,));
        source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;
        
        # get real-space scattered wavefunction
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
        print("Rhat =\n", Rhat[elecspin*n_mol_dof:(elecspin+1)*n_mol_dof, elecspin*n_mol_dof:(elecspin+1)*n_mol_dof]);
        
        # evaluate Rhat by its fidelity w/r/t various gates
        to_annotate = "";
        for gs in gate_strings:
            Uhat, dummy_ticks = get_U_gate(gs, myTwoS);
            F_gs = get_Fval(gs, myTwoS, Uhat, Rhat, elecspin, ferromagnetic);
            to_annotate += ", $F_{avg} [ \mathbf{R}, \mathbf{U}_{"+gs+"}] = $"+ "{:.4f}".format(F_gs);
        axes[-1].annotate(to_annotate, (jvals[1],0.0));

        # plot probability densities       
        for elecspin_index in range(len(axes)):
            for sigmai in range(n_mol_dof):
                axes[elecspin_index].plot(jvals, pdf[:,elecspin_index*n_mol_dof+sigmai],
                           label = "$\sigma = "+state_labels[elecspin_index*n_mol_dof+sigmai]+"$");
            axes[elecspin_index].legend();
            axes[elecspin_index].set_ylabel("$\langle j \sigma | \psi \\rangle$");
            #axes[elecspin_index].set_ylim(0,1);


        # format
        axes[-1].set_xlabel("$j$");
        axes[-1].set_xlim(0, np.max(jvals));
        axes[-1].set_xticks([0, np.max(jvals)]);
        axes[0].set_title("$J = {:.2f}, \lambda_i/a = {:.2f}, N_B a/\lambda_i = {:.2f}$".format(Jval, lambdaval, NBval/lambdaval));
                                   
        # show
        plt.tight_layout();
        plt.show();   
         
elif(case in ["offset"]): # move qubit locations with barrier enforcing LL-SR junction
    ferromagnetic = False;
    Lbarrier = True; # whether to put small barrier at LL-SR junction
    Lbarrier_val = 0.0001;
    show = False;
    
    # select NBval/lambda points
    if(  NBval == 100): xval = 0.13; 
    elif(NBval == 500): xval = 0.63;
    else: raise NotImplementedError;
    lambdaval = NBval/xval;

    # iter over cicc offset
    offsetvals = np.arange(0.1*NBval, 0.9*NBval, 0.1*NBval, dtype=int);
    Fvals = np.empty((len(offsetvals), len(gate_strings)),dtype=float);
    for offsetvali in range(len(offsetvals)):
        
        # plotting
        fig, axes = plt.subplots(2, sharex = True); # for elec spin up and elec spin down
        
        # construct hblocks from cicc-type spin ham
        hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VB, NBval,
                        the_offset = offsetvals[offsetvali],
                        is_Lbarrier=Lbarrier, bval=Lbarrier_val, verbose=1);

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
            gs = gate_strings[gatei];
            Uhat, dummy_ticks = get_U_gate(gs, myTwoS);
            F_gs = get_Fval(gs, myTwoS, Uhat, Rhat, elecspin, ferromagnetic);
            Fvals[offsetvali, gatei] = 1*F_gs;
            to_annotate += ", $F_{avg} [ \mathbf{R}, \mathbf{U}_{"+gs+"}] = $"+ "{:.4f}".format(F_gs);

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
            axes[0].set_title("$J = {:.2f}, \lambda_i/a = {:.2f}, N_B a/\lambda_i = {:.2f}$".format(Jval, lambdaval, NBval/lambdaval));
            plt.tight_layout();
            plt.show();
        else:
            plt.close();

    # plot F vs offset
    vs_fig, vs_ax = plt.subplots();
    for gatei in range(len(gate_strings)):
        vs_ax.plot(offsetvals, Fvals[:,gatei], label="$U_{"+gate_strings[gatei]+"}$");

    # format
    vs_ax.legend();
    vs_ax.set_xlabel("$N_{offset}$");
    vs_ax.set_ylabel("$F_{avg} [ \mathbf{R}, \mathbf{U}] $");
    the_ticks = [0.0,1.0];
    for tick in the_ticks: vs_ax.axhline(tick,color='lightgray',linestyle='dashed');
    vs_ax.set_title("$J =${:.2f}, $N_B =${:.2f}, $N_B/\lambda_i =${:.2f}, $B_L =${:.4f}".format(Jval, NBval, xval, Lbarrier_val));

    # show
    for row in Fvals: print(row);
    plt.tight_layout();
    plt.show();
        
elif(case in ["onsite"]):
    ferromagnetic = False;
    SRbarrier = True; # whether to change onsite energy of localized spins
    show = False;
    
    # select NBval/lambda points
    if(  NBval == 100): xval = 0.13; 
    elif(NBval == 500): xval = 0.63;
    else: raise NotImplementedError;
    lambdaval = NBval/xval;

    # iter over onsite energies
    bvals = np.linspace(-0.2*abs(Jval), 0.4*abs(Jval), 49);
    Fvals = np.empty((len(bvals), len(gate_strings)), dtype=float);
    for bvali in range(len(bvals)):
        
        # plotting
        fig, axes = plt.subplots(2, sharex = True); # for elec spin up and elec spin down
        
        # construct hblocks from cicc-type spin ham
        hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VB, NBval,
                        is_SRbarrier = SRbarrier, bval=bvals[bvali], verbose=0);

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
            gs = gate_strings[gatei];
            Uhat, dummy_ticks = get_U_gate(gs, myTwoS);
            F_gs = get_Fval(gs, myTwoS, Uhat, Rhat, elecspin, ferromagnetic);
            Fvals[bvali, gatei] = 1*F_gs;
            to_annotate += ", $F_{avg} [ \mathbf{R}, \mathbf{U}_{"+gs+"}] = $"+ "{:.4f}".format(F_gs);

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
            axes[0].set_title("$J = {:.2f}, \lambda_i/a = {:.2f}, N_B a/\lambda_i = {:.2f}$".format(Jval, lambdaval, NBval/lambdaval));
            plt.tight_layout();
            plt.show();
        else:
            plt.close();

    # plot F vs offset
    vs_fig, vs_ax = plt.subplots();
    for gatei in range(len(gate_strings)):
        vs_ax.plot(bvals, Fvals[:,gatei], label="$U_{"+gate_strings[gatei]+"}$");

    # format
    vs_ax.legend();
    vs_ax.set_xlabel("$\mu $");
    vs_ax.set_ylabel("$F_{avg} [ \mathbf{R}, \mathbf{U}] $");
    the_ticks = [0.0,1.0];
    for tick in the_ticks: vs_ax.axhline(tick,color='lightgray',linestyle='dashed');
    vs_ax.set_title("$J =${:.2f}, $N_B =${:.2f}, $N_B/\lambda_i =${:.2f}".format(Jval, NBval, xval));

    # show
    for row in Fvals: print(row);
    plt.tight_layout();
    plt.show();

elif(case in ["V0", "V0_zoom"]):
    ferromagnetic = False;
    V0barrier = True; # change onsite energies to the right of the qubits
    show = False;
    
    # select NBval/lambda points
    if(  NBval == 100): xval = 0.13; 
    elif(NBval == 500): xval = 0.63;
    else: raise NotImplementedError;
    lambdaval = NBval/xval;

    # iter over onsite energies
    bvals = np.linspace(-0.01, 0, 99);
    bvals = np.linspace(-0.05, -0.04, 99);
    if(case=="V0_zoom"): bvals = np.array([-0.15714, -0.157, -0.157/2, -0.157/4, 0.00]); show = True; assert(Jval == -0.2);
    Fvals = np.empty((len(bvals), len(gate_strings)), dtype=float);
    for bvali in range(len(bvals)):
        
        # plotting
        fig, axes = plt.subplots(2, sharex = True); # for elec spin up and elec spin down
        
        # construct hblocks from cicc-type spin ham
        hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VB, NBval,
                        is_V0 = V0barrier, bval=bvals[bvali], verbose=0);

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
        print("\n",bvals[bvali])
        for gatei in range(len(gate_strings)):
            gs = gate_strings[gatei];
            Uhat, dummy_ticks = get_U_gate(gs, myTwoS);
            F_gs = get_Fval(gs, myTwoS, Uhat, Rhat, elecspin, ferromagnetic);
            print(gs, F_gs)
            Fvals[bvali, gatei] = 1*F_gs;
            to_annotate += ", $F_{avg} [ \mathbf{R}, \mathbf{U}_{"+gs+"}] = $"+ "{:.4f}".format(F_gs);

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
            axes[0].set_title("$J = {:.2f}, \lambda_i/a = {:.2f}, N_B a/\lambda_i = {:.2f}$".format(Jval, lambdaval, NBval/lambdaval));
            plt.tight_layout();
            plt.show();
        else:
            plt.close();

    # plot F vs offset
    vs_fig, vs_ax = plt.subplots();
    for gatei in range(len(gate_strings)):
        vs_ax.plot(bvals, Fvals[:,gatei], label="$U_{"+gate_strings[gatei]+"}$");

    # format
    vs_ax.legend();
    vs_ax.set_xlabel("$V_0$");
    vs_ax.set_ylabel("$F_{avg} [ \mathbf{R}, \mathbf{U}] $");
    the_ticks = [0.0,1.0];
    for tick in the_ticks: vs_ax.axhline(tick,color='lightgray',linestyle='dashed');
    vs_ax.set_title("$J =${:.2f}, $N_B =${:.2f}, $N_B/\lambda_i =${:.2f}".format(Jval, NBval, xval));

    # show
    for row in Fvals: print(row);
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
    plt.show(); 
