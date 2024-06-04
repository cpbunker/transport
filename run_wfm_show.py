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
VB = 5.0*tl;
V0 = 0.0*tl; # just affects title, not implemented physically
ferromagnetic = False;

# iter over differnt NBval/lambda points
if(  NBval == 100): xvals = [0.13, 0.50, 0.70,1.00];
elif(NBval == 200): xvals = [0.26, 1.00];
elif(NBval == 500): xvals = [0.13, 0.30, 0.50, 0.63, 1.00, 1.17];
else: raise NotImplementedError;
for xval in xvals:
    lambdaval = NBval/xval;
    
    # plotting
    fig, axes = plt.subplots(2, sharex = True); # for elec spin up and elec spin down
    
    # construct hblocks from cicc-type spin ham
    hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VB, NBval, the_offset = NBval//2, verbose = 1);
    #assert False;

    # define source vector (boundary condition for incident wave)
    source = np.zeros((n_loc_dof,));
    source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;
    
     # FM leads = modify so only up-up hopping allowed
    if(ferromagnetic): 
        tnn[0] = np.zeros( (n_loc_dof,), dtype=float);
        for sigmai in range(n_mol_dof): tnn[0][sigmai, sigmai] = -tl;
                
        if(False): # printing
            for jindex in [0,1,2,len(tnn)-3, len(tnn)-2,len(tnn)-1]:
                print("j = {:.0f} <-> j = {:.0f}".format(jindex, jindex+1));
                print(tnn[jindex]);
            assert False;
    # new code < -------------- !!!!!!!!!!!!
    
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
    gate_strings = ["SWAP", "I"];
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
     
     
     
