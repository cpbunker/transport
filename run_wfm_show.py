from transport import wfm

from run_wfm_gate import get_hblocks, get_U_gate, get_Fval, get_suptitle, get_indep_vals, lorentzian_fit;

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 5, suppress = True);

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
if(case in ["onsite_NB100", "onsite_NB500", "onsite_NB100_show"]):

    show = False;
    if("_show" in case): show = True;
    if("_NB100" in case): NBval = 100;
    elif("_NB500" in case): NBval = 500;
    else: raise NotImplementedError;
    gate_strings = ["overlap", "conc", "overlap_sf", "I"];
    gate_labels = ["$P_{swap}$", "$C$", "$P_{sf}$", "$F(\mathbf{I})$"];
    
    # change onsite energy on the qubits
    Vqvals = np.linspace(-0.75*abs(Jval), 0.75*abs(Jval), 199);
    if(show): 
        Vqvals = np.array([-0.5*abs(Jval),-0.25*abs(Jval), 0.0*abs(Jval), 0.125*abs(Jval), 0.25*abs(Jval)]);
    
    # select NBval/lambda points
    if(  NBval == 100): xvals = np.array([0.13,0.50]); 
    elif(NBval == 500): 
        if(Jval == -0.2): xvals = np.array([1.17]);
        elif(Jval == -0.05): xvals = np.array([1.20]);
    else: raise NotImplementedError;
    VBval = 0.0; # just affects titles, not code
    
    # figure and axes
    nrows, ncols = len(xvals), 1;
    vs_fig, vs_axes = plt.subplots(nrows, ncols, sharex = True, sharey = True);
    if(nrows == 1): vs_axes = [vs_axes];
    vs_fig.set_size_inches(ncols*myfigsize[0], nrows*myfigsize[1]);
    
    # iter over NB/lambda (axes)
    Fvals = np.empty((len(Vqvals), len(xvals), len(gate_strings)), dtype=float);
    for xvali in range(len(xvals)):

        # iter over qubit onsite energies (x axis)
        for Vqvali in range(len(Vqvals)):
    
            # construct hblocks from cicc-type spin ham
            hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, Vqvals[Vqvali], VBval, Vend, NBval, verbose=0);

            # define source vector (boundary condition for incident wave)
            source = np.zeros((n_loc_dof,));
            source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;
            lambdaval = NBval/xvals[xvali];
        
            if(show): # get and plot real-space scattered wavefunction
                fig, axes = plt.subplots(2, sharex = True);
                psi = wfm.kernel(hblocks, tnn, tnnn, tl, -2*tl*np.cos(2*np.pi/lambdaval), 
                	 source, is_psi_jsigma = True, is_Rhat = False, all_debug = False);
                   
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
                Fvals[Vqvali, xvali, gatei] = 1*F_gs;
                to_annotate += ","+gate_labels[gatei]+" = "+ "{:.4f}".format(F_gs);

            if(show): # plot probability densities
                axes[-1].annotate(to_annotate, (jvals[1],0.0));
                for elecspin_index in range(len(axes)):
                    for sigmai in range(n_mol_dof):
                        axes[elecspin_index].plot(jvals, pdf[:,elecspin_index* n_mol_dof+sigmai], label = "$\sigma = "+state_labels[elecspin_index*n_mol_dof+sigmai]+"$");
                    axes[elecspin_index].legend();
                    axes[elecspin_index].set_ylabel("$\langle j \sigma | \psi \\rangle$");

                # format
                axes[-1].set_xlabel("$j$");
                axes[-1].set_xlim(0, np.max(jvals));
                axes[-1].set_xticks([0, np.max(jvals)]);
                axes[0].set_title("$V_B =${:.2f}, $N_B =${:.0f}, $N_B a/\lambda_i =${:.2f}, $V_q =${:.6f}".format(VBval, NBval, NBval/lambdaval, Vqvals[Vqvali]));
                plt.tight_layout();
                plt.show();
                
        #### end loop over onsite energies

        # plot F vs qubit onsite energies
        for gatei in range(len(gate_strings)):
            vs_axes[xvali].plot(Vqvals, Fvals[:,xvali,gatei], label=gate_labels[gatei]);
            # save Fvals to .npy
            if(final_plots > 1):
                XVAL_INT_CONVERSION = 100;
                xy_savename = "data/gate/"+case+"/"+gate_strings[gatei]+"_J{:.2f}_x{:.0f}".format(Jval, XVAL_INT_CONVERSION*xvals[xvali]);
                np.save(xy_savename + "_y.npy", Fvals[:,xvali,gatei]);
                np.save(xy_savename + "_x.npy",Vqvals);
                
        # format the V vs VB plot
        #vs_axes[xvali].set_yticks(dummy_ticks);
        vs_axes[xvali].set_ylim(-0.1+dummy_ticks[0],0.1+dummy_ticks[-1]);
        for tick in dummy_ticks:
            vs_axes[xvali].axhline(tick,color='lightgray',linestyle='dashed');
        vs_axes[xvali].annotate("$N_B a/\lambda_i =${:.2f}".format(xvals[xvali]), (Vqvals[1],1.01), fontsize = myfontsize);
    
    #### end loop over xvals                

    # format
    vs_axes[-1].legend();
    vs_axes[-1].set_xlabel("$V_q$");
    suptitle = "$s = ${:.1f}, $J =${:.4f}, $N_B =${:.0f}, $V_B =${:.2f}".format(myTwoS/2, Jval, NBval, VBval);
    vs_axes[0].set_title(suptitle);

    # show
    if(final_plots > 1): # save fig and legend
        # title and color values
        np.savetxt("data/gate/"+case+"/ALL_J{:.2f}_x{:.0f}_title.txt".format(Jval, XVAL_INT_CONVERSION*xvals[-1]),XVAL_INT_CONVERSION*xvals,header = suptitle);
    else:
        plt.tight_layout();
        plt.show(); 
        
# change onsite energy in the barrier region
elif(case in ["VB_NB100", "VB_NB500", "VB_NB100_show"]): 

    show = False; # whether to plot individual wavefunctions
    if("_show" in case): show = True;
    if("_NB100" in case): NBval = 100;
    elif("_NB500" in case): NBval = 500;
    else: raise NotImplementedError;
    gate_strings = ["overlap", "conc", "overlap_sf", "I"];
    gate_labels = ["$P_{swap}$", "$C$", "$P_{sf}$", "$F(\mathbf{I})$"];
    
    # change onsite energies in barrier region to the right of the qubits
    VBvals = np.linspace(-0.0006, 0.000,199)# 399);
    if(case=="VB_NB100_show"): 
        assert(Jval == -0.2);
        VBvals=(1)*np.array([0.0, -0.0002, -0.0004, -0.0006, -0.0008, -0.00083, -0.00084, -0.0010]); 
    
    # select NB/lambda points
    if(  NBval == 100): xvals = np.array([0.13,0.50]); 
    elif(NBval == 500): 
        if(Jval == -0.2): xvals = np.array([1.00,1.17,1.50]);
        elif(Jval == -0.05): xvals = np.array([1.00,1.20,1.50]);
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
                
        #### end loop over onsite energies
                
        # format the V vs VB plot
        #vs_axes[xvali].set_yticks(dummy_ticks);
        vs_axes[xvali].set_ylim(-0.1+dummy_ticks[0],0.1+dummy_ticks[-1]);
        for tick in dummy_ticks: vs_axes[xvali].axhline(tick,color='lightgray',linestyle='dashed');
        vs_axes[xvali].annotate("$N_B a/\lambda_i =${:.2f}".format(xvals[xvali]), (VBvals[1],1.01), fontsize = myfontsize);
    
    #### end loop over xvals

    # format
    vs_axes[-1].legend();
    vs_axes[-1].set_xlabel("$V_B$");
    suptitle = "$s = ${:.1f}, $J =${:.4f}, $N_B =${:.0f}, $V_q =${:.2f}".format(myTwoS/2, Jval, NBval, Vqval);
    vs_axes[0].set_title(suptitle);

    # show
    if(final_plots > 1): # save fig and legend
        # title and color values
        np.savetxt("data/gate/"+case+"/ALL_J{:.2f}_x{:.0f}_title.txt".format(Jval, XVAL_INT_CONVERSION*xvals[-1]),XVAL_INT_CONVERSION*xvals,header = suptitle);
    else:
        plt.tight_layout();
        plt.show(); 

# double barrier well with resonant level
elif(case in ["well", "well_show"]): 

    show = False; # whether to plot individual wavefunctions
    if("_show" in case): show = True;
    if("_NB30" in case): NBval = 30;
    elif("_NB500" in case): NBval = 500;
    else: pass #raise NotImplementedError;
    is_sqrt_K = False#True;
    
    # iter over onsite energy in well (axes). Should be > 0
    VBval = 1e-5;
    Barrierval = 0.5;
    Vend = 0*Vend;
    NBvals = np.array([60,70,80,90,100,110])
    
    # figure and axes
    nrows, ncols = len(NBvals), 1;
    vs_fig, vs_axes = plt.subplots(nrows, ncols, sharex=True, sharey = True);
    if(nrows==1): vs_axes = [vs_axes];
    vs_fig.set_size_inches(ncols*myfigsize[0], nrows*myfigsize[1]);
    
    # return values
    myxvals = 999;
    Tvals = np.empty((myxvals, len(NBvals), n_loc_dof), dtype=float); # transmission prob
    
    # iter over onsite energy in well (axes)
    for NBvali in range(len(NBvals)):

        # construct hblocks from cicc-type spin ham
        # NB both left and right barrier are nonzero and same height
        # NB both Vq and VB are in the well and so have same onsite energy
        hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VBval,
              VBval, Vend, NBvals[NBvali], 
              is_Lbarrier=True, is_Rbarrier=True, barrierval=Barrierval,
              verbose=1);

        # define source vector (boundary condition for incident wave)
        source = np.zeros((n_loc_dof,));
        source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;
        
        # iter over incident KE (x axis)
        xmax = -999;
        Kvals, Energies, _ = get_indep_vals(True,"K",myxvals, xmax, NBvals[NBvali], tl);
        if(is_sqrt_K):
            Kvals_sqrt = np.linspace( np.sqrt(Kvals[0]), np.sqrt(Kvals[-1]), myxvals);
            Kvals = Kvals_sqrt*Kvals_sqrt;
            Energies = Kvals - 2*tl;
            del Kvals_sqrt;
        if(show):
            assert(NBvals[NBvali]==30)
            Kvals = np.array([0.008,0.016,0.033]);
            Energies = Kvals -2*tl;
            
        # iter over incident KE (x axis) 
        for Kvali in range(len(Kvals)):
        
            if(show): # get and plot real-space scattered wavefunction
                fig, axes = plt.subplots(2, sharex = True);
                print("K = {:.5f}, E = {:.5f}, E+2t = {:.5f}".format(Kvals[Kvali], Energies[Kvali], Energies[Kvali]+2*tl));
                psi = wfm.kernel(hblocks,tnn,tnnn,tl, Energies[Kvali], source, 
                         is_psi_jsigma = True, is_Rhat = False, all_debug = False);
                   
                # probability densities
                jvals = np.arange(len(psi));
                pdf = np.conj(psi)*psi;
                if( np.max(abs(np.imag(pdf))) > 1e-10):
                    print(np.max(abs(np.imag(pdf)))); assert False;
                pdf = np.real(pdf);

            # get transmission prob
            Rdum,Tdum=wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, 
                     is_psi_jsigma = False, is_Rhat = False, all_debug = False);
            Tvals[Kvali, NBvali, :] = Tdum[:]; 

            if(show): # plot probability densities
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
                axes[0].set_title("$V_w=${:.6f}, $N_B =${:.0f}, $K_i =${:.5f} ".format(VBval, NBvals[NBvali], Kvals[Kvali]));
                plt.tight_layout();
                plt.show();

        # bound state energies
        nEnergies = 4;
        Hbound = wfm.Hmat(hblocks, tnn, tnnn)
        Hbound = Hbound[1:-1,1:-1];
        Ebound, _ = np.linalg.eigh(Hbound);
        Ebound = Ebound + 2*tl;
        if(is_sqrt_K): Ebound = np.sqrt(Ebound);
        for En in Ebound[:n_loc_dof*nEnergies:n_loc_dof]:
            print("En = {:.5f}".format(En));
            vs_axes[NBvali].axvline(En,linestyle="dashed",color="lightgray");

        # plot F vs onsite energies VB
        if(is_sqrt_K): indep_vals = np.sqrt(Kvals);
        else: indep_vals = 1*Kvals;
        for sigmai in range(elecspin*n_mol_dof, (elecspin+1)*n_mol_dof):
            vs_axes[NBvali].plot(indep_vals, Tvals[:,NBvali,sigmai], label="$\sigma =${:.0f}".format(sigmai), color = mycolors[sigmai]);        
            # fit to lorentzian
            indep_x0_ind = np.argmin(abs(indep_vals-Ebound[0]));
            x0_ind_bounds = [int(np.max([0, indep_x0_ind-myxvals//8])), int(np.min([myxvals-1, indep_x0_ind+myxvals//8]))];
            xpoints_to_fit = indep_vals[x0_ind_bounds[0]:x0_ind_bounds[1]];
            
            #sourcei = n_mol_dof*elecspin + elems_to_keep[1] # <---- !!!
            
            ypoints_to_fit = Tvals[:,NBvali,sigmai][x0_ind_bounds[0]:x0_ind_bounds[1]];
            fitted = lorentzian_fit(xpoints_to_fit, ypoints_to_fit,
                Ebound[0],Ebound[0]/2,Ebound[0]/(2*np.pi),Ebound[0]/(2*np.pi),verbose=1);
            vs_axes[NBvali].plot(xpoints_to_fit, fitted, color="cornflowerblue",linestyle="dashed");
            
            # save Fvals to .npy
            if(final_plots > 1):
                pass;
                
        #### end loop over Kvals
                
        # format the vs Kvals plot
        dummy_ticks = [0.0,1.0];
        #vs_axes[NBvali].set_yticks(dummy_ticks);
        vs_axes[NBvali].set_ylim(-0.1,0.1+dummy_ticks[-1]);
        for tick in dummy_ticks: vs_axes[NBvali].axhline(tick,color="lightgray",linestyle="dashed");
        vs_axes[NBvali].set_ylabel("$T$");
        vs_axes[NBvali].annotate("$N_B =${:.0f}".format(NBvals[NBvali]), (indep_vals[-1],1.01), fontsize = myfontsize);
    
    #### end loop over axes

    # format
    vs_axes[-1].legend();
    vs_axes[-1].set_xlim(np.min(indep_vals), np.max(indep_vals));
    if(is_sqrt_K):
        vs_axes[-1].set_xlabel("$\sqrt{K_i/t}$");
    else:
        vs_axes[-1].set_xlabel("$K_i/t$");
        vs_axes[-1].set_xscale("log", subs = []);
    suptitle = "$s = ${:.1f}, $J =${:.4f}, $B =${:.2f}, Vend = {:.2f}".format(myTwoS/2, Jval, Barrierval, Vend);
    vs_axes[0].set_title(suptitle);

    # show
    if(final_plots > 1): # save fig and legend
        # title and color values
        np.savetxt("data/gate/"+case+"/ALL_J{:.2f}_x{:.0f}_title.txt".format(),header = suptitle);
    else:
        plt.tight_layout();
        plt.show(); 

elif(case in ["vsJ_NB100", "vsJ_NB500"]):

    # iter over J values (x axis)
    Jvals = Jval*np.linspace(1, 1/20, 39); del Jval;
    if("_NB100" in case): NBval = 100;
    elif("_NB500" in case): NBval = 500;
    else: raise NotImplementedError;
    print(Jvals)
    
    # select NBval/lambda points (different axes)
    # NB half integers and integers are never interesting
    if(  NBval == 100): xvals = np.array([0.10,0.13,0.50,0.69]); 
    elif(NBval == 500): 
        if(Jvals[0] == -0.2): xvals = np.array([1.10,1.15,1.20,1.25]); 
    else: raise NotImplementedError;
    VBval = 0.0; # zero onsite energy in barrier

    # figure and axes
    nrows, ncols = len(xvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0], nrows*myfigsize[1]);
    
    # ??? (colors)
    bvals = np.array([0.0]) #, abs(Jvals[0])/50, abs(Jvals[0])/25, abs(Jvals[0])/4]);
    
    # iter over NB/lambda (axes)
    Fvals = np.empty((len(Jvals), len(xvals), len(bvals)), dtype=float);
    for xvali in range(len(xvals)):

        # iter over colors (onsite energies)
        for bvali in range(len(bvals)):

            # iter over x axis (J)
            for Jvali in range(len(Jvals)):
               
                # construct hblocks from cicc-type spin ham
                hblocks, tnn, tnnn = get_hblocks(myTwoS,tl,Jvals[Jvali], bvals[bvali], VBval,
                				Vend, NBval, verbose=0);

                # define source vector (boundary condition for incident wave)
                source = np.zeros((n_loc_dof,));
                source[n_mol_dof*elecspin + elems_to_keep[1]] = 1.0;
                lambdaval = NBval/xvals[xvali];

                # get Rhat
                Rhat = wfm.kernel(hblocks,tnn,tnnn,tl, -2*tl*np.cos(2*np.pi/lambdaval), source, 
                     is_psi_jsigma = False, is_Rhat = True, all_debug = False);
       
                # evaluate Rhat by its fidelity w/r/t various gates
                gs = sys.argv[4];
                Uhat, dummy_ticks = get_U_gate(gs, myTwoS);
                Fvals[Jvali, xvali, bvali] = get_Fval(gs, myTwoS, Uhat, Rhat, elecspin);
                
            #### end loop over J vals

            # plot F vs J
            axes[xvali].plot(Jvals, Fvals[:,xvali,bvali], label="$V_q =${:.4f}".format(bvals[bvali]));
            if(final_plots > 1):
                # save Fvals
                XVAL_INT_CONVERSION = 100;
                xy_savename = "data/gate/"+case+"/"+gs+"_J{:.2f}_x{:.0f}".format(Jval, XVAL_INT_CONVERSION*xvals[xvali]);
                np.save(xy_savename + "_y.npy", Fvals[:,xvali,gatei]);
                np.save(xy_savename + "_x.npy",Jvals);
                
        #### end loop over colors

        # format this axis
        #axes[xvali].set_yticks(dummy_ticks);
        axes[xvali].set_ylabel(gs);
        for tick in dummy_ticks: axes[xvali].axhline(tick,color='lightgray',linestyle='dashed');
        axes[xvali].annotate("$N_B a/\lambda_i =${:.2f}".format(xvals[xvali]), (Jvals[1],1.01), fontsize = myfontsize);
        
    # format overall
    axes[-1].legend();
    axes[-1].set_xlabel("$J$");
    suptitle = "$s = ${:.1f}, $N_B =${:.0f}, $V_B =${:.2f}".format(myTwoS/2, NBval, VBval);
    axes[0].set_title(suptitle);

    # show
    if(final_plots > 1):
        pass;
    else: 
        plt.tight_layout();
        plt.show();
    
    
    
    
