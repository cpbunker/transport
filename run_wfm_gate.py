'''
Christian Bunker
M^2QM at UF
November 2022

Scattering from two spin-s MSQs
Want to make a SWAP gate
solved in time-independent QM using wfm method in transport/wfm
'''

from transport import wfm, tdfci
from transport.tdfci import utils

import numpy as np
import matplotlib.pyplot as plt

import sys

# constructing the hamiltonian
def h_cicc(TwoS, J, i1, i2, verbose=0) -> np.ndarray: 
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
    mol_dof = (TwoS+1)*(TwoS+1);
    
    # Sd ops 
    TwoS_ladder = (2*np.arange(TwoS+1) -TwoS)[::-1];
    Seye = np.eye(TwoS+1);
    Sz = np.diagflat(0.5*TwoS_ladder)
    Splus = np.diagflat(np.sqrt(0.5*TwoS*(0.5*TwoS+1)-0.5*TwoS_ladder[1:]*(0.5*TwoS_ladder[1:]+1)),k=1);
    Sminus = np.diagflat(np.sqrt(0.5*TwoS*(0.5*TwoS+1)-0.5*TwoS_ladder[:-1]*(0.5*TwoS_ladder[:-1]-1)),k=-1);
    if(verbose and TwoS > 1):
        print("TwoS_ladder =\n",TwoS_ladder)
        print("Sz = \n",Sz)
        print("S+ = \n",Splus)
        print("S- = \n",Sminus)

    
    # heisenberg interaction matrices
    Se_dot_S1_z = tdfci.utils.mat_4d_to_2d(np.tensordot(Sz, Seye, axes=0));
    Se_dot_S1_z = tdfci.utils.mat_4d_to_2d(np.tensordot(np.array([[0.5,0],[0,-0.5]]), Se_dot_S1_z, axes=0));
    Se_dot_S1_pm = tdfci.utils.mat_4d_to_2d(np.tensordot(Sminus, Seye, axes=0));
    Se_dot_S1_pm = tdfci.utils.mat_4d_to_2d(np.tensordot(np.array([[0,1],[0,0]]), Se_dot_S1_pm, axes=0));
    Se_dot_S1_mp = tdfci.utils.mat_4d_to_2d(np.tensordot(Splus, Seye, axes=0));
    Se_dot_S1_mp = tdfci.utils.mat_4d_to_2d(np.tensordot(np.array([[0,0],[1,0]]), Se_dot_S1_mp, axes=0));
    Se_dot_S1 = J*(Se_dot_S1_z + 0.5*(Se_dot_S1_pm + Se_dot_S1_mp));
    Se_dot_S2_z = tdfci.utils.mat_4d_to_2d(np.tensordot(Seye, Sz, axes=0));
    Se_dot_S2_z = tdfci.utils.mat_4d_to_2d(np.tensordot(np.array([[0.5,0],[0,-0.5]]), Se_dot_S2_z, axes=0));
    Se_dot_S2_pm = tdfci.utils.mat_4d_to_2d(np.tensordot(Seye, Sminus, axes=0));
    Se_dot_S2_pm = tdfci.utils.mat_4d_to_2d(np.tensordot(np.array([[0,1],[0,0]]), Se_dot_S2_pm, axes=0));
    Se_dot_S2_mp = tdfci.utils.mat_4d_to_2d(np.tensordot(Seye, Splus, axes=0));
    Se_dot_S2_mp = tdfci.utils.mat_4d_to_2d(np.tensordot(np.array([[0,0],[1,0]]), Se_dot_S2_mp, axes=0));
    Se_dot_S2 = J*(Se_dot_S2_z + 0.5*(Se_dot_S2_pm + Se_dot_S2_mp));
    #print(Se_dot_S1);
    #print(Se_dot_S2);
    #assert False

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
    
def get_hblocks(TwoS, the_tl, the_J, the_VB, the_NB, verbose = 0):
    '''
    '''
    the_nlocdof = 2*(TwoS+1)*(TwoS+1);
    
    # ciccarello type interaction at beginning
    hblocks_cicc = h_cicc(TwoS, the_J, [1],[2]);
    if(verbose):
        print("hblocks_cicc = ");
        for block in hblocks_cicc: print(np.real(block));

    # add large barrier at end
    NC = len(hblocks_cicc); assert(NC==3); # num sites in central region
    hblocks_all, tnn_all = [], []; # new empty array all the way to barrier, will add cicc later
    for _ in range(NC+the_NB):
        hblocks_all.append(0.0*np.eye(the_nlocdof));
        tnn_all.append(-the_tl*np.eye(the_nlocdof));
    hblocks_all, tnn_all = np.array(hblocks_all,dtype=complex), np.array(tnn_all[:-1]);
    hblocks_all[0:NC] += hblocks_cicc;
    hblocks_all[-1] += the_VB*np.eye(the_nlocdof);
    tnnn_all = np.zeros_like(tnn_all)[:-1]; # no next nearest neighbor hopping
    
    # the diagonal term must be the same for all channels!
    for sigmai in range(the_nlocdof):
        assert(hblocks_all[0][sigmai, sigmai]==0.0);
    
    # return
    return hblocks_all, tnn_all, tnnn_all;
    
def get_U_gate(gate0, TwoS):
    '''
    '''
    if(gate0 in ["I", "RZI", "SeS12", "RZSeS12", "conc", "overlap"]): # identity and non-gate quantities
        ticks = [0.0,1.0];
        proj_choice = "identical";
        if(gate0 in ["I", "RZI"]): U_q = np.eye(4, dtype=complex);
        else: U_q = np.nan*np.eye(4, dtype=complex);
    elif(gate0=="SQRT"):
        ticks = [-1.0,0.0,1.0];
        proj_choice = "identical";
        U_q = np.array([[1,0,0,0],
                       [0,complex(0.5,0.5),complex(0.5,-0.5),0],
                       [0,complex(0.5,-0.5),complex(0.5,0.5),0],
                       [0,0,0,1]], dtype=complex); #  SWAP^1/2 gate
    elif(gate0=="SWAP"):
        ticks = [0.0,1.0];
        proj_choice = "identical";
        U_q = np.array([[1,0,0,0],
                   [0,0,1,0],
                   [0,1,0,0],
                   [0,0,0,1]], dtype=complex); # SWAP gate
    elif(gate0[:2]=="RZ" and ((gate0[-1] in ["1","2","3","4"]) and len(gate0)==3) ): # roots of SWAP
        ticks = [-1.0,0.0,1.0];
        proj_choice = "identical";
        root = int(gate0[-1]); # tells us it is nth root, ie angle = pi/n
        U_q = np.array([[1,0,0,0],
                       [0,0.5+0.5*np.exp(complex(0,np.pi/root)),0.5-0.5*np.exp(complex(0,np.pi/root)),0],
                       [0,0.5-0.5*np.exp(complex(0,np.pi/root)),0.5+0.5*np.exp(complex(0,np.pi/root)),0],
                       [0,0,0,1]], dtype=complex); # roots of SWAP, that achieves Z rotation on ST0 encoding
    elif(gate0=="CNZ"):
        ticks = [0.0,1.0];
        proj_choice = "treversal";
        U_q = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,-1,0],
                   [0,0,0,1]], dtype=complex); # controlled (-Z)
    elif(gate0=="RX"):
        ticks = [0.0,1.0];
        proj_choice = "treversal";
        root = 4; # results are misleading if smaller
        U_q = np.array([[1,0,0,0],
                   [0,np.exp(complex(0,-np.pi/(2*root))),0,0],
                   [0,0,np.exp(complex(0,np.pi/(2*root))),0],
                   [0,0,0,1]], dtype=complex); # roots of Z * phase, that achieves X rotation on ST0 encoding
    else:
        raise NotImplementedError("gate0 = "+gate0+" not supported");
        
    # from Uq to Ugate
    mol_dof = (TwoS+1)*(TwoS+1);
    U_mol = np.eye(mol_dof, dtype=complex); # acts on full space of 2 molecular spins
    # matrix elements to keep are |s,s>,|s,s-1>,|s-1,s>,|s-1,s-1>. Rest should be diagonal
    elems_to_keep = [0,1,TwoS+1,TwoS+1+1];
    for elemi in range(len(elems_to_keep)):
        for elemj in range(len(elems_to_keep)):
            U_mol[elems_to_keep[elemi],elems_to_keep[elemj]] = U_q[elemi,elemj]; 
    # project onto electron spin
    U = np.zeros( (2*mol_dof,2*mol_dof), dtype=complex);
    if(proj_choice == "identical"): # elec up and elec down blocks are exactly the same
        U[:mol_dof,:mol_dof] = U_mol[:,:];
        U[mol_dof:,mol_dof:] = U_mol[:,:];
    elif(proj_choice == "treversal"): # determine elec down block by time reversal symmetry
        assert(TwoS == 1);
        U[:mol_dof,:mol_dof] = U_mol[:,:];
        U[mol_dof:,mol_dof:] = np.diagflat(np.diag(U_mol)[::-1]);
    else:
        raise NotImplementedError("proj_choice = "+proj_choice+" not supported");

    # return
    return U, ticks;

def get_Fval(gate0, TwoS, U, R, the_espin, is_FM):
    '''
    '''
    assert(np.shape(R) == np.shape(U));
    assert(len(U)==2*(TwoS+1)*(TwoS+1)); # this affects results even when off-diagonal blocks are zero, due to 1/nd dependence

    # from Uq to Ugate
    mol_dof = (TwoS+1)*(TwoS+1); 
    elems_to_keep = [0,1,TwoS+1,TwoS+1+1]; # these are mol_dof elements, ie 0 represent up_1 up_2, no electron dofs

    if("SeS12" in gate0): # do not actually get fidelity, instead quantify R^out
        Rout = R[:mol_dof, mol_dof:]; 
        # maximize over rows
        R_rows = np.zeros((mol_dof,),dtype=float);
        for row_sigma in range(mol_dof):
            # sum over columns
            Rout_row_mags = np.real(np.conj(Rout[row_sigma])*Rout[row_sigma]);
            R_rows[row_sigma] = np.sum(Rout_row_mags);
        the_trace = np.max(R_rows);
        
    elif(gate0 == "conc"): # do not actually get fidelity, get qubit-qubit concurrence from *some* initial state
        in_state = np.zeros((len(R),),dtype=complex);
        in_state[elems_to_keep[1]] = 1.0;
        out_state = np.matmul(R, in_state);
        Y_otimes_Y_q = np.array([[ 0, 0, 0,-1],
                                 [ 0, 0, 1, 0],
                                 [ 0, 1, 0, 0],
                                 [-1, 0, 0, 0]]);
        Y_otimes_Y = np.zeros_like(R);
        Y_otimes_Y[:mol_dof, :mol_dof] = Y_otimes_Y_q[:,:];
        Y_otimes_Y[mol_dof:, mol_dof:] = Y_otimes_Y_q[:,:];
        the_trace = np.dot( np.conj(out_state), np.matmul(Y_otimes_Y, np.conj(out_state))); # inner product
        the_trace = np.sqrt(np.conj(the_trace)*the_trace); # norm of that inner product is the concurrence
        
    elif(gate0 == "overlap"): # do not actually get fidelity, get overlap with qubit-flipped partner of initial state
        in_state = np.zeros((len(R),),dtype=complex);
        in_state[elems_to_keep[1]] = 1.0;
        out_state = np.matmul(R, in_state);
        overlap_state = np.zeros_like(out_state, dtype=complex);
        overlap_state[elems_to_keep[2]] = 1.0;
        the_trace = np.dot(np.conj(overlap_state), out_state); # overlap
        the_trace = np.sqrt(np.conj(the_trace)*the_trace); # norm 

    else: # actually get fidelity
    
        # truncate for ferromagnetic leads 
        if(is_FM):
            U=U[the_espin*mol_dof:(the_espin+1)*mol_dof, the_espin*mol_dof:(the_espin+1)*mol_dof];
            R=R[the_espin*mol_dof:(the_espin+1)*mol_dof, the_espin*mol_dof:(the_espin+1)*mol_dof]; 
    
        # Molmer formula
        print(np.shape(U), np.shape(R));
        M_matrix = np.matmul(np.conj(U.T), R); # M = U^\dagger R
        the_trace = np.sqrt((np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U)*(len(U)+1)));

    # return
    if(abs(np.imag(the_trace)) > 1e-10): print(the_trace); assert False;
    return np.real(the_trace);
    
def get_suptitle(TwoS, theJ, theV0, theVB, is_FM, is_overt):
    '''
    '''
    suptitle = "$s=${:.1f}, $J=${:.2f}, $V_0=${:.1f}, $V_B=${:.1f}".format(0.5*TwoS, theJ, theV0, theVB);
    if(is_overt): "$s=${:.1f}, $J/t=${:.2f}, $V_0/t=${:.1f}, $V_B/t=${:.1f}".format(0.5*TwoS, theJ, theV0, theVB);
    
    # add-ons
    if(is_FM): suptitle += " (FM leads)";
    return suptitle;
    
def get_indep_vals(is_NB_fixed, is_K_indep, the_xvals, the_xmax, the_NB, the_tl):
    '''
    '''
    
    # most of the time, NB is fixed for a give color, axis, and only wavenumber changes on x axis
    if(is_NB_fixed):
        if(is_K_indep):
            the_Kpowers = np.array([-2,-3,-4,-5]); # incident kinetic energy/t = 10^Kpower
            the_wavelengths = 2*np.pi/np.sqrt(np.logspace(the_Kpowers[0], the_Kpowers[-1], num=the_xvals));
        else:
            the_NBoverLambda = np.linspace(0.001, the_xmax - 0.001, the_xvals);
            the_wavelengths = the_NB/the_NBoverLambda
        # dispersion
        the_Kvals = 2*the_tl - 2*the_tl*np.cos(2*np.pi/the_wavelengths);
        # -2t < Energy < 2t, the argument of self energies, Green's funcs, etc
        the_Energies = the_Kvals - 2*the_tl; 
    
        if(is_K_indep): the_indep_vals = 4*np.pi*np.pi/(the_wavelengths*the_wavelengths);
        else: the_indep_vals = 1*the_NBoverLambda;
        
        return the_Kvals, the_Energies, the_indep_vals;
        
    else: # change NB on the x axis
        assert(the_NB == None);
        the_Kpowers = np.array([-2,-3]) #,-4,-5]); # incident kinetic energy/t = 10^Kpower
        
        # wavenumbers
        the_knumbers = np.sqrt(np.logspace(the_Kpowers[0], the_Kpowers[-1], num=len(the_Kpowers)));
        
        # dispersion
        the_Kvals = 2*the_tl - 2*tl*np.cos(the_knumbers);
        # -2t < Energy < 2t, the argument of self energies, Green's funcs, etc
        the_Energies = the_Kvals - 2*the_tl; 
        
        return the_Kvals, the_Energies, the_knumbers, the_Kpowers;   
           
############################################################################ 
#### exec code
if(__name__ == "__main__"):

    #### top level
    np.set_printoptions(precision = 2, suppress = True);
    verbose = 1;
    case = sys.argv[2];
    final_plots = int(sys.argv[3]);
    #if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
    summed_columns = True;
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
    myTwoS = 1;
    n_mol_dof = (myTwoS+1)*(myTwoS+1); 
    n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
    Jval = float(sys.argv[1]);
    VB = 5.0*tl;
    V0 = 0.0*tl; # just affects title, not implemented physically

if(__name__ == "__main__" and case in ["swap_NB","swap_NB_lambda"]): # distance of the barrier NB on the x axis
    
    # axes
    nrows, ncols = 4, 4;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    
    # cases / other options
    if("_lambda" in case): NB_indep = False # whether to put NB, alternatively wavenumber*NB
    else: NB_indep = True;
    ferromagnetic = False;
    if("swap" in case): which_gate = "SWAP";
    else: raise NotImplementedError;
    U_gate, the_ticks = get_U_gate(which_gate, myTwoS);

    # iter over incident kinetic energy (colors)
    Kvals, Energies, knumbers, Kpowers = get_indep_vals(False, None, None, None, None, tl); 
    # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
    
    # return value
    rhatvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals),dtype=complex); 
    # by  init spin, final spin, energy, NB
    Fvals_min = np.empty((len(Kvals), myxvals),dtype=float);
    for Kvali in range(len(Kvals)):
        
        # iter over barrier distance (x axis)
        kNBmax = 0.75*np.pi #0.75*np.pi;
        if(NB_indep): 
            NBmax = 150;
            NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
            indep_vals = 1*NBvals;
        else: 
            NBmax = int(kNBmax/knumbers[Kvali]);
            NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
            indep_vals = NBvals/(2*np.pi/knumbers[Kvali]);
        if(verbose): print("k^2, NBmax = ", knumbers[Kvali]**2, NBmax); 
        
        # iter over barrier distance (x axis)
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # construct hblocks from cicc-type spin ham
            hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VB, NBval);

            # define source, although it doesn't function as a b.c. since we return Rhat matrix
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            
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
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, False, is_Rhat = True, all_debug = False);
            
            # fidelity w/r/t U_gate
            Fvals_min[Kvali, NBvali] = get_Fval(which_gate, myTwoS, U_gate, rhatvals[:,:,Kvali,NBvali], elecspin, ferromagnetic);  
            
        #### end loop over NB

        # process R output
        if(elecspin==1): # final e state (column, 2nd index) is spin up
            rhatvals_offdiag = rhatvals[:,np.array(range(n_loc_dof))<n_mol_dof]; 
        elif(elecspin==0): # final e state (column, 2nd index) is spin down
            rhatvals_offdiag = rhatvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        yvals = np.copy(rhatvals); rbracket = "";
        if(which_gate not in ["SQRT", "RX", "RZ"]): # make everything real
            rbracket = "|"
            yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals)); 
            
        # determine fidelity and kNB*, ie x val where the SWAP happens
        indep_argmax = np.argmax(Fvals_min[Kvali]);
        indep_star = indep_vals[indep_argmax];
        if(verbose):
            indep_comment = case+": indep_star, fidelity(indep_star) = {:.6f}, {:.4f}".format(indep_star, Fvals_min[Kvali,indep_argmax]);
            print(indep_comment,"\n",rhatvals[elecspin*n_mol_dof:(elecspin+1)*n_mol_dof, elecspin*n_mol_dof:(elecspin+1)*n_mol_dof,Kvali,indep_argmax]);
               
        # plot as a function of NBvals
        elems_to_keep = np.array([0,1,myTwoS+1,myTwoS+1+1]);
        for sourcei in range(len(elems_to_keep)):
            for sigmai in range(sourcei+1):
                # formatting
                if(myTwoS > 1): axes[sourcei,sigmai].set_title(str(n_mol_dof*elecspin+elems_to_keep[sourcei])+" $\\rightarrow$"+str(n_mol_dof*elecspin+elems_to_keep[sigmai]));
                else: axes[sourcei,sigmai].set_title("$"+rbracket+"\langle"+str(ylabels[4*elecspin+sourcei])+"| \mathbf{R} |"+str(ylabels[4*elecspin+sigmai])+"\\rangle"+rbracket+"$");
                axes[sourcei,sigmai].set_yticks(the_ticks);
                axes[sourcei,sigmai].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks: axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                axes[-1,sigmai].set_xlim(0,np.max(indep_vals));
                if(NB_indep):
                    axes[-1,sigmai].set_xlabel("$N_B$",fontsize=myfontsize);
                else:
                    axes[-1,sigmai].set_xlabel("$N_B a/\lambda_i$",fontsize=myfontsize);
 
                # plot rhat (real part = solid, imag part = dashed)
                if(abs(knumbers[1] - np.sqrt(10.0**Kpowers[1])) < 1e-10):
                    mylabel = "$k_i^2 a^2 = 10^{"+str(Kpowers[Kvali])+"}$"
                else: 
                    mylabel = "$k_i^2 a^2 = {:.6f} $".format((2*np.pi/wavelengths)[Kvali]**2);
                axes[sourcei, sigmai].plot(indep_vals, np.real(yvals)[n_mol_dof*elecspin+elems_to_keep[sourcei], n_mol_dof*elecspin+elems_to_keep[sigmai],Kvali], label = mylabel, color=mycolors[Kvali]);
                if(rbracket != "|"): axes[sourcei,sigmai].plot(indep_vals, np.imag(yvals)[n_mol_dof*elecspin+elems_to_keep[sourcei], n_mol_dof*elecspin+elems_to_keep[sigmai],Kvali], linestyle="dashed", color=mycolors[Kvali]);
 
                # plot fidelity, starred SWAP locations
                if(sourcei==2 and sigmai==1):
                    axes[sigmai,sourcei].plot(indep_vals,Fvals_min[Kvali], label = "$k_i^2 a^2= 10^{"+str(Kpowers[Kvali])+"}$",color=mycolors[Kvali]);

                    for tick in the_ticks: axes[sigmai,sourcei].axhline(tick,color='lightgray',linestyle='dashed');
                    axes[sigmai,sourcei].set_title("$F_{avg}(\mathbf{R}, \mathbf{U}_{"+which_gate+"})$");
                    axes[1,0].legend();
                   
                # plot reflection summed over final states (columns)
                if((sourcei in [1,2] and sigmai==0) and summed_columns):                
                    # < elec up row state| \sum |final states elec down>. to measure Se <-> S1(2)
                    # NB rhatvals_offdiag have shape (8,4)
                    which_rhatvals_offdiag = rhatvals_offdiag[elecspin*n_mol_dof + elems_to_keep[sourcei], :,Kvali];
                    axes[sourcei,sigmai+len(elems_to_keep)-1].plot(indep_vals, np.sum(np.real(np.conj(which_rhatvals_offdiag)*which_rhatvals_offdiag),axis=0), label="dummy", color=mycolors[Kvali]);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_title("$\sum_{\sigma_1 \sigma_2}| \langle"+str(ylabels[n_mol_dof*elecspin+sourcei])+"| \mathbf{R} |"+["\downarrow_e","\\uparrow_e"][elecspin]+" \sigma_1 \sigma_2 \\rangle |^2$");
                    
                    # format summed final states
                    axes[sourcei,sigmai+len(elems_to_keep)- 1].set_yticks(the_ticks);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                    for tick in the_ticks: axes[sourcei,sigmai+len(elems_to_keep) -1].axhline(tick,color='lightgray',linestyle='dashed');
                    
                    # difference between diagonal elements of r
                    axes[0,sigmai+len(elems_to_keep)-1].plot(indep_vals, np.real(yvals)[1,1,Kvali] - np.real(yvals)[2,2,Kvali], label = "$N_B$ = {:.0f}".format(NBvals[Kvali]),color=mycolors[Kvali]);
                    if(rbracket != "|"): axes[0,sigmai+len(elems_to_keep)-1].plot(indep_vals, np.imag(yvals)[1,1,Kvali] - np.imag(yvals)[2,2,Kvali], label = "$N_B$ = {:.0f}".format(NBvals[Kvali]),color=mycolors[Kvali],linestyle="dashed");
                    axes[0,sigmai+len(elems_to_keep)-1].set_title("$"+rbracket+"\langle"+str(1)+"| \mathbf{R} |"+str(1)+"\\rangle"+rbracket+" - "+rbracket+"\langle"+str(2)+"| \mathbf{R} |"+str(2)+"\\rangle"+rbracket+"$");
                    
    # show
    fig.suptitle(get_suptitle(myTwoS, Jval, V0, VB, ferromagnetic, False));
    plt.tight_layout();
    if(final_plots > 1): # save fig
        Jstring = "";
        if(Jval != -0.2): Jstring ="J"+ str(int(abs(100*Jval)))+"_";
        sstring = "";
        if(myTwoS != 1): sstring = "2s"+str(int(myTwoS))+"_";
        fname = "figs/gate/spin12_"+Jstring+sstring+case;
        plt.savefig(fname+".pdf")
    else:
        plt.show();

elif(__name__ == "__main__" and case in["swap_K","swap_lambda"]): # incident kinetic energy or wavenumber on the x axis
         # NB is now fixed !!!!

    # axes
    nrows, ncols = 4,4;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    
    # cases / other options
    if("_lambda" in case): K_indep = False;
    elif("_K" in case): K_indep = True; # whether to put ki^2 on x axis, alternatively NBa/lambda
    ferromagnetic = False;
    if("swap" in case): which_gate = "SWAP";
    else: raise NotImplementedError;
    U_gate, the_ticks = get_U_gate(which_gate, myTwoS);

    # iter over fixed NB (colors)
    NBvals = np.array([100]);
    Fvals_min = np.empty((myxvals, len(NBvals)),dtype=float); 
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):

        # set barrier distance
        NBval = int(NBvals[NBvali]); 
        if(verbose): print("NB = ",NBval);
        
        # iter over incident wavenumber (x axis)
        xmax = 1.1;
        Kvals, Energies, indep_vals = get_indep_vals(True, K_indep, myxvals, xmax, NBval, tl); 
        # -2t < Energy < 2t, the argument of self energies, Green's funcs, etc 
        for Kvali in range(len(Kvals)):

            # construct hblocks from cicc-type spin ham
            hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, VB, NBval);

            # define source, although it doesn't function as a b.c. since we return Rhat matrix
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            
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
            
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, False, is_Rhat = True, all_debug = False);            

            # fidelity w/r/t U_gate           
            Fvals_min[Kvali, NBvali] = get_Fval(which_gate, myTwoS, U_gate, rhatvals[:,:,Kvali,NBvali], elecspin, ferromagnetic); 

        #### end loop over Kvals

        # process R output
        if(elecspin==1): # final e state (column, 2nd index) is spin up
            rhatvals_offdiag = rhatvals[:,np.array(range(n_loc_dof))<n_mol_dof]; 
        elif(elecspin==0): # final e state (column, 2nd index) is spin down
            rhatvals_offdiag = rhatvals[:,np.array(range(n_loc_dof))>=n_mol_dof];
        yvals = np.copy(rhatvals); rbracket = "";
        the_ticks = [-1.0,0.0,1.0];
        if(which_gate not in ["SWAP","SQRT", "RX", "RZ"]): # make everything real
            rbracket = "|"
            yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals)); 
            
        # determine fidelity and K*, ie x val where the SWAP happens   
        indep_argmax = np.argmax(Fvals_min[:,NBvali]);
        indep_star = indep_vals[np.argmax(Fvals_min[:,NBvali])];
        if(verbose):
            indep_comment = "case = "+case+": indep_star, fidelity(indep_star) = {:.8f}, {:.4f}".format(indep_star, Fvals_min[indep_argmax, NBvali]);
            print(indep_comment, "\n", rhatvals[elecspin*n_mol_dof:(elecspin+1)*n_mol_dof, elecspin*n_mol_dof:(elecspin+1)*n_mol_dof, indep_argmax, NBvali]);
            
        # plot as a function of K
        elems_to_keep = np.array([0,1,myTwoS+1,myTwoS+1+1]);
        for sourcei in range(len(elems_to_keep)):
            for sigmai in range(sourcei+1):
                # formatting
                if(myTwoS > 1): 
                    axes[sourcei,sigmai].set_title(str(n_mol_dof*elecspin+elems_to_keep[sourcei])+" $\\rightarrow$"+str(n_mol_dof*elecspin+elems_to_keep[sigmai]));
                else: 
                    axes[sourcei,sigmai].set_title("$"+rbracket+"\langle"+ str(ylabels[4*elecspin+sourcei])+ "| \mathbf{R} |"+str(ylabels[4*elecspin+sigmai])+"\\rangle"+rbracket+"$")
                axes[sourcei,sigmai].set_yticks(the_ticks);
                axes[sourcei,sigmai].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks: axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                if(K_indep): 
                    axes[-1,sigmai].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
                    axes[-1,sigmai].set_xscale('log', subs = []);
                else:
                    axes[-1,sigmai].set_xlabel('$N_B a/\lambda_i$',fontsize=myfontsize);
 
                # plot rhat (real part = solid, imag part = dashed)
                axes[sourcei,sigmai].plot(indep_vals, np.real(yvals)[n_mol_dof*elecspin+ elems_to_keep[sourcei], n_mol_dof*elecspin+ elems_to_keep[sigmai],:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]), color=mycolors[NBvali]);
                if(rbracket != "|"): axes[sourcei,sigmai].plot(indep_vals, np.imag(yvals)[n_mol_dof*elecspin+ elems_to_keep[sourcei], n_mol_dof*elecspin+ elems_to_keep[sigmai],:,NBvali], linestyle="dashed", color=mycolors[NBvali]);
 
                # plot fidelity
                if(sourcei==2 and sigmai==1): # NB sourcei, sigmai reversed here
                    axes[sigmai,sourcei].plot(indep_vals, Fvals_min[:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali]);
                    for tick in the_ticks: axes[sigmai, sourcei].axhline(tick, color='lightgray', linestyle='dashed');
                    axes[sigmai,sourcei].set_title("$F_{avg}(\mathbf{R},\mathbf{U}_{"+which_gate+"})$");
                    axes[1,0].legend();

                # plot reflection summed over final states (column, 2nd index)
                if((sourcei in [1,2] and sigmai==0) and summed_columns):                    
                    # < elec up row state| \sum |final states elec down>. to measure Se <-> S1(2)
                    # NB rhatvals_offdiag have shape (8,4)
                    which_rhatvals_offdiag = rhatvals_offdiag[elecspin*n_mol_dof + elems_to_keep[sourcei], :,:,NBvali];
                    axes[sourcei,sigmai+len(elems_to_keep)-1].plot(indep_vals, np.sum(np.real(np.conj(which_rhatvals_offdiag)*which_rhatvals_offdiag),axis=0), label="dummy", color=mycolors[NBvali]);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_title("$\sum_{\sigma_1 \sigma_2}| \langle"+str(ylabels[n_mol_dof*elecspin+sourcei])+"| \mathbf{R} |"+["\downarrow_e","\\uparrow_e"][elecspin]+" \sigma_1 \sigma_2 \\rangle |^2$");
                    
                    # format summed reflection
                    axes[sourcei, sigmai+len(elems_to_keep)-1].set_yticks(the_ticks);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                    for tick in the_ticks: axes[sourcei,sigmai+len(elems_to_keep) -1].axhline(tick, color='lightgray', linestyle='dashed');
                    
                    # difference between diagonal elements of R
                    axes[0,sigmai+len(elems_to_keep)-1].plot(indep_vals, np.real(yvals)[1,1,:,NBvali] - np.real(yvals)[2,2,:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]), color=mycolors[NBvali]);
                    if(rbracket != "|"): axes[0,sigmai+len(elems_to_keep)-1].plot(indep_vals, np.imag(yvals)[1,1,:,NBvali] - np.imag(yvals)[2,2,:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali],linestyle="dashed");
                    axes[0,sigmai+len(elems_to_keep)-1].set_title("$"+rbracket+"\langle"+str(1)+"| \mathbf{R} |"+str(1)+"\\rangle"+rbracket+" - "+rbracket+"\langle"+str(2)+"| \mathbf{R} |"+str(2)+"\\rangle"+rbracket+"$");
                         
    # show
    fig.suptitle(get_suptitle(myTwoS, Jval, V0, VB, ferromagnetic, False));
    plt.tight_layout();
    if(final_plots > 1): # save fig
        Jstring = "";
        if(Jval != -0.2): Jstring ="J"+ str(int(abs(100*Jval)))+"_";
        sstring = "";
        if(myTwoS != 1): sstring = "2s"+str(int(myTwoS))+"_";
        fname = "figs/gate/spin12_"+Jstring+sstring+case;
        plt.savefig(fname+".pdf")
    else:
        plt.show();



























