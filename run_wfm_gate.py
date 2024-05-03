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
    
def get_U_gate(gate0, TwoS):
    '''
    '''
    if(gate0=="I"):
        ticks = [0.0,1.0];
        proj_choice = "identical";
        U_q = np.eye(4, dtype=complex); # Identity gate
    elif(gate0 in ["SeS12", "RZSeS12"]):
        ticks = [0.0,1.0];
        proj_choice = "identical";
        U_q = np.nan*np.eye(4, dtype=complex);
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
    elif(gate0[:2]=="RZ" and ((gate0[-1] in ["1","2","3","4"]) and len(gate0)==3) ):
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

def get_Fval(gate0, TwoS, U, R):
    '''
    '''
    assert(np.shape(R) == np.shape(U));
    assert(len(U)==8); # this affects results even when off-diagonal blocks are zero, due to 1/nd dependence

    # from Uq to Ugate
    mol_dof = (TwoS+1)*(TwoS+1); 
    elems_to_keep = [0,1,TwoS+1,TwoS+1+1];

    if("SeS12" in gate0): # do not actually get fidelity, instead quantify R^out
        Rout = R[:mol_dof, mol_dof:]; 
        # maximize over rows
        R_rows = np.zeros((mol_dof,),dtype=float);
        for row_sigma in range(mol_dof):
            # sum over columns
            Rout_row_mags = np.real(np.conj(Rout[row_sigma])*Rout[row_sigma]);
            R_rows[row_sigma] = np.sum(Rout_row_mags);
        the_trace = np.max(R_rows);

    else: # actually get fidelity
        M_matrix = np.matmul(np.conj(U.T), R); # M = U^\dagger R
        the_trace = np.sqrt((np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U)*(len(U)+1)));

    # return
    if(abs(np.imag(the_trace)) > 1e-10): print(the_trace); assert False;
    return np.real(the_trace);
           
############################################################################ 
#### exec code
if(__name__ == "__main__"):

    #### top level
    np.set_printoptions(precision = 4, suppress = True);
    verbose = 1;
    which_gate = sys.argv[1];
    case = sys.argv[2];
    final_plots = int(sys.argv[3]);
    #if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})
    vlines = not final_plots; # whether to highlight certain x vals with vertical dashed lines
    summed_columns = True;
    elecspin = 0; # itinerant e is spin up
    if(which_gate not in ["SWAP"]): assert(not final_plots);

    # fig standardizing
    myxvals = 49; 
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
    Jval = -0.2*tl;
    VB = 5.0*tl;
    V0 = 0.0*tl; # just affects title, not implemented physically
    U_gate, the_ticks = get_U_gate(which_gate, myTwoS);

if(__name__ == "__main__" and case in ["NB","kNB"]): # distance of the barrier NB on the x axis
    
    # axes
    nrows, ncols = 4, 4;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    if(case=="NB"): NB_indep = True;
    elif(case=="kNB"): NB_indep = False # whether to put NB, alternatively wavenumber*NB

    # iter over incident kinetic energy (colors)
    Kpowers = np.array([-3,-4]); # incident kinetic energy/t = 10^Kpower
    knumbers = np.sqrt(np.logspace(Kpowers[0],Kpowers[-1],num=len(Kpowers)));
    print("knumbers^2 = \n",knumbers*knumbers);
    Kvals = 2*tl - 2*tl*np.cos(knumbers);
    Energies = Kvals - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
    rhatvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals),dtype=complex); # by  init spin, final spin, energy, NB
    Fvals_Uchi = np.empty((len(Kvals), myxvals),dtype=float);
    for Kvali in range(len(Kvals)):
        
        # iter over barrier distance (x axis)
        kNBmax = 0.75*np.pi #0.75*np.pi;
        if(NB_indep): NBmax = 150;
        else: NBmax = int(kNBmax/knumbers[Kvali]);
        if(verbose): print("k^2, NBmax = ",knumbers[Kvali]**2, NBmax); 
        NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
        if(myxvals==100): NBvals = np.linspace(51,150,100,dtype=int); assert(myxvals==100);
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
            if(Kvali == 0 and NBvali == 0): print("\nhblocks = \n",np.real(hblocks));

            # since we don't iter over sources, ALL sources must have 0 chem potential
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            for sourcei in range(n_loc_dof):
                assert(hblocks[0][sourcei,sourcei]==0.0);
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, rhat = True, all_debug = False);
            # fidelity w/r/t U, chi
            assert(np.shape(rhatvals[:,:,0,0]) == np.shape(U_gate));
            M_matrix = np.matmul(np.conj(U_gate.T), rhatvals[:,:,Kvali,NBvali]);
            trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
            if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
            Fvals_Uchi[Kvali, NBvali] = np.real(trace_fidelity);
            if(verbose>4): print(Kvali, 2*NBvals[NBvali], "{:.6f}, {:.4f}".format(2*knumbers[Kvali]*NBvals[NBvali]/np.pi, Fvals_Uchi[Kvali, NBvali]));
            
        #### end loop over NB

        # determine fidelity and kNB*, ie x val where the SWAP happens
        rhatvals_up = rhatvals[:,np.array(range(n_loc_dof))<n_mol_dof]; # ref'd is e up
        rhatvals_down = rhatvals[:,np.array(range(n_loc_dof))>=n_mol_dof]; # ref'd is e down
        yvals = np.copy(rhatvals); rbracket = "";
        if(which_gate not in ["SQRT", "RX", "RZ"]): # make everything real
            rbracket = "|"
            yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals)); 
        if(NB_indep): indep_var = NBvals; # what to put on x axis
        else: indep_var = 2*knumbers[Kvali]*NBvals/np.pi;
        indep_argmax = np.argmax(Fvals_Uchi[Kvali]);
        indep_star = indep_var[indep_argmax];
        if(verbose):
            indep_comment = case+": indep_star, fidelity(indep_star) = {:.6f}, {:.4f}".format(indep_star, Fvals_Uchi[Kvali,indep_argmax]);
            print(indep_comment,"\n",rhatvals[:,:,Kvali,indep_argmax]);
            np.savetxt("data/gate/rhat_{:.0f}_{:.0f}.txt".format(Kvali,indep_argmax),rhatvals[:,:,Kvali,indep_argmax], fmt="%.4f", header=indep_comment);
        if(False and Kvali==1):
            the_sourceindex, the_NBindex = 1, np.argmax(Fvals_Uchi[Kvali]);
            the_y, the_x = np.imag(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]), np.real(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]);
            comp_phase = 2*np.arctan( the_y/(the_x+np.sqrt(the_x*the_x+the_y*the_y)));
            print(yvals[:,:,Kvali,the_NBindex]); # |r|
            print(rhatvals[:,:,Kvali,the_NBindex])
            print(np.exp(complex(0,-comp_phase))*rhatvals[:,:,Kvali,the_NBindex]) # r*e^-i\phi =? |r|
            assert False;
            
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
                axes[-1,sigmai].set_xlim(0,np.max(indep_var));
                if(NB_indep):
                    axes[-1,sigmai].set_xlabel('$N_B$',fontsize=myfontsize);
                else:
                    axes[-1,sigmai].set_xlabel('$2k_i aN_B /\pi$',fontsize=myfontsize);
 
                # plot rhat
                if(abs(knumbers[1] - np.sqrt(10.0**Kpowers[1])) < 1e-10):
                    mylabel = "$k_i^2 a^2 = 10^{"+str(Kpowers[Kvali])+"}$"
                else: 
                    mylabel = "$k_i^2 a^2 = {:.6f} $".format(knumbers[Kvali]**2);
                axes[sourcei, sigmai].plot(indep_var, np.real(yvals)[n_mol_dof*elecspin+elems_to_keep[sourcei],n_mol_dof*elecspin+elems_to_keep[sigmai],Kvali], label = mylabel, color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery);
                axes[sourcei,sigmai].plot(indep_var, np.imag(yvals)[n_mol_dof*elecspin+elems_to_keep[sourcei],n_mol_dof*elecspin+elems_to_keep[sigmai],Kvali], linestyle="dashed", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery);
                if(vlines): axes[sourcei,sigmai].axvline(indep_star, color=mycolors[Kvali], linestyle="dotted");
 
                # plot fidelity, starred SWAP locations
                if(sourcei==2 and sigmai==1):
                    axes[sigmai,sourcei].plot(indep_var,Fvals_Uchi[Kvali], label = "$k_i^2 a^2= 10^{"+str(Kpowers[Kvali])+"}$",color=mycolors[Kvali],marker=mymarkers[1+Kvali],markevery=mymarkevery);
                    axes[sigmai,sourcei].axvline(indep_star, color=mycolors[Kvali], linestyle="dotted");
                    for tick in the_ticks: axes[sigmai,sourcei].axhline(tick,color='lightgray',linestyle='dashed');
                    axes[sigmai,sourcei].set_title("$F_{avg}(\mathbf{R}, \mathbf{U}_{"+which_gate+"})$");
                    axes[1,0].legend();
                   
                # plot reflection summed over final states (columns)
                if((sourcei in [1,2] and sigmai==0) and summed_columns and elecspin==0):
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_yticks(the_ticks);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                    for tick in the_ticks: axes[sourcei,sigmai+len(elems_to_keep)-1].axhline(tick,color='lightgray',linestyle='dashed');
                    # < elec up row state| \sum |final states elec down>. to measure Se <-> S1(2)
                    axes[sourcei,sigmai+len(elems_to_keep)-1].plot(indep_var, np.sum(np.real(np.conj(rhatvals_down[n_mol_dof*elecspin+elems_to_keep[sourcei],:,Kvali])*rhatvals_down[n_mol_dof*elecspin+elems_to_keep[sourcei],:,Kvali]),axis=0), label="dummy", color=mycolors[Kvali], marker=mymarkers[1+Kvali], markevery=mymarkevery);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_title("$\sum_{\sigma_1 \sigma_2}| \langle"+str(ylabels[4*elecspin+sourcei])+"| \mathbf{R} |\downarrow_e \sigma_1 \sigma_2 \\rangle |^2$")
                    # difference between diagonal elements of r
                    axes[0,sigmai+len(elems_to_keep)-1].plot(indep_var, np.real(yvals)[1,1,Kvali] - np.real(yvals)[2,2,Kvali], label = "$N_B$ = {:.0f}".format(NBvals[Kvali]),color=mycolors[Kvali],marker=mymarkers[1+Kvali], markevery=mymarkevery);
                    axes[0,sigmai+len(elems_to_keep)-1].plot(indep_var, np.imag(yvals)[1,1,Kvali] - np.imag(yvals)[2,2,Kvali], label = "$N_B$ = {:.0f}".format(NBvals[Kvali]),color=mycolors[Kvali],marker=mymarkers[1+Kvali], markevery=mymarkevery,linestyle="dashed");
                    axes[0,sigmai+len(elems_to_keep)-1].set_title("$"+rbracket+"\langle"+str(1)+"| \mathbf{R} |"+str(1)+"\\rangle"+rbracket+" - "+rbracket+"\langle"+str(2)+"| \mathbf{R} |"+str(2)+"\\rangle"+rbracket+"$");
                    
    # show
    suptitle = which_gate+" gate, $s=${:.1f}, $J/t=${:.2f}, $V_0/t=${:.2f}, $V_B/t=${:.2f}".format(0.5*myTwoS, Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle);
    plt.tight_layout();
    if(final_plots): # save fig
        Jstring = "";
        if(Jval != -0.2): Jstring ="J"+ str(int(abs(100*Jval)))+"_";
        sstring = "";
        if(myTwoS != 1): sstring = "2s"+str(int(myTwoS))+"_";
        fname = "figs/gate/spin12_"+Jstring+sstring+case;
        plt.savefig(fname+".pdf")

    else:
        plt.show();

elif(__name__ == "__main__" and case in["K","ki"]): # incident kinetic energy or wavenumber on the x axis
         # NB is now fixed !!!!

    # axes
    nrows, ncols = 4,4;
    fig, axes = plt.subplots(nrows, ncols, sharex=True);
    fig.set_size_inches(ncols*7/2,nrows*3/2);
    if(case=="ki"): K_indep = False;
    elif(case=="K"): K_indep = True; # whether to put ki^2 on x axis, alternatively ki a Nb/pi 

    # iter over fixed NB (colors)
    NBvals = np.array([50,100,131,150]);
    Fvals_Uchi = np.empty((myxvals, len(NBvals)),dtype=float); 
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):

        # set barrier distance
        NBval = int(NBvals[NBvali])
        if(verbose): print("NB = ",NBval); 

        # iter over incident kinetic energy (x axis)
        kNBmax = 2.0*np.pi #0.75*np.pi;
        Kpowers = np.array([-2,-4]); # incident kinetic energy/t = 10^Kpower
        if(K_indep): knumbers = np.sqrt(np.logspace(Kpowers[0],Kpowers[-1],num=myxvals)); del kNBmax;
        else: knumbers = np.linspace(0.001, kNBmax/NBval, myxvals); 
        print(knumbers**2); #assert False
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
            if(Kvali == 0 and NBvali == 0): print("\nhblocks = \n",np.real(hblocks));

            # since we don't iter over sources, ALL sources must have 0 chem potential
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
            for sourcei in range(n_loc_dof):
                assert(hblocks[0][sourcei,sourcei]==0.0);
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, rhat = True, all_debug = False);

            # fidelity w/r/t U, chi
            assert(np.shape(rhatvals[:,:,0,0]) == np.shape(U_gate));
            M_matrix = np.matmul(np.conj(U_gate.T), rhatvals[:,:,Kvali,NBvali]);
            trace_fidelity = (np.trace(np.matmul(M_matrix,np.conj(M_matrix.T) ))+np.conj(np.trace(M_matrix))*np.trace(M_matrix))/(len(U_gate)*(len(U_gate)+1));
            if(np.imag(trace_fidelity) > 1e-10): print(trace_fidelity); assert False;
            Fvals_Uchi[Kvali, NBvali] = np.real(trace_fidelity);

        #### end loop over Kvals

        # determine fidelity and K*, ie x val where the SWAP happens
        rhatvals_up = rhatvals[:,np.array(range(n_loc_dof))<n_mol_dof]; # ref'd is e up
        rhatvals_down = rhatvals[:,np.array(range(n_loc_dof))>=n_mol_dof]; # ref'd is e down
        yvals = np.copy(rhatvals); rbracket = "";
        if(which_gate not in ["SQRT", "RX", "RZ"]): # make everything real
            rbracket = "|"
            yvals = np.sqrt(np.real(np.conj(rhatvals)*rhatvals)); 
        if(K_indep): indep_var = knumbers*knumbers; # what to put on x axis
        else: indep_var = 2*knumbers*NBvals[NBvali]/np.pi;
        indep_argmax = np.argmax(Fvals_Uchi[Kvali]);
        indep_star = indep_var[np.argmax(Fvals_Uchi[:,NBvali])];
        if(verbose):
            indep_comment = case+": indep_star, fidelity(indep_star) = {:.6f}, {:.4f}".format(indep_star, Fvals_Uchi[indep_argmax,NBvali]);
            print(indep_comment,"\n",rhatvals[:,:,indep_argmax,Kvali]);
        if(False and Kvali==1):
            the_sourceindex, the_NBindex = 1, np.argmax(Fvals_Uchi[Kvali]);
            the_y, the_x = np.imag(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]), np.real(rhatvals[the_sourceindex,the_sourceindex,Kvali,the_NBindex]);
            comp_phase = 2*np.arctan( the_y/(the_x+np.sqrt(the_x*the_x+the_y*the_y)));
            print(yvals[:,:,Kvali,the_NBindex])
            print(rhatvals[:,:,Kvali,the_NBindex])
            print(np.exp(complex(0,-comp_phase))*rhatvals[:,:,Kvali,the_NBindex])
            assert False;
            
        # plot as a function of K
        elems_to_keep = np.array([0,1,myTwoS+1,myTwoS+1+1]);
        for sourcei in range(len(elems_to_keep)):
            for sigmai in range(sourcei+1):
                # formatting
                if(myTwoS > 1): axes[sourcei,sigmai].set_title(str(n_mol_dof*elecspin+elems_to_keep[sourcei])+" $\\rightarrow$"+str(n_mol_dof*elecspin+elems_to_keep[sigmai]));
                else: axes[sourcei,sigmai].set_title("$"+rbracket+"\langle"+str(ylabels[4*elecspin+sourcei])+"| \mathbf{R} |"+str(ylabels[4*elecspin+sigmai])+"\\rangle"+rbracket+"$")
                axes[sourcei,sigmai].set_yticks(the_ticks);
                axes[sourcei,sigmai].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                for tick in the_ticks: axes[sourcei,sigmai].axhline(tick,color='lightgray',linestyle='dashed');
                if(K_indep): 
                    axes[-1,sigmai].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
                    axes[-1,sigmai].set_xscale('log', subs = []);
                else:
                    axes[-1,sigmai].set_xlabel('$2k_i a N_B/\pi$',fontsize=myfontsize);
 
                # plot rhat
                axes[sourcei,sigmai].plot(indep_var, np.real(yvals)[n_mol_dof*elecspin+elems_to_keep[sourcei],n_mol_dof*elecspin+elems_to_keep[sigmai],:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]), color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery);
                axes[sourcei,sigmai].plot(indep_var, np.imag(yvals)[n_mol_dof*elecspin+elems_to_keep[sourcei],n_mol_dof*elecspin+elems_to_keep[sigmai],:,NBvali], linestyle="dashed", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery);
                if(vlines): axes[sourcei,sigmai].axvline(indep_star, color=mycolors[NBvali], linestyle="dotted");
 
                # plot fidelity
                if(sourcei==2 and sigmai==1): # NB sourcei, sigmai reversed here
                    axes[sigmai,sourcei].plot(indep_var,Fvals_Uchi[:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali],marker=mymarkers[1+NBvali], markevery=mymarkevery);
                    axes[sigmai,sourcei].axvline(indep_star, color=mycolors[NBvali], linestyle="dotted");
                    for tick in the_ticks: axes[sigmai,sourcei].axhline(tick,color='lightgray',linestyle='dashed');
                    axes[sigmai,sourcei].set_title("$F_{avg}(\mathbf{R},\mathbf{U}_{"+which_gate+"})$");
                    axes[1,0].legend();

                # plot reflection summed over final states (columns)
                if((sourcei in [1,2] and sigmai==0) and summed_columns and elecspin==0):
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_yticks(the_ticks);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
                    for tick in the_ticks: axes[sourcei,sigmai+len(elems_to_keep)-1].axhline(tick,color='lightgray',linestyle='dashed');
                    # < elec up row state| \sum |final states elec down>. to measure Se <-> S1(2)
                    axes[sourcei,sigmai+len(elems_to_keep)-1].plot(indep_var, np.sum(np.real(np.conj(rhatvals_down[n_mol_dof*elecspin+elems_to_keep[sourcei],:,:,NBvali])*rhatvals_down[n_mol_dof*elecspin+elems_to_keep[sourcei],:,:,NBvali]),axis=0), label="dummy", color=mycolors[NBvali], marker=mymarkers[1+NBvali], markevery=mymarkevery);
                    axes[sourcei,sigmai+len(elems_to_keep)-1].set_title("$\sum_{\sigma_1 \sigma_2}| \langle"+str(ylabels[4*elecspin+sourcei])+"| \mathbf{R} |\downarrow_e \sigma_1 \sigma_2 \\rangle |^2$")
                    # difference between diagonal elements of r
                    axes[0,sigmai+len(elems_to_keep)-1].plot(indep_var, np.real(yvals)[1,1,:,NBvali] - np.real(yvals)[2,2,:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali],marker=mymarkers[1+NBvali], markevery=mymarkevery);
                    axes[0,sigmai+len(elems_to_keep)-1].plot(indep_var, np.imag(yvals)[1,1,:,NBvali] - np.imag(yvals)[2,2,:,NBvali], label = "$N_B$ = {:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali],marker=mymarkers[1+NBvali], markevery=mymarkevery,linestyle="dashed");
                    axes[0,sigmai+len(elems_to_keep)-1].set_title("$"+rbracket+"\langle"+str(1)+"| \mathbf{R} |"+str(1)+"\\rangle"+rbracket+" - "+rbracket+"\langle"+str(2)+"| \mathbf{R} |"+str(2)+"\\rangle"+rbracket+"$");
                         
    # show
    suptitle = "$s=${:.1f}, $J/t=${:.2f}, $V_0/tl={:.2f}$, $V_B/t=${:.2f}".format(0.5*myTwoS, Jval/tl, V0/tl, VB/tl);
    fig.suptitle(suptitle);
    plt.tight_layout();
    if(final_plots): # save fig
        Jstring = "";
        if(Jval != -0.2): Jstring ="J"+ str(int(abs(100*Jval)))+"_";
        sstring = "";
        if(myTwoS != 1): sstring = "2s"+str(int(myTwoS))+"_";
        fname = "figs/gate/spin12_"+Jstring+sstring+case;
        plt.savefig(fname+".pdf")
    else:
        plt.show();



























