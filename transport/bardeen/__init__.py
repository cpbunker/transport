'''
Christian Bunker
M^2QM at UF
November 2022

Bardeen tunneling theory in 1D
'''

#from transport import fci_mod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##################################################################################
#### driver of transmission coefficient calculations

def kernel(h, tnn, tnnn, tl, E, Ajsigma, verbose = 0, all_debug = True):
    '''

    '''

    return;

def Hsysmat(tL, tb, tR, Vinfty, VL, Vb, VR, dims):
    '''
    Make the TB Hamiltonian for the full system
    '''
    NL, NC, NR = dims;
    for N in dims:
        if(not isinstance(N, int)): raise TypeError;
    if NL <= 0: raise ValueError;
    if NR <= 0: raise ValueError;
    minusinfty = -NC - 2*NL;
    plusinfty = NC + 2*NR;
    Nsites = -minusinfty + plusinfty + 1;

    # Hamiltonian matrix
    Hmat = np.zeros((Nsites,Nsites));
    for j in range(minusinfty, plusinfty+1):

        # diag
        if(j < -NL - NC):           
            Hmat[j-minusinfty,j-minusinfty] = Vinfty
        elif(j >= -NL-NC and j < -NC):
            Hmat[j-minusinfty,j-minusinfty] = VL;
        elif(j >= -NC and j <= NC):
            Hmat[j-minusinfty,j-minusinfty] = Vb;
        elif(j > NC and j <= NC+NR):
            Hmat[j-minusinfty,j-minusinfty] = VR;
        elif(j > NC+NR):
            Hmat[j-minusinfty,j-minusinfty] = Vinfty;

        # off diag
        if(j > minusinfty and j <= -NC):
            Hmat[j-minusinfty,j-1-minusinfty] = -tL;
            Hmat[j-1-minusinfty,j-minusinfty] = -tL;
        if(j >= -NC and j < NC):
            Hmat[j-minusinfty,j+1-minusinfty] = -tb;
            Hmat[j+1-minusinfty,j-minusinfty] = -tb;
        elif(j >= NC and j < plusinfty):
            Hmat[j-minusinfty,j+1-minusinfty] = -tR;
            Hmat[j+1-minusinfty,j-minusinfty] = -tR;           
            
    return Hmat, minusinfty;

def HLmat(tL, tb, Vinfty, VL, Vb, dims):
    '''
    Make the TB Hamiltonian for a left side finite quantum well of NL sites
    '''

    return Hsysmat(tL, tb, tb, Vinfty, VL, Vb, Vb, dims);


def HRmat(tb, tR, Vinfty, Vb, VR, dims):
    '''
    Make the TB Hamiltonian for a right side finite quantum well of NR sites
    '''          
            
    return Hsysmat(tb, tb, tR, Vinfty, Vb, Vb, VR, dims);

##################################################################################
#### util functions

def plot_wfs(tL, tb, tR, Vinfty, VL, Vb, VR, dims):
    '''
    Visualize the problem by plotting some LL wfs against Hsys
    '''
    fig, axes = plt.subplots(4, sharex = True);
    mycolors = cm.get_cmap('Set1');

    # plot left well eigenstates
    HL, offset = HLmat(tL, tb, Vinfty, VL, Vb, dims);
    jvals = np.array(range(len(HL))) + offset;
    axes[0].plot(jvals, np.diag(HL), color = 'black', linestyle = 'dashed', linewidth = 2);
    Ems, psims = np.linalg.eigh(HL);
    Ems_bound = Ems[Ems + 2*tL < Vb];
    ms_bound = np.linspace(0,len(Ems_bound)-1,3,dtype = int);
    for counter in range(len(ms_bound)):
        m = ms_bound[counter]
        axes[0].plot(jvals[jvals <= dims[1]+dims[2]], psims[:,m][jvals <= dims[1]+dims[2]], color = mycolors(counter));
        axes[0].plot([dims[1]+dims[2],jvals[-1]],(2*tL+ Ems[m])*np.ones((2,)), color = mycolors(counter));
    axes[0].set_ylabel('$V_j/t_L$');
    axes[0].set_ylim(VL-2*Vb,VL+2*Vb);

    # plot system ham
    Hsys, _ = Hsysmat(tL, tb, tR, Vinfty, VL, Vb, VR, dims);
    axes[1].plot(jvals, np.diag(Hsys), color = 'black', linestyle = 'dashed', linewidth = 2);
    axes[1].plot(jvals, np.diag(Hsys-HL), color = 'darkblue', linestyle = 'dashed', linewidth = 2);
    axes[1].set_ylabel('$V_j/t_L$');

    # plot (Hsys-HL)*psi_m
    for counter in range(len(ms_bound)):
        m = ms_bound[counter];
        axes[2].plot(jvals, np.dot(Hsys-HL,psims[:,m]), color = mycolors(counter));
    axes[2].set_ylabel('$(H_{sys}-H_L) \psi_m $');

    # plot right well eigenstates
    HR, offset = HRmat(tb, tR, Vinfty, Vb, VR, dims);
    axes[3].plot(jvals, np.diag(HR), color = 'black', linestyle = 'dashed', linewidth = 2);
    Emprimes, psimprimes = np.linalg.eigh(HR);
    for counter in range(len(ms_bound)):
        mprime = ms_bound[counter];
        axes[3].plot(jvals[jvals > -dims[0]-dims[1]], psimprimes[:,mprime][jvals > -dims[0]-dims[1]], color = mycolors(counter));
        axes[3].plot([jvals[0],-dims[0]-dims[1]],(2*tL+ Ems[mprime])*np.ones((2,)), color = cmycolors(counter));
    axes[3].set_ylabel('$V_j/t_L$');
    axes[3].set_ylim(VR-2*Vb,VR+2*Vb);
        
    # format
    axes[-1].set_xlabel('$j$');
    plt.tight_layout();
    plt.show();
    
def TvsE(tL, tb, tR, Vinfty, VL, Vb, VR, dims):
    '''
    Calculate a transmission coefficient for each LL eigenstate and return
    these as a function of their energy
    '''

    # left well eigenstates
    HL, _ = HLmat(tL, tb, Vinfty, VL, Vb, dims);
    Ems, psims = np.linalg.eigh(HL);

    # right well eigenstates
    HR, _ = HRmat(tb, tR, Vinfty, Vb, VR, dims);
    Emprimes, psimprimes = np.linalg.eigh(HR);

    # operator
    Hsys, offset = Hsysmat(tL, tb, tR, Vinfty, VL, Vb, VR, dims);
    op = Hsys - HL;

    # compute T
    Tms = np.zeros_like(Ems);
    for m in range(len(Ems)):
        M = np.dot(psimprimes[:,m],np.dot(op,psims[:,m]));
        Tms[m] = M*np.conj(M) #/(Ems[m]+2*tL);
        
    return Ems, Tms;

##################################################################################
#### test code

if __name__ == "__main__":

    # left lead quantum well test
    # tb params, in tL
    tL = 1.0;
    tb = 1.0;
    tR = 1.0;
    Vinfty = tL/2;
    VL = -0.0;
    Vb = tL/10;
    VR = VL;
    NL = 100;
    NC = NL//20;
    NR = 100;
    dims = (NL, NC, NR);
    sys_args = (tL, tb, tR, Vinfty, VL, Vb, VR, dims);

    # visualize the problem
    if False:
        plot_wfs(*sys_args);

    # test matrix elements
    if True:
        numplots = 1;
        fig, axes = plt.subplots(numplots, sharex = True);
        if numplots == 1: axes = [axes];
        fig.set_size_inches(7/2,3*numplots/2);

        # bardeen results
        Es, Ts = TvsE(tL, tb, tR, Vinfty, VL, Vb, VR, dims);
        axes[0].plot(Es+2*tL, Ts);

        # compare
        kas = np.arccos((Es-VL)/(-2*tL));
        kappas = np.arccosh((Es-Vb)/(-2*tb));
        ideal_exp = np.exp(-2*(2*NC+1)*kappas);
        #axes[0].plot(Es, np.power(4*kas/kappas,2)*ideal_exp);

        # format and show
        axes[-1].set_xscale('log', subs = []);
        axes[-1].set_xlabel('$K_i/t_L$');
        plt.tight_layout();
        plt.show();

    





    
    


    








