'''
Christian Bunker
M^2QM at UF
November 2022

Bardeen tunneling theory in 1D
'''

#from transport import fci_mod

import numpy as np

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

def matrix_element(m, mprime, tL, tb, tR, Vinfty, VL, Vb, VR, dims):

    # get well eigenstates
    HL, _ = HLmat(tL, tb, Vinfty, VL, Vb, dims);
    Es, psis = np.linalg.eigh(HL);
    psi_m = psis[:,m];
    HR, _ = HRmat(tb, tR, Vinfty, Vb, VR, dims);
    Es, psis = np.linalg.eigh(HR);
    psi_mprime = psis[:,mprime];
    del psis, Es;

    # operator
    Hsys, offset = Hsysmat(tL, tb, tR, Vinfty, VL, Vb, VR, dims);
    op = Hsys - HL;

    return psi_m, 

def TvsE(tL, tb, tR, Vinfty, VL, Vb, VR, dims):

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
    import matplotlib.pyplot as plt;

    # left lead quantum well test
    tL = 1.0;
    tb = 1.0;
    tR = 1.0;
    Vinfty = 1.0;
    VL = -0.0;
    Vb = Vinfty/10;
    VR = VL;
    NL = 1000;
    NC = 0;
    NR = 1000;
    dims = (NL, NC, NR);
    sys_args = (tL, tb, tR, Vinfty, VL, Vb, VR, dims);

    # test Hsys
    if False:
        Hsys, offset = Hsysmat(tL, tb, tR, Vinfty, VL, Vb, VR, dims);
        print(Hsys);
        jvals = np.array(range(len(Hsys))) + offset;
        Vvals = np.diag(Hsys);
        plt.plot(jvals, Vvals);
        plt.show();

    # test HL
    if False:
        HL, offset = HLmat(tL, tb, Vinfty, VL, Vb, dims);
        print(HL);
        jvals = np.array(range(len(HL))) + offset;
        Vvals = np.diag(HL);
        plt.plot(jvals, Vvals);
        Es, psiEs = np.linalg.eigh(HL);
        for psii in range(3):
            plt.plot(jvals, psiEs[:,psii]);
        plt.show();

    # test matrix elements
    if True:
        Es, Ts = TvsE(tL, tb, tR, Vinfty, VL, Vb, VR, dims);
        plt.plot(Es, Ts);
        plt.show();

    





    
    


    








