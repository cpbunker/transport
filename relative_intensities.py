'''
Compare different methods for determining the relative intensities
of the dI/dV steps from DFT params
'''

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 4, suppress = True);

def get_spin_Hamiltonian(s,Bx,By,Bz,D,E):
    '''
    for a spin-s particle w/ 2s+1 spin dofs, get a 2s+1 x 2s+1 matrix
    for the spin Hamiltonian, parameterized in the conventional form
    H = g*mu_B* B \cdot S + D S_z^2 + E(S_x^2 - S_y^2)
    '''
    n_loc_dof = int(2*s+1);
    S0 = np.eye(n_loc_dof,dtype=complex);
    identity = np.eye(2*n_loc_dof);

    # construct the spin-s operators according to
    # https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
    Sx = np.zeros_like(S0);
    Sy = np.zeros_like(S0);
    Sz = np.zeros_like(S0);
    for a in range(1,1+n_loc_dof):
        for b in range(1,1+n_loc_dof):
            Sx[a-1,b-1] = (1/2)*(identity[a,b+1]+identity[a+1,b])*np.lib.scimath.sqrt((s+1)*(a+b-1)-a*b);
            Sy[a-1,b-1] = complex(0,1/2)*(identity[a,b+1]-identity[a+1,b])*np.lib.scimath.sqrt((s+1)*(a+b-1)-a*b);
            Sz[a-1,b-1] = (s+1-a)*identity[a,b];

    if False:
        Sops = [Sx,Sy,Sz];
        for op in Sops: print(op);

    # construct Ham
    return Bx*Sx + By*Sy + Bz*Sz + D*np.matmul(Sz,Sz) + E*(np.matmul(Sx,Sx)-np.matmul(Sy,Sy));
    

# spin-2 Fe eigenvectors as in Heinrich Large Magnetic Anisotropy Tab 1
# all energies in meV
mu_Bohr = 5.788*1e-2; # bohr magneton in meV/Tesla
gfactor = 2.11;
myBx, myBy, myBz = 0,0,gfactor*mu_Bohr*7; # 7 Tesla field in the z direction
myD = -1.55;
myE = 0.31;
mys = 2;
Fe_ham = get_spin_Hamiltonian(mys, myBx, myBy, myBz, myD, myE);
eigvals, eigvecs = np.linalg.eigh(Fe_ham);
print(eigvecs.T);


