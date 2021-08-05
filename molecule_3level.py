'''
Christian Bunker
M^2QM at UF
June 2021
    
Template:
Solve exact diag problem with given 1-electron and 2-electron Hamiltonian

Formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

Specific case:
- Silas' model of molecule (SOC and spatial anisotropy)
- 3 L_z levels: m=-1,0,1 (6 spin orbitals)
'''

import ops

import numpy as np
import matplotlib.pyplot as plt

##########################################
#### create hamiltonians

def h1e(norbs,D,E,alpha):
    '''
    Create one electron part of Silas' model hamiltonian from physical params
    norbs = # spin orbs
    D = z axis spatial anisotropy
    E = xy plane spatial anisotropy
    alpha = SOC strength
    Returns: 2D np array
    '''
    
    # make empty 2d matrix
    # since code is not flexible right now (bad practice)
    my_norbs = 6; # spin orbs
    assert(norbs == my_norbs);
    h = np.zeros((norbs, norbs));

    # single electron terms
    # best practice to use += thru out since we may modify elements twice
    h[0,0] += -D; # z axis anisotropy
    h[1,1] += -D;
    h[4,4] += -D;
    h[5,5] += -D;
    h[0,4] += 2*E; # xy plane anisotropy
    h[4,0] += 2*E;
    h[1,5] += 2*E;
    h[5,1] += 2*E;
    h[0,0] += -alpha; # diagonal SOC
    h[1,1] += alpha;
    h[4,4] += alpha;
    h[5,5] += -alpha;
    h[0,3] += alpha; # off diagonal SOC
    h[3,0] += alpha;
    h[2,5] += alpha;
    h[5,2] += alpha;
    
    return h;
    
    
def h2e(norbs, U):
    '''
    Create two electron part of Silas' model hamiltonian from physical params
    norbs = # spin orbs
    U = hubbard repulsion
    Returns: 4D np array
    '''
    
    # make empty 4d matrix
    # since code is not flexible right now (bad practice)
    my_norbs = 6; # spin orbs
    assert(norbs == my_norbs);
    h = np.zeros((norbs, norbs,norbs,norbs));

    # double electron terms
    h[0,0,1,1] = 2*U; # hubbard
    h[2,2,3,3] = 2*U;
    h[4,4,5,5] = 2*U;
    
    return h;

##########################################
#### wrapper funcs and test code

def Test():

    from pyscf import fci

    verbose = 2;
    np.set_printoptions(suppress=True); # no sci notatation printing

    # parameters in the hamiltonian
    alpha = 0.5;
    D = -10.0;
    E = 1e-4
    U = 100.0;
    E_shift = U - 2*D
    if(verbose):
        print("\nInputs:","\nalpha = ",alpha,"\nD = ",D,"\nE = ",E,"\nU = ",U);
        print("\nalpha/D = ",alpha/D,"E/U = ",E/U,"\nalpha/(4E^2/U) = ", alpha*U/(4*E*E),"\nalpha^2/(4E^2/U) = ",alpha*alpha*U/(4*E*E) );

    #### get analytical energies

    # diagonalize numerically
    H_T = np.array([[-2*alpha,0,-2*E,2*E], [0,2*alpha,2*E,-2*E],[-2*E,2*E,U,0],[2*E,-2*E,0,U]]); # T sector
    H_O = np.array([[-D-alpha,0,0,2*E,0],[0,-D+alpha,2*E,0,0],[0,2*E,-D-alpha,0,0],[2*E,0,0,-D+alpha,0],[0,0,0,0,-2*D+U]]); # other sector
    E_exact = np.append(np.linalg.eigh(H_T)[0], np.linalg.eigh(H_O)[0]);
    H_eff = np.array([[-2*alpha- 8*E*E/U, 8*E*E/U], [8*E*E/U, 2*alpha - 8*E*E/U - 2*alpha*alpha/(alpha-D)] ])
    E_S, E_T0 = np.linalg.eigh(H_eff)[0];

    # sort and print
    E_exact.sort();
    if(verbose):
        print("\n0. Analytical solution, exact as alpha --> 0:")
        print("Exact energies = \n",E_exact);
        print("Expected energies as E/U, alpha/(E^2/U), alpha/D --> 0:\n- Singlet energy = ", E_S, "\n- T0 energy = ", E_T0, "\n- T+/- energy = ", alpha*alpha/(D+alpha) );
        
    #### solve with spin blind method
    nelecs = (4,0);
    norbs = 6; # spin orbs
    myroots = 15; # size of full basis
    
    # get h1e and h2e maatrices
    h1e_mat = h1e(norbs,D,E,alpha);
    h2e_mat = h2e(norbs,U);

    # solve with FCISolver object
    cisolver = fci.direct_spin1.FCI()
    E_fci, v_fci = cisolver.kernel(h1e_mat, h2e_mat, norbs, nelecs,nroots=15);
    E_fci.sort();
    if(verbose):
        print("\n1. ASU solution, nelecs = ",nelecs, ", nroots = ",myroots);
        for i in range(myroots):
            Sx = np.dot(v_fci[i].T, fci.direct_spin1.contract_1e(ops.Sx([0,1,2,3,4,5],6),v_fci[i], norbs, nelecs) );
            Sz = np.dot(v_fci[i].T, fci.direct_spin1.contract_1e(ops.Sz([0,1,2,3,4,5],6),v_fci[i], norbs, nelecs) );
            print("- E = ",E_fci[i] - E_shift, ", <S_x> = ",Sx," <S_z> = ", Sz);
            if(verbose > 2):
                print("     ",v_fci[i].T);
        
        
##########################################
#### exec code

if __name__ == "__main__":

    # compare DS and alpha when E=0, D<0
    # mark pederson predicts DS linear in alpha in this case
    alphas = np.array([0.1,0.2,0.3,0.4,0.5,1.0])
    Ds = np.array([-0.199,-0.396,-0.59,-0.784,-0.974,-1.89])
    plt.plot(alphas, Ds, label = "data");
    m, b = tuple(np.polyfit(alphas, Ds, 1))
    fit = m*alphas + b
    plt.plot(alphas, fit, label = "m = "+str(m));
    plt.legend();
    plt.xlabel(r"$\alpha$")
    plt.ylabel("$D_S$")
    plt.title("Linear $D_{S}$ when D < 0, E = 0");
    plt.show()

    

