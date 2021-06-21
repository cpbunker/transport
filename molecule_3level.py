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

import utils

import numpy as np
from pyscf import fci

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

    verbose = True;
    np.set_printoptions(suppress=True); # no sci notatation printing

    # parameters in the hamiltonian
    alpha = 0.001;
    D = 100.0;
    E = 10.0;
    U = 1000.0;
    E_shift = U - 2*D
    if(verbose):
        print("\nInputs:","\nalpha = ",alpha,"\nD = ",D,"\nE = ",E,"\nU = ",U);
        print("E/U = ",E/U,"\nalpha/(E^2/U) = ", alpha*U/(E*E),"\nalpha/D = ",alpha/D );

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
    cisolver = fci.direct_nosym.FCI()
    E_fci, v_fci = cisolver.kernel(h1e_mat, h2e_mat, norbs, nelecs,nroots=15);
    E_fci.sort();
    spinexps = utils.Spin_exp(v_fci, norbs, nelecs)
    if(verbose):
        print("\n1. Spin blind solution, nelecs = ",nelecs, ", nroots = ",myroots);
        for i in range(myroots):
            print("- E = ",E_fci[i] - E_shift, ", <S^2> = ",np.linalg.norm(spinexps[i]),", <S_z> = ", spinexps[i][2]);
        
        
##########################################
#### exec code

if __name__ == "__main__":

    Test();

