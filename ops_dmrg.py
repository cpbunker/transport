'''
Christian Bunker
M^2QM at UF
June 2021

ops.py

Representation of operators, hamiltonian and otherwise, in DMRG
ie as generating functions (with yield statements) which are
then passed to the Hamiltonian.hamiltonian.build_mpo() method

pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr


'''

import numpy as np

#######################################################
####


def occ(site_i, norbs):
    '''
    Operator for the occupancy of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    def occ_yield(norbs, adag, a):

        # iter over site(s) we want occupancies of
        for i in site_i:
            spin = 0; # ASU formalism
            yield adag[i,spin]*a[i,spin]; # yield number operator of this site

    return occ_yield;


def Sz(site_i, norbs):
    '''
    Operator for the z spin of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    def Sz_yield(norbs, adag, a):

        # iter over site(s) we want occupancies of
        for i in site_i:
            spin = 0; # ASU formalism
            if(i % 2 == 0): # spin up orb
                yield (1/2)*adag[i,spin]*a[i,spin]; 
            else: # spin down orb
                yield (-1/2)*adag[i,spin]*a[i,spin];

    return Sz_yield;


def Jup(site_i, norbs):
    '''
    Current of up spin e's thru sitei
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( len(site_i) == 2); # should be only 1 site ie 2 spatial orbs

    def J_yield(norbs, adag, a):

        # even spin index is up spins
        upi = site_i[0];
        spin = 0; # ASU formalism
        assert(upi % 2 == 0); # check even
        yield (-1/2)*adag[upi-2,spin]*a[upi,spin] # dot up spin to left up spin #left moving is negative current
        yield (1/2)*adag[upi,spin]*a[upi-2,spin]# left up spin to dot up spin # hc of above # right moving is +
        yield (1/2)*adag[upi+2,spin]*a[upi,spin]  # up spin to right up spin
        yield (-1/2)*adag[upi,spin]*a[upi+2,spin] # hc

    return J_yield;


def Jdown(site_i, norbs):
    '''
    Current of down spin e's thru sitei
    ASU formalism only !!!
    Args:
    - site_i, list of (usually spin orb) site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( len(site_i) == 2); # should be only 1 site ie 2 spatial orbs

    def J_yield(norbs, adag, a):

        # odd spin index is down spins
        dwi = site_i[1];
        spin = 0; # ASU formalism
        assert(dwi % 2 == 1); # check odd
        yield (-1/2)*adag[dwi-2,spin]*a[dwi,spin] # dot dw spin to left dw spin #left moving is negative current
        yield (1/2)*adag[dwi,spin]*a[dwi-2,spin]  # left dw spin to dot dw spin # hc of above # right moving is +
        yield (1/2)*adag[dwi+2,spin]*a[dwi,spin]  # dot dw spin to right dw spin
        yield (-1/2)*adag[dwi,spin]*a[dwi+2,spin]

    return J_yield;


def h_B(B, theta, site_i, norbs, verbose=0):
    '''
    Turn on a magnetic field of strength B in the theta hat direction, on site i
    This has the effect of preparing the spin state of the site
    e.g. large, negative B, theta=0 yields an up electron

    Args:
    - B, float, mag field strength
    - theta, float, mag field direction
    - norbs, int, num spin orbs in whole system
    - site_i, list, spin indices (even up, odd down) of site that feels mag field

    Returns 2d np array repping magnetic field on given sites
    '''

    assert(isinstance(site_i, list) );

    hB = np.zeros((norbs,norbs));
    for i in range(site_i[0],site_i[-1],2): # i is spin up, i+1 is spin down
        hB[i,i+1] = B*np.sin(theta)/2; # implement the mag field, x part
        hB[i+1,i] = B*np.sin(theta)/2;
        hB[i,i] = B*np.cos(theta)/2;    # z part
        hB[i+1,i+1] = -B*np.cos(theta)/2;
        
    if (verbose > 3): print("h_B:\n", hB);
    return hB;


#####################################
#### wrapper functions, test code
    
#####################################
#### exec code

if(__name__ == "__main__"):

    pass;


