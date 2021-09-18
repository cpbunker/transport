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
#### 1 e operators, yield form

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


def Sx(site_i, norbs):
    '''
    Operator for the x spin of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of spin orb site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    def Sx_yield(norbs, adag, a):

        # iter over site(s) we want occupancies of
        for i in site_i:
            spin = 0; # ASU formalism
            if(i % 2 == 0): # spin up orb
                yield (1/2)*adag[i,spin]*a[i+1,spin];
                yield (1/2)*adag[i+1,spin]*a[i,spin];

    return Sx_yield;


def Sy(site_i, norbs):
    '''
    Operator for the x spin of sites specified by site_i
    ASU formalism only !!!
    Args:
    - site_i, list of spin orb site indices
    - norbs, total num orbitals in system
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));

    def Sy_yield(norbs, adag, a):

        # iter over site(s) we want occupancies of
        for i in site_i:
            spin = 0; # ASU formalism
            if(i % 2 == 0): # spin up orb
                yield (-1/2)*adag[i,spin]*a[i+1,spin];
                yield (1/2)*adag[i+1,spin]*a[i,spin];

    return Sy_yield;


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
                yield (-1/2)*adag[i+1,spin]*a[i+1,spin];

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

    def JL_yield(norbs, adag, a):

        # even spin index is up spins
        upi = site_i[0];
        spin = 0; # ASU formalism
        assert(upi % 2 == 0); # check even
        yield -adag[upi-2,spin]*a[upi,spin] # dot up spin to left up spin #left moving is negative current
        yield adag[upi,spin]*a[upi-2,spin]# left up spin to dot up spin # hc of above # right moving is +

    def JR_yield(norbs, adag, a):

        # even spin index is up spins
        upi = site_i[0];
        spin = 0; # ASU formalism
        yield adag[upi+2,spin]*a[upi,spin]  # up spin to right up spin
        yield -adag[upi,spin]*a[upi+2,spin] # hc

    return JL_yield, JR_yield;


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

    def JL_yield(norbs, adag, a):

        # odd spin index is down spins
        dwi = site_i[1];
        spin = 0; # ASU formalism
        assert(dwi % 2 == 1); # check odd
        yield -adag[dwi-2,spin]*a[dwi,spin] # dot dw spin to left dw spin #left moving is negative current
        yield adag[dwi,spin]*a[dwi-2,spin]  # left dw spin to dot dw spin # hc of above # right moving is +

    def JR_yield(norbs, adag, a):

        # odd spin index is down spins
        dwi = site_i[1];
        spin = 0; # ASU formalism
        yield adag[dwi+2,spin]*a[dwi,spin]  # dot dw spin to right dw spin
        yield -adag[dwi,spin]*a[dwi+2,spin]

    return JL_yield, JR_yield;


#######################################################
#### 2 e operators, yield form

def spinflip(site_i, norbs):
    '''
    define the "spin flip operator \sigma_y x \sigma_y for two qubits
    abs val of exp of spin flip operator gives concurrence
    '''

    # check inputs
    assert( isinstance(site_i, list) or isinstance(site_i, np.ndarray));
    #assert( len(site_i) == 4); # concurrence def'd for 2 qubits

    def sf_yield(norbs, adag, a): # just a copy of Sz for now
    
        # iter over site(s) we want occupancies of
        for i in site_i:
            spin = 0; # ASU formalism
            if(i % 2 == 0): # spin up orb
                yield (1/2)*adag[i,spin]*a[i,spin]; 
                yield (-1/2)*adag[i+1,spin]*a[i+1,spin];

    return sf_yield;


#####################################
#### wrapper functions, test code
    
#####################################
#### exec code

if(__name__ == "__main__"):

    pass;


