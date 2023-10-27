'''
Ruojing Peng
Chan group, Caltech

This code uses RK4 exact diagonalization to do discrete time evolution on a
quantum state.

Christian Bunker has adapted this code from Ruojing to
benchmark model Hamiltonians, where the spin degrees of freedom are more
important than the original quantum chemistry setting. Therefore the "all
spin up" formalism is used:
- instead of N spatial orbitals with up to double occupancy, we have 2N
    fermionic orbitals with up to single occupancy. The even ones are spin
    up and the odd ones are spin down
- The electron tuple is always (Ne, 0), ie code sees no down electrons
- the Hamiltonian matrix elements are not spin-degenerate, but only the
    up-up elements are nonzero (ie only h1e_aa and g2e_aa)
- because of the previous point, the direct_uhf solver must always be used

Other notes:
- kernel is main driver
- observables should be calculated within kernel
- the Hamiltonian for time propagation (the dynamic Hamiltonian) must include
    a perturbation relative to the ground state Hamiltonian. Often this is
    turning on a bias voltage, or hopping through the central region
'''

from pyscf import lib, fci, scf, gto, ao2mo
from pyscf.fci import direct_uhf, cistring

import numpy as np
import functools


################################################################
#### kernel

def kernel(ci_inst, eris_inst, tf, dt):
    '''
    Main driver of time evolution

    Args:
    ci_inst, a CIObject (def'd below) which contains the FCI state. This
        state is time evolved IN PLACE
    eris_inst, an ERIs object (def'd below) which contains the matrix elements
        of the dynamic Hamiltonian

    Calculation of observables:
    '''

    Nsteps = int(tf/dt+1e-6); # number of time steps beyond t=0
    for i in range(Nsteps+1):
        # update state
        dr, dr_imag = compute_update(ci_inst, eris_inst, dt) # update state (r, an fcivec) at each time step
        r = ci_inst.r + dt*dr
        r_imag = ci_inst.i + dt*dr_imag # imag part of fcivec
        norm = np.linalg.norm(r + 1j*r_imag) # normalize complex vector
        ci_inst.r = r/norm # update cisolver attributes
        ci_inst.i = r_imag/norm

    return ci_inst;

################################################################
#### util functions

def make_hop(eris, norb, nelec):
    h2e = direct_uhf.absorb_h1e(eris.h1e, eris.g2e, norb, nelec,.5)
    def _hop(c):
        return direct_uhf.contract_2e(h2e, c, norb, nelec)
    return _hop

def compute_update(ci, eris, h, RK=4):
    hop = make_hop(eris, ci.norb, ci.nelec)
    dr1 =  hop(ci.i)
    di1 = -hop(ci.r)
    if RK == 1:
        return dr1, di1
    if RK == 4:
        r = ci.r+dr1*h*0.5
        i = ci.i+di1*h*0.5
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr2 =  hop(i)
        di2 = -hop(r)

        r = ci.r+dr2*h*0.5
        i = ci.i+di2*h*0.5
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr3 =  hop(i)
        di3 = -hop(r)

        r = ci.r+dr3*h
        i = ci.i+di3*h
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr4 =  hop(i)
        di4 = -hop(r)

        dr = (dr1+2.0*dr2+2.0*dr3+dr4)/6.0
        di = (di1+2.0*di2+2.0*di3+di4)/6.0
        return dr, di      
    
def compute_energy(d1, d2, eris, time=None):
    raise NotImplementedError("see ompute_obs below");

def compute_obs(ci_inst, op_eris):
    '''
    Ruojing's code
    Computes <H> by
        1) getting h1e, h2e from eris object
        2) contracting these with density matrices from co object

    I overload this function by passing it eris w/ arb op x stored
    then ruojings code gets <x> for any eris operator x

    Args:
    ci_obj, object which contains a particular many body state
    eris_obj, object which contains hamiltonians
    '''

    # set up return values
    h1e_a, h1e_b = op_eris.h1e
    g2e_aa, g2e_ab, g2e_bb = op_eris.g2e
    h1e_a = np.array(h1e_a,dtype=complex)
    h1e_b = np.array(h1e_b,dtype=complex)
    g2e_aa = np.array(g2e_aa,dtype=complex)
    g2e_ab = np.array(g2e_ab,dtype=complex)
    g2e_bb = np.array(g2e_bb,dtype=complex)

    # get density matrices
    (d1a, d1b), (d2aa, d2ab, d2bb) = ci_inst.compute_rdm12();

    # convert to physicts notation
    g2e_aa = g2e_aa.transpose(0,2,1,3)
    g2e_ab = g2e_ab.transpose(0,2,1,3)
    g2e_bb = g2e_bb.transpose(0,2,1,3)
    d2aa = d2aa.transpose(0,2,1,3)
    d2ab = d2ab.transpose(0,2,1,3)
    d2bb = d2bb.transpose(0,2,1,3)

    # antisymmetrize integral
    g2e_aa -= g2e_aa.transpose(1,0,2,3)
    g2e_bb -= g2e_bb.transpose(1,0,2,3)

    # calculate observable
    e  = lib.einsum('pq,qp',h1e_a,d1a)
    e += lib.einsum('PQ,QP',h1e_b,d1b)
    e += 0.25 * lib.einsum('pqrs,rspq',g2e_aa,d2aa)
    e += 0.25 * lib.einsum('PQRS,RSPQ',g2e_bb,d2bb)
    e +=        lib.einsum('pQrS,rSpQ',g2e_ab,d2ab)

    if(abs(np.imag(e)) > op_eris.imag_cutoff): print(e); raise ValueError;
    return np.real(e);

class ERIs():
    def __init__(self, h1e, g2e, mo_coeff, imag_cutoff = 1e-12):
        '''
        h1e: 1-elec Hamiltonian in site basis
        g2e: 2-elec Hamiltonian in site basis
              chemists notation (pq|rs)=<pr|qs>
        mo_coeff: moa, mob
        '''
        moa, mob = mo_coeff

        h1e_a = lib.einsum('uv,up,vq->pq',h1e,moa,moa)
        h1e_b = lib.einsum('uv,up,vq->pq',h1e,mob,mob)
        g2e_aa = lib.einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_aa = lib.einsum('prxy,xq,ys->prqs',g2e_aa,moa,moa)
        g2e_ab = lib.einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_ab = lib.einsum('prxy,xq,ys->prqs',g2e_ab,mob,mob)
        g2e_bb = lib.einsum('uvxy,up,vr->prxy',g2e,mob,mob)
        g2e_bb = lib.einsum('prxy,xq,ys->prqs',g2e_bb,mob,mob)

        self.mo_coeff = mo_coeff
        self.h1e = h1e_a, h1e_b
        self.g2e = g2e_aa, g2e_ab, g2e_bb
        self.imag_cutoff = imag_cutoff

class CIObject():
    def __init__(self, fcivec, norb, nelec):
        '''
           fcivec: ground state uhf fcivec
           norb: size of site basis
           nelec: nea, neb
        '''       
        self.r = fcivec.copy() # ie r is the state in slater det basis
        self.i = np.zeros_like(fcivec)
        self.norb = norb
        self.nelec = nelec

    def dot(self, ket):
        if(not isinstance(ket, CIObject)): raise TypeError;
        if(self.norb != ket.norb or self.nelec != ket.nelec): raise ValueError;
        return np.dot( np.conj(self.r + complex(0,1)*self.i)[:,0], (ket.r + complex(0,1)*ket.i)[:,0]);

    def __str__(self):
        return str((self.r + complex(0,1)*self.i)[:,0]);

    def compute_rdm1(self):
        rr = direct_uhf.make_rdm1s(self.r, self.norb, self.nelec) # tuple of 1 particle density matrices for alpha, beta spin. self.r is fcivec
        # dm1_alpha_pq = <a_p alpha ^dagger a_q alpha
        ii = direct_uhf.make_rdm1s(self.i, self.norb, self.nelec)
        ri = direct_uhf.trans_rdm1s(self.r, self.i, self.norb, self.nelec) # tuple of transition density matrices for alpha, beta spin. 1st arg is a bra and 2nd arg is a ket
        d1a = rr[0] + ii[0] + 1j*(ri[0]-ri[0].T)
        d1b = rr[1] + ii[1] + 1j*(ri[1]-ri[1].T)
        return d1a, d1b

    def compute_rdm12(self):
        # 1pdm[q,p] = \langle p^\dagger q\rangle
        # 2pdm[p,r,q,s] = \langle p^\dagger q^\dagger s r\rangle
        rr1, rr2 = direct_uhf.make_rdm12s(self.r, self.norb, self.nelec)
        ii1, ii2 = direct_uhf.make_rdm12s(self.i, self.norb, self.nelec)
        ri1, ri2 = direct_uhf.trans_rdm12s(self.r, self.i, self.norb, self.nelec)
        # make_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2bb)
        # trans_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2ba, d2bb)
        d1a = rr1[0] + ii1[0] + 1j*(ri1[0]-ri1[0].T)
        d1b = rr1[1] + ii1[1] + 1j*(ri1[1]-ri1[1].T)
        d2aa = rr2[0] + ii2[0] + 1j*(ri2[0]-ri2[0].transpose(1,0,3,2))
        d2ab = rr2[1] + ii2[1] + 1j*(ri2[1]-ri2[2].transpose(3,2,1,0))
        d2bb = rr2[2] + ii2[2] + 1j*(ri2[3]-ri2[3].transpose(1,0,3,2))
        # 2pdm[r,p,s,q] = \langle p^\dagger q^\dagger s r\rangle
        d2aa = d2aa.transpose(1,0,3,2)
        d2ab = d2ab.transpose(1,0,3,2)
        d2bb = d2bb.transpose(1,0,3,2)
        return (d1a, d1b), (d2aa, d2ab, d2bb)


#####################################################################################
#### run code
    
if __name__ == "__main__":

    pass;
