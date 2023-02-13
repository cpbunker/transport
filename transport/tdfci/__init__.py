'''
Time dependent fci code 
Author: Ruojing Peng

tdfci module:
- have to run thru direct_uhf solver
- I use all spin up formalism: only alpha electrons input, only h1e_a and g1e_aa matter to solver
- benchmarked with dot impurity model from ruojings_td_fci.py
- TimeProp is main driver
- turn on bias in leads, pass hamiltonians, molecule, and scf object to time prop
- outputs current and energy vs time
'''

from transport import fci_mod
from transport.fci_mod import ops

from pyscf import lib, fci, scf, gto, ao2mo
from pyscf.fci import direct_uhf, direct_nosym, cistring

import numpy as np
import functools
import os
import time
einsum = lib.einsum

################################################################
#### kernel

def kernel(h1e, g2e, i_state, mol_inst, scf_inst, tf, dt, verbose=0):
    '''
    Kernel for getting observables at each time step, for plotting
    Outputs 1d arrays of time, energy, current, occupancy, Sz

    Returns
    init_obs, 2d arr, rows are sites and columns are occ, Sx, Sy, Sz at t=0
    observables, 2d arr, rows are time and columns are all observables
    '''

    # assertion statements to check inputs
    assert( np.shape(h1e)[0] == np.shape(g2e)[0]);
    assert( type(mol_inst) == type(gto.M()));
    assert( type(scf_inst) == type(scf.UHF(mol_inst)));

    # unpack
    Norbs = np.shape(h1e)[0];
    Nelecs = (mol_inst.nelectron,0);
    N = int(tf/dt+1e-6); # number of time steps beyond t=0
    
    # time propagation requires
    # - ERIS object to encode hamiltonians
    # - CI object to encode ci states
    Eeris = ERIs(h1e, g2e, scf_inst.mo_coeff); # hamiltonian info
    ci = CIObject(i_state, Norbs, Nelecs); # wf info

    # from the time prop we want to record
    # - ci object at each step ? (time, vec)
    # - observables  at each step (time, total E, occ for each orb)
    civecs = np.zeros((N+1, 1+len(i_state)), dtype = complex);
    observables = np.zeros((N+1, 2+Norbs), dtype = complex );
    
    # operators for observables
    obs_ops = [];
    for orbi in range(Norbs): # orb occupancies
        obs_ops.append(ops.occ([orbi], Norbs) );
    assert( 2+len(obs_ops) == np.shape(observables)[1] ); # 2 is time and energy

    # convert ops (which are ndarrays) to ERIs objects
    obs_eris = [Eeris];
    for op in obs_ops:

        # depends on if a 1e or 2e operator
        if(len(np.shape(op)) == 2): # 1e op
            obs_eris.append(ERIs(op, np.zeros((Norbs, Norbs, Norbs, Norbs)), Eeris.mo_coeff) );
        elif(len(np.shape(op)) == 4): # 2e op
            obs_eris.append(ERIs(np.zeros((Norbs, Norbs)), op, Eeris.mo_coeff) );
        else: assert(False);
    
    # time step loop
    for i in range(N+1):

        if(verbose > 3): print(" - time: ", i*dt);
    
        # density matrices
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12();
        
        # time step
        dr, dr_imag = compute_update(ci, Eeris, dt) # update state (r, an fcivec) at each time step
        r = ci.r + dt*dr
        r_imag = ci.i + dt*dr_imag # imag part of fcivec
        norm = np.linalg.norm(r + 1j*r_imag) # normalize complex vector
        ci.r = r/norm # update cisolver attributes
        ci.i = r_imag/norm

        # store ci
        civecs[i,0] = i*dt;
        civecs[i, 1:] = (ci.r + complex(0,1)*ci.i).flatten()
        
        # compute observables
        observables[i,0] = i*dt; # time
        for ei in range(len(obs_eris)): # iter over eris list
            observables[i, ei+1] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), obs_eris[ei] );

        # before any time stepping, get initial state

    # return val is array of observables
    # column ordering is always t, E, JupL, JupR, JdownL, JdownR, concurrence, (occ, Sx, Sy, Sz for each site)
    return civecs, observables

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
    '''
    Ruojing's code
    Computes <H> by
        1) getting h1e, h2e from eris object
        2) contracting with density matrix

    I overload this function by passing it eris w/ arb op x stored
    then ruojings code gets <x> for any eris operator x

    Args:
    d1, d2, 1 and 2 particle density matrices
    eris, object which contains hamiltonians
    ''' 

    h1e_a, h1e_b = eris.h1e
    g2e_aa, g2e_ab, g2e_bb = eris.g2e
    h1e_a = np.array(h1e_a,dtype=complex)
    h1e_b = np.array(h1e_b,dtype=complex)
    g2e_aa = np.array(g2e_aa,dtype=complex)
    g2e_ab = np.array(g2e_ab,dtype=complex)
    g2e_bb = np.array(g2e_bb,dtype=complex)
    d1a, d1b = d1
    d2aa, d2ab, d2bb = d2
    # to physicts notation
    g2e_aa = g2e_aa.transpose(0,2,1,3)
    g2e_ab = g2e_ab.transpose(0,2,1,3)
    g2e_bb = g2e_bb.transpose(0,2,1,3)
    d2aa = d2aa.transpose(0,2,1,3)
    d2ab = d2ab.transpose(0,2,1,3)
    d2bb = d2bb.transpose(0,2,1,3)
    # antisymmetrize integral
    g2e_aa -= g2e_aa.transpose(1,0,2,3)
    g2e_bb -= g2e_bb.transpose(1,0,2,3)

    e  = einsum('pq,qp',h1e_a,d1a)
    e += einsum('PQ,QP',h1e_b,d1b)
    e += 0.25 * einsum('pqrs,rspq',g2e_aa,d2aa)
    e += 0.25 * einsum('PQRS,RSPQ',g2e_bb,d2bb)
    e +=        einsum('pQrS,rSpQ',g2e_ab,d2ab)
    return e

class ERIs():
    def __init__(self, h1e, g2e, mo_coeff):
        ''' SIAM-like model Hamiltonian
            h1e: 1-elec Hamiltonian in site basis 
            g2e: 2-elec Hamiltonian in site basis
                 chemists notation (pr|qs)=<pq|rs>
            mo_coeff: moa, mob 
        '''
        moa, mob = mo_coeff
        
        h1e_a = einsum('uv,up,vq->pq',h1e,moa,moa)
        h1e_b = einsum('uv,up,vq->pq',h1e,mob,mob)
        g2e_aa = einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_aa = einsum('prxy,xq,ys->prqs',g2e_aa,moa,moa)
        g2e_ab = einsum('uvxy,up,vr->prxy',g2e,moa,moa)
        g2e_ab = einsum('prxy,xq,ys->prqs',g2e_ab,mob,mob)
        g2e_bb = einsum('uvxy,up,vr->prxy',g2e,mob,mob)
        g2e_bb = einsum('prxy,xq,ys->prqs',g2e_bb,mob,mob)

        self.mo_coeff = mo_coeff
        self.h1e = h1e_a, h1e_b
        self.g2e = g2e_aa, g2e_ab, g2e_bb

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
