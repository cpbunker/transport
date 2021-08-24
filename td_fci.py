'''
Time dependent fci code and SIAM example
Author: Ruojing Peng

td fci module:
- have to run thru direct_uhf solver
- I use all spin up formalism: only alpha electrons input, only h1e_a and g1e_aa matter to solver
- benchmarked with dot impurity model from ruojings_td_fci.py
- TimeProp is main driver
- turn on bias in leads, pass hamiltonians, molecule, and scf object to time prop
- outputs current and energy vs time
'''
import ops

from pyscf import lib, fci, scf, gto, ao2mo
from pyscf.fci import direct_uhf, direct_nosym, cistring
import numpy as np
import functools
import os
einsum = lib.einsum

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
        
        
################################################################
#### measure observables from density matrices

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
    
def compute_occ(site_i, d1, d2, mocoeffs, norbs, ASU = False):
    '''
    Compute the occupancy of the sites in list site_i
    by encoding n_i as an h1e

    ASU tells whether we are dealing with molecular orbitals (max occ 2) or spin orbitals (max occ 1)
    
    Generally to compute observable here and below follow this formula:
    - put operator in h1e, h2e format
    - use these to construct an eris
    - pass eris and density matrices to compute energy
    '''

    if(len(site_i) == 0): return 0.0;

    # operator
    occ = ops.occ(site_i, norbs);
    
    # have to store this operator as an eris object to compute "energy"
    occ_eris = ERIs(occ, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
    return compute_energy(d1,d2, occ_eris);


def compute_S(site_i, civec, norbs, nelecs, ASU = True):

    ciobj = CIObject(civec, norbs, nelecs);
    (d1a, d1b), (d2aa, d2ab, d2bb) = ciobj.compute_rdm12();
    

def compute_Sz(site_i, d1, d2, mocoeffs, norbs, ASU = False):
    '''
    Compute Sz for the impurity. See compute_occ doc above
    '''

    if(len(site_i) == 0): return 0.0;

    if ASU: # just put Sz operator in h1e form
        Sz = ops.Sz(site_i, norbs);

        # have to store this operator as an eris object
        Sz_eris = ERIs(Sz, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
        return compute_energy(d1,d2, Sz_eris);
    
    else: # more complicated, have to contract 1e density matrices seperately
        Sz = np.zeros((norbs,norbs));
        for i in site_i:
            Sz[i,i] = 1; # pick out relevant sites
        d1a, d1b = d1; # unpack 1 elec density matrix
        Sz_eris = ERIs(Sz, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs);
        Szup = compute_energy((d1a,np.zeros_like(d1b)),d2,Sz_eris); # contract with d1a only
        Szdw = compute_energy((np.zeros_like(d1a),d1b),d2,Sz_eris); # now with d1b only
        #print("up, dw = ",Szup, Szdw);
        return (1/2)*(Szup - Szdw);


def compute_Sx(site_i, d1, d2, mocoeffs, norbs, ASU = False):
    '''
    Compute Sx for the impurity. See compute_occ doc above
    '''

    if(len(site_i) == 0): return 0.0;

    if ASU: # just put Szxoperator in h1e form
        Sx = ops.Sx(site_i, norbs);

        # have to store this operator as an eris object
        Sx_eris = ERIs(Sx, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
        return compute_energy(d1,d2, Sx_eris);

    else:
        return 0.0;


def compute_Sy(site_i, d1, d2, mocoeffs, norbs, ASU = False):
    '''
    Compute Sy for the impurity. See compute_occ doc above
    '''

    if(len(site_i) == 0): return 0.0;

    if ASU: # just put Sy operator in h1e form
        Sy = ops.Sy(site_i, norbs);

        # have to store this operator as an eris object
        Sy_eris = ERIs(Sy, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
        return compute_energy(d1,d2, Sy_eris);

    else:
        return 0.0;
    
    
def compute_current(site_i,d1,d2,mocoeffs,norbs, ASU = False):
    '''
    Compute current through the impurity. See compute_occ docstring above

    Notes:
    - Does not multiply by t_hyb anymore! have to do later
    - spin free returns all current as up current, ASU actually gets both
    '''

    assert( isinstance(site_i, list) or isinstance(site__i, np.ndarray));

    if ASU:
        return compute_current_ASU(site_i,d1,d2,mocoeffs,norbs);

    # operator
    J = np.zeros((norbs,norbs));
    for doti in range(site_i[0], site_i[-1]+1, 1):
        if site_i[0] != 0: # if left lead exists, do left lead coupling
            J[doti - 1,doti] = -1/2;
            J[doti,doti - 1] =  1/2;
        if site_i[-1] != norbs-1: # if right lead exists, right lead coupling
            J[doti + 1,doti] =  1/2;
            J[doti,doti + 1] = -1/2;
    
    # have to store this operator as an eris object
    J_eris = ERIs(J, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs)
    J_val = compute_energy(d1,d2, J_eris);
    J_val = -np.imag(J_val);
    return J_val, 0;
    
def compute_current_ASU(site_i,d1,d2,mocoeffs,norbs):
    '''
    ASU formalism version of above
    '''

    # up, down current operators (1 e operators)
    # each op now returns tuple of left and right

    
    # have to store as an eris object
    Jup_eris = ERIs(Jup, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs);
    Jdown_eris = ERIs(Jdown, np.zeros((norbs,norbs,norbs,norbs)), mocoeffs);
    Jup_val = compute_energy(d1,d2, Jup_eris); # this func of ruojings gets <x> for whatever operator x is stored in eris arg
    Jdown_val = compute_energy(d1,d2, Jdown_eris);
    return -np.imag(Jup_val), -np.imag(Jdown_val); # -imag is same as *i

################################################################
#### kernel

def kernel_bleh(mode, eris, ci, tf, dt, RK=4, i_dot = None, t_dot = None, spinblind = False, verbose = 0):
    '''
    Wrapper for the different kernel implementations
    Lots of assertions to prevent confusion
    
    Kernel implementations:
    - std, taken from ruojing, outputs density matrices
    - plot, outputs values of obervables vs time for plotting
    
    All kernel funcs drive time propagation
    Args:
    - eris is an instance of ERIs (see below)
    - ci is an instance of CIObject (see below)
    - tf, dt, floats are final time and time step
    - RK is order of runge kutta, default 4th
    - dot_i is site (MO) index of the dot, not needed if not doing plot, hence defaults to None
    - t_dot is hopping strength between dot, leads, likewise defaults to None
    - verbose prints helpful debugging stuff

    Returns whatever kernel_std or kernel_plot returns
    '''

    
    # select specific kernel implementation from mode input
    modes = ["std","plot"];
    assert(mode in modes); # make sure mode input is supported
    
    if(mode == "std"):
    
        return kernel_std(eris,ci,tf,dt,RK);
        
    if(mode == "plot"):
    
        # check inputs
        assert(i_dot != None);
        assert(t_dot != None);
    
        return kernel_plot(eris, ci, tf, dt, i_dot, t_dot, RK, spinblind, verbose);

def kernel(eris, ci, tf, dt, dot_i, t_hyb, ASU = True, RK = 4, verbose= 0):
    '''
    Kernel for getting observables at each time step, for plotting
    Outputs 1d arrays of time, energy, dot occupancy, current
    Access thru calling kernel (see above) with mode=plot

    Returns
    timevals, 1d arr of time steps
    observables, tuple of arrs of observable values at each time: E(t), J(t), Occ(t), Sz(t)
    '''

    N = int(tf/dt+1e-6)
    all_i = np.arange(0,ci.norb, 1, dtype = int);
    left_i = all_i[:dot_i[0] ];
    right_i = all_i[dot_i[-1]+1:];
    norbs = ci.norb;


    # operators for observables
    JupL, JupR = ops.Jup(dot_i, norbs); # up e current
    JdownL, JdownR = ops.Jdown(dot_i, norbs); # down e current
    occL, occD, occR = ops.occ(left_i, norbs), ops.occ(dot_i, norbs), ops.occ(right_i, norbs);
    SzL, SzD, SzR = ops.Sz(left_i, norbs), ops.Sz(dot_i, norbs), ops.Sz(right_i, norbs);

    # eris for observables
    JupL_eris = ERIs(JupL, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    JupR_eris = ERIs(JupR, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    JdownL_eris = ERIs(JdownL, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    JdownR_eris = ERIs(JdownR, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    occL_eris = ERIs(occL, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    occD_eris = ERIs(occD, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    occR_eris = ERIs(occR, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    SzL_eris = ERIs(SzL, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    SzD_eris = ERIs(SzD, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    SzR_eris = ERIs(SzR, np.zeros((norbs,norbs,norbs,norbs)), eris.mo_coeff);
    
    #  observable return vals
    t_vals = np.zeros(N+1);
    energy_vals = np.zeros(N+1);
    current_vals = np.zeros((4,N+1)); # up and down e current separate
    occ_vals = np.zeros( (3,N+1), dtype = complex ); # occ list has [left lead occ, dot occ, right lead occ]
    Sz_vals = np.zeros( (3,N+1), dtype = complex ); # see below
    
    # time step loop
    for i in range(N+1):
    
        # density matrices
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12();

        # before any time stepping, get initial state
        if(i==0):
            occ_init = np.zeros(len(all_i), dtype = complex);
            Sz_init = np.zeros(len(all_i), dtype = complex);
            Sx_init = np.zeros(len(all_i), dtype = complex);
            Sy_init = np.zeros(len(all_i), dtype = complex);
            for sitej in all_i:# iter over sites
                if ASU: # sites are pairs of spin orbs
                    if sitej % 2 == 0: # do up and down together
                        sitejlist = [sitej, sitej+1];
                        occ_init[sitej] = compute_occ(sitejlist,(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, norbs, ASU = ASU);
                        Sz_init[sitej] = compute_Sz(sitejlist,(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, norbs, ASU = ASU);
                        Sx_init[sitej] = compute_Sx(sitejlist,(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, norbs, ASU = ASU);
                        Sy_init[sitej] = compute_Sy(sitejlist,(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, norbs, ASU = ASU); 
                else: # sites are just sites
                    occ_init[sitej] = compute_occ([sitej],(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, norbs, ASU = ASU);
                    Sz_init[sitej] = compute_Sz([sitej],(d1a,d1b),(d2aa,d2ab,d2bb),eris.mo_coeff, norbs, ASU = ASU);
            initstatestr = "\nInitial state:"
            initstatestr += "\n    occ = "+str(np.real(occ_init));
            initstatestr += "\n    Sz = "+str(Sz_init);
            initstatestr += "\n    Sx = "+str(Sx_init);
            initstatestr += "\n    Sy = "+str(Sy_init);
        
        # time step
        dr, dr_imag = compute_update(ci, eris, dt, RK) # update state (r, an fcivec) at each time step
        r = ci.r + dt*dr
        r_imag = ci.i + dt*dr_imag # imag part of fcivec
        norm = np.linalg.norm(r + 1j*r_imag) # normalize complex vector
        ci.r = r/norm # update cisolver attributes
        ci.i = r_imag/norm
        
        # compute observables
        t_vals[i] = i*dt;
        energy_vals[i]  = np.real(compute_energy((d1a,d1b),(d2aa,d2ab,d2bb),eris));
        JupL_val = -np.imag(compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), JupL_eris));
        JupR_val = -np.imag(compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), JupR_eris));
        JdownL_val = -np.imag(compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), JdownL_eris));
        JdownR_val = -np.imag(compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), JdownR_eris));
        current_vals[0][i] = t_hyb*JupL_val;
        current_vals[1][i] = t_hyb*JupR_val;
        current_vals[2][i] = t_hyb*JdownL_val;
        current_vals[3][i] = t_hyb*JdownR_val; # add in hopping strength
        
        # occupancy of left lead, dot, right lead
        occ_vals[0][i] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), occL_eris);
        occ_vals[1][i] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), occD_eris);
        occ_vals[2][i] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), occR_eris);

        # total z spin of left lead, dot, right lead
        Sz_vals[0][i] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), SzL_eris);
        Sz_vals[1][i] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), SzD_eris);      
        Sz_vals[2][i] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), SzR_eris);

        if(verbose > 2): print("    time: ", i*dt);

    # return val is array of observables
    # ordering is always t, E, JupL, JupR, JdownL, JdownR, occ left, occ dot, occ right, Sz left, Sz dot, Sz right
    observables = [t_vals, energy_vals, current_vals[0], current_vals[1], current_vals[2], current_vals[3], occ_vals[0], occ_vals[1], occ_vals[2], Sz_vals[0], Sz_vals[1], Sz_vals[2]]
    return initstatestr, np.array(observables)
    
def kernel_old(eris, ci, tf, dt, RK):
    '''
    Kernel for td calc copied straight from ruojing
    Outputs density matrices in form (1e alpha, 1e beta), (2e aa, 2e ab, 2e bb)
    Equivalent to calculating wf at every time step instead of just some observables and discarding
    Not in use at moment
    '''
    N = int(tf/dt+1e-6)
    d1as = []
    d1bs = []
    d2aas = []
    d2abs = []
    d2bbs = []
    for i in range(N+1):
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12()
        d1as.append(d1a)
        d1bs.append(d1b)
        d2aas.append(d2aa)
        d2abs.append(d2ab)
        d2bbs.append(d2bb)

        print('time: ', i*dt)
        dr, di = compute_update(ci, eris, dt, RK)
        r = ci.r + dt*dr
        i = ci.i + dt*di
        norm = np.linalg.norm(r + 1j*i)
        ci.r = r/norm
        ci.i = i/norm
    d1as = np.array(d1as,dtype=complex)
    d1bs = np.array(d1bs,dtype=complex)
    d2aas = np.array(d2aas,dtype=complex)
    d2abs = np.array(d2abs,dtype=complex)
    d2bbs = np.array(d2bbs,dtype=complex)
    return (d1as, d1bs), (d2aas, d2abs, d2bbs)

#####################################################################################
#### class definitions

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


##########################################################################################################
#### time propagation 

def TimeProp(h1e, h2e, fcivec, mol,  scf_inst, time_stop, time_step, dot_i, t_hyb, verbose = 0):
    '''
    Time propagate an FCI gd state
    The physics of the FCI gd state is encoded in an scf instance

    Kernel is driver of time prop
    Kernel gets hamiltonian, and ci wf, which is coeffs of slater dets of HF-determined molecular orbs
    Then updates ci wf at each time step, this in turn updates density matrices
    Contract density matrices at each time step to compute obervables (e.g. compute_energy, compute_current functions)
    Set kernel_mode to std to call kernel_std which returns density matrices
    Set kernel_mode to plot to call kernel_plot which returns arrays of time, observable vals (default)
    Defaults to kernel mode plot, in which case returns
    timevals, observables (tuple of E(t), J(t), Occ(t), Sz(t) )
    '''

    # assertion statements to check inputs
    assert( np.shape(h1e)[0] == np.shape(h2e)[0]);
    assert( type(mol) == type(gto.M() ) );
    assert( type(scf_inst) == type(scf.UHF(mol) ) );
    assert( isinstance(dot_i, list));

    # unpack
    norbs = np.shape(h1e)[0];
    nelecs = (mol.nelectron,0);
    if(verbose > 1):
        print("\n- Time Propagation, norbs = ", norbs, ", nelecs = ", nelecs);

    # time propagation kernel requires
    # - ERIS object to encode hamiltonians
    # - CI object to encode ci states
    eris = ERIs(h1e, h2e, scf_inst.mo_coeff);
    ci = CIObject(fcivec, norbs, nelecs);
    
    # kernel does time prop, NB we assume a spin blind formalism
    return kernel(eris, ci, time_stop, time_step, dot_i, t_hyb, verbose = verbose);


###########################################################################################################
#### test code and wrapper funcs

def SpinfreeTest(nleads, nelecs, tf, dt, phys_params = None, verbose = 0):
    '''
    Spin free calculation of SIAM. Impurity is dot

    Args:
    - nleads, tuple of left lead sites, right lead sites
    - nelecs, tuple of up es, down es
    - tf, float, stop time run
    - dt, float, time step of time run
    - phys_params, tuple of all the physical inputs to the model, explained in code
                defaults to None, meaning std inputs

    Saves all observables as single array to .npy
    returns name of .npy file
    '''

    # inputs
    ll = nleads[0] # number of left leads
    lr = nleads[1] # number of right leads
    nelec =  nelecs
    norb = ll+lr+1 # total number of sites
    idot = ll # dot index

    # physical params, should always be floats
    if( phys_params == None): # defaults
        t = 1.0 # lead hopping
        td = 0.0 # dot-lead hopping not turned on yet, but still nonzero to make code more robust
        td_noneq = 0.4 # for when it is turned on
        V = -0.005 # bias
        Vg = -0.5 # gate voltage
        U = 1.0 # dot interaction

    else: # custom
        td = 0.0 # dot-lead hopping not turned on yet!
        t, td_noneq, V, Vg, U = phys_params

    if(verbose):
        print("\nInputs:\n- Left, right leads = ",(ll,lr),"\n- nelecs = ", nelec,"\n- Gate voltage = ",Vg,"\n- Bias voltage = ",V,"\n- Lead hopping = ",t,"\n- Dot lead hopping = ",td,"\n- U = ",U);

    #### make hamiltonian matrices, spin free formalism
    # remember impurity is just one level doto

    # quick fox for run_110
    if nelec == (0,1): Vg = -10.0
    
    # make ground state Hamiltonian, equilibrium (ie t_hyb and Vbias not turned on yet)
    if(verbose): print("1. Construct hamiltonian")
    h1e = np.zeros((norb,)*2)
    for i in range(norb):
        if i < norb-1:
            dot = (i==idot or i+1==idot)
            h1e[i,i+1] = -td if dot else -t
        if i > 0:
            dot = (i==idot or i-1==idot)
            h1e[i,i-1] = -td if dot else -t
    h1e[idot,idot] = Vg # input gate voltage on dot
    g2e = np.zeros((norb,norb, norb, norb)); # 2 body terms = hubbard
    g2e[idot,idot,idot,idot] = U
    
    if(verbose > 2):
        print("- Full one electron hamiltonian:\n", h1e)
        
    # code straight from ruojing, don't understand yet
    Pa = np.zeros(norb)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    Pb = np.zeros(norb)
    Pb[1::2] = 1.0
    Pb = np.diag(Pb)
    # UHF
    mol = gto.M(spin = 0)
    mol.incore_anyway = True
    mol.nelectron = sum(nelec)
    mol.spin = nelec[0] - nelec[1]
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args:h1e # put h1e into scf solver
    mf.get_ovlp = lambda *args:np.eye(norb) # init overlap as identity
    mf._eri = g2e # put h2e into scf solver
    if sum(nelecs) == 1: mf.kernel();
    else: mf.kernel(dm0=(Pa,Pb))

    # ground state FCI
    mo_a = mf.mo_coeff[0]
    mo_b = mf.mo_coeff[1]
    cisolver = direct_uhf.FCISolver(mol)
    h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
    h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
    g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
    g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
    g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
    g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
    g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
    h1e_mo = (h1e_a, h1e_b)
    g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
    eci, fcivec = cisolver.kernel(h1e_mo, g2e_mo, norb, nelec)
    if(verbose):
        print("2. FCI solution");
        print("- gd state energy, zero bias = ", eci);
        #print("- direct spin 1 gd state, zero bias = ",myE," (norbs, nelecs = ",norb,nelec,")")
    #############
        
    #### do time propagation

    # intro nonequilibrium terms (t_hyb = td nonzero)
    if(verbose): print("3. Time propagation")
    if nelec == (0,1): h1e[idot, idot] = 0.0;
    if nleads[0] != 0: # left lead coupling
        h1e[idot, idot-1] += -td_noneq; 
        h1e[idot-1, idot] += -td_noneq;
    if nleads[1] != 0: # right lead coupling
        h1e[idot+1, idot] += -td_noneq;
        h1e[idot, idot+1] += -td_noneq;
    for i in range(idot): # bias for leftward current (since V < 0)
        h1e[i,i] += V/2
    for i in range(idot+1,norb):
        h1e[i,i] += -V/2

    if(verbose > 2 ): print("- Nonequilibrium terms:\n", h1e);

    if True: # get noneq energies
        mycisolver = fci.direct_spin1.FCI();
        myE, myv = mycisolver.kernel(h1e, g2e, norb, nelec, nroots = 10);
        print("- Noneq energies = ",myE);

    eris = ERIs(h1e, g2e, mf.mo_coeff) # diff h1e than in uhf, thus time dependence
    ci = CIObject(fcivec, norb, nelec)
    kernel_mode = "plot"; # tell kernel whether to return density matrices or arrs for plotting
    init_str, observables = kernel(kernel_mode, eris, ci, tf, dt, i_dot = [idot], t_dot = td_noneq, verbose = verbose);
    print(init_str);

    return observables;
    

if __name__ == "__main__":

    pass;
