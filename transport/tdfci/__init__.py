'''
Time dependent fci code and SIAM example
Author: Ruojing Peng

tdfci module:
- have to run thru direct_uhf solver
- I use all spin up formalism: only alpha electrons input, only h1e_a and g1e_aa matter to solver
- benchmarked with dot impurity model from ruojings_td_fci.py
- TimeProp is main driver
- turn on bias in leads, pass hamiltonians, molecule, and scf object to time prop
- outputs current and energy vs time
'''

from transport import ops
from transport import fci_mod

from pyscf import lib, fci, scf, gto, ao2mo
from pyscf.fci import direct_uhf, direct_nosym, cistring

import numpy as np
import functools
import os
import time
einsum = lib.einsum

################################################################
#### kernel

def kernel(h1e, g2e, fcivec, mol, scf_inst, tf, dt, dot_i, ASU = True, RK = 4, verbose= 0):
    '''
    Kernel for getting observables at each time step, for plotting
    Outputs 1d arrays of time, energy, current, occupancy, Sz

    Returns
    init_obs, 2d arr, rows are sites and columns are occ, Sx, Sy, Sz at t=0
    observables, 2d arr, rows are time and columns are all observables
    '''

    # assertion statements to check inputs
    assert( np.shape(h1e)[0] == np.shape(g2e)[0]);
    assert( type(mol) == type(gto.M() ) );
    assert( type(scf_inst) == type(scf.UHF(mol) ) );
    assert( isinstance(dot_i, list));

    # unpack
    norbs = np.shape(h1e)[0];
    nelecs = (mol.nelectron,0);
    ndots = int( (dot_i[-1] - dot_i[0] + 1)/2 );
    N = int(tf/dt+1e-6); # number of time steps
    n_generic_obs = 7; # 7 are time, E, 4 J's, concurrence
    sites = np.array(range(norbs)).reshape(int(norbs/2),2); # all indices, sep'd by site

    # time propagation requires
    # - ERIS object to encode hamiltonians
    # - CI object to encode ci states
    # - observables array to observables (columns) vs time (rows)
    Eeris = ERIs(h1e, g2e, scf_inst.mo_coeff); # hamiltonian info
    ci = CIObject(fcivec, norbs, nelecs); # wf info
    observables = np.zeros((N+1, n_generic_obs+4*len(sites) ), dtype = complex ); # generic plus occ, Sx, Sy, Sz per site
    
    # operators for observables
    obs_ops = [];
    obs_ops.extend(ops.Jup(dot_i, norbs) ); # up e current
    obs_ops.extend(ops.Jdown(dot_i, norbs) ); # down e current
    obs_ops.append(ops.spinflip(dot_i,norbs) ); # concurrence btwn dots or dot and RL
    for site in sites: # site specific observables
        obs_ops.append(ops.occ(site, norbs) );
        obs_ops.append(ops.Sx(site, norbs) );
        obs_ops.append(ops.Sy(site, norbs) );
        obs_ops.append(ops.Sz(site, norbs) );
    assert( 2+len(obs_ops) == np.shape(observables)[1] ); # 2 is time and energy

    # convert ops (which are ndarrays) to ERIs objects
    obs_eris = [Eeris];
    for op in obs_ops:

        # depends on if a 1e or 2e operator
        if(len(np.shape(op)) == 2): # 1e op
            obs_eris.append(ERIs(op, np.zeros((norbs,norbs,norbs,norbs)), Eeris.mo_coeff) );
        elif(len(np.shape(op)) == 4): # 2e op
            obs_eris.append(ERIs(np.zeros((norbs, norbs)), op, Eeris.mo_coeff) );
        else: assert(False);
    
    # time step loop
    for i in range(N+1):

        if(verbose > 2): print("    time: ", i*dt);
    
        # density matrices
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12();
        
        # time step
        dr, dr_imag = compute_update(ci, Eeris, dt, RK) # update state (r, an fcivec) at each time step
        r = ci.r + dt*dr
        r_imag = ci.i + dt*dr_imag # imag part of fcivec
        norm = np.linalg.norm(r + 1j*r_imag) # normalize complex vector
        ci.r = r/norm # update cisolver attributes
        ci.i = r_imag/norm
        
        # compute observables
        observables[i,0] = i*dt; # time
        for ei in range(len(obs_eris)): # iter over eris list
            observables[i, ei+1] = compute_energy((d1a,d1b),(d2aa,d2ab,d2bb), obs_eris[ei] );

        # before any time stepping, get initial state
        if(i==0):
            # get site specific observables at t=0 in array where rows are sites
            initobs = np.real(np.reshape(observables[i,n_generic_obs:],(len(sites), 4) ) );

    # return val is array of observables
    # column ordering is always t, E, JupL, JupR, JdownL, JdownR, concurrence, (occ, Sx, Sy, Sz for each site)
    return initobs, observables;
    


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


############################################################################
#### wrappers

def DotData(nleads, nelecs, ndots, timestop, deltat, phys_params, spinstate = "", prefix = "dat/temp/", namevar="Vg", verbose = 0):
    '''
    Walks thru all the steps for plotting current thru a SIAM, using FCI for equil state
    and td-FCI for nonequil dynamics. Impurity is a single quantum dot w/ gate voltage and hubbard U
    - construct the eq hamiltonian, 1e and 2e parts, as np arrays
    - encode hamiltonians in an scf.UHF inst
    - do FCI on scf.UHF to get exact gd state
    - turn on thyb to intro nonequilibrium (current will flow)
    - use ruojing's code (td_fci module) to do time propagation

    Args:
    nleads, tuple of ints of left lead sites, right lead sites
    nelecs, tuple of num up e's, 0 due to ASU formalism
    ndots, int, number of dots in impurity
    timestop, float, how long to run for
    deltat, float, time step increment
    physical params, tuple of t, thyb, Vbias, mu, Vgate, U, B, theta, phi
    	if None, gives defaults vals for all (see below)
    prefix: assigns prefix (eg folder) to default output file name

    Returns:
    none, but outputs t, observable data to /dat/DotData/ folder
    '''

    # check inputs
    assert( isinstance(nleads, tuple) );
    assert( isinstance(nelecs, tuple) );
    assert( isinstance(ndots, int) );
    assert( isinstance(timestop, float) );
    assert( isinstance(deltat, float) );
    assert( isinstance(phys_params, tuple) or phys_params == None);

    # set up the hamiltonian
    imp_i = [nleads[0]*2, nleads[0]*2 + 2*ndots - 1 ]; # imp sites start and end, inclusive
    norbs = 2*(nleads[0]+nleads[1]+ndots); # num spin orbs
    # nelecs left as tunable
    t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, B, theta = phys_params;

    # get 1 elec and 2 elec hamiltonian arrays for siam, dot model impurity
    if(verbose): print("1. Construct hamiltonian")
    eq_params = t_leads, 0.0, t_dots, 0.0, mu, V_gate, U, B, theta; # thyb, Vbias turned off, mag field in theta to prep spin
    h1e, g2e, input_str = ops.dot_hams(nleads, nelecs, ndots, eq_params, spinstate, verbose = verbose);
        
    # get scf implementation siam by passing hamiltonian arrays
    if(verbose): print("2. FCI solution");
    mol, dotscf = fci_mod.arr_to_scf(h1e, g2e, norbs, nelecs, verbose = verbose);
    
    # from scf instance, do FCI, get exact gd state of equilibrium system
    E_fci, v_fci = fci_mod.scf_FCI(mol, dotscf, verbose = verbose);
    if( verbose > 3): print("|initial> = ",v_fci);
    
    # prepare in nonequilibrium state by turning on t_hyb (hopping onto dot)
    if(verbose > 3 ): print("- Add nonequilibrium terms");
    neq_params = t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, 0.0, 0.0; # thyb, Vbias turned on, no mag field
    neq_h1e, neq_g2e, input_str_noneq = ops.dot_hams(nleads, nelecs, ndots, neq_params, "", verbose = verbose);

    # from fci gd state, do time propagation
    if(verbose): print("3. Time propagation")
    init, observables = kernel(neq_h1e, neq_g2e, v_fci, mol, dotscf, timestop, deltat, imp_i, verbose = verbose);
    
    # write results to external file
    fname = os.getcwd()+"/";
    if namevar == "Vg":
        fname += prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vg"+str(V_gate)+".npy";
    elif namevar == "U":
        fname += prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_U"+str(U)+".npy";
    elif namevar == "Vb":
        fname += prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vb"+str(V_bias)+".npy";
    elif namevar == "th":
        fname += prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_th"+str(t_hyb)+".npy";
    else: assert(False); # invalid option
    hstring = time.asctime();
    hstring += "\ntf = "+str(timestop)+"\ndt = "+str(deltat);
    hstring += "\nASU formalism, t_hyb noneq. term"
    hstring += "\nEquilibrium"+input_str; # write input vals to txt
    hstring += "\nNonequlibrium"+input_str_noneq;
    np.savetxt(fname[:-4]+".txt", init, header = hstring); # saves info to txt
    np.save(fname, observables);
    if (verbose): print("4. Saved data to "+fname);
    
    return fname; # end dot data


def Data(source, leadsites, h1e, g2e, tf, dt, fname = "fci_data.npy", verbose = 0):
    '''
    Wrapper for taking a system setup (geometry spec'd by leadsites, physics by
    h1e, g2e, and electronic config by source) and going through the entire
    tdfci process.

    Args:
    source, list, spin orbs to fill with an electron initially
    leadsites, tuple of how many sites in left, right lead
    h1e, 2d arr, one body interactions
    g2e, 4d arr, two body interactions
    tf, float, time to stop propagation
    dt, float, time step for propagation
    '''

    # check inputs
    assert(np.shape(h1e) == np.shape(g2e)[:2]);
    
    # unpack
    nelecs = (len(source), 0);
    norbs = np.shape(h1e)[0];
    ndets = int(np.math.factorial(norbs)/(np.math.factorial(nelecs[0])*np.math.factorial(norbs - nelecs[0])));
    imp_i = [2*leadsites[0],norbs - 2*leadsites[1]-1];

    # get scf implementation siam by passing hamiltonian arrays
    if(verbose): print("1. FCI solution");
    mol, dotscf = fci_mod.arr_to_scf(h1e, g2e, norbs, nelecs, verbose = verbose);
    
    # from scf instance, do FCI, get exact gd state of equilibrium system
    E_fci, v_fci = fci_mod.scf_FCI(mol, dotscf, verbose = verbose);
    if( verbose > 3): print("\n - |initial> = \n",v_fci);

    # prep initial state
    ci0 = np.zeros((ndets,1));
    ci0 = v_fci;

    assert False;
    # from fci gd state, do time propagation
    if(verbose): print("2. Time propagation");
    init, observables = td_fci.kernel(h1e, g2e, v_fci, mol, dotscf, tf, dt, imp_i, verbose = verbose);
    
    hstring = time.asctime();
    hstring += "\ntf = "+str(tf)+"\ndt = "+str(dt);
    hstring += "\n"+str(h1e);
    np.savetxt(fname[:-4]+".txt", init, header = hstring); # saves info to txt
    np.save(fname, observables);
    if (verbose): print("3. Saved data to "+fname);
    
    return fname; # end custom data



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
