
'''
Main routine to set up DMFT parameters and run DMFT
'''

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

try:
    import block2
    from block2.su2 import MPICommunicator
    dmrg_ = True
except:
    dmrg_ = False
    pass
import numpy as np
import scipy, h5py
from fcdmft.utils import write
from fcdmft.dmft import gwdmft
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def dmft_abinitio():
    '''
    List of DMFT parameters
    gw_dmft : choose to run GW+DMFT (True) or HF+DMFT (False)
    opt_mu : whether to optimize chemical potential during DMFT cycles
    solver_type : choose impurity solver ('cc', 'ucc', 'dmrg', 'dmrgsz', 'fci')
    disc_type : choose bath discretization method ('opt', 'direct', 'linear', 'gauss', 'log')
    max_memory : maximum memory for DMFT calculation (per MPI process for CC, per node for DMRG)
    dmft_max_cycle : maximum number of DMFT iterations (set to 0 for one-shot DMFT)
    chkfile : chkfile for saving DMFT self-consistent quantities (hyb and self-energy)
    diag_only : choose to only fit diagonal hybridization (optional)
    orb_fit : special orbitals (e.g. 3d/4f) with x5 weight in bath optimization (optional)
    delta : broadening for discretizing hybridization (often 0.02-0.1 Ha)
    nbath : number of bath energies (can be any integer)
    nb_per_e : number of bath orbitals per bath energy (should be no greater than nval-ncore)
                total bath number = nb_per_e * nbath
    mu : initial chemical potential
    nval : number of valence (plus core) impurity orbitals (only ncore:nval orbs coupled to bath)
    ncore : number of core impurity orbitals
    nelectron : electron number per cell
    gmres_tol : GMRES/GCROTMK convergence criteria for solvers in production run (often 1e-3)
    wl0, wh0: (optional) real-axis frequency range [wl0+mu, wh0+mu] for bath discretization
                in DMFT self-consistent iterations (defualt: -0.4, 0.4)
    wl, wh : real-axis frequnecy range [wl, wh] for production run
    eta : spectral broadening for production run (often 0.1-0.4 eV)
    '''
    # DMFT self-consistent loop parameters
    gw_dmft = True
    opt_mu = False
    solver_type = 'cc'
    disc_type = 'opt'
    max_memory = 32000
    dmft_max_cycle = 10
    chkfile = 'DMFT_chk.h5'
    diag_only = False
    orb_fit = None

    delta = 0.1
    mu = 0.267
    nbath = 12
    nb_per_e = 8
    wl0 = -0.4
    wh0 = 0.4

    nval = 8
    ncore = 0
    nelectron = 8

    # DMFT production run parameters
    Ha2eV = 27.211386
    wl = 2./Ha2eV
    wh = 13./Ha2eV
    eta = 0.1/Ha2eV
    gmres_tol = 1e-3

    '''
    specific parameters for CAS treatment of impurity problem:
        cas : use CASCI or not (default: False)
        casno : natural orbital method for CASCI
                (choices: 'gw': GW@HF, 'cc': CCSD)
        composite : whether to use GW or CCSD Green's function as the low-level GF
                    for impurity problem; if False, use HF Green's function as low-level GF
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        ea_cut : float
            Energy cutoff for determining number of EA charged density matrices.
        ip_cut : float
            Energy cutoff for determining number of IP charged density matrices.
        ea_no : int
            Number of negatively charged density matrices included for making NOs.
        ip_no : int
            Number of positively charged density matrices included for making NOs.
        vno_only : bool
            Only construct virtual natural orbitals. Default is True.
    '''
    cas = False
    casno = 'gw'
    composite = False
    thresh = None
    nvir_act = None
    nocc_act = None
    ea_cut = None
    ip_cut = None
    ea_no = None
    ip_no = None
    vno_only = False

    # specific parameters for DMRG solvers (see fcdmft/solver/gfdmrg.py for detailed comments)
    gs_n_steps = 20
    gf_n_steps = 10
    gs_tol = 1E-12
    gf_tol = 1E-4
    gs_bond_dims = [400] * 5 + [800] * 5 + [1500] * 5 + [2000] * 5
    gs_noises = [1E-3] * 7 + [1E-4] * 5 + [1e-7] * 5 + [0]
    gf_bond_dims = [200] * 2 + [500] * 8
    gf_noises = [1E-4] * 1 + [1E-5] * 1 + [1E-7] * 1 + [0]
    dmrg_gmres_tol = 1E-9
    dmrg_verbose = 2
    reorder_method = 'gaopt'

    ### Finishing parameter settings ###

    # read hcore
    fn = 'hcore_JK_iao_k_dft.h5'
    feri = h5py.File(fn, 'r')
    hcore_k = np.asarray(feri['hcore'])
    feri.close()

    # read HF-JK matrix
    fn = 'hcore_JK_iao_k_hf.h5'
    feri = h5py.File(fn, 'r')
    JK_k = np.asarray(feri['JK'])
    feri.close()

    # read density matrix
    fn = 'DM_iao_k.h5'
    feri = h5py.File(fn, 'r')
    DM_k = np.asarray(feri['DM'])
    feri.close()

    # read 4-index ERI
    fn = 'eri_imp111_iao.h5'
    feri = h5py.File(fn, 'r')
    eri = np.asarray(feri['eri'])
    feri.close()
    eri_new = eri
    if eri_new.shape[0] == 3:
        eri_new = np.zeros_like(eri)
        eri_new[0] = eri[0]
        eri_new[1] = eri[2]
        eri_new[2] = eri[1]
    del eri

    # run self-consistent DMFT
    mydmft = gwdmft.DMFT(hcore_k, JK_k, DM_k, eri_new, nval, ncore, nbath,
                       nb_per_e, disc_type=disc_type, solver_type=solver_type)
    mydmft.gw_dmft = gw_dmft
    mydmft.verbose = 5
    mydmft.diis = True
    mydmft.gmres_tol = gmres_tol
    mydmft.max_memory = max_memory
    mydmft.chkfile = chkfile
    mydmft.diag_only = diag_only
    mydmft.orb_fit = orb_fit
    mydmft.dmft_max_cycle = dmft_max_cycle
    if solver_type == 'dmrg' or solver_type == 'dmrgsz':
        if not dmrg_:
            raise ImportError

    if cas:
        mydmft.cas = cas
        mydmft.casno = casno
        mydmft.composite = composite
        mydmft.thresh = thresh
        mydmft.nvir_act = nvir_act
        mydmft.nocc_act = nocc_act
        mydmft.ea_cut = ea_cut
        mydmft.ip_cut = ip_cut
        mydmft.ea_no = ea_no
        mydmft.ip_no = ip_no
        mydmft.vno_only = vno_only

    if solver_type == 'dmrg' or solver_type == 'dmrg_sz':
        mydmft.gs_n_steps = gs_n_steps
        mydmft.gf_n_steps = gf_n_steps
        mydmft.gs_tol = gs_tol
        mydmft.gf_tol = gf_tol
        mydmft.gs_bond_dims = gs_bond_dims
        mydmft.gs_noises = gs_noises
        mydmft.gf_bond_dims = gf_bond_dims
        mydmft.gf_noises = gf_noises
        mydmft.dmrg_gmres_tol = dmrg_gmres_tol
        mydmft.dmrg_verbose = dmrg_verbose
        mydmft.reorder_method = reorder_method

    mydmft.kernel(mu0=mu, wl=wl0, wh=wh0, delta=delta, occupancy=nelectron, opt_mu=opt_mu)
    occupancy = np.trace(mydmft.get_rdm_imp())
    if rank == 0:
        print ('At mu =', mydmft.mu, ', occupancy =', occupancy)

    mydmft.verbose = 5
    mydmft._scf.mol.verbose = 5
    nw = int(round((wh-wl)/eta))+1
    freqs = np.linspace(wl, wh, nw)

    # Get impurity DOS (production run)
    #ldos = mydmft.get_ldos_imp(freqs, eta)

    # Get lattice DOS (production run)
    ldos, ldos_gw = mydmft.get_ldos_latt(freqs, eta)
    spin = mydmft.spin

    filename = 'mu-%0.3f_n-%0.2f_%d-%d_eta-%.2f_d-%.2f_%s'%(
                mu,occupancy,nval,nbath,eta*Ha2eV,delta,solver_type)
    if rank == 0:
        write.write_dos(filename, freqs, ldos, occupancy=occupancy)

    if mydmft.gw_dmft:
        filename = 'mu-%0.3f_n-%0.2f_%d-%d_eta-%.2f_d-%.2f_gw'%(
                    mu,occupancy,nval,nbath,eta*Ha2eV,delta)
    else:
        filename = 'mu-%0.3f_n-%0.2f_%d-%d_eta-%.2f_d-%.2f_hf'%(
                    mu,occupancy,nval,nbath,eta*Ha2eV,delta)
    if rank == 0:
        write.write_dos(filename, freqs, ldos_gw, occupancy=occupancy)


if __name__ == '__main__':
    dmft_abinitio()
