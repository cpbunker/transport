'''
Christian Bunker
M^2QM at UF
June 2021

fci_mod.py

Helpful funcs for using pySCF, pyblock3
Imports are within functions since some machines can run only pyblock3 or pyscf

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr
'''

import ops

import numpy as np
import functools


##########################################################################################################
#### conversions


def arr_to_scf(h1e, g2e, norbs, nelecs, verbose = 0):
    '''
    Converts hamiltonians in array form to scf object
    
    Args:
    - h1e, 2d np array, 1e part of siam ham
    - g2e, 2d np array, 2e part of siam ham
    - norbs, int, total num spin orbs
    - nelecs, tuple of number es, 0 due to All spin up formalism
    
    Returns: tuple of
    mol, gto.mol object which holds some physical params
    scf inst, holds physics: h1e, h2e, mo coeffs etc
    '''

    from pyscf import gto, scf
    
    # initial guess density matrices
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    
    # put everything into UHF scf object
    if(verbose):
        print("\nUHF energy calculation")
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
    scf_inst = scf.UHF(mol)
    scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
    scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
    scf_inst._eri = g2e # put h2e into scf solver
    if( nelecs == (1,0) ):
        scf_inst.kernel(); # no dm
    else:
        scf_inst.kernel(dm0=(Pa, Pa)); # prints HF gd state but this number is meaningless
                                   # what matter is h1e, h2e are now encoded in this scf instance

    return mol, scf_inst;


def scf_to_arr(mol, scf_obj):
    '''
    Converts physics of an atomic/molecular system, as contained in an scf inst
    ie produced by passing molecular geometry object mol to
    - scf.RHF(mol) restricted hartree fock
    - scf.UHF(mol) unrestricted hartree fock
    - scf.RKS(mol).run() restricted Kohn sham
    - etc
    to ab initio hamiltonian arrays h1e and g2e
    '''

    from pyscf import ao2mo

    # unpack scf object
    hcore = scf_obj.get_hcore();
    coeffs = scf_obj.mo_coeff;
    norbs = np.shape(coeffs)[0];

    # convert to h1e and h2e array reps in molecular orb basis
    h1e = np.dot(coeffs.T, hcore @ coeffs);
    g2e = ao2mo.restore(1, ao2mo.kernel(mol, coeffs), norbs);

    return h1e, g2e;


def fd_to_mpe(fd, bdim_i, cutoff = 1e-9):
    '''
    Convert physics contained in an FCIDUMP object or file to a Matrix
    Product Expectation (MPE) for doing DMRG

    Args:
    fd, a pyblock3.fcidump.FCIDUMP object, or filename of such an object
    bdim_i, int, initial bond dimension of the MPE

    Returns:
    MPE object
    '''

    from pyblock3 import fcidump, hamiltonian, algebra
    from pyblock3.algebra.mpe import MPE

    # convert fcidump to hamiltonian obj
    if( isinstance(fd, string) ): # fd is file, must be read
        hobj = hamiltonian.Hamiltonian(FCIDUMP().read(fd), flat=True);
    else: # fcidump obj already
        h_obj = hamiltonian.Hamiltonian(fd, flat=True);

    # Matrix Product Operator
    h_mpo = h_obj.build_qc_mpo();
    h_mpo, _ = h_mpo.compress(cutoff = cutoff);
    psi_mps = h_obj.build_mps(bdim_i);

    # MPE
    return MPE(psi_mps, h_mpo, psi_mps);
    
    
def mol_model(nleads, nsites, norbs, nelecs, physical_params,verbose = 0):
    # WILL NEED OVERHAUL 

    # checks
    assert norbs == 2*(nsites + nleads[0]+nleads[1]);
    assert nelecs[1] == 0;
    assert nelecs[0] <= norbs;

    # unpack inputs
    V_leads, V_imp_leads, V_bias, mu, mol_params = physical_params;
    D, E, alpha, U = mol_params;

    if(verbose): # print inputs
        try:
            print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias,"\n- mu = ",mu,"\n- D = ",D,"\n- E = ",E, "\n- alpha = ",alpha, "\n- U = ",U, "\n- E/U = ",E/U,"\n- alpha/D = ",alpha/D,"\n- alpha/(E^2/U) = ",alpha*U/(E*E),"\n- alpha^2/(E^2/U) = ",alpha*alpha**U/(E*E) );
        except: 
            print("\nInputs:\n- Num. leads = ",nleads,"\n- Num. impurity sites = ",nsites,"\n- nelecs = ",nelecs,"\n- V_leads = ",V_leads,"\n- V_imp_leads = ",V_imp_leads,"\n- V_bias = ",V_bias)
    #### make full system ham from inputs

    # make, combine all 1e hamiltonians
    hl = ops.h_leads(V_leads, nleads); # leads only
    hb = ops.h_chem(mu, nleads);   # chem potential on leads
    hdl = ops.h_imp_leads(V_imp_leads, nsites); # leads talk to dot
    hd = molecule_5level.h1e(nsites*2,D,E,alpha); # Silas' model
    h1e = ops.stitch_h1e(hd, hdl, hl, hb, nleads, verbose = verbose); # syntax is imp, imp-leads, leads, bias/chem potential
    if(verbose > 2):
        print("\n- Full one electron hamiltonian = \n",h1e);
        
    # 2e hamiltonian only comes from impurity
    if(verbose > 2):
        print("\n- Nonzero h2e elements = ");
    hd2e = molecule_5level.h2e(2*nsites, U);
    h2e = ops.stitch_h2e(hd2e, nleads, verbose = verbose);

    #### encode physics of dot model in an SCF obj

    # initial guess density matrices
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)

    # put everything into UHF scf object
    if(verbose):
        print("\nUHF energy calculation")
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
    scf_inst = scf.UHF(mol)
    scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
    scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
    scf_inst._eri = h2e # put h2e into scf solver
    scf_inst.kernel(dm0=(Pa, Pa)); # prints HF gd state but this number is meaningless
                                   # what matter is h1e, h2e are now encoded in this scf instance
        
    return h1e, h2e, mol, scf_inst;


def direct_FCI(h1e, h2e, norbs, nelecs, nroots = 1, verbose = 0):
    '''
    solve gd state with direct FCI
    '''

    from pyscf import fci
    
    cisolver = fci.direct_spin1.FCI();
    E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
    if(verbose):
        print("\nDirect FCI energies, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ",E_fci);

    return E_fci, v_fci;


def scf_FCI(mol, scf_inst, nroots = 1, verbose = 0):
    '''
    '''

    from pyscf import fci, ao2mo

    # init ci solver with ham from molecule inst
    cisolver = fci.direct_uhf.FCISolver(mol);

    # get unpack from scf inst
    h1e = scf_inst.get_hcore(mol);
    norbs = np.shape(h1e)[0];
    nelecs = (mol.nelectron,0);

    # slater determinant coefficients
    mo_a = scf_inst.mo_coeff[0]
    mo_b = scf_inst.mo_coeff[1]
   
    # since we are in UHF formalism, need to split all hams by alpha, beta
    # but since everything is spin blind, all beta matrices are zeros
    h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
    h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
    h2e_aa = ao2mo.incore.general(scf_inst._eri, (mo_a,)*4, compact=False)
    h2e_aa = h2e_aa.reshape(norbs,norbs,norbs,norbs)
    h2e_ab = ao2mo.incore.general(scf_inst._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    h2e_ab = h2e_ab.reshape(norbs,norbs,norbs,norbs)
    h2e_bb = ao2mo.incore.general(scf_inst._eri, (mo_b,)*4, compact=False)
    h2e_bb = h2e_bb.reshape(norbs,norbs,norbs,norbs)
    h1e_tup = (h1e_a, h1e_b)
    h2e_tup = (h2e_aa, h2e_ab, h2e_bb)
    
    # run kernel to get exact energy
    E_fci, v_fci = cisolver.kernel(h1e_tup, h2e_tup, norbs, nelecs, nroots = nroots)
    if(verbose):
        print("\nFCI from UHF, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ", E_fci);

    return E_fci, v_fci;


##########################################################################################################
#### exec code

if __name__ == "__main__":

    pass;
