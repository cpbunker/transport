'''
Christian Bunker
M^2QM at UF
October 2022

Examining time-dep perturbation theory using td-FCI tools

Wrappers and util functions here
'''

from transport import fci_mod, ops

import numpy as np

def initial_state(samp_energies, tip_energies, verbose = 0):
    '''
    Construct the initial state of the sample + stm tip system,
    which has the following features:
    - sample consists of orthonormal spinless states, lowest energy of
      which is occuppied by a single spinless fermion
    - tip consists of nondegenerate orthonormal spinless states, none
      of which are occupied, all of which are orthogonal to tip states
    '''
    if( not isinstance(samp_energies, np.ndarray)): raise TypeError;
    if( not isinstance(samp_energies, np.ndarray)): raise TypeError;

    # unpack
    N_samp = len(samp_energies);
    N_tip = len(tip_energies);
    Norbs = N_samp+N_tip; # total num spin orbs
    samp_energies = np.sort(samp_energies);
    tip_energies = np.sort(tip_energies);

    # lower the lowest sample state to make sure it is filled
    fill_shift = -1000
    samp_energies[0] += fill_shift
    assert samp_energies[0] < tip_energies[0];

    # 2nd qu'd matrices
    h1e = np.diagflat(np.append(samp_energies, tip_energies));
    h1e = h1e.astype(float); # change dtype to float
    g2e = np.zeros((Norbs, Norbs, Norbs, Norbs), dtype = float);

    # scf implementation
    mol_inst, scf_inst = fci_mod.arr_to_scf(h1e, g2e, Norbs, (1,0), verbose = verbose);

    # from scf instance, do FCI, get exact gd state of equilibrium system
    E_fci, v_fci = fci_mod.scf_FCI(mol_inst, scf_inst, verbose = verbose);
    if( verbose > 0): print("|initial> = ",v_fci);

    # undo fill shift
    h1e[0,0] += -fill_shift;
    return h1e, g2e, v_fci, mol_inst, scf_inst;
    



