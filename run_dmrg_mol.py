'''
Christian Bunker
M^2QM at UF
October 2023


'''

from transport import fci_mod

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
import pyblock3
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP

import time

# top level
np.set_printoptions(precision = 1, suppress = True);
verbose = 5;
myxvals = 15;
do_fci = True; # whether to do fci
atoms = ["N","N"];
mol_spin = 0;


######################################################################
#### main code
start = time.time();

# dmrg params
bdims = [100, 200, 200, 300];
noises = [1e-4, 1e-5, 1e-6, 0.0];

# iter over bond lengths
rvals = np.linspace(1.0,8.0,myxvals);
Evals_rhf = np.empty_like(rvals);
Evals_fci = np.empty_like(rvals);
Evals_dmrg = np.empty_like(rvals);
for rvali in range(len(rvals)):

    # set up molecule geometry
    coords = [[atoms[0], (0, 0, 0)]];
    coords.append([atoms[1], (0, 0, rvals[rvali])]);
    basis = "sto-3g"; # minimal
    X_mol = gto.M(atom=coords, basis=basis, unit="bohr", spin = mol_spin); # mol geometry

    # do Restricted Hartree Fock
    X_rhf = scf.RHF(X_mol).run(verbose=0);
    Evals_rhf[rvali] = X_rhf.energy_tot();

    # convert RHF results to spinless electron matrix elements
    h1e, g2e = fci_mod.rhf_to_arr(X_mol, X_rhf);
    fci_solver = fci.direct_nosym.FCI();
    if(do_fci):
        Evals_fci[rvali], _ = fci_solver.kernel(h1e, g2e, len(h1e), X_mol.nelec, ecore = X_rhf.energy_nuc());
    else:
        Evals_fci[rvali] = np.nan;
    
    #### do DMRG ####

    # convert electron integrals to MPO and run DMRG algorithm
    dmrg_ham, dmrg_mpo, dmrg_mps = fci_mod.arr_to_mpo(h1e, g2e, X_mol.nelec, bdims[0],
                                  energy_nuc = X_rhf.energy_nuc(), cutoff=1e-6, verbose=verbose);
    dmrg_mpe = fci_mod.mpo_to_mpe(dmrg_mpo, dmrg_mps);
    dmrg_mpe_output = dmrg_mpe.dmrg(bdims=bdims, noises=noises, iprint=0);
    Evals_dmrg[rvali] = dmrg_mpe_output.energies[-1];
    
stop = time.time();

# plot
reference = np.zeros_like(Evals_fci); # to look at deviations rather than abs energies
fig, ax = plt.subplots();
ax.plot(rvals,Evals_rhf - reference,label="RHF - ref");
ax.plot(rvals,Evals_fci - reference,label="FCI - ref");
ax.scatter(rvals,Evals_dmrg - reference,label="DMRG - ref", marker='o', edgecolors='tab:red', facecolors='none');  
ax.axvline(rvals[np.argmin(Evals_fci)], color="black", linestyle="dashed");

# format
ax.set_xlabel("$R$ (bohr)");
ax.set_ylabel("$E$ (hartree)");
ax.set_title(atoms[0]+"$-$"+atoms[1]+" bond (compute time = {:.1f} minutes)".format((stop-start)/60));
plt.legend();
plt.tight_layout();
fname = "figs/dmrg/"+atoms[0]+"-"+atoms[1]+".pdf";
if True: ("Saving plot to "+fname); plt.savefig(fname);
else: plt.show();
