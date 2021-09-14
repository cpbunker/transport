'''
Christian Bunker
M^2QM at UF
September 2021

Itinerant electron scattering from a single dot
'''

import ops
import fci_mod

import numpy as np

import sys

##################################################################################
#### set up physics of dot

# top level
verbose = 3;
nleads = (2,2);
nelecs = (2,0); # one electron on dot and one itinerant
ndots = 1; 
spinstate = "ab";
coef_cutoff = 0.5

# phys params, must be floats
tl = 1.0;
th = 1.0; 
td = 1.0; 
Vb = 0.0;
mu = 0.0;
Vg = -10.0;
U = abs(4*Vg);
B = 1000;
theta = 0.0;

# investigate 1 dot scattering

# get eq initial state
eq_params = tl, 0.0, td, 0.0, mu, Vg, U, B, 0.0;
h1e, g2e, input_str = ops.dot_hams(nleads, nelecs, ndots, eq_params, spinstate, verbose = verbose);
psi_i, observables = fci_mod.arr_to_initstate(h1e, g2e, nleads, nelecs, ndots, verbose = verbose);
print("\nInitial state:-\n-Occ = "+str(observables[6:6+sum(nleads)+ndots].T)+"\n-Sz = "+str(observables[6+sum(nleads)+ndots:-1].T));
print("\n- Concur = "+str(observables[-1]) );

# get neq eigstates
neq_params = tl, th, td, Vb, mu, Vg, U, 0.0, 0.0; # thyb, Vbias turned on, no mag field
neq_h1e, neq_g2e, input_str_noneq = ops.dot_hams(nleads, nelecs, ndots, neq_params, "", verbose = verbose);
E_neq, v_neq = fci_mod.arr_to_eigen(neq_h1e, neq_g2e, nelecs, verbose = verbose);

# get coefs of initial state in neq eigenstate basis
coefs = np.zeros_like(E_neq)
for vi in range(len(v_neq) ):
    coefs[vi] = np.dot(psi_i.T, v_neq[vi]);

# look at weighted eigenstates
print("\nEigenstates");
for ci in range(len(coefs)):
    if(abs(coefs[ci]) > coef_cutoff):
        print("- |E = "+str(E_neq[ci])+"> : \n\t Occ = "
        
    








