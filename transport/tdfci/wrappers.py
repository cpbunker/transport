'''
Christian Bunker
M^2QM at UF
June 2021

siam_current.py

Use FCI exact diag to solve single impurity anderson model (siam)
Then use td FCI or td DMRG to time propagate

pyscf formalism:
- h1e_pq = (p|h|q) p,q spatial orbitals
- h2e_pqrs = (pq|h|rs) chemists notation, <pr|h|qs> physicists notation
- all direct_x solvers assume 4fold symmetry from sum_{pqrs} (don't need to do manually)
- 1/2 out front all 2e terms, so contributions are written as 1/2(2*actual ham term)
- hermicity: h_pqrs = h_qpsr can absorb factor of 1/2

pyscf fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr

'''

import numpy as np
import time

#################################################
#### get current data

def SiamData(nleads, nelecs, ndots, timestop, deltat, phys_params, 
spinstate = "", prefix = "dat/", namevar="Vg", verbose = 0) -> str:
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
    name of observables vs t data file
    '''
    from transport import tdfci, fci_mod
    from transport.fci_mod import ops
    from pyscf import fci

    # check inputs
    if(not isinstance(nleads, tuple) ): raise TypeError;
    if(not isinstance(nelecs, tuple) ): raise TypeError;
    if(not isinstance(ndots, int) ): raise TypeError;

    # set up the hamiltonian
    imp_i = [nleads[0]*2, nleads[0]*2 + 2*ndots - 1 ]; # imp sites start and end, inclusive
    norbs = 2*(nleads[0]+nleads[1]+ndots); # num spin orbs
    # nelecs left as tunable
    t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, B, theta = phys_params;

    # get 1 elec and 2 elec hamiltonian arrays for siam, dot model impurity
    if(verbose): print("1. Construct hamiltonian")
    eq_params = t_leads, 0.0, t_dots, 0.0, mu, V_gate, U, B, theta; # thyb, Vbias turned off, mag field in theta to prep spin
    h1e, g2e, input_str = ops.dot_hams(nleads, ndots, eq_params, spinstate, verbose = verbose);
        
    # get scf implementation siam by passing hamiltonian arrays
    if(verbose): print("2. FCI solution");
    mol, dotscf = fci_mod.arr_to_uhf(h1e, g2e, norbs, nelecs, verbose = verbose);

    # from scf instance, do FCI, get exact gd state of equilibrium system
    E_fci, v_fci = fci_mod.scf_FCI(mol, dotscf, verbose = verbose);
    if( verbose > 3): print("|initial> = ",v_fci);
    x=ops.spinflip([0],len(h1e))
    print(x)
    raise NotImplementedError;
    # prepare in nonequilibrium state by turning on t_hyb (hopping onto dot)
    if(verbose > 3 ): print("- Add nonequilibrium terms");
    neq_params = t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, 0.0, 0.0; # thyb, Vbias turned on, no mag field
    neq_h1e, neq_g2e, input_str_noneq = ops.dot_hams(nleads, ndots, neq_params, spinstate, verbose = verbose);
    
    # from fci gd state, do time propagation
    if(verbose): print("3. Time propagation")
    init, observables = tdfci.kernel(neq_h1e, neq_g2e, v_fci, mol, dotscf, timestop, deltat, verbose = verbose);

    return;

    # write results to external file
    if namevar == "Vg":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vg"+str(V_gate)+".npy";
    elif namevar == "U":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_U"+str(U)+".npy";
    elif namevar == "Vb":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vb"+str(V_bias)+".npy";
    elif namevar == "th":
        fname = prefix+"siam_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_th"+str(t_hyb)+".npy";
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
    
#################################################
#### exec code

if(__name__ == "__main__"):

    pass;


