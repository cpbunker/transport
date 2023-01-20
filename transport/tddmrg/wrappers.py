'''
Christian Bunker
M^2QM at UF
June 2021

tddmrg/wrappers.py

use dmrg for time evol of model ham systems
- single impurity anderson model

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

def siam_data(nleads, nelecs, ndots, timestop, deltat, phys_params, bond_dims, noises, 
spinstate = "", prefix = "data/", namevar = "Vg", verbose = 0) -> str:
    '''
    Walks thru all the steps for plotting current thru a SIAM, using DMRG for equil state
    and td-DMRG for nonequilibirum dynamics. Impurity is a quantum dot w/ gate voltage, hubbard U
    - construct the eq hamiltonian, 1e and 2e parts, as np arrays
    - store eq ham in FCIDUMP object which allows us to access it w/ pyblock3
    - from FCIDUMP create a pyblock3.hamiltonian.Hamiltonian object\
    - use this to build a Matrix Product Operator (MPO) and initial guess MPS
    - use these to construct Matrix Product Expectation (MPE) which calls dmrg() to get gd state
    - construct noneq ham (thyb = 1e-5 -> 0.4 default) and repeat to get MPE (in td_dmrg module)
    - then MPE.tddmrg() method updates wf in time and we can get observables (in td_dmrg module)
    	NB tddmrg uses max bonddim of dmrg as of now

    Args:
    nleads, tuple of ints of left lead sites, right lead sites
    nelecs, tuple of num up e's, 0 due to ASU formalism
    ndots, int, number of dots in impurity
    timestop, float, how long to run for
    deltat, float, time step increment
    bond_dims, list of increasing bond dim over dmrg sweeps, optional
    noises, list of decreasing noises over dmrg sweeps, optional
    physical params, tuple of t, thyb, Vbias, mu, Vgate, U, B, theta, phi
    	if None, gives defaults vals for all (see below)
    prefix: assigns prefix (eg folder) to default output file name

    Returns:
    name of observables vs t data file
    '''
    from transport import tddmrg, fci_mod
    from transport.fci_mod import ops, ops_dmrg
    from pyblock3 import fcidump, hamiltonian
    from pyblock3.algebra.mpe import MPE

    # check inputs
    if(not isinstance(nleads, tuple) ): raise TypeError;
    if(not isinstance(nelecs, tuple) ): raise TypeError;
    if(not isinstance(ndots, int) ): raise TypeError;
    if(not isinstance(timestop, float) ): raise TypeError;
    if(not isinstance(deltat, float) ): raise TypeError;
    if(not isinstance(phys_params, tuple) or phys_params == None): raise TypeError;
    if(not bond_dims[0] <= bond_dims[-1]): raise ValueError; # bdims must have increasing behavior 
    if(not noises[0] >= noises[-1] ): raise ValueError; # noises must have decreasing behavior 

    # set up the hamiltonian
    imp_i = [nleads[0]*2, nleads[0]*2 + 2*ndots - 1 ]; # imp sites, inclusive
    norbs = 2*(nleads[0]+nleads[1]+ndots); # num spin orbs
    # nelecs left as tunable
    t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, B, theta = phys_params;


    # get h1e and h2e for siam, h_imp = h_dot
    if(verbose): print("1. Construct hamiltonian")
    ham_params = t_leads, 1e-5, t_dots, 0.0, mu, V_gate, U, B, theta; # thyb, Vbias turned off, mag field in theta direction
    h1e, g2e, input_str = fci_mod.ops.dot_hams(nleads, ndots, ham_params, spinstate, verbose = verbose);

    # store physics in fci dump object
    hdump = fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]); # twos = 2S tells spin    

    # instead of np array, dmrg wants ham as a matrix product operator (MPO)
    h_obj = hamiltonian.Hamiltonian(hdump,flat=True);
    h_mpo = h_obj.build_qc_mpo();
    h_mpo, _ = h_mpo.compress(cutoff=1E-15); # compressing saves memory
    if verbose: print("- Built H as compressed MPO: ", h_mpo.show_bond_dims() );

    # initial ansatz for wf, in matrix product state (MPS) form
    psi_mps = h_obj.build_mps(bond_dims[0]);
    E_mps_init = tddmrg.compute_obs(h_mpo, psi_mps);
    if verbose: print("- Initial gd energy = ", E_mps_init);

    # ground-state DMRG
    # runs thru an MPE (matrix product expectation) class built from mpo, mps
    if(verbose): print("2. DMRG solution");
    MPE_obj = MPE(psi_mps, h_mpo, psi_mps);

    # solve system by doing dmrg sweeps
    # MPE.dmrg method takes list of bond dimensions, noises, threads defaults to 1e-7
    # can also control verbosity (iprint) sweeps (n_sweeps), conv tol (tol)
    # noises[0] = 1e-3 and tol = 1e-8 work best from trial and error
    dmrg_obj = MPE_obj.dmrg(bdims=bond_dims, noises = noises, tol = 1e-8, iprint=0);
    E_dmrg = dmrg_obj.energies;
    if verbose: print("- Final gd energy = ", E_dmrg[-1]);

    # nonequil hamiltonian (as MPO)
    if(verbose > 2 ): print("- Add nonequilibrium terms");
    ham_params_neq = t_leads, t_hyb, t_dots, V_bias, mu, V_gate, U, 0.0, 0.0; # thyb and Vbias on, no zeeman splitting
    h1e_neq, g2e_neq, input_str_neq = fci_mod.ops.dot_hams(nleads, ndots, ham_params_neq, "", verbose = verbose);
    hdump_neq = fcidump.FCIDUMP(h1e=h1e_neq,g2e=g2e_neq,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]); 
    h_obj_neq = hamiltonian.Hamiltonian(hdump_neq, flat=True);
    h_mpo_neq = h_obj_neq.build_qc_mpo(); # got mpo
    h_mpo_neq, _ = h_mpo_neq.compress(cutoff=1E-15); # compression saves memory

    try:
        from pyblock3.algebra import flat
        assert( isinstance(h_obj_neq.FT, flat.FlatFermionTensor) );
        assert( isinstance(h_mpo_neq, flat.FlatFermionTensor) );
    except:
        pass;

    # time propagate the noneq state
    # td dmrg uses highest bond dim
    if(verbose): print("3. Time propagation");
    init, observables = tddmrg.kernel(h_mpo_neq, h_obj_neq, psi_mps, timestop, deltat, imp_i, [bond_dims[-1]], verbose = verbose);

    # write results to external file
    if namevar == "Vg":
        fname = prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vg"+str(V_gate)+".npy";
    elif namevar == "U":
        fname = prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_U"+str(U)+".npy";
    elif namevar == "Vb":
        fname = prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_Vb"+str(V_bias)+".npy";
    elif namevar == "th":
        fname = prefix+"fci_"+str(nleads[0])+"_"+str(ndots)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_th"+str(t_hyb)+".npy";
    else: assert(False); # invalid option
    hstring = time.asctime(); # header has lots of important info: phys params, bond dims, etc
    hstring += "\ntf = "+str(timestop)+"\ndt = "+str(deltat);
    hstring += "\nASU formalism, t_hyb noneq. term, td-DMRG,\nbdims = "+str(bond_dims)+"\n noises = "+str(noises); 
    hstring += "\nEquilibrium"+input_str; # write input vals to txt
    hstring += "\nNonequlibrium"+input_str_neq;
    np.savetxt(fname[:-4]+".txt", init, header = hstring); # saves info to txt
    np.save(fname, observables);
    if(verbose): print("4. Saved data to "+fname);
    
    return fname; # end dot data dmrg

def kernel(mpo, h_obj, mps, tf, dt, i_dot, bdims, verbose = 0):
    '''
    Drive time prop for dmrg
    Use real time time dependent dmrg method outlined here:
    https://pyblock3.readthedocs.io/en/latest/Documentation/rttddmrg.html

    Args:
    -mpo, a matrix product operator form of the hamiltonian
    -h_obj, a pyblock3.hamiltonian.Hamiltonian form of the hamiltonian
    -mps, a matrix product state
    -tf, float, the time to end the time evolution at
    -dt, float, the time step of the time evolution
    -i_dot, int, the site index of the impurity
    =bdims, list of ints, bond dimension of the DMRG solver
    '''

    # check inputs
    assert(isinstance(bdims, list));

    # unpack
    norbs = mps.n_sites
    N = int(tf/dt+1e-6); # num steps
    n_generic_obs = 7; # 7 are time, E, 4 J's, concurrence
    sites = np.array(range(norbs)).reshape(int(norbs/2),2); # all indices, sep'd by site
    mpe_obj = MPE(mps, mpo, mps); # init mpe obj
    # return vals
    observables = np.zeros((N+1, n_generic_obs+4*len(sites) ), dtype = complex ); # generic plus occ, Sx, Sy, Sz per site

    # mpos for observables
    obs_mpos = [];
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jup(i_dot, norbs)[0] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jup(i_dot, norbs)[1] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jdown(i_dot, norbs)[0] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.Jdown(i_dot, norbs)[1] ) );
    obs_mpos.append(h_obj.build_mpo(ops_dmrg.spinflip(i_dot, norbs) ) );
    for site in sites: # site specific observables
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.occ(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sx(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sy(site, norbs) ) );
        obs_mpos.append( h_obj.build_mpo(ops_dmrg.Sz(site, norbs) ) );

    # loop over time
    for i in range(N+1):

        if(verbose>2): print("    time: ", i*dt);

        # mpe.tddmrg method does time prop, outputs energies but also modifies mpe obj
        energies = mpe_obj.tddmrg(bdims,-np.complex(0,dt), n_sweeps = 1, iprint=0, cutoff = 0).energies
        mpst = mpe_obj.ket; # update wf

        # compute observables
        observables[i,0] = i*dt; # time
        observables[i,1] = energies[-1]; # energy
        for mi in range(len(obs_mpos)): # iter over mpos
            print(mi);
            observables[i,mi+2] = compute_obs(obs_mpos[mi], mpst);
        
        # before any time stepping, get initial state
        if(i==0):
            # get site specific observables at t=0 in array where rows are sites
            initobs = np.real(np.reshape(observables[i,n_generic_obs:],(len(sites), 4) ) );

    # return tuple of observables at t=0 and observables as arrays vs time
    return initobs, observables;
    
#################################################
#### exec code

if(__name__ == "__main__"):

    pass;


