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

import ops

import time
import numpy as np

#################################################
#### get current data

def DotData(n_leads, nelecs, timestop, deltat, phys_params=None, prefix = "dat/", namevar="Vg", verbose = 0):
    '''
    Walks thru all the steps for plotting current thru a SIAM, using FCI for equil state
    and td-FCI for nonequil dynamics. Impurity is a quantum dot w/ gate voltage and hubbard U
    - construct the eq hamiltonian, 1e and 2e parts, as np arrays
    - encode hamiltonians in an scf.UHF inst
    - do FCI on scf.UHF to get exact gd state
    - turn on thyb to intro nonequilibrium (current will flow)
    - use ruojing's code (td_fci module) to do time propagation

    Args:
    nleads, tuple of ints of left lead sites, right lead sites
    nelecs, tuple of num up e's, 0 due to ASU formalism
    timestop, float, how long to run for
    deltat, float, time step increment
    physical params, tuple of t, thyb, Vbias, mu, Vgate, U, B, theta, phi
    	if None, gives defaults vals for all (see below)
    prefix: assigns prefix (eg folder) to default output file name

    Returns:
    none, but outputs t, observable data to /dat/DotData/ folder
    '''
    
    # imports here so dmrg can be run even if pyscf not on machine
    from pyscf import fci
    import fci_mod
    import td_fci

    # check inputs
    assert( isinstance(n_leads, tuple) );
    assert( isinstance(nelecs, tuple) );
    assert( isinstance(timestop, float) );
    assert( isinstance(deltat, float) );
    assert( isinstance(phys_params, tuple) or phys_params == None);

    # set up the hamiltonian
    n_imp_sites = 1 # dot
    imp_i = [n_leads[0]*2, n_leads[0]*2 + 2*n_imp_sites - 1 ]; # imp sites, inclusive
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    # nelecs left as tunable

    # physical params, should always be floats
    if( phys_params == None): # defaults
        V_leads = 1.0; # hopping
        V_imp_leads = 0.4; # hopping t dot, allows current flow
        V_bias = -0.005; # induces current flow
        mu = 0.0;
        V_gate = -0.5;
        U = 1.0; # hubbard repulsion
        B = 0.0; # magnetic field strength
        theta = 0.0;
        phi = 0.0;
        thyb_eq = 0.0; # small but nonzero val is more robust
    else: # customized
        V_leads, V_imp_leads, V_bias, mu, V_gate, U, B, theta, phi = phys_params;
        thyb_eq = 0.0; # small but nonzero val is more robust

    # get 1 elec and 2 elec hamiltonian arrays for siam, dot model impurity
    if(verbose): print("1. Construct hamiltonian")
    eq_params = V_leads, thyb_eq, 0.0, mu, V_gate, U, B, theta, phi; # dot hopping turned off, but nonzero = more robust
    h1e, g2e, input_str = ops.dot_hams(n_leads, n_imp_sites, nelecs, eq_params, verbose = verbose);
        
    # get scf implementation siam by passing hamiltonian arrays
    if(verbose): print("2. FCI solution");
    mol, dotscf = fci_mod.arr_to_scf(h1e, g2e, norbs, nelecs, verbose = verbose);
    
    # from scf instance, do FCI, get exact gd state of equilibrium system
    E_fci, v_fci = fci_mod.scf_FCI(mol, dotscf, verbose = verbose);

    # remove spin prep terms
    h1e += ops.h_B(-B, theta, phi, imp_i, norbs, verbose = verbose);
    
    # prepare in nonequilibrium state by turning on t_hyb (hopping onto dot)
    if(verbose > 2 ): print("- Add nonequilibrium terms");
    neq_params = 0.0, V_imp_leads, V_bias, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    neq_h1e, dummy, input_str_noneq = ops.dot_hams(n_leads, n_imp_sites, nelecs, neq_params, verbose = verbose);
    h1e += neq_h1e; # updated to include thyb and Vbias

    # from fci gd state, do time propagation
    if(verbose): print("3. Time propagation")
    if verbose > 4: print(h1e);
    init_str, observables = td_fci.TimeProp(h1e, g2e, v_fci, mol, dotscf, timestop, deltat, imp_i, V_imp_leads, verbose = verbose);
    
    # write results to external file
    if namevar == "Vg":
        fname = prefix+"fci_"+str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_Vg"+str(V_gate)+".npy";
    elif namevar == "U":
        fname = prefix+"fci_"+str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_U"+str(U)+".npy";
    else: assert(False);
    hstring = time.asctime();
    hstring += "\nASU formalism, t_hyb noneq. term"
    hstring += "\nEquilibrium"+input_str; # write input vals to txt
    hstring += "\nNonequlibrium"+input_str_noneq;
    hstring += init_str; # write initial state to txt
    print(fname[:-4]+".txt");
    np.savetxt(fname[:-4]+".txt", np.array([1,2,3]), header = hstring); # saves info to txt
    np.save(fname, observables);
    print("4. Saved data to "+fname);
    
    return fname; # end dot data


def DotDataDmrg(n_leads, nelecs, timestop, deltat, bond_dims = [100, 200, 300, 400], noises = [1e-3,1e-4,1e-5,0], phys_params=None, prefix = "dat/", verbose = 0):
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
    timestop, float, how long to run for
    deltat, float, time step increment
    bond_dims, list of increasing bond dim over dmrg sweeps, optional
    noises, list of decreasing noises over dmrg sweeps, optional
    physical params, tuple of t, thyb, Vbias, mu, Vgate, U, B, theta, phi
    	if None, gives defaults vals for all (see below)
    prefix: assigns prefix (eg folder) to default output file name

    Returns:
    none, but outputs t, observable data to /dat/DotDataDMRG/ folder
    '''
    
    from pyblock3 import fcidump, hamiltonian
    from pyblock3.algebra.mpe import MPE
    import td_dmrg

    # check inputs
    assert( isinstance(n_leads, tuple) );
    assert( isinstance(nelecs, tuple) );
    assert( isinstance(timestop, float) );
    assert( isinstance(deltat, float) );
    assert( isinstance(phys_params, tuple) or phys_params == None);
    assert( bond_dims[0] <= bond_dims[-1]); # checks bdims has increasing behavior and is list
    assert( noises[0] >= noises[-1] ); # checks noises has decreasing behavior and is list

    # set up the hamiltonian
    n_imp_sites = 1 # dot
    imp_i = [n_leads[0]*2, n_leads[0]*2 + 2*n_imp_sites - 1 ]; # imp sites, inclusive
    norbs = 2*(n_leads[0]+n_leads[1]+n_imp_sites); # num spin orbs
    # nelecs left as tunable

    # physical params, should always be floats
    if( phys_params == None): # defaults
        V_leads = 1.0; # hopping
        V_imp_leads = 0.4; # hopping t dot, allows current flow
        V_bias = -0.005; # induces current flow
        mu = 0.0;
        V_gate = -0.5;
        U = 1.0; # hubbard repulsion
        B = 0.0; # magnetic field strength
        theta = 0.0;
        phi = 0.0;
        thyb_eq = 0.0; # small but nonzero val is more robust
    else: # customized
        V_leads, V_imp_leads, V_bias, mu, V_gate, U, B, theta, phi = phys_params;
        thyb_eq = 0.0; # small but nonzero val is more robust


    # get h1e and h2e for siam, h_imp = h_dot
    if(verbose): print("1. Construct hamiltonian")
    ham_params = V_leads, thyb_eq, 0.0, mu, V_gate, U, B, theta, phi; # thyb, Vbias turned off
    h1e, g2e, input_str = ops.dot_hams(n_leads, n_imp_sites, nelecs, ham_params, verbose = verbose);

    # store physics in fci dump object
    hdump = fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]); # twos = 2S tells spin    

    # instead of np array, dmrg wants ham as a matrix product operator (MPO)
    h_obj = hamiltonian.Hamiltonian(hdump,flat=True);
    h_mpo = h_obj.build_qc_mpo();
    h_mpo, _ = h_mpo.compress(cutoff=1E-15); # compressing saves memory
    if verbose: print("- Built H as compressed MPO: ", h_mpo.show_bond_dims() );

    # initial ansatz for wf, in matrix product state (MPS) form
    psi_mps = h_obj.build_mps(bond_dims[0]);
    E_mps_init = td_dmrg.compute_obs(h_mpo, psi_mps);
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
    ham_params_neq = V_leads, V_imp_leads, V_bias, mu, V_gate, U, 0.0, 0.0, 0.0; # dot hopping on now, B field off
    h1e_neq, g2e_neq, input_str_neq = ops.dot_hams(n_leads, n_imp_sites, nelecs, ham_params_neq, verbose = verbose);
    hdump_neq = fcidump.FCIDUMP(h1e=h1e_neq,g2e=g2e_neq,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]); 
    h_obj_neq = hamiltonian.Hamiltonian(hdump_neq,True);
    h_mpo_neq = h_obj_neq.build_qc_mpo(); # got mpo
    h_mpo_neq, _ = h_mpo_neq.compress(cutoff=1E-15); # compression saves memory

    # time propagate the noneq state
    # td dmrg uses highest bond dim
    init_str, observables = td_dmrg.kernel(h_mpo_neq, h_obj_neq, psi_mps, timestop, deltat, imp_i, V_imp_leads, [bond_dims[-1]], verbose = verbose);

    # write results to external file
    fname = prefix+"dmrg_"+str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_Vg"+str(V_gate)+".npy";
    hstring = time.asctime(); # header has lots of important info: phys params, bond dims, etc
    hstring += "\nASU formalism, t_hyb noneq. term, td-DMRG,\nbdims = "+str(bond_dims)+"\n noises = "+str(noises); 
    hstring += "\nEquilibrium"+input_str; # write input vals to txt
    hstring += "\nNonequlibrium"+input_str_neq;
    hstring += init_str; # write initial state to txt
    print(fname[:-4]+".txt");
    np.savetxt(fname[:-4]+".txt", np.array([1,2,3]), header = hstring); # saves info to txt
    np.save(fname, observables);
    print("4. Saved data to "+fname);
    
    return fname; # end dot data dmrg


#################################################
#### manipulate current data

def Fourier(signal, samplerate, angular = False, dominant = 0, shorten = False):
    '''
    Uses the discrete fourier transform to find the frequency composition of the signal

    Args:
    - signal, 1d np array of info vs time
    - samplerate, num data pts per second. Necessary for freq to make any sense

    Returns: tuple of
    1d array of |FT|^2, 1d array of freqs
    '''

    # get vals
    nx = len(signal);
    dx = 1/samplerate;

    # perform fourier transform
    FT = np.fft.fft(signal)
    nu = np.fft.fftfreq(nx, dx); # gets accompanying freqs

    # manipulate data
    FT = FT/nx; # norm missing in np.fft.fft
    FT, nu = np.fft.fftshift(FT), np.fft.fftshift(nu); # puts zero freq at center
    FT = np.absolute(FT)*np.absolute(FT); # get norm squared
    if np.isrealobj(signal): # real signals have only positive freqs
        # truncate FT, nu to nu > 0
        FT, nu = FT[int(nx/2):], nu[int(nx/2):]
        
    # show freq resolution # dnu = 1/(tf - ti)
    #print("dnu = ", nu[1] - nu[0] );

    # if asked, convert to omega
    if angular: nu = nu*2*np.pi;

    # if asked, get and return dominant frequencies
    if dominant:

        # get as many of the highest freqs as asked for
        nu_maxvals = np.zeros(dominant);
        for i in range(dominant):

            # get current largest FT val
            imax = np.argmax(FT); # where dominant freq occurs
            nu_maxvals[i] = nu[imax]; # place dominant freq in array

            # get current max out of FT
            FT = np.delete(FT, imax);
            nu = np.delete(nu, imax);

        return nu_maxvals; # end here instead

    return  FT, nu;

    
#################################################
#### exec code

if(__name__ == "__main__"):

    pass;


