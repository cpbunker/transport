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

import plot
import ops
import fci_mod
import td_fci
import td_dmrg

import time
import numpy as np
import matplotlib.pyplot as plt
from pyscf import fci
from pyblock3 import fcidump, hamiltonian
from pyblock3.algebra.mpe import MPE

#################################################
#### get current data

def DotData(n_leads, nelecs, timestop, deltat, phys_params=None, Rlead_pol = 0, prefix = "", verbose = 0):
    '''
    More flexible version of siam.py DotCurrentWrapper with inputs allowing tuning of nelecs, mu, Vgate

    Walks thru all the steps for plotting current thru a SIAM. Impurity is a quantum dot
    - construct the biasless hamiltonian, 1e and 2e parts
    - encode hamiltonians in an scf.UHF inst
    - do FCI on scf.UHF to get exact gd state
    - turn on bias to induce current
    - use ruojing's code to do time propagation

    Args:
    nleads, tuple of ints of left lead sites, right lead sites
    nelecs, tuple of num up e's, 0 due to ASU formalism
    timestop, float, how long to run for
    deltat, float, time step increment
    physical params, tuple of t, thyb, Vbias, mu, Vgate, U, B, theta. if None gives defaults
    Rlead_pol, int -1, 0, 1
        if +/- 1, will polarize right lead spins to up/down state
        if 0, does nothing (default)
        also does nothing if B=0 no matter what rlead_pol actually is
    prefix: assigns prefix (eg folder) to default output file name

    Returns:
    none, but outputs t, observable data to /dat/Data/ folder
    '''

    # check inputs
    assert( isinstance(n_leads, tuple) );
    assert( isinstance(nelecs, tuple) );
    assert( isinstance(timestop, float) );
    assert( isinstance(deltat, float) );
    assert( isinstance(phys_params, tuple) or phys_params == None);
    if(Rlead_pol == 1 or Rlead_pol == -1): prefix = "spinpol/";

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
    else: # customized
        V_leads, V_imp_leads, V_bias, mu, V_gate, U, B, theta = phys_params;


    # get 1 elec and 2 elec hamiltonian arrays for siam, dot model impurity
    if(verbose): print("1. Construct hamiltonian")
    thyb_eq = 1e-5; # small but nonzero = more robust
    eq_params = V_leads, thyb_eq, V_bias, mu, V_gate, U, B, theta; # dot hopping turned off, but nonzero = more robust
    h1e, g2e, input_str = fci_mod.dot_hams(n_leads, n_imp_sites, nelecs, eq_params, Rlead_pol = Rlead_pol, verbose = verbose);
        
    # get scf implementation siam by passing hamiltonian arrays
    if(verbose): print("2. FCI solution");
    mol, dotscf = fci_mod.dot_model(h1e, g2e, norbs, nelecs, verbose = verbose);
    
    # from scf instance, do FCI, get exact gd state of equilibrium system
    E_fci, v_fci = fci_mod.scf_FCI(mol, dotscf, verbose = verbose);

    # remove spin prep terms
    h1e += ops.h_B(-B, theta, imp_i, norbs, verbose = verbose);
    h1e += ops.h_B(abs(B)*Rlead_pol,0.0,np.arange(0,norbs, 1, dtype = int)[imp_i[-1]+1:],norbs, verbose = verbose);
    
    # prepare in nonequilibrium state by turning on t_hyb (hopping onto dot)
    if(verbose > 2 ): print("- Add nonequilibrium terms");
    neq_params = 0.0, V_imp_leads, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    neq_h1e, dummy, input_str_noneq = fci_mod.dot_hams(n_leads, n_imp_sites, nelecs, neq_params, verbose = verbose);
    h1e += neq_h1e; # updated to include thyb

    # from fci gd state, do time propagation
    if(verbose): print("3. Time propagation")
    init_str, observables = td_fci.TimeProp(h1e, g2e, v_fci, mol, dotscf, timestop, deltat, imp_i, V_imp_leads, kernel_mode = "plot", verbose = verbose);
    
    # write results to external file
    folder = "dat/DotData/";
    fname = folder+prefix+ str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_Vg"+str(V_gate)+".npy";
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


def DotDataDmrg(n_leads, nelecs, timestop, deltat, bond_dims_i = 50, phys_params=None, Rlead_pol = 0, prefix = "", verbose = 0):
    '''
    Exactly as above, but uses dmrg instead of td-fci
    '''

    setup_start = time.time();

    # check inputs
    assert( isinstance(n_leads, tuple) );
    assert( isinstance(nelecs, tuple) );
    assert( isinstance(timestop, float) );
    assert( isinstance(deltat, float) );
    assert( isinstance(phys_params, tuple) or phys_params == None);
    if(Rlead_pol == 1 or Rlead_pol == -1): prefix = "spinpol/";
    

    # dmrg controls
    bond_dims = [bond_dims_i,bond_dims_i+25,bond_dims_i+50];
    noises = [1e-3,1e-4,1e-5,0] # need to give noise on order of smallest energy

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
    else: # customized
        V_leads, V_imp_leads, V_bias, mu, V_gate, U, B, theta = phys_params;


    # get h1e and h2e for siam, h_imp = h_dot
    if(verbose): print("1. Construct hamiltonian")
    thyb_eq = 1e-5; # small but nonzero val is more robust
    ham_params = V_leads, thyb_eq, V_bias, mu, V_gate, U, B, theta; # dot hopping turned off, but nonzero to fix numerical errors
    h1e, g2e, input_str = fci_mod.dot_hams(n_leads, n_imp_sites, nelecs, ham_params, Rlead_pol = Rlead_pol, verbose = verbose);

    # store physics in fci dump object
    hdump = fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]);      

    # instead of np array, dmrg wants ham as a matrix product operator (MPO)
    h_obj = hamiltonian.Hamiltonian(hdump,flat=True);
    h_mpo = h_obj.build_qc_mpo();
    h_mpo, _ = h_mpo.compress(cutoff=1E-15); # compressing saves memory
    if verbose: print("- Built H as MPO");

    # initial ansatz for wf, in matrix product state (MPS) form
    psi_mps = h_obj.build_mps(bond_dims_i);
    E_mps_init = td_dmrg.compute_obs(h_mpo, psi_mps);
    if verbose: print("- Initial gd energy = ", E_mps_init);

    # ground-state DMRG
    # runs thru an MPE (matrix product expectation) class built from mpo, mps
    if(verbose): print("2. DMRG solution");
    MPE_obj = MPE(psi_mps, h_mpo, psi_mps);

    # solve system by doing dmrg sweeps
    # MPE.dmrg method takes list of bond dimensions, noises, threads defaults to 1e-7
    # can also control verbosity (iprint) sweeps (n_sweeps), conv tol (tol)
    # noises[0] = 1e-3 and tol = 1e-8 needed here
    dmrg_obj = MPE_obj.dmrg(bdims=bond_dims, noises = noises, tol = 1e-8, iprint=0); # will print sweep output
    E_dmrg = dmrg_obj.energies;
    if verbose: print("- Final gd energy = ", E_dmrg[-1]);

    setup_stop = time.time();
    print("\n\n\n\n\n",60*"*","Setup time = ", setup_stop - setup_start,60*"*","\n\n\n\n\n")

    # nonequil hamiltonian (as MPO)
    if(verbose > 2 ): print("- Add nonequilibrium terms");
    ham_params_neq = V_leads, V_imp_leads, V_bias, mu, V_gate, U, 0.0, theta; # dot hopping on now, B field off
    h1e_neq, g2e_neq, input_str_neq = fci_mod.dot_hams(n_leads, n_imp_sites, nelecs, ham_params_neq, verbose = verbose);
    hdump_neq = fcidump.FCIDUMP(h1e=h1e_neq,g2e=g2e_neq,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]);
    h_obj_neq = hamiltonian.Hamiltonian(hdump_neq,True);
    h_mpo_neq = h_obj_neq.build_qc_mpo(); # got mpo
    h_mpo_neq, _ = h_mpo_neq.compress(cutoff=1E-15); # compression saves memory

    # time propagate the noneq state
    # td dmrg uses highest bond dim
    init_str, observables = td_dmrg.kernel(h_mpo_neq, h_obj_neq, psi_mps, timestop, deltat, imp_i, V_imp_leads, [bond_dims[-1]], verbose = verbose);

    # write results to external file
    folder = "dat/DotDataDMRG/";
    fname = folder+prefix+ str(n_leads[0])+"_"+str(n_imp_sites)+"_"+str(n_leads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_Vg"+str(V_gate)+".npy";
    hstring = time.asctime();
    hstring += "\nASU formalism, t_hyb noneq. term, td-DMRG, bdims = "+str(bond_dims)
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
#### wrappers and test code

def MolCurrentPlot():

    # get current data from txt
    nleads = (2,2);
    nimp = 5;
    fname = "dat/MolCurrentWrapper_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1]);
    xJ, yJ = plot.PlotTxt2D(fname+"_J.txt"); #current
    xE, yE = plot.PlotTxt2D(fname+"_E.txt"); # energy
    yE = yE/yE[0] - 1; # normalize energy to 0
    ti, tf = xJ[0], xJ[-1]
    dt = (tf-ti)/len(xJ);

    # control layout of plots
    ax1 = plt.subplot2grid((5, 3), (0, 0), rowspan = 2)               # energy spectrum, top left
    ax2 = plt.subplot2grid((5, 3), (0, 1), rowspan = 2, colspan = 2)               # freqs top right
    ax3 = plt.subplot2grid((5, 3), (2, 0), colspan=3, rowspan=2) # J vs t
    ax4 = plt.subplot2grid((5, 3), (4, 0), colspan=3)            # E vs t

    # plot energy spectrum 
    Energies = [-11.96,-11.56,-11.55, -11.34, -11.31];
    xElevels, yElevels = plot.ESpectrumPlot(Energies);
    for E in yElevels: # each energy level is sep line
        ax1.plot(xElevels, E);
    ax1.set_ylabel("Energy (a.u.)")
    ax1.grid(which = 'both')
    ax1.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False);

    # plot frequencies
    Fnorm, freq = Fourier(yJ, 1/dt, angular = True); # gives dominant frequencies # not yet working
    ax2.plot(freq, Fnorm);
    ax2.set_xlabel("$\omega$ (2$\pi$/s)")

    if False: # compare with dominant freq
        wmax = Fourier(yJ, 1/dt, angular = True, dominant = 1); # gets dominant freq
        ampl_formax = np.amax(yJ)/2; # amplitude when plotting dominant freq
        ax3.plot(xJ, ampl_formax*(np.sin(wmax[0]*xJ) ), linestyle = "dashed");

    # plot J vs t on bottom 
    ax3.plot(xJ, yJ);
    ax3.set_title("td-FCI through $d$ orbital, 2 lead sites on each side");
    ax3.set_xlabel("time (dt = 0.1 s)");
    ax3.set_ylabel("Current*$\pi/V_{bias}$");

    # plot E vs t on bottom  
    ax4.plot(xE, yE);
    ax4.set_xlabel("time (dt = 0.1 s)");
    ax4.set_ylabel("$E/E_i$ - 1");
    #ax4.get_yaxis().get_major_formatter()._set_offset(1)

    # config and show
    plt.tight_layout();
    plt.show();

    return; # end mol current plot


def DotCurrentPlot():

    # control layout of plots
    ax1 = plt.subplot2grid((5, 3), (0, 0), rowspan = 2)               # energy spectrum, top left
    ax2 = plt.subplot2grid((5, 3), (0, 1), rowspan = 2, colspan = 2)               # freqs top right
    ax3 = plt.subplot2grid((5, 3), (2, 0), colspan=3, rowspan=2) # J vs t
    ax4 = plt.subplot2grid((5, 3), (4, 0), colspan=3)            # E vs t

    # global formatting
    mystyle = 'solid';
    mycolor = 'tab:blue'

    # get current data from txt
    nleads = (3,3);
    nimp = 1;
    nelecs = (sum(nleads)+1,0); # half filling
    mu, Vg = [0], np.linspace(-1.0, 1.0, 9); # physical inputs
    mu, Vg = [0], [-1.0, -0.5];
    fname = "dat/DotCurrentData/";
    xJ, yJ, xE, yE = UnpackDotData(fname, nleads, nimp, nelecs, mu, Vg);

    # e spectrum must be done manually
    Energies = [-10.24, -9.68, -9.52, -9.30, -9.19]; 
    xElevels, yElevels = plot.ESpectrumPlot(Energies);
    for E in yElevels: # each energy level is sep line
        ax1.plot(xElevels, E, color = mycolor);
    # repeat for Vg = -0.5
    Energies = [-9.70, -8.91, -8.87, -8.70, -8.47]; 
    xElevels, yElevels = plot.ESpectrumPlot(Energies);
    xElevels += 1;
    for E in yElevels: # each energy level is sep line
        ax1.plot(xElevels, E, color = 'tab:orange');
    ax1.set_ylabel("Energy (a.u.)")
    ax1.grid(which = 'both')
    ax1.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False);

    for i in range(len(Vg)):

        if Vg[i] == 0.0 or Vg[i] == -0.25:
            mystyle = "dashed";
            mys = 0.1
        else:
            mystyle = "solid"
            mys = 0.0

        yE[i] = yE[i]/yE[i][0] - 1; # normalize energy to 0
        ti, tf = xJ[i][0], xJ[i][-1]
        dt = (tf-ti)/len(xJ[i]);

        # plot frequencies
        Fnorm, freq = Fourier(yJ[i], 1/dt, angular = True); # gives dominant frequencies
        ax2.plot(freq, Fnorm);
        ax2.set_xlabel("$\omega$ (2$\pi$/s)")
        ax2.set_ylabel("Fourier amp.")
        ax2.axvline(x = 2*np.pi/(sum(nleads)+1), color = "grey", linestyle = "dashed" );

        # plot J vs t on bottom 
        ax3.plot(xJ[i], yJ[i]+mys, linestyle = mystyle, label = "$V_g$ = "+str(Vg[i]));
        ax3.set_title("td-FCI through dot, "+str(nleads[0])+" lead sites on each side");
        ax3.set_xlabel("time (dt = 0.005 s)");
        ax3.set_ylabel("$J*\pi/V_{bias}$");

        # plot E vs t on bottom  
        ax4.plot(xE[i], yE[i] );
        ax4.set_xlabel("time (dt = 0.005 s)");
        ax4.set_ylabel("$E/E_i$ - 1");


    # config and show
    ax2.grid();
    plt.tight_layout();
    ax3.legend();
    plt.show();

    return; # end dot current plot


def DebugPlot():

    # plot data from txt
    nleads = (1,1);
    nimp = 5;
    fname = "dat/Debug_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_J.txt";
    x, J = plot.PlotTxt2D(fname)

    # compare with fine time step data
    fname_fine = "dat/Debug_fine_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_J.txt";
    xfine, Jfine = plot.PlotTxt2D(fname_fine)

    # plot
    plt.plot(x,J, label = "dt = 0.1 s");
    plt.plot(xfine, Jfine, label = "dt = 0.01 s", linestyle = "dashed");

    # plot data from txt
    fname = "dat/Debug_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_E.txt";
    x, E = plot.PlotTxt2D(fname)
    E = E/E[0] # norm

    # compare with fine time step data
    fname_fine = "dat/Debug_fine_"+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_E.txt";
    xfine, Efine = plot.PlotTxt2D(fname_fine)
    Efine = Efine/Efine[0] # norm

    # plot
    fig, (ax1, ax2) = plt.subplots(2,1);
    ax1.plot(x,J, label = "dt = 0.1 s");
    ax1.plot(xfine, Jfine, label = "dt = 0.01 s", linestyle = "dashed");
    ax2.plot(x,E, label = "E, dt = 0.1 s");
    ax2.plot(xfine, Efine, label = "E, dt = 0.01 s", linestyle = "dashed");

    # config and show
    labels = ["Time (s)", "Current*$\pi/V_{bias}$", "td-FCI: $d$ orbital, 1 lead site on each side"]
    #ax1.set_xlabel(labels[0]);
    ax1.set_ylabel(labels[1]);
    ax1.set_title(labels[2]);
    ax1.legend();
    ax2.set_ylabel("Normalized energy");
    ax2.set_xlabel("Time (s)");
    #ax2.legend();
    plt.show();
    
    
def dtVsdE():

    # system inputs
    nleads = (4,4);
    nelecs = (nleads[0] + nleads[1] + 1,0); # half filling
    mu = 0;
    Vg = -1.0;

    # time step is variable
    tf = 1.0;
    dts = [0.2, 0.1, 0.02, 0.01, 0.002, 0.001];
    dts = [0.167, 0.0167]
    
    for dt in dts:
    
        # run code
        print("****    ",dt);
        prefix = "dt"+str(dt)+"_";
        DotCurrentData(nleads, nelecs, tf, dt, mu, Vg, prefix = prefix, verbose = 5);


def DotDataVsVgate():
    '''
    Get current data thru DotCurrentData which generates E, J vs time txt files
    Tune gate voltage
    '''

    # system inputs
    nleads = (3,3);
    nelecs = (nleads[0] + nleads[1] + 1,0); # half filling
    tf = 40.0
    dt = 0.005

    # tunable phys params
    mu = 0
    for Vg in np.linspace(-1.0, 1.0, 9):

        # run code
        DotCurrentData(nleads, nelecs, tf, dt, mu, Vg, verbose = 5);

    Vg = -0.5;
    for mu in np.linspace(-1.0,0.0,9):

        DotCurrentData(nleads, nelecs, tf, dt, mu, Vg, verbose = 5);

    return; # end dot data vs V gate

    
#################################################
#### exec code

if(__name__ == "__main__"):

    import time
    start_t = time.time(); #time in sec

    verbose = 4;

    # system inputs
    nleads = (3,2);
    nelecs = (sum(nleads)+1,0); # half filling
    tf = 8.0
    dt = 0.01

    # dmrg run, std params, for benchmark
    datafile = DotDataDmrg(nleads,nelecs,tf,dt,phys_params = None, verbose = verbose);

    stop_t = time.time();
    print("total elapsed time = "+str((stop_t-start_t)/60)+" minutes");
    print("Real sec./comp. sec = "+str((stop_t-start_t)/tf) );
    
    # plot results
    datafile = "dat/DotDataDMRG/3_1_2_e6_B0_t0_Vg-0.5.npy"
    splots = ['Jtot','occ','delta_occ','Sz'];
    plot.PlotObservables(nleads, 0.4, datafile, splots = splots);


