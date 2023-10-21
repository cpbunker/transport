'''
Christian Bunker
M^2QM at UF
October 2023


'''

import numpy as np
import matplotlib.pyplot as plt

import time
import json
import sys

##################################################################################
#### wrappers

def get_occ(N, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or ERIs) representing the occupancy of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): # construct ExprBuilder
        builder = eris_or_driver.expr_builder()
    else:
      h1e, g2e = np.zeros((N,N),dtype=float), np.zeros((N,N,N,N),dtype=float);

    # construct
    for spin in spin_inds:
        if(block):
            builder.add_term(spin_strs[spin],[whichsite,whichsite],1.0);
        else:
            h1e[nloc*whichsite+spin,nloc*whichsite+spin] += 1.0;

    # return
    if(block):
        mpo_from_builder = eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
        return mpo_from_builder;
    else:
        occ_eri = tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);
        return occ_eri;

def get_sz(Nspinorbs, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sz> of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): # construct ExprBuilder
        builder = eris_or_driver.expr_builder()
    else:
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cd",[whichsite,whichsite], 0.5);
        builder.add_term("CD",[whichsite,whichsite],-0.5);
    else:
        h1e[nloc*whichsite+spin_inds[0],nloc*whichsite+spin_inds[0]] += 0.5;
        h1e[nloc*whichsite+spin_inds[1],nloc*whichsite+spin_inds[1]] +=-0.5;

    # return
    if(block):
        mpo_from_builder = eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
        return mpo_from_builder;
    else:
        occ_eri = tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);
        return occ_eri;

def get_sx(Nspinorbs, eris_or_driver, whichsite, block, verbose=0):
    '''
    Constructs an operator (either MPO or matrix) representing <Sx> of site whichsite
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): # construct ExprBuilder
        builder = eris_or_driver.expr_builder()
    else:
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    if(block):
        builder.add_term("cD",[whichsite,whichsite], 0.5);
        builder.add_term("Cd",[whichsite,whichsite],0.5);
    else:
        h1e[nloc*whichsite+spin_inds[0],nloc*whichsite+spin_inds[1]] += 0.5;
        h1e[nloc*whichsite+spin_inds[1],nloc*whichsite+spin_inds[0]] += 0.5;
        
    # return
    if(block):
        mpo_from_builder = eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
        return mpo_from_builder;
    else:
        occ_eri = tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);
        return occ_eri;

def get_concurrence(Nspinorbs, eris_or_driver, whichsites, block, g2e_only=False, verbose=0):
    '''
    '''
    spin_inds=[0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);

    # return objects
    if(block): # construct ExprBuilder
        builder = eris_or_driver.expr_builder()
    else:
        h1e, g2e = np.zeros((Nspinorbs,Nspinorbs),dtype=float), np.zeros((Nspinorbs,Nspinorbs,Nspinorbs,Nspinorbs),dtype=float);

    # construct
    print("Concurrence("+str(whichsites)+")");
    which1, which2 = whichsites;
    if(block):
        #builder.add_term("cDcD",[which1,which1,which2,which2],-1.0);
        builder.add_term("cDCd",[which1,which1,which2,which2], 1.0);
        builder.add_term("CdcD",[which1,which1,which2,which2], 1.0);
        #builder.add_term("CdCd",[which1,which1,which2,which2],-1.0);
    else:
        #g2e[nloc*which1+spin_inds[0],nloc*which1+spin_inds[1],nloc*which2+spin_inds[0],nloc*which2+spin_inds[1]] += -1.0;
        g2e[nloc*which1+spin_inds[0],nloc*which1+spin_inds[1],nloc*which2+spin_inds[1],nloc*which2+spin_inds[0]] += 1.0;
        g2e[nloc*which1+spin_inds[1],nloc*which1+spin_inds[0],nloc*which2+spin_inds[0],nloc*which2+spin_inds[1]] += 1.0;
        #g2e[nloc*which1+spin_inds[1],nloc*which1+spin_inds[0],nloc*which2+spin_inds[1],nloc*which2+spin_inds[0]] += -1.0;
        # switch particle labels
        #g2e[nloc*which2+spin_inds[0],nloc*which2+spin_inds[1],nloc*which1+spin_inds[0],nloc*which1+spin_inds[1]] += -1.0;
        g2e[nloc*which2+spin_inds[1],nloc*which2+spin_inds[0],nloc*which1+spin_inds[0],nloc*which1+spin_inds[1]] += 1.0;
        g2e[nloc*which2+spin_inds[0],nloc*which2+spin_inds[1],nloc*which1+spin_inds[1],nloc*which1+spin_inds[0]] += 1.0;
        #g2e[nloc*which2+spin_inds[1],nloc*which2+spin_inds[0],nloc*which1+spin_inds[1],nloc*which1+spin_inds[0]] += -1.0;

    if(g2e_only): return g2e;

    # return
    if(block):
        mpo_from_builder = eris_or_driver.get_mpo(builder.finalize(), iprint=verbose);
        return mpo_from_builder;
    else:
        occ_eri = tdfci.ERIs(h1e, g2e, eris_or_driver.mo_coeff);
        return occ_eri;

def get_energy_fci(h1e, g2e, nelec, nroots=1, verbose=0):
    # convert from arrays to uhf instance
    mol_inst, uhf_inst = utils.arr_to_uhf(h1e, g2e, len(h1e), nelec, verbose = verbose);
    # fci solution
    E_fci, v_fci = utils.scf_FCI(mol_inst, uhf_inst, nroots);
    if(nroots>1): E_fci, v_fci = E_fci[0], v_fci[0];
    # ci object
    CI_inst = tdfci.CIObject(v_fci, len(h1e), nelec);
    return CI_inst, E_fci, uhf_inst;

def get_energy_dmrg(driver, mpo, verbose=0):
    bond_dims = [250] * 4 + [500] * 4
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    threads = [1e-10] * 8
    if(driver is None and mpo is None): return bond_dims, noises, threads;
    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dims[0], nroots=1)
    bond_dims = [250] * 4 + [500] * 4
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-10] * 8
    ret = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=threads, cutoff=0, iprint=verbose);
    return ket, ret;

def vs_site(psi,eris_or_driver,block,which_obs):
    obs_funcs = {"occ":get_occ, "sz":get_sz, "sx":get_sx}
    if(which_obs not in obs_funcs.keys()): raise ValueError;

    # site array
    if(block):
        Nspinorbs = eris_or_driver.n_sites*2;
        impo = eris_or_driver.get_identity_mpo();
    else:
        Nspinorbs = len(eris_or_driver.h1e[0]);
    js = np.arange(Nspinorbs//2);
    vals = np.empty_like(js,dtype=float)
    for j in js:
        op = obs_funcs[which_obs](Nspinorbs,eris_or_driver,j,block);
        if(block):
            vals[j] = np.real(tddmrg.compute_obs(psi, op, eris_or_driver)/eris_or_driver.expectation(psi, impo, psi));
        else:
            vals[j] = np.real(tdfci.compute_obs(psi, op));

    return js, vals;

def plot_wrapper(psi_ci, psi_mps, eris_inst, driver_inst, concur_sites, title_s = ""):
    '''
    '''
    import matplotlib.pyplot as plt
    if(not isinstance(eris_inst, tdfci.ERIs)): raise TypeError;

    # plot charge and spin vs site
    obs_strs = ["occ","sz"];
    ylabels = ["$\langle n_j \\rangle $","$ \langle S_j^{\mu} \\rangle $"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);

    if(psi_ci is not None): # with fci
        E_ci = tdfci.compute_obs(psi_ci, eris_inst);
        C_ci = tdfci.compute_obs(psi_ci,
                    get_concurrence(len(eris_inst.h1e[0]), eris_inst, concur_sites, False));
        for obsi in range(len(obs_strs)):
            x, y = vs_site(psi_ci,H_eris,False,obs_strs[obsi])
            axes[obsi].plot(x,y, label = ("FCI ($E=${:.2f}, $C"+str(concur_sites)+"=${:.2f})").format(E_ci,C_ci), color='tab:blue');

    if(psi_mps is not None): # with dmrg
        E_dmrg = -999# tddmrg.compute_obs(psi_mps, driver_inst.get_mpo(), driver_inst);
        C_dmrg = tddmrg.compute_obs(psi_mps,
                    get_concurrence(len(eris_inst.h1e[0]), driver_inst, concur_sites, True),
                    driver_inst);
        for obsi in range(len(obs_strs)):
            x, y = vs_site(psi_mps,driver_inst,True,obs_strs[obsi])
            axes[obsi].scatter(x,y, label = ("DMRG ($E=${:.2f}, $C"+str(concur_sites)+"=${:.2f})").format(E_dmrg,C_dmrg),
                               marker='o', edgecolors='tab:red', facecolors='none');

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j$");
    axes[0].set_title(title_s);
    plt.legend();
    plt.tight_layout();
    plt.show();

##################################################################################
#### run code

# top level
verbose = 5;
np.set_printoptions(precision = 4, suppress = True);
do_dmrg = True;
json_name = sys.argv[1];
params = json.load(open(json_name));

from transport import tdfci, tddmrg
from transport.tdfci import utils

# some unpacking
myNL, myNFM, myNR, myNe = params["NL"], params["NFM"], params["NR"], params["Ne"],
mynelec = (myNFM+myNe,0);

# checks
assert(params["Jz"]==params["Jx"]);
espin = myNe*np.sign(params["Be"]);
locspin = myNFM*np.sign(params["BFM"]);
myTwoSz = params["TwoSz"];
#assert(espin+locspin == myTwoSz);

#### FCI initialization
####
####

# construct arrays with terms there for all times
H_1e, H_2e = tddmrg.Hsys_builder(params, False, verbose=verbose);

# add in t<0 terms
H_1e, H_2e = tddmrg.Hsys_polarizer(params, False, (H_1e, H_2e), verbose=verbose);
print("H_1e = ");print(H_1e[:2*(myNL+myNFM),:2*(myNL+myNFM)]);print(H_1e[2*(myNL+myNFM):,2*(myNL+myNFM):]);

# gd state
gdstate_ci_inst, gdstate_E, gdstate_scf_inst = get_energy_fci(H_1e, H_2e, mynelec, nroots=1, verbose=verbose);
H_eris = tdfci.ERIs(H_1e, H_2e, gdstate_scf_inst.mo_coeff);
print("Ground state energy (FCI) = {:.6f}".format(gdstate_E))

# check gd state
check_E = tdfci.compute_obs(gdstate_ci_inst, H_eris)
print("Manually computed energy (FCI) = {:.6f}".format(check_E));

#### DMRG initialization
####
####

if(do_dmrg):
    # init ExprBuilder object with terms that are there for all times
    H_driver, H_builder = tddmrg.Hsys_builder(params, True, verbose=verbose); # returns DMRGDriver, ExprBuilder

    # add in t<0 terms
    H_driver, H_mpo_initial = tddmrg.Hsys_polarizer(params, True, (H_driver,H_builder), verbose=0);

    if False: # from fcidump
        H_driver.write_fcidump(H_1e, H_2e, 0.0, n_sites=H_driver.n_sites, n_elec=myNe, spin=myTwoSz, filename="from_driver.fd")
        H_driver.read_fcidump(filename="from_driver.fd")
        H_driver.initialize_system(n_sites=H_driver.n_sites, n_elec=myNe,
                             spin=myTwoSz)
        H_mpo_initial = H_driver.get_qc_mpo(h1e=H_driver.h1e, g2e=H_driver.g2e, ecore=H_driver.ecore, iprint=1)
        print(np.shape(H_1e))
        print(np.shape(H_driver.h1e))
    
    # gd state
    gdstate_mps_inst, gdstate_E_dmrg = get_energy_dmrg(H_driver, H_mpo_initial, verbose=0);
    print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));

    # check gd state
    check_E_dmrg = tddmrg.compute_obs(gdstate_mps_inst, H_mpo_initial, H_driver);
    print("Manually computed energy (DMRG) = {:.6f}".format(check_E_dmrg));

else:
    H_driver, gdstate_mps_inst = None, None;

#### Observables
####
####

# site 0 spin
my_sites = tuple(params["ex_sites"]);
s0_eris = get_sz(len(H_1e), H_eris, my_sites[0], False);
gd_s0 = tdfci.compute_obs(gdstate_ci_inst, s0_eris);
print("Site {:.0f} <Sz> (FCI) = {:.6f}".format(my_sites[0],gd_s0));
if(do_dmrg):
    s0_mpo = get_sz(len(H_1e), H_driver, my_sites[0], True);
    gd_s0_dmrg = tddmrg.compute_obs(gdstate_mps_inst, s0_mpo, H_driver);
    print("Site {:.0f} <Sz> (DMRG) = {:.6f}".format(my_sites[0],gd_s0_dmrg));

# site 5 (ie the impurity site) spin
sdot_eris = get_sz(len(H_1e), H_eris, my_sites[1], False);
gd_sdot = tdfci.compute_obs(gdstate_ci_inst, sdot_eris);
print("Site {:.0f} <Sz> (FCI) = {:.6f}".format(my_sites[1],gd_sdot));
if(do_dmrg):
    sdot_mpo = get_sz(len(H_1e), H_driver, my_sites[1], True);
    gd_sdot_dmrg = tddmrg.compute_obs(gdstate_mps_inst, sdot_mpo, H_driver);
    print("Site {:.0f} <Sz> (DMRG) = {:.6f}".format(my_sites[1], gd_sdot_dmrg));

# plot observables
mytime=0;
plot_wrapper(gdstate_ci_inst, gdstate_mps_inst, H_eris, H_driver, my_sites, title_s = "$t = ${:.2f}".format(mytime));

#### Time evolution
####
####
time_step = 0.01;
time_update = 0.4*np.pi;
time_update = time_step*int(abs(time_update/time_step) + 0.1); # round to discrete # time steps
mytime += time_update;
        
# dynamics FCI
H_1e_dyn, H_2e_dyn = tddmrg.Hsys_builder(params, False, verbose=verbose);
print("H_1e_dyn = ");print(H_1e_dyn[:2*(myNL+myNFM),:2*(myNL+myNFM)]);print(H_1e_dyn[2*(myNL+myNFM):,2*(myNL+myNFM):]);
H_eris_dyn = tdfci.ERIs(H_1e_dyn, H_2e_dyn, gdstate_scf_inst.mo_coeff);
t1_ci_inst = tdfci.kernel(gdstate_ci_inst, H_eris_dyn, time_update, time_step);

if(do_dmrg): # dynamics DMRG
    H_driver_dyn, H_builder_dyn = tddmrg.Hsys_builder(params, True, verbose=verbose);
    H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=0);

    if False: # from fcidump
        H_driver_dyn.write_fcidump(H_1e_dyn, H_2e_dyn, 0.0, filename="from_driver_dyn.fd")
        H_driver_dyn.read_fcidump(filename="from_driver_dyn.fd")
        H_driver_dyn.initialize_system(n_sites=H_driver_dyn.n_sites, n_elec=myNe,
                             spin=myTwoSz)
        H_mpo_dyn = H_driver.get_qc_mpo(h1e=H_driver_dyn.h1e, g2e=H_driver_dyn.g2e, ecore=H_driver_dyn.ecore, iprint=1)
        print(np.shape(H_1e_dyn))
        print(np.shape(H_driver_dyn.h1e))

    # time evol
    bdims = None;

    t1_mps_inst = H_driver_dyn.td_dmrg(H_mpo_dyn, gdstate_mps_inst, delta_t=time_step, target_t=time_update,
                    bond_dims=bdims, hermitian=True, normalize_mps=True, cutoff=0.0, iprint=0);
else:
    t1_mps_inst, H_driver_dyn = None, None;

# observables
plot_wrapper(t1_ci_inst, t1_mps_inst, H_eris_dyn, H_driver_dyn, my_sites, title_s = "$t = ${:.2f}".format(mytime));

# time evol 2nd time
time_update = 0.6*np.pi;
time_update = time_step*int(abs(time_update/time_step) + 0.1); # round to discrete # time steps
mytime += time_update;

if(do_dmrg): # dynamics dmrg
    t2_mps_inst = H_driver_dyn.td_dmrg(H_mpo_dyn, t1_mps_inst, delta_t=time_step, target_t=time_update,
                bond_dims=bdims, hermitian=True, normalize_mps=True, cutoff=0.0, iprint=0);
else:
    t2_mps_inst = None;
    
# dynamics fci
t2_ci_inst = tdfci.kernel(t1_ci_inst, H_eris_dyn, time_update, time_step);

# observables
plot_wrapper(t2_ci_inst, t2_mps_inst, H_eris_dyn, H_driver_dyn, my_sites, title_s = "$t = ${:.2f}".format(mytime));

# time evol 3rd time
time_update = 1.0*np.pi;
time_update = time_step*int(abs(time_update/time_step) + 0.1); # round to discrete # time steps
mytime += time_update;

if(do_dmrg): # dynamics dmrg
    t3_mps_inst = H_driver_dyn.td_dmrg(H_mpo_dyn, t2_mps_inst, delta_t=time_step, target_t=time_update,
                bond_dims=bdims, hermitian=True, normalize_mps=True, cutoff=0.0, iprint=0);
else:
    t3_mps_inst = None;
    
# dynamics fci
t3_ci_inst = tdfci.kernel(t2_ci_inst, H_eris_dyn, time_update, time_step);

# observables
plot_wrapper(t3_ci_inst, t3_mps_inst, H_eris_dyn, H_driver_dyn, my_sites, title_s = "$t = ${:.2f}".format(mytime));









