'''
Christian Bunker
M^2QM at UF
October 2023

Use density matrix renormalization group (DMRG) code from Huanchen Zhai (block2)
to study a 1D array of localized spins interacting with itinerant electrons in a
nanowire. In spintronics, this system is of interest because elecrons can impart
angular momentum on the localized spins, exerting spin transfer torque (STT).
'''

import numpy as np
import matplotlib.pyplot as plt

import time
import json
import sys
import os
print(">>> PWD: ",os.getcwd());

##################################################################################
#### wrappers

def get_energy_fci(h1e, g2e, nelec, nroots=1, verbose=0):
    # convert from arrays to uhf instance
    mol_inst, uhf_inst = utils.arr_to_uhf(h1e, g2e, len(h1e), nelec, verbose = verbose);
    # fci solution
    E_fci, v_fci = utils.scf_FCI(mol_inst, uhf_inst, nroots);
    if(nroots>1): E_fci, v_fci = E_fci[0], v_fci[0];
    # ci object
    CI_inst = tdfci.CIObject(v_fci, len(h1e), nelec);
    return CI_inst, E_fci, uhf_inst;

def get_energy_dmrg(driver, mpo):
    bond_dims = [250] * 4 + [500] * 4
    noises = [1e-2] * 2 + [1e-3] * 2 + [1e-4]*4 + [0]
    threads = [1e-10] * 8
    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dims[0], nroots=1)
    ret = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=threads, cutoff=-1.0, iprint=2); # set to 2 to see Mmps
    return ket, ret;

def concurrence_wrapper(psi,eris_or_driver, whichsites, block):
    '''
    Need to combine operators from TwoSz=+2, 0, -2 symmetry blocks
    to get concurrence
    '''
    # unpack
    spin_inds = [0,1];
    spin_strs = ["cd","CD"];
    nloc = len(spin_strs);
    if(block): Nspinorbs = eris_or_driver.n_sites*2;
    else: Nspinorbs = len(eris_or_driver.h1e[0]);
    which1, which2 = whichsites;

    # test code !!!!
    if(not block): return -999;

    # new gd state MPS bras
    b_builder = eris_or_driver.expr_builder();
    b_builder.add_term("cd",[which1, which1],-1);# up 1
    b_builder.add_term("CD",[which2,which2],-1);# down 2
    c_builder = eris_or_driver.expr_builder();
    c_builder.add_term("CD",[which1,which1],-1);# down 1
    c_builder.add_term("cd",[which2,which2],-1);# up 2
    for b in [b_builder,c_builder]: # add hopping noise
        for j in range(Nspinorbs//2-1):
            for spin in spin_strs:
                b.add_term(spin,[j,j+1],-1e-4);
                b.add_term(spin,[j+1,j],-1e-4);
    noises = [1e-3,1e-4,1e-5,0.0];
    b_mpo = eris_or_driver.get_mpo(b_builder.finalize());
    c_mpo = eris_or_driver.get_mpo(c_builder.finalize());
    b_mps = eris_or_driver.get_random_mps(tag="b_mps");
    c_mps = eris_or_driver.get_random_mps(tag="c_mps");
    eris_or_driver.dmrg(b_mpo, b_mps, noises=noises);
    eris_or_driver.dmrg(c_mpo, c_mps, noises=noises);
    #snapshot(None, b_mps, None, eris_or_driver, params, time=40)
    #snapshot(None, c_mps, None, eris_or_driver, params, time=50)
    #assert False

    # expectation values
    op_for_b = tddmrg.get_concur_coef("b",Nspinorbs,eris_or_driver, whichsites, block);
    b_coef = eris_or_driver.expectation(b_mps,op_for_b,psi);
    print("b_coef",b_coef);
    op_for_c = tddmrg.get_concur_coef("c",Nspinorbs,eris_or_driver, whichsites, block);
    c_coef = eris_or_driver.expectation(c_mps,op_for_c,psi);
    print("c_coef = ",c_coef);
    concur_norm = 2*np.conj(b_coef)*np.conj(c_coef)
    return np.sqrt( np.conj(concur_norm)*concur_norm);

    # calculate term for each symmetry block

    # Sz = 2
    op_p2 = tddmrg.get_concurrence(Nspinorbs, eris_or_driver, whichsites, block, 2);
    if(block): term_p2 = tddmrg.compute_obs(psi, op_p2, eris_or_driver);
    else: term_p2 = tdfci.compute_obs(psi, op_p2);
    
    # Sz = 0
    op_0 = tddmrg.get_concurrence(Nspinorbs, eris_or_driver, whichsites, block, 0);
    if(block): term_0 = tddmrg.compute_obs(psi, op_0, eris_or_driver);
    else: term_0 = tdfci.compute_obs(psi, op_0);

    # Sz =-2
    op_m2 = tddmrg.get_concurrence(Nspinorbs, eris_or_driver, whichsites, block, -2);
    if(block): term_m2 = tddmrg.compute_obs(psi, op_m2, eris_or_driver);
    else: term_m2 = tdfci.compute_obs(psi, op_m2);
    print("block = ",block)
    print("term_p2 = ",term_p2);
    print("term_0 = ",term_0)
    print("term_m2 = ",term_m2)
    return term_0

    return term_p2 + term_0 + term_m2;

def check_observables(the_sites,psi,eris_or_driver,block):
    if(not block):
        # site 0 spin
        s0_eris = tddmrg.get_sz(len(eris_or_driver.h1e[0]), eris_or_driver, the_sites[0], block);
        gd_s0 = tdfci.compute_obs(psi, s0_eris);
        print("Site {:.0f} <Sz> (FCI) = {:.6f}".format(the_sites[0],gd_s0));
        # site 5 (ie the impurity site) spin
        sdot_eris = tddmrg.get_sz(len(eris_or_driver.h1e[0]), eris_or_driver, the_sites[1], block);
        gd_sdot = tdfci.compute_obs(psi, sdot_eris);
        print("Site {:.0f} <Sz> (FCI) = {:.6f}".format(the_sites[1],gd_sdot));       
        # concurrence between
        C_ci = concurrence_wrapper(psi, eris_or_driver, the_sites, False);
        print("C"+str(the_sites)+" = ",C_ci);
        print((psi.r+complex(0,1)*psi.i).T)
    else:
        s0_mpo = tddmrg.get_sz(eris_or_driver.n_sites*2, eris_or_driver, the_sites[0], block);
        gd_s0_dmrg = tddmrg.compute_obs(psi, s0_mpo, eris_or_driver);
        print("Site {:.0f} <Sz> (DMRG) = {:.6f}".format(the_sites[0],gd_s0_dmrg));
        sdot_mpo = tddmrg.get_sz(eris_or_driver.n_sites*2, eris_or_driver, the_sites[1], block);
        gd_sdot_dmrg = tddmrg.compute_obs(psi, sdot_mpo, eris_or_driver);
        print("Site {:.0f} <Sz> (DMRG) = {:.6f}".format(the_sites[1], gd_sdot_dmrg));
        # concurrence between 
        C_dmrg = concurrence_wrapper(psi, eris_or_driver, the_sites, True);
        print("C"+str(the_sites)+" = ",C_dmrg);

def vs_site(psi,eris_or_driver,block,which_obs):
    obs_funcs = {"occ":tddmrg.get_occ, "sz":tddmrg.get_sz, "sx01":tddmrg.get_sx01, "sx10":tddmrg.get_sx10}
    if(which_obs not in obs_funcs.keys()): raise ValueError;

    # site array
    if(block): Nspinorbs = eris_or_driver.n_sites*2;
    else: Nspinorbs = len(eris_or_driver.h1e[0]);
    js = np.arange(Nspinorbs//2);
    vals = np.empty_like(js,dtype=float)
    for j in js:
        op = obs_funcs[which_obs](Nspinorbs,eris_or_driver,j,block);
        if(block):
            vals[j] = np.real(tddmrg.compute_obs(psi, op, eris_or_driver));
        else:
            vals[j] = np.real(tdfci.compute_obs(psi, op));

    return js, vals;

# fig standardizing
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mylinewidth = 3.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def snapshot(psi_ci, psi_mps, eris_inst, driver_inst, params_dict, time = 0.0, draw_arrow=False):
    '''
    '''
    #return;
    if(not isinstance(eris_inst, tdfci.ERIs) and (driver_inst == None)): raise TypeError;

    # unpack
    concur_sites = params_dict["ex_sites"];
    Jsd, Jx, Jz = params_dict["Jsd"], params_dict["Jx"], params_dict["Jz"];
    NL, NFM, NR, Ne = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Ne"];
    Ndofs = NL+2*NFM+NR;
    central_sites = [j for j in range(NL,Ndofs-NR)  if j%2==0];
    loc_spins = [sitei for sitei in range(NL,Ndofs-NR)  if sitei%2==1];

    # plot charge and spin vs site
    obs_strs = ["occ","sz"];
    ylabels = ["$\langle n_j \\rangle $","$ \langle s_j^{\mu} \\rangle $"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);

    if(psi_ci is not None): # with fci
        C_ci =-9999 # concurrence_wrapper(psi_ci, eris_inst, concur_sites, False);
        for obsi in range(len(obs_strs)):
            x, y = vs_site(psi_ci,eris_inst,False,obs_strs[obsi]);
            y_js = y[np.isin(x,loc_spins,invert=True)];# on chain sites
            y_ds = y[np.isin(x,loc_spins)];# off chain impurities
            js = np.array(range(len(y_js)));
            ds_j = np.array(range(central_sites[0],central_sites[0]+len(central_sites)));
            # delocalized spins
            axes[obsi].plot(js,y_js,color=mycolors[0],marker='o',
                            label = ("FCI ($C"+str(concur_sites)+"=${:.2f})").format(C_ci),linewidth=mylinewidth);
            # localized spins
            if(draw_arrow and obs_strs[obsi] != "occ"):
                for di in range(len(central_sites)):
                    axes[obsi].arrow(ds_j[di],0,0,y_ds[di],color=mycolors[1],
                                     width=0.01*mylinewidth,length_includes_head=True);
            else:
                axes[obsi].scatter(ds_j, y_ds, color=mycolors[1], marker="^", s=(3*mylinewidth)**2);

        # get sx
        x, sx01 = vs_site(psi_ci,eris_inst,False,"sx01");
        x, sx10 = vs_site(psi_ci,eris_inst,False,"sx10");
        sx = sx01+sx10;
        sx_js = sx[np.isin(x,loc_spins,invert=True)];# on chain sites
        axes[-1].plot(np.array(range(len(sx_js))), sx_js, color="purple",marker='s', linewidth=mylinewidth);
    if(psi_mps is not None): # with dmrg
        C_dmrg =-9999 #concurrence_wrapper(psi_mps, driver_inst, concur_sites, True);
        for obsi in range(len(obs_strs)):
            x, y = vs_site(psi_mps,driver_inst,True,obs_strs[obsi]);
            y_js = y[np.isin(x,loc_spins,invert=True)];# on chain sites
            y_ds = y[np.isin(x,loc_spins)];# off chain impurities
            js = np.array(range(len(y_js)));
            ds_j = np.array(range(central_sites[0],central_sites[0]+len(central_sites)));
            # delocalized spins
            axes[obsi].scatter(js,y_js,marker=mymarkers[0], edgecolors=accentcolors[1],
                               s=(3*mylinewidth)**2, facecolors='none',label = ("DMRG ($C"+str(concur_sites)+"=${:.2f})").format(C_dmrg));
            # localized spins
            axes[obsi].scatter(ds_j, y_ds, marker="^", edgecolors=accentcolors[1],
                               s=(3*mylinewidth)**2, facecolors='none');
                
    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j$");
    axes[-1].set_xlim(np.min(js), np.max(js));
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title("$J_{sd} = $"+"{:.4f}$t_l$".format(Jsd)+", $J_x = ${:.4f}$t_l$, $J_z = ${:.4f}$t_l$, $N_e = ${:.0f}".format(Jx, Jz, Ne));
    plt.tight_layout();
    plt.show();
    #plt.savefig(json_name[:-4]+"_time{:.2f}.pdf".format(time));

##################################################################################
#### run code

# top level
verbose = 1; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
params = json.load(open(json_name));
do_fci = bool(int(sys.argv[2]));
do_dmrg = bool(int(sys.argv[3]));
assert(do_fci or do_dmrg);
print(">>> Do FCI  = ",do_fci);
print(">>> Do DMRG = ",do_dmrg);
fci_from_block = False; # skip g2e
from transport import tdfci, tddmrg
from transport.tdfci import utils

# some unpacking
myNL, myNFM, myNR, myNe = params["NL"], params["NFM"], params["NR"], params["Ne"],
mynelec = (myNFM+myNe,0);
my_sites = params["ex_sites"];

# checks
assert(params["Jz"]==params["Jx"]);
espin = myNe*np.sign(params["Be"]);
locspin = myNFM*np.sign(params["BFM"]);
myTwoSz = params["TwoSz"];
special_cases = ["BFM_first", "Bsd", "Bsd_x"];
special_cases_flag = False;
for case in special_cases:
    if(case in params.keys()):print(">>> special case: ",case); special_cases_flag = True;
if(not special_cases_flag): assert(espin+locspin == myTwoSz);

#### Initialization
####
####
init_start = time.time();

if(do_fci): # fci gd state

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

else:
    H_eris, gdstate_ci_inst = None, None;

if(do_dmrg): # dmrg gd state
    
    # init ExprBuilder object with terms that are there for all times
    H_driver, H_builder = tddmrg.Hsys_builder(params, True, scratch_dir=json_name, verbose=0); # returns DMRGDriver, ExprBuilder

    # add in t<0 terms
    H_driver, H_mpo_initial = tddmrg.Hsys_polarizer(params, True, (H_driver,H_builder), verbose=0);

    if fci_from_block: # from fcidump
        #H_driver.write_fcidump(H_1e, H_2e, 0.0, n_sites=H_driver.n_sites, n_elec=myNe, spin=myTwoSz, filename="from_driver.fd")
        #H_driver.read_fcidump(filename="from_driver.fd")
        #H_driver.initialize_system(n_sites=H_driver.n_sites, n_elec=myNe,spin=myTwoSz)
        print(H_driver.n_sites, H_driver.n_elec, H_driver.spin)
        H_mpo_initial = H_driver.get_qc_mpo(h1e=H_1e, g2e=H_2e, ecore=0, iprint=verbose-1)
        print(np.shape(H_1e))
        #print(np.shape(H_driver.h1e))
    
    # gd state
    gdstate_mps_inst, gdstate_E_dmrg = get_energy_dmrg(H_driver, H_mpo_initial);
    print("Ground state energy (DMRG) = {:.6f}".format(gdstate_E_dmrg));

    # check gd state
    check_E_dmrg = tddmrg.compute_obs(gdstate_mps_inst, H_mpo_initial, H_driver);
    print("Manually computed energy (DMRG) = {:.6f}".format(check_E_dmrg));

else:
    H_driver, gdstate_mps_inst = None, None;

init_end = time.time();
print(">>> Init compute time (FCI = "+str(do_fci)+", DMRG="+str(do_dmrg)+") = "+str(init_end-init_start));

#### Observables
####
####
mytime=0;

# plot observables
if(do_fci): check_observables(my_sites, gdstate_ci_inst, H_eris, False);
if(do_dmrg): check_observables(my_sites, gdstate_mps_inst, H_driver, True);
snapshot(gdstate_ci_inst, gdstate_mps_inst, H_eris, H_driver, params, time = mytime);

#### Time evolution
####
####
evol1_start = time.time();
time_step = 0.01;
time_update = params["t1"];
time_update = time_step*int(abs(time_update/time_step) + 0.1); # round to discrete # time steps
mytime += time_update;
        
if(do_fci): # FCI dynamics 
    H_1e_dyn, H_2e_dyn = tddmrg.Hsys_builder(params, False, verbose=verbose);
    print("H_1e_dyn = ");print(H_1e_dyn[:2*(myNL+myNFM),:2*(myNL+myNFM)]);print(H_1e_dyn[2*(myNL+myNFM):,2*(myNL+myNFM):]);
    H_eris_dyn = tdfci.ERIs(H_1e_dyn, H_2e_dyn, gdstate_scf_inst.mo_coeff);
    t1_ci_inst = tdfci.kernel(gdstate_ci_inst, H_eris_dyn, time_update, time_step);
else:
    t1_ci_inst, H_eris_dyn = None, None;
    
if(do_dmrg): # DMRG dynamics
    bdims = [600];
    H_driver_dyn, H_builder_dyn = tddmrg.Hsys_builder(params, True, scratch_dir = json_name, verbose=verbose);
    H_mpo_dyn = H_driver_dyn.get_mpo(H_builder_dyn.finalize(), iprint=0);
    if(fci_from_block): H_mpo_dyn = H_driver_dyn.get_qc_mpo(h1e=H_1e_dyn, g2e=H_2e_dyn, ecore=0, iprint=1)
    t1_mps_inst = H_driver_dyn.td_dmrg(H_mpo_dyn, gdstate_mps_inst, delta_t=time_step*1j, target_t=time_update*1j,
                    bond_dims=bdims, hermitian=False, normalize_mps=False, iprint=verbose-1);
else:
    t1_mps_inst, H_driver_dyn = None, None;

evol1_end = time.time();
print(">>> Evol1 compute time (FCI = "+str(do_fci)+", DMRG="+str(do_dmrg)+") = "+str(evol1_end-evol1_start));

# observables
if(do_fci): check_observables(my_sites, t1_ci_inst, H_eris_dyn, False);
if(do_dmrg): check_observables(my_sites, t1_mps_inst, H_driver_dyn, True);
snapshot(t1_ci_inst, t1_mps_inst, H_eris_dyn, H_driver_dyn, params, time=mytime);

# time evol 2nd time
evol2_start = time.time();
time_update = params["t2"];
time_update = time_step*int(abs(time_update/time_step) + 0.1); # round to discrete # time steps
mytime += time_update;

if(do_dmrg): # DMRG dynamics
    t2_mps_inst = H_driver_dyn.td_dmrg(H_mpo_dyn, t1_mps_inst, delta_t=time_step*1j, target_t=time_update*1j,
                bond_dims=bdims, hermitian=False, normalize_mps=False, cutoff=0.0, iprint=verbose-1);
else:
    t2_mps_inst = None;
    
if(do_fci): # FCI dynamics
    t2_ci_inst = tdfci.kernel(t1_ci_inst, H_eris_dyn, time_update, time_step);
else:
    t2_ci_inst = None;

evol2_end = time.time();
print(">>> Evol2 compute time (FCI = "+str(do_fci)+", DMRG="+str(do_dmrg)+") = "+str(evol2_end-evol2_start));

# observables
if(do_fci): check_observables(my_sites, t2_ci_inst, H_eris, False);
if(do_dmrg): check_observables(my_sites, t2_mps_inst, H_driver, True);
snapshot(t2_ci_inst, t2_mps_inst, H_eris_dyn, H_driver_dyn, params, time=mytime);

# time evol 3rd time
evol3_start = time.time();
time_update = params["t3"];
time_update = time_step*int(abs(time_update/time_step) + 0.1); # round to discrete # time steps
mytime += time_update;

if(do_dmrg): # DMRG dynamics
    t3_mps_inst = H_driver_dyn.td_dmrg(H_mpo_dyn, t2_mps_inst, delta_t=time_step*1j, target_t=time_update*1j,
                bond_dims=bdims, hermitian=False, normalize_mps=False, cutoff=0.0, iprint=verbose-1);
else:
    t3_mps_inst = None;
    
if(do_fci): # FCI dynamics
    t3_ci_inst = tdfci.kernel(t2_ci_inst, H_eris_dyn, time_update, time_step);
else:
    t3_ci_inst = None;
    
evol3_end = time.time();
print(">>> Evol3 compute time (FCI = "+str(do_fci)+", DMRG="+str(do_dmrg)+") = "+str(evol3_end-evol3_start));

# observables
if(do_fci): check_observables(my_sites, t3_ci_inst, H_eris, False);
if(do_dmrg): check_observables(my_sites, t3_mps_inst, H_driver, True);
snapshot(t3_ci_inst, t3_mps_inst, H_eris_dyn, H_driver_dyn, params, time=mytime);









