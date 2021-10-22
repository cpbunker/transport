'''
Demonstrate weird behavior in td-fci code as t_hyb is turned off
'''

import td_fci
from pyscf import lib, fci, scf, gto, ao2mo

import numpy as np
import matplotlib.pyplot as plt
import functools

# top level inputs
verbose = 3;
nleads = (2,1);
dt = 0.01;
tf = 1.0;

##################################################################################
#### ruojings normal code - copy pasted with some additions
#### V_bias is turned on for nonequilibrium, and all works normally

# physical inputs
ll = nleads[0] # number of left leads
lr = nleads[1] # number of right leads
t = 1.0 # lead hopping
td = 0.4 # dot-lead hopping
U = 1.0 # dot interaction
Vg = -0.5 # gate voltage
V = -0.005 # bias
norb = ll+lr+1 # total number of sites
idot = ll # dot index
if(verbose):
    print("\nInputs:\n- Left, right leads = ",(ll,lr),"\n- nelecs = ", (int(norb/2), int(norb/2)),"\n- Gate voltage = ",Vg,"\n- Bias voltage = ",V,"\n- Lead hopping = ",t,"\n- Dot lead hopping = ",td,"\n- U = ",U);

#### make hamiltonian matrices, spin free formalism
# remember impurity is just one level dot

# make ground state Hamiltonian with zero bias
if(verbose):
    print("1. Construct hamiltonian")
h1e = np.zeros((norb,)*2)
for i in range(norb):
    if i < norb-1:
        dot = (i==idot or i+1==idot)
        h1e[i,i+1] = -td if dot else -t
    if i > 0:
        dot = (i==idot or i-1==idot)
        h1e[i,i-1] = -td if dot else -t
h1e[idot,idot] = Vg # input gate voltage
g2e = np.zeros((norb,)*4) # hubbard
g2e[idot,idot,idot,idot] = U

if(verbose > 2):
    print("- Full one electron hamiltonian:\n", h1e)
    
# code straight from ruojing, don't understand yet
nelec = int(norb/2), int(norb/2) # half filling
Pa = np.zeros(norb)
Pa[::2] = 1.0
Pa = np.diag(Pa)
Pb = np.zeros(norb)
Pb[1::2] = 1.0
Pb = np.diag(Pb)
# UHF
mol = gto.M()
mol.incore_anyway = True
mol.nelectron = sum(nelec)
mf = scf.UHF(mol)
mf.get_hcore = lambda *args:h1e # put h1e into scf solver
mf.get_ovlp = lambda *args:np.eye(norb) # init overlap as identity
mf._eri = g2e # put h2e into scf solver
mf.kernel(dm0=(Pa,Pb))
# ground state FCI
mo_a = mf.mo_coeff[0]
mo_b = mf.mo_coeff[1]
cisolver = fci.direct_uhf.FCISolver(mol)
h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
h1e_mo = (h1e_a, h1e_b)
g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
eci, fcivec = cisolver.kernel(h1e_mo, g2e_mo, norb, nelec)
mycisolver = fci.direct_spin1.FCI();
myE, myv = mycisolver.kernel(h1e, g2e, norb, nelec);
if(verbose):
    print("2. FCI solution");
    print("- gd state energy, zero bias = ", eci);
    print("- direct spin 1 gd state, zero bias = ",myE," (norbs, nelecs = ",norb,nelec,")")
#############
    
#### do time propagation
if(verbose):
    print("3. Time propagation")
for i in range(idot): # introduce bias on leads to get td current
    h1e[i,i] = V/2
for i in range(idot+1,norb):
    h1e[i,i] = -V/2
eris = td_fci.ERIs(h1e, g2e, mf.mo_coeff) # diff h1e than in uhf, thus time dependence
ci = td_fci.CIObject(fcivec, norb, nelec)
kernel_mode = "plot"; # tell kernel whether to return density matrices or arrs for plotting
t, observables = td_fci.kernel(kernel_mode, eris, ci, tf, dt, i_dot = [idot], t_dot = td, verbose = verbose);
E, J, occ, Sz = observables;

# normalize vals
J = J*np.pi/abs(V);
E = E/E[0] - 1;

# plot current vs time
fig, axes = plt.subplots(2, sharex = True);
axes[0].plot(t, J);
axes[0].set_xlabel("time (dt = "+str(dt)+")");
axes[0].set_ylabel("J*$\pi / |V_{bias}|$");
axes[0].set_title("$V_{bias}$ turned on: "+str(nleads[0])+" left sites, "+str(nleads[1])+" right sites");
axes[1].plot(t, E);
axes[1].set_xlabel("time (dt = "+str(dt)+")");
axes[1].set_ylabel("$E/E_{i} - 1$");
plt.show();

##################################################################################
#### Now, intro nonequilibrium by turning on t_hyb (here td), instead of V_bias
#### constructing an scf.UHF(mol) and running kernel behaves strangely in this case:
####    1) <S^2> = 1 even though nelecs = (2,2), so it should be zero (and is zero if t_hyb != 0)
####    2) 2S + 1 = 2.236068, which is not consistent with S^2 = 1 or 0 (again, 2S+1 = 1 if t_hyb != 0)
####    3) direct_spin1 no longer agrees with direct_uhf or SCF
####    4) current readout no longer agrees with my code (not seen here)
#### NB turning t_hyb off in first example reproduces issues 1,2,3 exactly, which
#### makes me think t_hyb is the culprit, not V_bias
####
#### td_vals =            [0.0, 1e-8, 1e-6, 1e-4, 1e-2,  0.1, 0.2, 0.3, 0.4]
#### corresponding <S^2>= [1,   1,  1,0.99999984,0.9984,0.837,0.315,0.0,0.0]
#### direct_spin1 energy agrees at and above td = 1e-6, so this might patch
####
#### weirdly, issue 4 goes away (though 1-3 remain) in the nleads = 3,2 case
####
#### setting td = 1e-6 instead of 0 gets rid of problem 4 for 2,1 case but brings it back for 3,2 case

# physical inputs
ll = nleads[0] # number of left leads
lr = nleads[1] # number of right leads
t = 1.0 # lead hopping
td = 0.0 # dot-lead hopping
U = 1.0 # dot interaction
Vg = -0.5 # gate voltage
V = -0.005 # bias
norb = ll+lr+1 # total number of sites
idot = ll # dot index
if(verbose):
    print("\nInputs:\n- Left, right leads = ",(ll,lr),"\n- nelecs = ", (int(norb/2), int(norb/2)),"\n- Gate voltage = ",Vg,"\n- Bias voltage = ",V,"\n- Lead hopping = ",t,"\n- Dot lead hopping = ",td,"\n- U = ",U);

#### make hamiltonian matrices, spin free formalism
# remember impurity is just one level dot

# make ground state Hamiltonian with bias, without hopping onto dot (td=0)
if(verbose):
    print("1. Construct hamiltonian")
h1e = np.zeros((norb,)*2)
for i in range(norb):
    if i < norb-1:
        dot = (i==idot or i+1==idot)
        h1e[i,i+1] = -td if dot else -t
    if i > 0:
        dot = (i==idot or i-1==idot)
        h1e[i,i-1] = -td if dot else -t
h1e[idot,idot] = Vg # input gate voltage
g2e = np.zeros((norb,)*4) # hubbard
g2e[idot,idot,idot,idot] = U;
for i in range(idot): # bias the leads
    h1e[i,i] = V/2
for i in range(idot+1,norb):
    h1e[i,i] = -V/2

if(verbose > 2):
    print("- Full one electron hamiltonian:\n", h1e)
    
# code straight from ruojing, don't understand yet
nelec = int(norb/2), int(norb/2) # half filling
Pa = np.zeros(norb)
Pa[::2] = 1.0
Pa = np.diag(Pa)
Pb = np.zeros(norb)
Pb[1::2] = 1.0
Pb = np.diag(Pb)
# UHF
mol = gto.M()
mol.incore_anyway = True
mol.nelectron = sum(nelec)
mf = scf.UHF(mol)
mf.get_hcore = lambda *args:h1e # put h1e into scf solver
mf.get_ovlp = lambda *args:np.eye(norb) # init overlap as identity
mf._eri = g2e # put h2e into scf solver
mf.kernel(dm0=(Pa,Pb))
# ground state FCI
mo_a = mf.mo_coeff[0]
mo_b = mf.mo_coeff[1]
cisolver = fci.direct_uhf.FCISolver(mol)
h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
g2e_aa = ao2mo.incore.general(mf._eri, (mo_a,)*4, compact=False)
g2e_aa = g2e_aa.reshape(norb,norb,norb,norb)
g2e_ab = ao2mo.incore.general(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
g2e_ab = g2e_ab.reshape(norb,norb,norb,norb)
g2e_bb = ao2mo.incore.general(mf._eri, (mo_b,)*4, compact=False)
g2e_bb = g2e_bb.reshape(norb,norb,norb,norb)
h1e_mo = (h1e_a, h1e_b)
g2e_mo = (g2e_aa, g2e_ab, g2e_bb)
eci, fcivec = cisolver.kernel(h1e_mo, g2e_mo, norb, nelec)
mycisolver = fci.direct_spin1.FCI();
myE, myv = mycisolver.kernel(h1e, g2e, norb, nelec);
if(verbose):
    print("2. FCI solution");
    print("- gd state energy, zero dot hopping = ", eci);
    print("- direct spin 1 gd state, zero dot hopping = ",myE," (norbs, nelecs = ",norb,nelec,")")
#############
    
#### do time propagation - nonequilibrium comes from allowing hopping thru dot
if(verbose):
    print("3. Time propagation")
td = 0.4;
h1e[idot, idot+1] += -td; # row
h1e[idot, idot-1] += -td;
h1e[idot+1, idot] += -td; # column
h1e[idot-1, idot] += -td;

eris = td_fci.ERIs(h1e, g2e, mf.mo_coeff) # diff h1e than in uhf, thus time dependence
ci = td_fci.CIObject(fcivec, norb, nelec)
kernel_mode = "plot"; # tell kernel whether to return density matrices or arrs for plotting
t, observables = td_fci.kernel(kernel_mode, eris, ci, tf, dt, i_dot = [idot], t_dot = td, verbose = verbose);
E, J, occ, Sz = observables;

# normalize vals
J = J*np.pi/abs(V);
E = E/E[0] - 1;

# plot current vs time
fig, axes = plt.subplots(2, sharex = True);
axes[0].plot(t, J);
axes[0].set_xlabel("time (dt = "+str(dt)+")");
axes[0].set_ylabel("J*$\pi / |V_{bias}|$");
axes[0].set_title("$t_{hyb}$ turned on: "+str(nleads[0])+" left sites, "+str(nleads[1])+" right sites");
axes[1].plot(t, E);
axes[1].set_xlabel("time (dt = "+str(dt)+")");
axes[1].set_ylabel("$E/E_{i} - 1$");
plt.show();









