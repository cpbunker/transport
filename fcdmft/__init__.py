'''
Compute the many body impurity Green's function using DMFT
For DMFT overview see: https://arxiv.org/pdf/1012.3609.pdf (Zgid, Chan paper)

fcdmft package due to Tianyu Zhu et al, Caltech

Wrapper functions due to Christian Bunker, UF
'''

#### setup the fcdmft the package

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

from fcdmft import dmft
from fcdmft.solver import scf_mu as scf
from transport import fci_mod
from pyscf import gto

import numpy as np


#### my wrappers that access package routines

def kernel(SR_1e, SR_2e, coupling, leadsite, verbose = 0):

    # check inputs and unpack
    assert( isinstance(leadsite, site) );
    n_imp_orbs = np.shape(SR_1e)[0];
    SR_1e = np.array([SR_1e]); # up spin only
    SR_2e = np.array([SR_2e]);

    # for now just put defaults here
    Ha2eV = 27.211386; # hartree to eV
    iter_depth = 10;
    n_bath_orbs = 3;
    n_core = 0; # core orbitals
    filling = 0.5; # e's per spin orb
    chem_pot = 4*np.pi*np.pi*filling*filling; # fermi energy, at zero temp
    chem_pot = 0.5/Ha2eV; # orbitals below will be filled, above empty
    nao = 1; # pretty sure this does not do anything
    max_mem = 8000;
    n_orbs = n_imp_orbs + 2*n_bath_orbs;

    # noninteracting green's function in the leads
    if(verbose): print("\n1. Noninteracting Green's function");
    g_non = leadsite.surface_gf(iter_depth, verbose = verbose);
    # returns array of noninteracting gf vals across tb band energies
    # ie from -2 + iE to 2 + iE
    # for hopping != 1, just do E -> E/th

    # now need to expand g (just a number) to operator in coupling space
    g_nona = np.zeros((np.shape(coupling)[0], np.shape(coupling)[1], len(g_non)), dtype = complex); # unfilled
    for coupi in range(np.shape(coupling)[0]):
        for coupj in range(np.shape(coupling)[1]):
            if(coupi == coupj): g_nona[coupi, coupj, :] = g_non; # I in coupling space
    g_nona = np.array([g_nona]); # up spin only

    # higher level green's function
    if(verbose): print("\n2. Interacting Green's function");
    imp_meanfield = dmft.dmft_solver.mf_kernel(SR_1e, SR_2e, chem_pot, nao, np.array([np.eye(n_imp_orbs)]), max_mem, verbose = verbose);
    g_inta = dmft.dmft_solver.mf_gf(imp_meanfield, leadsite.energies, leadsite.iE);

    # understand how the mean field green's function works
    print(">> energy", imp_meanfield.mo_energy);
    print(">> occ", imp_meanfield.mo_occ);
    assert False;
    
    # get hybridization from dyson eq
    # ie, hybridization defines interaction between imp and leads
    if(verbose): print("\n3. Bath discretization");
    hyb = dmft.dmft_solver.get_sigma(g_nona, g_inta);
    if(verbose): print(" - hyb(E) = \n", hyb[0,:,:,0]);

    # start convergence loop here

    # first attempt at bath disc
    # outputs n_bath_orbs bath energies, for each imp orb
    bath = dmft.gwdmft.get_bath_direct(hyb, leadsite.energies, n_bath_orbs);
    if(verbose): print(" - bath energies = ", bath[1]);

    # optimize bath disc

    # construct manybody hamiltonian of imp + bath
    if(verbose): print("\n4. Combine impurity and bath");
    h1e_imp, h2e_imp = dmft.gwdmft.imp_ham(SR_1e, SR_2e, *bath, n_core); # adds in bath states
        
    # find manybody gf of imp + bath
    # I hope this is equivalent to Zgid paper eq 28
    if(verbose): print("\n5. Impurity Green's function");
    meanfield = dmft.dmft_solver.mf_kernel(h1e_imp, h2e_imp, chem_pot, nao, np.array([np.eye(n_orbs)]), max_mem, verbose = 0);
    print(">> energies = ", meanfield.mo_energy);
    print(">> nelecs = ", np.count_nonzero(meanfield.mo_occ));
    
    # use fci (which assumes only one kind of spin) to get Green's function
    assert(len(np.shape(meanfield.mo_coeff)) == 2); # ie no spin dof
    Gimp = dmft.dmft_solver.fci_gf(meanfield, leadsite.energies, leadsite.iE);
    


def h1e_to_gf(E, h1e, g2e, nelecs, bdims, noises):
    '''
    Use dmrg routines in the solvers module to extract a green's function from
    a second quantized hamiltonian
    '''

    # check inputs

    # unpack
    nsites = np.shape(h1e)[0];

    # init GFDMRG object
    dmrg_obj = solver.gfdmrg.GFDMRG();

    # input hams
    pointgroup = 'c1';
    Ecore = 0.0;
    isym = None;
    orb_sym = None;
    dmrg_obj.init_hamiltonian(pointgroup, nsites, sum(nelecs), nelecs[0] - nelecs[1], isym, orb_sym, Ecore, h1e, g2e);

    # get greens function
        # default params taken from fcdmft/examples/DMRG_GF_test/run_dmrg.py
    gmres_tol = 1e-9;
    conv_tol = 1e-8;
    nsteps = 10;
    cps_bond_dims=[1500];
    cps_noises=[0];
    cps_tol=1E-13;
    cps_n_steps=20;
    idxs = None;
    eta = None;
    dfparams = gmres_tol, conv_tol, nsteps, cps_bond_dims, cps_noises, cps_conv_tol, cps_n_steps, idxs, eta;
    G = dmrg_obj.greens_function(bdims, noises, *dfparams, E, None);

    return G;



########################################################################
#### calculation of the surface dos in noninteracting lead
#### recursive formula due to Haydock, 1972

class site(object):
    '''
    Represents tight binding basis vector, in discrete 1d chain
    1d chain is truncated by ends but these can be made arbitrarily large
    
    Quick implementation of how such vectors add, multiply
    Computes action of 1d nearest neighbor hopping hamiltonian H on such vectors
    Thus by defining the H() method can make general
    '''

    def __init__(self, j, c, iE, ends, ham):
        '''
        j, ndarray, nonzero sites
        c, ndarray, coefficients multiplying sites, defaults to 1 for each site
        iE, float, imag offset from real axis
        ends = first and last allowed sites in the chain, defaults to 0, a million
        ham, str, which hamiltonian method to use, defines lead physics
        '''

        # inputs
        assert(isinstance(j,np.ndarray));
        assert(isinstance(c,np.ndarray));
        assert(len(j) == len(c) );
        assert(len(ends) == 2 and isinstance(ends[0], int) );
        
        # screen out anything beyond ends
        self.j = j[j >= ends[0]][ j[j >= ends[0]] <= ends[1] ];

        # coefficients of nonzero sites, similarly screened
        self.c = c[j >= ends[0]][ j[j >= ends[0]] <= ends[1] ];

        # off real axis
        self.iE = iE;

        # endpts
        self.ends = ends;

        # hamiltonian
        self.ham = ham; # keyword for calling a function via H() method


    #### overloads

    def __str__(self):
        stri = "";
        for ji in range(len(self.j)):
            stri += str(self.c[ji])+"|"+str(self.j[ji])+"> + ";
        return stri;
    

    def __add__(self, other): # add two sites, not in place

        assert(self.ends == other.ends); # need to correspond to same chain

        return site(np.append(self.j, other.j), np.append(self.c, other.c), self.iE, self.ends, self.ham);


    def __mul__(self, other): # inner product as orthonormal basis

        if( isinstance(other, int) or isinstance(other, float) ): # multiply coefficients, not in place
            return site(self.j, self.c*other, self.iE, self.ends, self.ham);

        else:
            
            assert(self.ends == other.ends); # need to correspond to same chain
            
            mysum = 0.0;
            for ji in range(len(self.j)):
                for ii in range(len(other.j)):
                    if(self.j[ji] == other.j[ii]):
                        mysum += self.c[ji]*other.c[ii];
            return mysum;

    #### hamiltonians

    def H(self):
        if self.ham == "defH":
            return self.defH();
        else: raise(TypeError);
        
    def defH(self):
        '''
        hamiltonian for uniform nearest neighbor hopping, mu=0
        default H option
        not in place

        assume hopping between all sites is same
        assume onsite energy of all sites is same
        Acts as if mu = 0 and t = -1, but this can be generalized by treating
        E as a scaled energy, E -> (E-mu)/t
        
        '''
        
        newj = [];
        newc = [];
        for ji in range(len(self.j)):
            newj.append(self.j[ji]-1);
            newc.append(self.c[ji]);
            newj.append(self.j[ji]+1);
            newc.append(self.c[ji]);
        return site(np.array(newj), -np.array(newc), self.iE, self.ends, self.ham);
    
        
    #### util
    
    def condense(self):
        '''
        Combine repeated states, in place
        '''

        newj = [];
        newc = [];
        for ji in range(len(self.j)):
            if self.j[ji] not in newj: # add site and coef
                newj.append(self.j[ji]);
                newc.append(self.c[ji]);
            else: # only update coef
                whichi = newj.index(self.j[ji]);
                newc[whichi] += self.c[ji];

        self.j = np.array(newj);
        self.c = np.array(newc);
        return self;
        

    #### calculation of the green's function
    
    def gen_as_bs(self, depth, verbose = 0):
        '''
        generate coefficients a_n', b_n' of fictitious states
        fictitious represent local environment of site of interest, site0

        local environment encoded in hamiltonian, which is represented by the
        site.H method. This method assumse nearest neighbor tight binding
        hamiltonian char'd only by:
        - onsite energy mu, same on each site, set to 0
        - hopping t

        stops iterating at n' = depth
        '''

        # return variables
        a_s = [];
        b_s = [];

        # zeroth iteration
        mu = 0.0; # choose energy shift
        a_s.append(mu); # by definition
        b_s.append(1); # b_(-1) set to one by definition

        # first iteration
        nmin1prime = self; # |0'>
        nprime = nmin1prime.H() + nmin1prime*(-a_s[0]); # |1'>
        if(False):
            print("|0'> = ",nmin1prime);
            print("|1'> = ",nprime);
            a1p = -4*mu*t*t/(2*t*t + mu*mu);
            print("predicted a1' = ", a1p);
            a2p = -4*t*t*mu*a1p*(mu+a1p)
            a2p = a2p/(2*t*t*np.power(mu+a1p,2) + mu*mu*a1p*a1p);
            print("predicted a2 = ", a2p);

        # now iterate
        for itr in range(depth-1):

            # a_n'
            divisor = nprime*nprime;
            if( divisor == 0.0): divisor = 1.0; # to avoid nans
            a_s.append((nprime*nprime.H())/divisor );

            #print(">>>", nprime);

            # b_(n-1)'
            divisor = nmin1prime*nmin1prime;
            if( divisor == 0.0): divisor = 1.0; # to avoid nans
            b_s.append((nmin1prime*nprime.H())/divisor);

            # update states
            nplus1prime = nprime.H() + nprime*(-a_s[-1]) + nmin1prime*(-b_s[-1]);
            nmin1prime = nprime*1; # to copy
            nprime = nplus1prime.condense()*1;

        return np.array(a_s), np.array(b_s);


    def surface_gf(self, depth, verbose = 0):
        '''
        Calculate resolvent green's function according to Haydock 2.6
        (continued fraction form)

        Args:
        - self, site object which contains physics of system thru H method
        - depth, int,  how far to go recursively in continuing fraction
        '''

        # get coefs
        a_s, b_s = self.gen_as_bs(depth, verbose = verbose);

        # define energy domain
        # recall site describes tight binding band
        # assume mu=0, t=-1 otherwise use scaled energy (E-mu)/t
        # ie band goes from -2 to 2 always
        E = np.linspace(-1.999, 1.999, 17) + self.iE; # off real axis
        self.energies = E;

        # start from bottom
        bG = (E-a_s[-1])/2 *(1- np.lib.scimath.sqrt(1-4*b_s[-1]/((E-a_s[-1])*(E-a_s[-1])) ) );

        # iter over rest
        for i in range(2, depth+1):

            bG = b_s[-i]/(E - a_s[-i] - bG); # update recursively
            if(verbose > 2 and (depth-i) < 3): print("g (n = "+str(depth - i)+") = ",bG[:2]);

        return bG;
