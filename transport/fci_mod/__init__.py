'''
Christian Bunker
M^2QM at UF
June 2021

fci_mod.py

Helpful funcs for using pySCF, pyblock3
Imports are within functions since some machines can run only pyblock3 or pyscf

pyscf/fci module:
- configuration interaction solvers of form fci.direct_x.FCI()
- diagonalize 2nd quant hamiltonians via the .kernel() method
- .kernel takes (1e hamiltonian, 2e hamiltonian, # spacial orbs, (# alpha e's, # beta e's))
- direct_nosym assumes only h_pqrs = h_rspq (switch r1, r2 in coulomb integral)
- direct_spin1 assumes h_pqrs = h_qprs = h_pqsr = h_qpsr
'''

import numpy as np


##########################################################################################################
#### type conversions

def arr_to_scf(h1e, g2e, norbs, nelecs, verbose = 0):
    '''
    Converts hamiltonians in array form to scf object
    
    Args:
    - h1e, 2d np array, 1e part of siam ham
    - g2e, 2d np array, 2e part of siam ham
    - norbs, int, total num spin orbs
    - nelecs, tuple of number es, 0 due to All spin up formalism
    
    Returns: tuple of
    mol, gto.mol object which holds some physical params
    scf inst, holds physics: h1e, h2e, mo coeffs etc
    '''
    from pyscf import gto, scf
    if(not isinstance(h1e, np.ndarray)): raise TypeError;
    if(not isinstance(g2e, np.ndarray)): raise TypeError;
    
    # initial guess density matrices
    Pa = np.zeros(norbs)
    Pa[::2] = 1.0
    Pa = np.diag(Pa)
    
    # put everything into UHF scf object
    if(verbose):
        print("\nUHF energy calculation")
    mol = gto.M(); # geometry is meaningless
    mol.incore_anyway = True
    mol.nelectron = sum(nelecs)
    mol.spin = nelecs[1] - nelecs[0]; # in all spin up formalism, mol is never spinless!
    scf_inst = scf.UHF(mol)
    scf_inst.get_hcore = lambda *args:h1e # put h1e into scf solver
    scf_inst.get_ovlp = lambda *args:np.eye(norbs) # init overlap as identity matrix
    scf_inst._eri = g2e # put h2e into scf solver
    if( nelecs == (1,0) ):
        scf_inst.kernel(); # no dm
    else:
        scf_inst.kernel(dm0=(Pa, Pa)); # prints HF gd state but this number is meaningless
                                   # what matter is h1e, h2e are now encoded in this scf instance

    return mol, scf_inst;

def arr_to_eigen(h1e, g2e, nelecs, verbose = 0):
    '''
    '''
    if(not isinstance(h1e, np.ndarray)): raise TypeError;
    if(not isinstance(g2e, np.ndarray)): raise TypeError;

    # unpack
    norbs = np.shape(h1e)[0];

    # to scf
    mol, scfo = arr_to_scf(h1e, g2e, norbs, nelecs);

    # to eigenstates
    e, v = scf_FCI(mol, scfo, nroots = norbs, verbose = verbose);

    return e,v;

def arr_to_mpo(h1e, g2e, nelecs, bdim_i, cutoff = 1e-15):
    '''
    Convert physics contained in an FCIDUMP object or file
    to a MatrixProduct Expectation (MPE) for doing DMRG

    Args:
    fd, a pyblock3.fcidump.FCIDUMP object, or filename of such an object
    bdim_i, int, initial bond dimension of the MPE

    Returns:
    tuple of ham as MPO, state as mps
    '''
    from pyblock3 import hamiltonian, fcidump
    from pyblock3.algebra.mpe import MPE
    
    if(not isinstance(h1e, np.ndarray)): raise TypeError;
    if(not isinstance(g2e, np.ndarray)): raise TypeError;

    # unpack
    norbs = np.shape(h1e)[0];

    # convert arrays to fcidump
    fd = fcidump.FCIDUMP(h1e=h1e,g2e=g2e,pg='c1',n_sites=norbs,n_elec=sum(nelecs), twos=nelecs[0]-nelecs[1]);

    # convert fcidump to hamiltonian obj
    h_obj = hamiltonian.Hamiltonian(fd,flat=True);

    # from hamiltonian obj, build Matrix Product Operator
    h_mpo = h_obj.build_qc_mpo();
    h_mpo, _ = h_mpo.compress(cutoff = cutoff);
    
    # from hamiltonian onj, build Matrix Product State
    psi_mps = h_obj.build_mps(bdim_i);

    # MPE
    return h_obj, h_mpo, psi_mps;


def scf_to_arr(mol, scf_obj):
    '''
    Converts physics of an atomic/molecular system, as contained in an scf inst
    ie produced by passing molecular geometry object mol to
    - scf.RHF(mol) restricted hartree fock
    - scf.UHF(mol) unrestricted hartree fock
    - scf.RKS(mol).run() restricted Kohn sham
    - etc
    to ab initio hamiltonian arrays h1e and g2e
    '''
    from pyscf import ao2mo, gto
    if(not isinstance(mol, gto.mol)): raise TypeError;

    # unpack scf object
    hcore = scf_obj.get_hcore();
    coeffs = scf_obj.mo_coeff;
    norbs = np.shape(coeffs)[0];

    # convert to h1e and h2e array reps in molecular orb basis
    h1e = np.dot(coeffs.T, hcore @ coeffs);
    g2e = ao2mo.restore(1, ao2mo.kernel(mol, coeffs), norbs);

    return h1e, g2e;

#### array dimensionality

def scal_to_vec(scal, n_dof):
    '''
    Take a number or operator, which is a scalar in real space,
    energy space, etc and make it a constant vector in that space
    '''

    return np.full((n_dof, *np.shape(scal)), scal).T;

def vec_1d_to_2d(vec, n_loc_dof):
    '''
    Take a 1d vector (ie with spatial and spin dofs mixed)
    to a 2d vector(ie with spatial and spin dofs separated)
    '''
    if( not isinstance(vec, np.ndarray)): raise TypeError;
    if( len(vec) % n_loc_dof != 0): raise ValueError;

    # unpack
    n_spatial_dof = len(vec) // n_loc_dof;
    new_vec = np.zeros((n_spatial_dof,n_loc_dof), dtype=vec.dtype);
    assert(n_spatial_dof > n_loc_dof); # hacky but works

    # convert
    for sitei in range(n_spatial_dof): # iter site dof only               
        for loci in range(n_loc_dof): # iter over local dofs

            # site, loc indices -> overall indices
            ovi = sitei*n_loc_dof + loci;

            # update
            new_vec[sitei, loci] = vec[ovi];

    return new_vec;

def vec_2d_to_1d(vec):
    '''
    Take a 2d vector (ie with spatial and spin dofs separated)
    to a 1d vector(ie with spatial and spin dofs mixed)
    '''
    if( not isinstance(vec, np.ndarray)): raise TypeError;

    # unpack
    n_spatial_dof, n_loc_dof = np.shape(vec);
    n_ov_dof = n_spatial_dof*n_loc_dof;
    new_vec = np.zeros((n_ov_dof,), dtype=vec.dtype);
    assert(n_spatial_dof > n_loc_dof); # hacky but works

    # convert
    for sitei in range(n_spatial_dof): # iter site dof only               
        for loci in range(n_loc_dof): # iter over local dofs

            # site, loc indices -> overall indices
            ovi = sitei*n_loc_dof + loci;

            # update
            new_vec[ovi] = vec[sitei, loci];

    return new_vec;

def mat_2d_to_4d(mat, n_loc_dof):
    '''
    Take a 2d matrix (ie with spatial and spin dofs mixed)
    to a 4d matrix (ie with spatial and spin dofs separated)
    '''
    if( not isinstance(mat, np.ndarray)): raise TypeError;
    if( len(mat) % n_loc_dof != 0): raise ValueError;

    # unpack
    n_spatial_dof = len(mat) // n_loc_dof;
    new_mat = np.zeros((n_spatial_dof, n_spatial_dof, n_loc_dof, n_loc_dof), dtype=mat.dtype);

    # convert
    for sitei in range(n_spatial_dof): # iter site dof only
        for sitej in range(n_spatial_dof): # same
                
            for loci in range(n_loc_dof): # iter over local dofs
                for locj in range(n_loc_dof):

                    # site, loc indices -> overall indices
                    ovi = sitei*n_loc_dof + loci;
                    ovj = sitej*n_loc_dof + locj;

                    # update
                    new_mat[sitei, sitej, loci, locj] = mat[ovi,ovj];

    return new_mat;

def mat_4d_to_2d(mat):
    '''
    Take a 4d matrix (ie with spatial and spin dofs separated)
    to a 2d matrix (ie with spatial and spin dofs mixed)
    '''
    if( not isinstance(mat, np.ndarray)): raise TypeError;
    if( np.shape(mat)[0] != np.shape(mat)[1]): raise ValueError;
    if( np.shape(mat)[2] != np.shape(mat)[3]): raise ValueError;

    # unpack
    n_loc_dof = np.shape(mat)[-1];
    n_spatial_dof = np.shape(mat)[0];
    n_ov_dof = n_loc_dof*n_spatial_dof;
    new_mat = np.zeros((n_ov_dof,n_ov_dof), dtype=mat.dtype);

    # convert
    for sitei in range(n_spatial_dof): # iter site dof only
        for sitej in range(n_spatial_dof): # same
                
            for loci in range(n_loc_dof): # iter over local dofs
                for locj in range(n_loc_dof):

                    # site, loc indices -> overall indices
                    ovi = sitei*n_loc_dof + loci;
                    ovj = sitej*n_loc_dof + locj;

                    # update
                    new_mat[ovi,ovj] = mat[sitei, sitej, loci, locj];

    return new_mat;

#### second quant

def terms_to_g2e(g2e, terms1, coefs1, terms2, coefs2):
    '''
    '''
    if(not isinstance(g2e, np.ndarray)): raise TypeError;

    for termi in range(len(terms1)):
        for termj in range(len(terms2)):
            #print(terms1[termi][0], terms1[termi][1], terms2[termj][0], terms2[termj][1]);
            g2e[terms1[termi][0], terms1[termi][1], terms2[termj][0], terms2[termj][1]] += coefs1[termi]*coefs2[termj];
            g2e[terms2[termj][0], terms2[termj][1], terms1[termi][0], terms1[termi][1]] += coefs1[termi]*coefs2[termj];

    return g2e;


def single_to_det(h1e, g2e, Nps, states, dets_interest = [], verbose = 0):
    '''
    transform h1e, g2e arrays, ie matrix elements in single particle basis rep
    to basis of slater determinants

    Args:
    - h1e, 2d np array, 1 particle matrix elements
    - g2e, 4d np array, 2 particle matrix elements
    - Nps, 1d array, number of particles of each species
    - states, list of lists of 1p basis states for each species
    - dets_interest, list of determinants to pick out matrix elements of
        only if asked
        only if dets of interest do not couple with other dets (blocked off)
    '''
    import itertools
    
    if(not isinstance(Nps, np.ndarray)): raise TypeError;
    if(not isinstance(states, list)): raise TypeError;
    if(not isinstance(dets_interest, list)): raise TypeError;
    if(not len(states) == len(Nps)): raise TypeError;
    if(not states[-1][-1]+1 == np.shape(h1e)[0] ): raise TypeError;

    # 1 particle basis to N particle slater determinants
    # dets start as cartesian products
    dets = np.array([xi for xi in itertools.product(*tuple(states))]);

    if verbose: print("Det. basis:\n",dets);

    # put one particle matrix elements into determinantal matrix
    H = np.zeros((len(dets), len(dets) ), dtype = complex);
    for deti in range(len(dets)):
        for detj in range(len(dets)):

            # how many 1p states the dets differ by, under maximum coincidence
            ndiff = 0;
            for pi in dets[deti]:
                if( pi not in dets[detj]):
                    ndiff += 1;

            if( ndiff == 0):
                
                # h1e
                for pi in dets[deti]: # sum over one particle states shared between both dets
                    H[deti, detj] += h1e[pi, pi]; # diagonal elements of 1p matrix

                # g2e
                mysum = 0.0;
                for pi in dets[deti]: # all shared states
                    for pj in dets[detj]:
                        mysum += g2e[pi, pi, pj, pj] - g2e[pi, pj, pj, pi]
                H[deti, detj] += (1/2)*mysum;

            elif( ndiff == 1):
                
                # have to figure out which two orbs are different:
                for pi in range(len(dets[deti])):
                    if dets[deti,pi] not in dets[detj]: whichi = pi; # index
                for pj in range(len(dets[detj])):
                    if dets[detj,pj] not in dets[deti]: whichj = pj; # index
                    
                # have to figure out fermi sign
                deltais = [abs(whichi - whichj)]
                for el in dets[deti]:
                    if el in dets[detj]:
                        deltais.append(abs(np.argmax(dets[detj] == el) - np.argmax(dets[deti] == el)));
                sign = np.power(-1, np.sum(deltais )/2 );

                # h1e
                H[deti, detj] += sign*h1e[dets[deti, whichi], dets[detj, whichj]];

                # g2e
                mysum = 0.0;
                for pi in dets[deti][dets[deti] != whichi]: # all shared orbs
                    mysum += g2e[dets[deti,whichi],dets[detj,whichj],pi,pi] - g2e[dets[deti,whichi],pi,pi,dets[detj,whichj]];
                H[deti, detj] += sign*mysum;

            elif( ndiff == 2):

                # have to figure out which two orbs are different:
                for pi2 in range(len(dets[deti])):
                    if dets[deti,pi2] not in dets[detj]: whichi2 = pi2;
                for pi1 in range(len(dets[deti])):
                    if dets[deti,pi1] not in dets[detj] and pi1 != whichi2: whichi1 = pi1;
                for pj2 in range(len(dets[deti])):
                    if dets[deti,pj2] not in dets[detj]: whichj2 = pj2;
                for pj1 in range(len(dets[deti])):
                    if dets[deti,pj1] not in dets[detj] and pj1 != whichj2: whichj1 = pj1;

                # have to figure out fermi sign
                deltais = [abs(whichi1 - whichj1),abs(whichi2-whichj2)]
                for el in dets[deti]:
                    if el in dets[detj]:
                        deltais.append(abs(np.argmax(dets[detj] == el) - np.argmax(dets[deti] == el)));
                sign = np.power(-1, np.sum(deltais )/2 );

                # no h1e contribution

                # g2e
                #print(dets[deti,whichi1],dets[detj,whichj1],dets[deti,whichi2],dets[detj,whichj2]);
                H[deti,detj] += sign*g2e[dets[deti,whichi1],dets[detj,whichj1],dets[deti,whichi2],dets[detj,whichj2]];
                H[deti,detj] += -sign*g2e[dets[deti,whichi1],dets[detj,whichj2],dets[deti,whichi2],dets[detj,whichj1]];
                         
            else: pass; # otherwise det matrix element is zero

    # if requested, choose dets of interest only
    if(len(dets_interest)):

        # make sure requested dets are valid
        for det in dets_interest:
            assert(det in dets);
        dets_interest = np.array(dets_interest);

        # get indices of dets of interest
        is_interest = [];
        for deti in range(len(dets)): # all determinants
            for det in dets_interest: # only ones equal to one of interest
                if not np.any(dets[deti] - det): # dets same
                    is_interest.append(deti);

        # check that requested dets do not couple to other dets
        for deti in range(len(dets)): # all determinants
            if deti in is_interest: # must be one of interest
                for detj in range(len(dets)): # all others
                    if np.any(dets[deti] - dets[detj]): # dets not same
                        coupling = H[deti,detj];
                        if( (coupling != 0) and (detj not in is_interest) ):
                            # bad: nonzero coupling outside subspace of interest
                            print("\nWARN: bad coupling: ",dets[deti], dets[detj], coupling);
                            #pass;
                        
        # transfer desired matrix elements
        newH = np.zeros((len(is_interest),len(is_interest) ), dtype = complex);
        for i in range(len(is_interest)):
            for j in range(len(is_interest)):
                newH[i,j] += H[is_interest[i], is_interest[j] ];
        H = newH;
        
    return H;

#### under construction

def cj_to_ck(h1e, nleads):
    '''
    Transform hams which are second quantized in real space (ie c_j)
    into hams which are second quantized in momentum space (ie c_k)

    Note on ASU formalism
    For an L sight chain we have tight binding sites 0,1,...2L in real space.
    Each are single e only, even are up and odd are down.
    In k space k = (2\pi)/La * integer < \pi/a -> L//2 states of each spin
    where m = \pm 1, ... \pm L/2 e.g. for a 2 site lead, m = \pm 1
    corresponding to a left and right moving state of each spin (4 total states)
    '''
    if(not np.shape(h1e)[0] % 2 == 0): raise TypeError;
    raise NotImplementedError

    # unpack
    norbs = np.shape(h1e)[0];
    nimp = norbs - 2*sum(nleads);
    iLL = 2*nleads[0];
    iRL = 2*nleads[0] + nimp;

    # full k state ham
    hk = np.zeros_like(h1e, dtype = complex);
    for j in range(norbs): # iter over spin orbs
        for jp in range(norbs):

            # to ref molecular orbs
            jmo = j // 2;
            jpmo = jp // 2;
            
            # replace pure lead j states with k states, left lead
            if( j < iLL and jp < iLL ):
                # k is just index in ham, m reps ka=2\pi m/L, w/ m <= L/2
                for k, m in enumerate(np.append(range(-nleads[0]//2,0),range(1,nleads[0]//2+1))):
                    if(j == 0 and jp == 0): # do only once
                        ka = 2*np.pi*m/nleads[0];
                        hk[2*k,2*k] += -2*np.cos(ka);
                        hk[2*k+1,2*k+1] += -2*np.cos(ka);
                    '''
                    for kp, mp in enumerate(np.append(range(-nleads[0]//2,0),range(1,nleads[0]//2+1))):
                        if(j % 2 == 0 and jp % 2 == 0): # up states
                            hk[2*k, 2*kp] += h1e[j, jp]*(1/nleads[0])*np.exp(complex(0,2*np.pi*m/nleads[0]*(jmo % nleads[0]) - 2*np.pi*mp/nleads[0]*(jpmo % nleads[0])));
                        elif(j % 2 == 1 and jp % 2 == 1): # down states
                            hk[2*k+1, 2*kp+1] += h1e[j, jp]*(1/nleads[0])*np.exp(complex(0,2*np.pi*m/nleads[0]*(jmo % nleads[0]) - 2*np.pi*mp/nleads[0]*(jpmo % nleads[0])));
                        else: assert(h1e[j,jp] == 0); # no spin flip terms in lead
                    '''
            # replace pure lead j states with k states, right lead
            elif(j >= iRL and jp >= iRL ):
                for k, m in enumerate(np.append(range(-nleads[1]//2,0),range(1,nleads[1]//2+1))):
                    if(j == 0 and jp == 0): # do only once
                        ka = 2*np.pi*m/nleads[1];
                        hk[iRL+2*k,iRL+2*k] += -2*np.cos(ka);
                        hk[iRL+2*k+1,iRL+2*k+1] += -2*np.cos(ka);
                    '''
                    for kp, mp in enumerate(np.append(range(-nleads[1]//2,0),range(1,nleads[1]//2+1))):
                        if(j % 2 == 0 and jp % 2 == 0): # up states
                            hk[iRL + 2*k, iRL + 2*kp] += h1e[j, jp]*(1/nleads[1])*np.exp(complex(0,2*np.pi*m/nleads[1]*(jmo % (nimp+nleads[1])) - 2*np.pi*mp/nleads[1]*(jpmo % (nimp+nleads[1]))));
                        elif(j % 2 == 1 and jp % 2 == 1): # down states
                            hk[iRL+2*k+1, iRL+2*kp+1] += h1e[j, jp]*(1/nleads[1])*np.exp(complex(0,2*np.pi*m/nleads[1]*(jmo % (nimp+nleads[1])) - 2*np.pi*mp/nleads[1]*(jpmo % (nimp+nleads[1]))));
                        else: assert(h1e[j,jp] == 0); # no spin flip terms in lead
                    '''
            # jp coupling to LL
            elif( j >= iLL and j < iRL and jp < iLL):

                # jp column elements go to kp
                for kp, mp in enumerate(np.append(range(-nleads[0]//2,0),range(1,nleads[0]//2+1))):
                    if(jp % 2 == 0): # up states
                        hk[j,2*kp] += h1e[j,jp]*(1/np.sqrt(nleads[0]))*np.exp(complex(0,2*np.pi*mp/nleads[0]*(jpmo % nleads[0])));
                    elif(jp % 2 == 1): # down states
                        hk[j,2*kp+1] += h1e[j,jp]*(1/np.sqrt(nleads[0]))*np.exp(complex(0,2*np.pi*mp/nleads[0]*(jpmo % nleads[0])));

            # j coupling to LL
            elif( jp >= iLL and jp < iRL and j < iLL):

                # j row elements go to k
                for k, m in enumerate(np.append(range(-nleads[0]//2,0),range(1,nleads[0]//2+1))):
                    if(j % 2 == 0): # up states
                        hk[2*k,jp] += h1e[j,jp]*(1/np.sqrt(nleads[0]))*np.exp(complex(0,2*np.pi*m/nleads[0]*(jmo % nleads[0])));
                    elif(j % 2 == 1): # down states
                        hk[2*k+1,jp] += h1e[j,jp]*(1/np.sqrt(nleads[0]))*np.exp(complex(0,2*np.pi*m/nleads[0]*(jmo % nleads[0])));

            # jp coupling to RL
            elif( j >= iLL and j < iRL and jp >= iRL):

                # jp column elements go to kp
                for kp, mp in enumerate(np.append(range(-nleads[1]//2,0),range(1,nleads[1]//2+1))):
                    if(jp % 2 == 0): # up states
                        hk[j,iRL+2*kp] += h1e[j,jp]*(1/np.sqrt(nleads[1]))*np.exp(complex(0,2*np.pi*mp/nleads[1]*(jpmo % (nimp+nleads[1]))));
                    elif(jp % 2 == 1): # down states
                        hk[j,iRL+2*kp+1] += h1e[j,jp]*(1/np.sqrt(nleads[1]))*np.exp(complex(0,2*np.pi*mp/nleads[1]*(jpmo % (nimp+nleads[1]))));

            # j coupling to RL
            elif( jp >= iLL and jp < iRL and j >= iRL):

                # j row elements go to k
                for k, m in enumerate(np.append(range(-nleads[1]//2,0),range(1,nleads[1]//2+1))):
                    if(j % 2 == 0): # up states
                        hk[iRL+2*k,jp] += h1e[j,jp]*(1/np.sqrt(nleads[1]))*np.exp(complex(0,2*np.pi*m/nleads[1]*(jmo % (nimp+nleads[1]))));
                    elif(j % 2 == 1): # down states
                        hk[iRL+2*k+1,jp] += h1e[j,jp]*(1/np.sqrt(nleads[1]))*np.exp(complex(0,2*np.pi*m/nleads[1]*(jmo % (nimp+nleads[1]))));
                        
            # pure SR states unchanged
            elif( j >= iLL and j < iRL and jp >= iLL and jp < iRL):
                hk[j,jp] += h1e[j,jp];

            else: assert(h1e[j,jp] == 0);

    # resulting ham should be real
    #assert(np.max(abs(np.imag(hk))) < 1e-10);
    return hk;

##########################################################################################################
#### solvers

def direct_FCI(h1e, h2e, norbs, nelecs, nroots = 1, verbose = 0):
    '''
    solve gd state with direct FCI
    '''

    from pyscf import fci
    
    cisolver = fci.direct_spin1.FCI();
    E_fci, v_fci = cisolver.kernel(h1e, h2e, norbs, nelecs, nroots = nroots);
    if(verbose):
        print("\nDirect FCI energies, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ",E_fci);

    return E_fci, v_fci;

def scf_FCI(mol, scf_inst, nroots = 1, verbose = 0):
    '''
    '''
    from pyscf import fci, ao2mo, gto
    import functools
    if( not isinstance(mol, gto.mol)): raise TypeError;

    # init ci solver with ham from molecule inst
    cisolver = fci.direct_uhf.FCISolver(mol);

    # get unpack from scf inst
    h1e = scf_inst.get_hcore(mol);
    norbs = np.shape(h1e)[0];
    nelecs = (mol.nelectron,0);

    # slater determinant coefficients
    mo_a = scf_inst.mo_coeff[0]
    mo_b = scf_inst.mo_coeff[1]
   
    # since we are in UHF formalism, need to split all hams by alpha, beta
    # but since everything is spin blind, all beta matrices are zeros
    h1e_a = functools.reduce(np.dot, (mo_a.T, h1e, mo_a))
    h1e_b = functools.reduce(np.dot, (mo_b.T, h1e, mo_b))
    h2e_aa = ao2mo.incore.general(scf_inst._eri, (mo_a,)*4, compact=False)
    h2e_aa = h2e_aa.reshape(norbs,norbs,norbs,norbs)
    h2e_ab = ao2mo.incore.general(scf_inst._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    h2e_ab = h2e_ab.reshape(norbs,norbs,norbs,norbs)
    h2e_bb = ao2mo.incore.general(scf_inst._eri, (mo_b,)*4, compact=False)
    h2e_bb = h2e_bb.reshape(norbs,norbs,norbs,norbs)
    h1e_tup = (h1e_a, h1e_b)
    h2e_tup = (h2e_aa, h2e_ab, h2e_bb)
    
    # run kernel to get exact energy
    E_fci, v_fci = cisolver.kernel(h1e_tup, h2e_tup, norbs, nelecs, nroots = nroots)
    if(verbose):
        print("\nFCI from UHF, zero bias, norbs = ",norbs,", nelecs = ",nelecs);
        print("- E = ", E_fci);

    return E_fci, v_fci;

##########################################################################################################
#### exec code

if __name__ == "__main__":

    pass;
