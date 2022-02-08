'''
utils for wave function matching
'''

from transport import fci_mod

import numpy as np

##################################################################################
#### general

def subspace(m):

    if(m==1/2):
        # pick out m=1/2 subspace
        picks = [[0,2,9],[0,3,8],[0,4,7],[0,5,6],[1,2,8],[1,3,7],[1,4,6]]; 
        pickis = [3,6,9,12,18,21,24]; 
        pickstrs = ["|up, 3/2, -3/2>","|up, 1/2, -1/2>","|up, -1/2, 1/2>","|up, -3/2, 3/2>","|down, 3/2, -1/2>","|down, 1/2, 1/2>","|down, -1/2, 3/2>"];

    elif(m==3/2):
        # pick out m=3/2 subspace
        picks = [[0,2,8],[0,3,7],[0,4,6],[1,2,7],[1,3,6]];
        pickis = [2, 5, 8, 17, 20,];
        pickstrs = ["|up, 3/2, -1/2>", "|up, 1/2, 1/2>", "|up, -1/2, 3/2>","|down, 3/2, 1/2>","|down, 1/2, 3/2>"];

    elif(m==5/2):
        picks = [[0,2,7],[0,3,6],[1,2,6]];
        pickis = [1, 4, 16];
        pickstrs = ["|up, 3/2, 1/2>", "|up, 1/2, 3/2>","|down, 3/2, 3/2>"];

    else:
        raise(ValueError);

    return picks, pickis, pickstrs;


def entangle(H,bi,bj):
    '''
    Perform a change of basis on a matrix such that basis vectors bi, bj become entangled (unentangled)
    in new ham, index bi -> + entangled state, bj -> - entangled state
    '''

    # check inputs
    assert(bi < bj);
    assert(bj < max(np.shape(H)));

    # rotation matrix
    R = np.zeros_like(H);
    for i in range(np.shape(H)[0]):
        for j in range(np.shape(H)[1]):
            if( i != bi and i != bj):
                if(i == j):
                    R[i,j] = 1; # identity
            elif( i == bi and j == bi):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j == bj):
                R[i,j] = -1/np.sqrt(2);
            elif( i == bi and j == bj):
                R[i,j] = 1/np.sqrt(2);
            elif( i == bj and j== bi):
                R[i,j] = 1/np.sqrt(2);

    return np.matmul(np.matmul(R.T,H),R);


def sweep_pairs(dets, sourcei):
    '''
    Given a list of all the determinants in the problem and a single source
    determinant, make a list of all the other state pairs to entangle
    Pairs must have diff electron spin from source, but same as each other!
    '''

    # return value
    result = [];

    for i in range(len(dets)):
        for j in range(len(dets)):
            if(i != sourcei and j != sourcei and i < j): # distinct pair
                # high spin pairs only
                if(True):
                #if(dets[i][1] in [2,5] and dets[j][1] in [2,5] and dets[i][2] in [6,9] and dets[j][2] in [6,9]):
                    if((dets[sourcei][0] != dets[i][0]) and (dets[i][0] == dets[j][0])):
                        result.append((i,j));

    return result;


def sweep_param_space(ps, d, n):
    '''
    Given initial vals of params, make a list covering all mesh points
    in the param space (ps)
    each param is allowed to deviate from its initial val to init*(1 +/- d)
    total of n points in the sweep

    Returns 2d list of combos
    '''

    # check inputs
    assert(d > 0);
    assert(isinstance(n, int));

    # recursive
    if(len(ps) > 2): # truncate param space until 2 by 2
        result = []
        inner = sweep_param_space(ps[1:], d, n);
        for pi in np.linspace(ps[0]*(1 - d), ps[0]*(1 + d), n):
            for el in inner:
                result.append([pi, *tuple(el) ]);
    else: # do 2 by 2 directly
        result = [];
        for pi in np.linspace(ps[0]*(1 - d), ps[0]*(1 + d), n):
            for pj in np.linspace(ps[1]*(1 - d), ps[1]*(1 + d), n):
                result.append([pi,pj]);
                
    return result;                   


##################################################################################
#### functions specific to a model

def h_cicc_eff(J, t, i1, i2, Nsites, Jz = True):
    '''
    Construct tight binding blocks (each block has many body dofs) to implement
    cicc model in quasi many body GF method

    Args:
    - J, float, eff heisenberg coupling
    - t, float, hopping btwn sites - corresponds to t' in my setup
    = i1, int, site of 1st imp
    - i2, int, site of 2nd imp
    - Nsites, int, total num sites in SR
    - Jz, bool, whether to include diagonal (Jz Se^z Si^z) terms
    '''

    # check inputs
    assert(i1 < i2);
    assert(i2 < Nsites);
    
    # heisenberg interaction matrices
    Se_dot_S1 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to first spin impurity
                        [0,1,0,0,0,0,0,0],
                        [0,0,-1,0,2,0,0,0],
                        [0,0,0,-1,0,2,0,0],
                        [0,0,2,0,-1,0,0,0],
                        [0,0,0,2,0,-1,0,0],
                        [0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,1] ]);

    Se_dot_S2 = (J/4.0)*np.array([ [1,0,0,0,0,0,0,0], # coupling to second spin impurity
                        [0,-1,0,0,2,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,-1,0,0,2,0],
                        [0,2,0,0,-1,0,0,0],
                        [0,0,0,0,0,1,0,0],
                        [0,0,0,2,0,0,-1,0],
                        [0,0,0,0,0,0,0,1] ]);

    # insert these local interactions
    h_cicc =[];
    for sitei in range(Nsites): # iter over all sites
        if(sitei == i1):
            h_cicc.append(Se_dot_S1);
        elif(sitei == i2):
            h_cicc.append(Se_dot_S2);
        else:
            h_cicc.append(np.zeros_like(Se_dot_S1) );
    h_cicc = np.array(h_cicc);

    # hopping connects like spin orientations only, ie is identity
    tblocks = []
    for sitei in range(Nsites-1):
        tblocks.append(-t*np.eye(*np.shape(Se_dot_S1)) );
    tblocks = np.array(tblocks);

    return h_cicc, tblocks;

def h_cicc_hacked(J,t,N, dimer = False):
    '''
    Version of h_cicc_eff
    '''

    # J is anisotropic
    Jz = 0;
    
    # heisenberg interaction matrices
    Se_dot_S1 = (1/4.0)*np.array([ [Jz,0,0,0,0,0,0,0], # coupling to first spin impurity
                        [0,Jz,0,0,0,0,0,0],
                        [0,0,-Jz,0,2*J,0,0,0],
                        [0,0,0,-Jz,0,2*J,0,0],
                        [0,0,2*J,0,-Jz,0,0,0],
                        [0,0,0,2*J,0,-Jz,0,0],
                        [0,0,0,0,0,0,Jz,0],
                        [0,0,0,0,0,0,0,Jz] ]);

    Se_dot_S2 = (1/4.0)*np.array([ [Jz,0,0,0,0,0,0,0], # coupling to second spin impurity
                        [0,-Jz,0,0,2*J,0,0,0],
                        [0,0,Jz,0,0,0,0,0],
                        [0,0,0,-Jz,0,0,2*J,0],
                        [0,2*J,0,0,-Jz,0,0,0],
                        [0,0,0,0,0,Jz,0,0],
                        [0,0,0,2*J,0,0,-Jz,0],
                        [0,0,0,0,0,0,0,Jz] ]);

    if(dimer): # Se dot S1 then Se dot S2
        assert(N==4);
        h_cicc = np.array([np.zeros_like(Se_dot_S1), Se_dot_S1, Se_dot_S2, np.zeros_like(Se_dot_S1)]);
        tblocks = np.array([-t*np.eye(*np.shape(Se_dot_S1)),-t*np.eye(*np.shape(Se_dot_S1)),-t*np.eye(*np.shape(Se_dot_S1))]);
        return h_cicc, tblocks;
    
    h_cicc =[];
    for sitei in range(N): # iter over all sites
        if(sitei > 0 and sitei < N - 1):
            h_cicc.append(Se_dot_S1 + Se_dot_S2);
        else:
            h_cicc.append(np.zeros_like(Se_dot_S1) );

    tblocks = []
    for sitei in range(N-1):
        tblocks.append(-t*np.eye(*np.shape(Se_dot_S1)) );

    return np.array(h_cicc), np.array(tblocks);


def h_kondo_2e(J,s2):
    '''
    Kondo interaction between spin 1/2 and spin s2
    '''

    # m2 states
    ms = [];
    m2 = s2;
    while(m2 >= -s2):
        ms.append(m2);
        m2 -= 1;

    assert( len(ms) == 2*s2+1);
    Nstates = 2 + len(ms);
    h = np.zeros((Nstates,Nstates,Nstates,Nstates));

    if(s2 == 0.5):

        # S \pm parts
        h[0,1,3,2] = 2;
        h[3,2,0,1] = 2;
        h[1,0,2,3] = 2;
        h[2,3,1,0] = 2;

        # Sz parts
        h[0,0,2,2] = 1;
        h[2,2,0,0] = 1;
        h[0,0,3,3] = -1;
        h[3,3,0,0] = -1;
        h[1,1,2,2] = -1;
        h[2,2,1,1] = -1;
        h[1,1,3,3] = 1;
        h[3,3,1,1] = 1;

        # scale with J
        h = (J/4.0)*h;

    elif(s2 == 1.0):

        # S \pm parts
        h[2,3,1,0] = 1/np.sqrt(2);
        h[1,0,2,3] = 1/np.sqrt(2);
        h[3,2,0,1] = 1/np.sqrt(2);
        h[0,1,3,2] = 1/np.sqrt(2);
        h[3,4,1,0] = 1/np.sqrt(2);
        h[1,0,3,4] = 1/np.sqrt(2);
        h[4,3,0,1] = 1/np.sqrt(2);
        h[0,1,4,3] = 1/np.sqrt(2);

        # Sz parts
        h[2,2,0,0] = 1/2;
        h[0,0,2,2] = 1/2;
        h[2,2,1,1] = -1/2;
        h[1,1,2,2] = -1/2;
        h[4,4,0,0] = -1/2;
        h[0,0,4,4] = -1/2;
        h[4,4,1,1] = 1/2;
        h[1,1,4,4] = 1/2;

        # scale with J
        h = J*h;

    else: raise Exception;

    return h;


def h_switzer(D1, D2, JH, JK1, JK2):
    '''
    Eric's model for spin coupling of itinerant spin 1/2 to two spin 1
    impurities, in second quantized form
    '''

    h = np.zeros((8,8));
    g = np.zeros((8,8,8,8));

    # spin anisotropy
    h[2,2] = D1;
    h[4,4] = D1;
    h[5,5] = D2;
    h[7,7] = D2;

    # heisenberg
    g[2,3,6,5] += JH;
    g[6,5,2,3] += JH;
    g[2,3,7,6] += JH;
    g[7,6,2,3] += JH;
    g[3,2,5,6] += JH;
    g[5,6,3,2] += JH;
    g[3,2,6,7] += JH;
    g[6,7,3,2] += JH;
    g[3,4,6,5] += JH;
    g[6,5,3,4] += JH;
    g[3,4,7,6] += JH;
    g[7,6,3,4] += JH;
    g[4,3,5,6] += JH;
    g[5,6,3,4] += JH;
    g[4,3,6,7] += JH;
    g[6,7,4,3] += JH;
    g[2,2,5,5] += JH;
    g[5,5,2,2] += JH;
    g[2,2,7,7] += -JH;
    g[7,7,2,2] += -JH;
    g[4,4,5,5] += -JH;
    g[5,5,4,4] += -JH;
    g[4,4,7,7] += JH;
    g[7,7,4,4] += JH;

    # K1
    g[2,3,1,0] += JK1/np.sqrt(2);
    g[1,0,2,3] += JK1/np.sqrt(2);
    g[3,2,0,1] += JK1/np.sqrt(2);
    g[0,1,3,2] += JK1/np.sqrt(2);
    g[3,4,1,0] += JK1/np.sqrt(2);
    g[1,0,3,4] += JK1/np.sqrt(2);
    g[4,3,0,1] += JK1/np.sqrt(2);
    g[0,1,4,3] += JK1/np.sqrt(2);
    g[2,2,0,0] += JK1/2;
    g[0,0,2,2] += JK1/2;
    g[2,2,1,1] += -JK1/2;
    g[1,1,2,2] += -JK1/2;
    g[4,4,0,0] += -JK1/2;
    g[0,0,4,4] += -JK1/2;
    g[4,4,1,1] += JK1/2;
    g[1,1,4,4] += JK1/2;

    # K2
    g[5,6,1,0] += JK2/np.sqrt(2);
    g[1,0,5,6] += JK2/np.sqrt(2);
    g[6,5,0,1] += JK2/np.sqrt(2);
    g[0,1,6,5] += JK2/np.sqrt(2);
    g[6,7,1,0] += JK1/np.sqrt(2);
    g[1,0,6,7] += JK2/np.sqrt(2);
    g[7,6,0,1] += JK2/np.sqrt(2);
    g[0,1,7,6] += JK2/np.sqrt(2);
    g[5,5,0,0] += JK2/2;
    g[0,0,5,5] += JK2/2;
    g[5,5,1,1] += -JK2/2;
    g[1,1,5,5] += -JK2/2;
    g[7,7,0,0] += -JK2/2;
    g[0,0,7,7] += -JK2/2;
    g[7,7,1,1] += JK2/2;
    g[1,1,7,7] += JK2/2;

    return h, g;


def h_dimer_2q(params):
    '''
    Generate second quantized form of the Co dimer spin hamiltonian

    Returns:
    h1e, one body part of second quantized ham
    g2e, two body part of second quantized ham
    '''

    # basis size
    Nb = 2+8; # e + 4 m states each imp

    # unpack params
    Jx, Jy, Jz, DO, DT, An, JK1, JK2 = params;

    # 1 particle terms
    h1e = np.zeros((Nb, Nb), dtype = complex);

    # octo spin anisitropy
    h1e[2,2] += DO*9/4;
    h1e[3,3] += DO*1/4;
    h1e[4,4] += DO*1/4;
    h1e[5,5] += DO*9/4;

    # tetra spin anisotropy
    h1e[6,6] += DT*9/4;
    h1e[7,7] += DT*1/4;
    h1e[8,8] += DT*1/4;
    h1e[9,9] += DT*9/4;

    # 2 particle terms
    g2e = np.zeros((Nb,Nb,Nb,Nb), dtype = complex);

    # isotropic terms
    xOcoefs = np.array([np.sqrt(3),np.sqrt(3),2,2,np.sqrt(3),np.sqrt(3)])/2;
    xOops = [(2,3),(3,2),(3,4),(4,3),(4,5),(5,4)];
    xTcoefs = np.copy(xOcoefs);
    xTops = [(6,7),(7,6),(7,8),(8,7),(8,9),(9,8)];
    g2e = fci_mod.terms_to_g2e(g2e, xOops, Jx*xOcoefs, xTops, xTcoefs);

    yOcoefs = complex(0,1)*np.array([-np.sqrt(3),np.sqrt(3),-2,2,-np.sqrt(3),np.sqrt(3)])/2;
    yOops = xOops;
    yTcoefs = np.copy(yOcoefs);
    yTops = xTops;
    g2e = fci_mod.terms_to_g2e(g2e, yOops, Jy*yOcoefs, yTops, yTcoefs);

    zOcoefs = np.array([3,1,-1,3])/2;
    zOops = [(2,2),(3,3),(4,4),(5,5)];
    zTcoefs = np.copy(zOcoefs);
    zTops = [(6,6),(7,7),(8,8),(9,9)];
    g2e = fci_mod.terms_to_g2e(g2e, zOops, Jz*zOcoefs, zTops, zTcoefs);

    # anisotropic terms
    g2e = fci_mod.terms_to_g2e(g2e, xOops, An*xOcoefs, zTops, zTcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, yOops,-An*yOcoefs, zTops, zTcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, zOops, -An*zOcoefs, xTops, xTcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, zOops, An*zOcoefs, yTops, yTcoefs);

    # Kondo terms
    xeops = [(0,1),(1,0)];
    xecoefs = np.array([1,1])/2;
    g2e = fci_mod.terms_to_g2e(g2e, xeops, JK1*xecoefs, xOops, xOcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, xeops, JK2*xecoefs, xTops, xTcoefs);
    yeops = [(0,1),(1,0)];
    yecoefs = complex(0,1)*np.array([-1,1])/2;
    g2e = fci_mod.terms_to_g2e(g2e, yeops, JK1*yecoefs, yOops, yOcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, yeops, JK2*yecoefs, yTops, yTcoefs);
    zeops = [(0,0),(1,1)];
    zecoefs = np.array([1,-1])/2;
    g2e = fci_mod.terms_to_g2e(g2e, zeops, JK1*zecoefs, zOops, zOcoefs);
    g2e = fci_mod.terms_to_g2e(g2e, zeops, JK2*zecoefs, zTops, zTcoefs);

    # check hermicity
    assert(not np.any(h1e - np.conj(h1e.T)));
    assert(not np.any(g2e - np.conj(g2e.T)));
    return h1e, g2e;

