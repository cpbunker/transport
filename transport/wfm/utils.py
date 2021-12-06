'''
utils for wave function matching
'''

from transport import fci_mod

import numpy as np

##################################################################################
#### general

def E_disp(k,a,t):
    # vectorized conversion from k to E(k), measured from bottom of band
    return -2*t*np.cos(k*a);


def k_disp(E,a,t):
    return np.arccos(E/(-2*t))/a;


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
    each param is allowed to deviate from its initial val to init +/- d
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

def h_cicc_eff(J, t, i1, i2, Nsites):
    '''
    construct hams
    formalism works by
    1) having 3 by 3 block's each block is differant site for itinerant e
          H_LL T    0
          T    H_SR T
          0    T    H_RL        T is hopping between leads and scattering region
    2) all other dof's encoded into blocks

    Args:
    - J, float, eff heisenberg coupling
    - t, float, hopping btwn sites
    = i1, int, site of 1st imp
    - i2, int, site of 2nd imp
    - Nsites, int, total num sites in SR
    '''
    
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
    tl_arr = []
    for sitei in range(Nsites-1):
        tl_arr.append(-t*np.eye(*np.shape(Se_dot_S1)) );
    tl_arr = np.array(tl_arr);

    return h_cicc, tl_arr;

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

    return h1e, g2e;

