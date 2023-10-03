#### dmrg utils
import numpy as np
import itertools

from transport import fci_mod

def print_H_alpha(H):
    assert(len(np.shape(H)) == 4);
    numj = np.shape(H)[0];
    for i in range(numj):
        for j in [max(0,i-1),i,min(numj-1,i+1)]:
            print("H["+str(i)+","+str(j)+"] =\n",np.real(H[i,j,:,:]));

def get_h1e(n_mols,s_mols,spatial_orbs,mytm, myB_mm, myB_elec, myJH, myJK, my_chiral, verbose = 0):
    '''
    make the 1body and 2body parts of the 2nd qu'd ham
    The Ham is Kumar 2017 Eqs (1)-(3)
    We only have 1 electron so all the interactions are 1body
    Electron has n_loc_dof channels which account for mol spin dofs as well
    Args:
    -mytm, hopping between mols
    -myB_mm, zeeman strength for the mol spins. Contains g*\mu_B !
    -myB_elec, zeeman strength for the electron. Contains g*\mu_B !
    -myJH, Heisenberg exchange between mol spins
    -myJK, Kondo exchange between elec and mol spins
    -my_chiral, strength of chiral operator acting on mol spins,
       should benonzero to lift chiral degeneracy, which avoids numerical issues
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    
    # return var
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    n_loc_dof = 2*len(mol_states);
    h1e = np.zeros((spatial_orbs, spatial_orbs, n_loc_dof, n_loc_dof),dtype=complex);

    #### fermionic terms

    # hacky code to break fermion spatial symetry
    for loci in range(n_loc_dof):
        h1e[1,1,loci,loci] += -0.0;

    # spin-independent hopping between n.n. sys orbs
    for sysi in range(spatial_orbs-1):
        # iter over local dofs (up, down, etc)
        for loci in range(n_loc_dof):
            h1e[sysi,sysi+1,loci,loci] += -mytm; 
            h1e[sysi+1,sysi,loci,loci] += -mytm;
    if(spatial_orbs > 2): # last to first hopping
        for loci in range(n_loc_dof): 
            h1e[0,-1,loci,loci] += -mytm; 
            h1e[-1,0,loci,loci] += -mytm;

    # fermionic Zeeman
    for sysi in range(spatial_orbs):
        h1e += myB_elec*get_sigz(n_loc_dof, spatial_orbs, sysi);

    #### spin terms

    # chiral breaking
    h1e += my_chiral*get_chiral_op(n_mols,s_mols,spatial_orbs);

    # spin Zeeman terms
    if(verbose): print("Zeeman"); 
    for sysi in range(spatial_orbs):
        # have to iter over local dofs paticle-by-particle
        # iter over all (2s+1)^n_mols many-body mol spin states
        for mol_statei in range(len(mol_states)):
            Sztot = sum(mol_states[mol_statei]);
            # iter over electron spin 
            for sigma in range(2):
                loci = 2*mol_statei+sigma;
                h1e[sysi,sysi,loci,loci] += myB_mm*Sztot;
            if(verbose>1 and sysi==0): print("->",2*mol_statei,mol_states[mol_statei],'->',Sztot);

    # Heisenberg - regardless of elec location, couples mol spins
    if(verbose): print("Heisenberg"); 
    for mola in range(n_mols):
        for molb in range(n_mols):
            if(molb-mola==1 or (mola==n_mols-1 and molb==0)): # nn only
                h1e += myJH*get_SaSb(n_mols, s_mols, spatial_orbs, mola, molb, verbose=verbose);

    # Kondo exchange - couples elec to molecule it is on
    if(n_mols != spatial_orbs): assert(myJK == 0); raise Exception; return h1e;
    if(verbose): print("Kondo");
    for mola in range(n_mols):
        h1e += myJK*get_SaSigb(n_mols, s_mols, spatial_orbs, mola, mola, verbose=verbose);

    # return
    return h1e;

def get_occ(n_loc_dof,spatial_orbs,aindex):
    '''
    Get the operator for the occupancy of site a
    '''
    occ = np.zeros((spatial_orbs,spatial_orbs,n_loc_dof,n_loc_dof));
    for spacei in range(spatial_orbs):
        if(spacei == aindex):
            occ[spacei,spacei] += np.eye(n_loc_dof);

    return occ;

def get_sigz(n_loc_dof,spatial_orbs,aindex):
    '''
    Get the operator for the electron z spin of site a
    '''
    sigz = np.zeros((spatial_orbs,spatial_orbs,n_loc_dof,n_loc_dof));
    for spacei in range(spatial_orbs):
        if(spacei == aindex):
            diag = [1 if i%2==0 else -1 for i in range(n_loc_dof)];
            sigz[spacei,spacei] += np.diagflat(diag)/2;

    return sigz;
 

def get_SaSb(n_mols,s_mols,spatial_orbs,aindex,bindex, verbose = 0):
    '''
    Get the operator mol spin S_a dotted into mol spin S_b
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    
    # construct as 4d in the spatial orbs, mol_states basis
    SaSb = np.zeros((spatial_orbs,spatial_orbs,2*len(mol_states),2*len(mol_states)));

    # iter over many-body mol spin states twice
    for mol_statei in range(len(mol_states)):
        for mol_statej in range(len(mol_states)):
            # difference between states
            n_different = np.count_nonzero(mol_states[mol_statei]-mol_states[mol_statej]);
            if(n_different in [0,2]):
                               
                # quantum numbers
                Szi_a = mol_states[mol_statei][aindex];
                Szi_b = mol_states[mol_statei][bindex];
                Szj_a = mol_states[mol_statej][aindex];
                Szj_b = mol_states[mol_statej][bindex];

                # S^z_a S^z_b - couples state to itself
                if(mol_statei == mol_statej):
                    # add term to diag of all spatial blocks
                    for spacei in range(spatial_orbs):
                        # for both elec spins
                        for sigma in [0,1]:
                            SaSb[spacei,spacei,2*mol_statei+sigma,2*mol_statej+sigma] += Szi_a*Szi_b;

                # S^+_a S^-_b couples spin flipped states
                if(Szi_a - Szj_a==1 and Szi_b-Szj_b==-1):
                    # add term to diag of all spatial blocks
                    for spacei in range(spatial_orbs):
                        if(verbose>1 and spacei==0): print("-> S_"+str(aindex)+" S_"+str(bindex)); print("->",2*mol_statei,mol_states[mol_statei],"->",2*mol_statej,mol_states[mol_statej],'->',1/2); 
                        # for both elec spins
                        for sigma in [0,1]:
                            SaSb[spacei,spacei,2*mol_statei+sigma,2*mol_statej+sigma] += (1/2);
                            # hc
                            SaSb[spacei,spacei,2*mol_statej+sigma,2*mol_statei+sigma] += (1/2);

    # return                       
    return SaSb;

def get_SaSigb(n_mols, s_mols, spatial_orbs, aindex, bindex, verbose=0):
    '''
    Get the operator mol spin S_a dotted into elec spin sigma on site b
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    
    # construct as 4d in the spatial orbs, mol_states basis
    SaSigb = np.zeros((spatial_orbs,spatial_orbs,2*len(mol_states),2*len(mol_states)));

    # iter over many-body mol spin states twice
    for mol_statei in range(len(mol_states)):
        for mol_statej in range(len(mol_states)):
            # difference between states
            n_different = np.count_nonzero(mol_states[mol_statei]-mol_states[mol_statej]);
            if(n_different in [0,1]):
                
                # S^z - couples state to itself
                if(mol_statei == mol_statej):
                    for sigma in range(2):
                        loci = 2*mol_statei+sigma;
                        SaSigb[aindex,bindex,loci,loci] += mol_states[mol_statei][aindex]*(1/2-sigma);
                        if(verbose>1 and aindex==0 and bindex==0): print("-> S_"+str(aindex)+" \sigma_"+str(bindex)); print("->",loci,mol_states[mol_statei],(1/2-sigma),'->',mol_states[mol_statei][aindex]*(1/2-sigma));

                # S^+ - couples statei to statej with moli flipped up by one
                if(mol_states[mol_statei][aindex]+1 == mol_states[mol_statej][aindex]):
                    # all other have to be the same
                    if(n_different == 1):
                        # couple statei with elec up to statej wth elec down
                        SaSigb[aindex,bindex,2*mol_statei,2*mol_statej+1] += (1/2);
                        # hc
                        SaSigb[aindex,bindex,2*mol_statej+1,2*mol_statei] += (1/2);
                        if(verbose>1 and aindex==0 and bindex==0): print("-> S_"+str(aindex)+" \sigma_"+str(bindex)); print("->",2*mol_statei+0,mol_states[mol_statei],0.5,"->",2*mol_statej+1,mol_states[mol_statej],-0.5,'->',1/2);                   

    # return
    return SaSigb;

def get_chiral_op(n_mols,s_mols, spatial_orbs):
    '''
    get the operator S_1 \cdot (S_2 \times S_3)
    '''
    assert n_mols == 3;
    assert s_mols == 1/2;
    hilbert_space = int((2*s_mols+1)**n_mols);
    chiral_op = np.zeros((hilbert_space,hilbert_space),dtype=complex);

    # define spin ops
    Sx = (1/2)*np.array([[0,1],[1,0]],dtype=complex);
    Sy = (1/2)*np.array([[0,-complex(0,1)],[complex(0,1),0]],dtype=complex);
    Sz = (1/2)*np.array([[1,0],[0,-1]],dtype=complex);

    # add in tensor products
    S2yS3z = fci_mod.mat_4d_to_2d(np.tensordot(Sy,Sz,axes=0));
    S2zS3y = fci_mod.mat_4d_to_2d(np.tensordot(Sz,Sy,axes=0));
    cross_x = fci_mod.mat_4d_to_2d(np.tensordot(Sx,S2yS3z-S2zS3y,axes=0));
    S2xS3z = fci_mod.mat_4d_to_2d(np.tensordot(Sx,Sz,axes=0));
    S2zS3x = fci_mod.mat_4d_to_2d(np.tensordot(Sz,Sx,axes=0));
    cross_y = fci_mod.mat_4d_to_2d(np.tensordot(Sy,S2xS3z-S2zS3x,axes=0));
    S2xS3y = fci_mod.mat_4d_to_2d(np.tensordot(Sx,Sy,axes=0));
    S2yS3x = fci_mod.mat_4d_to_2d(np.tensordot(Sy,Sx,axes=0));
    cross_z = fci_mod.mat_4d_to_2d(np.tensordot(Sz,S2xS3y-S2yS3x,axes=0));
    chiral_op += cross_x - cross_y + cross_z;

    # convert from (2*s_mols+1)^n_mols dimensional to full dimensionality
    chiral_op_sigma = np.zeros((spatial_orbs,spatial_orbs,2*hilbert_space,2*hilbert_space),dtype=complex);
    # iter over spatial
    for spacei in range(spatial_orbs):
        # iter over mol dimensionality
        for oldi in range(hilbert_space):
            for oldj in range(hilbert_space):
                # add in spin block
                for sigma in [0,1]:
                    chiral_op_sigma[spacei,spacei,2*oldi+sigma,2*oldj+sigma] += chiral_op[oldi,oldj];

    # return
    return chiral_op_sigma;

def get_Stot2(n_mols,s_mols, spatial_orbs):
    '''
    Get the total spin operator squared
    '''
    assert s_mols == 1/2; # have to add factors of sqrt(s) on S^\pm otherwise!
    mol_projections = tuple(np.linspace(-s_mols,s_mols,int(2*s_mols+1))[::-1]);
    mol_states = np.array([x for x in itertools.product(*(n_mols*(mol_projections,)))]);
    
    # construct as 4d in the spatial orbs, mol_states basis
    Stot2 = np.zeros((spatial_orbs,spatial_orbs,2*len(mol_states),2*len(mol_states)));

    # add mol spins squared
    S1 = get_SaSb(s_mols, n_mols, spatial_orbs, 0, 0)
    Stot2 += np.matmul(S1,S1);
    S2 = get_SaSb(s_mols, n_mols, spatial_orbs, 1, 1)
    Stot2 += np.matmul(S2,S2);
    S3 = get_SaSb(s_mols, n_mols, spatial_orbs, 2, 2)
    Stot2 += np.matmul(S3,S3);

    # add in unique spin-spin correlation
    for mola in range(n_mols):
        for molb in range(n_mols):
            if(mola < molb):
                Stot2 += get_SaSb(n_mols,s_mols,spatial_orbs,mola,molb);

    


