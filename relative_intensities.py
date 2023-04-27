'''
Compare different methods for determining the relative intensities
of the dI/dV steps from DFT params
'''

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 4, suppress = True);

def get_spin_ops(s):
    '''
    construct the spin-s operators according to
    https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
    '''
    n_loc_dof = int(2*s+1);
    S0 = np.eye(n_loc_dof,dtype=complex);
    identity = np.eye(2*n_loc_dof);

    # spin operators
    Sx = np.zeros_like(S0);
    Sy = np.zeros_like(S0);
    Sz = np.zeros_like(S0);
    for a in range(1,1+n_loc_dof):
        for b in range(1,1+n_loc_dof):
            Sx[a-1,b-1] = (1/2)*(identity[a,b+1]+identity[a+1,b])*np.lib.scimath.sqrt((s+1)*(a+b-1)-a*b);
            Sy[a-1,b-1] = complex(0,1/2)*(identity[a,b+1]-identity[a+1,b])*np.lib.scimath.sqrt((s+1)*(a+b-1)-a*b);
            Sz[a-1,b-1] = (s+1-a)*identity[a,b];

    if False:
        Sops = [Sx,Sy,Sz];
        for op in Sops: print(op);

    return Sx, Sy, Sz;

def get_spin_Hamiltonian(s,Bx,By,Bz,D,E):
    '''
    for a spin-s particle w/ 2s+1 spin dofs, get a 2s+1 x 2s+1 matrix
    for the spin Hamiltonian, parameterized in the conventional form
    H = g*mu_B* B \cdot S + D S_z^2 + E(S_x^2 - S_y^2)
    '''
    n_loc_dof = int(2*s+1);
    Sx, Sy, Sz = get_spin_ops(s);

    # construct Ham
    return Bx*Sx + By*Sy + Bz*Sz + D*np.matmul(Sz,Sz) + E*(np.matmul(Sx,Sx)-np.matmul(Sy,Sy));

def get_intensities_CG(s,j,eigvecs,verbose=0):
    '''
    transition intensities, a la Guayacq
    '''
    # expand the S_z eigenstates in terms of Ham eiegenstates
    # total spin mag is j = Fe spin mag (2) + elec spin mag (1/2) = 5/2
    C_n_mmol = np.zeros((int(2*s+1),int(2*s+1)), dtype=complex); # expansion coefs
    for nindex in range(int(2*s+1)): # eigenstate index
        for mmolindex in range(int(2*s+1)): # Sz index
            mmol_state = np.zeros_like(eigvecs.T[0]);
            mmol_state[mmolindex] = 1.0;
            C_n_mmol[nindex,mmolindex] = np.dot(np.conj(eigvecs.T[nindex]),mmol_state);
     
    # clebsch gordon coefficients
    if(s==2):
        # total spin z projection mj = +5/2,...-5/2
        # CGs are a function of mT and me
        CGs = np.array([[1.0, 0.0],                   # mT = 5/2
                        [np.sqrt(4/5), np.sqrt(1/5)], # mT = 3/2
                        [np.sqrt(3/5), np.sqrt(2/5)], # mT = 1/2
                        [np.sqrt(2/5), np.sqrt(3/5)], # mT =-1/2
                        [np.sqrt(1/5), np.sqrt(4/5)], # mT =-3/2
                        [0.0,1.0]]);                  # mT =-5/2
    elif(s==5/2):
        # total spin z projection mj = +3,...-3
        # CGs are a function of mT and me
        CGs = np.array([[1.0, 0.0],                   # mT = 3
                        [np.sqrt(5/6), np.sqrt(1/6)], # mT = 2
                        [np.sqrt(2/3), np.sqrt(1/3)], # mT = 1
                        [np.sqrt(1/2), np.sqrt(1/2)], # mT = 0
                        [np.sqrt(1/3), np.sqrt(2/3)], # mT =-1
                        [np.sqrt(1/6), np.sqrt(5/6)], # mT =-2
                        [0.0,1.0]]);                  # mT =-3
    else:
        raise NotImplementedError;

    # transition intensities combine A_n_mmol and CGs
    A_n_me_mT = np.zeros((int(2*s+1),2,int(2*j+1)),dtype=complex);
    for nindex in range(np.shape(A_n_me_mT)[0]):
        for meindex in range(np.shape(A_n_me_mT)[1]):
            for mTindex in range(np.shape(A_n_me_mT)[2]):
                meval = 1/2-1*meindex;
                mTval = j-1*mTindex;
                mmolval = mTval-meval;
                mmolindex = int( (j-1/2)-mmolval);
                if(verbose): print(meval, mTval, mmolval, mmolindex);
                if(abs(mmolval) > s): # unphysical combo
                    assert(CGs[mTindex,meindex] == 0.0);
                    A_n_me_mT[nindex,meindex,mTindex] = 0.0; 
                else:
                    dummy = A_n_me_mT[nindex,meindex,mTindex]
                    dummy = CGs[mTindex,meindex]
                    dummy = C_n_mmol[nindex,mmolindex]
                    A_n_me_mT[nindex,meindex,mTindex] = CGs[mTindex,meindex]*C_n_mmol[nindex,mmolindex];  
        #print(A_n_me_mT[0].T);

    # then we sum over the electron spin states
    W_0f = np.zeros((len(eigvecs.T[0]),),dtype=float);
    for findex in range(len(eigvecs.T[0])):
        sum_me = 0;
        for meindex in range(2):
            for mejndex in range(2):
                sum_mT = 0;
                for mTindex in range(int(2*j+1)):
                    sum_mT += A_n_me_mT[0,meindex,mTindex]*A_n_me_mT[findex,mejndex,mTindex];
                sum_me += np.real(np.conj(sum_mT)*sum_mT);
        W_0f[findex] = sum_me;

    return W_0f/sum(W_0f);

#####################################################
#### results

if False: # Gauyacq Fe results
    # spin-2 Fe eigenvectors
    # see Heinrich Large Magnetic Anisotropy Tab 1
    # all energies in meV
    mu_Bohr = 5.788*1e-2; # bohr magneton in meV/Tesla
    gfactor = 2.11;
    myD = -1.55;
    myE = 0.31;
    mys = 2;
    myj = mys+1/2;
    print("Model of a spin-"+str(mys)+" atom with D = "+str(myD)+" meV and E = "+str(myE)+" meV"); 

    # intensities vs B
    direction = 'N'; # b applied along N ie z axis, as in Gauyacq Figs 2 and 3
    Bvals = np.linspace(0,7,8);
    for Bvali in range(len(Bvals)):
        Bval = Bvals[Bvali];
        if direction == 'N': # z axis
            myBx, myBy, myBz = 0.0,0.0,gfactor*mu_Bohr*Bval;
        else:
            raise NotImplementedError;
        print("-B = "+str(Bval)+" Tesla in the "+direction+" direction");

        # get eigvecs
        Fe_ham = get_spin_Hamiltonian(mys, myBx, myBy, myBz, myD, myE);
        myeigvals, myeigvecs = np.linalg.eigh(Fe_ham);
        #print("- eigenvectors:\n",myeigvecs.T);

        # get intensities
        Ws = get_intensities_CG(mys,myj,myeigvecs);
        if(Bval == 0.0 and True):
            print("\nGuayacq transition intensities from the ground state:\n",Ws);
            print("Normalized:\n",Ws/Ws[0]);
            steps_CG = np.zeros_like(Ws);
            for stepi in range(len(steps)):
                steps[stepi] += sum( (Ws/Ws[0])[0:stepi+1]);
            print("Steps, if zero bias value is 0.01 (Guayacg Fig 2b):\n",0.01*steps);
        assert False;

if True: # Gauyacq Mn results
    # all energies in meV
    mu_Bohr = 5.788*1e-2; # bohr magneton in meV/Tesla
    gfactor = 1.90;
    myD = -0.039;
    myE = 0.007;
    mys = 5/2;
    myj = mys+1/2;
    print("Model of a spin-"+str(mys)+" atom with D = "+str(myD)+" meV and E = "+str(myE)+" meV"); 

    # intensities vs B
    direction = 'N'; # b applied along N ie z axis, as in Gauyacq Figs 2 and 3
    Bvals = np.linspace(0,7,8);
    for Bvali in range(len(Bvals)):
        Bval = Bvals[Bvali];
        if direction == 'N': # z axis
            myBx, myBy, myBz = 0.0,0.0,gfactor*mu_Bohr*Bval;
        else:
            raise NotImplementedError;
        print("-B = "+str(Bval)+" Tesla in the "+direction+" direction");

        # get eigvecs
        Fe_ham = get_spin_Hamiltonian(mys, myBx, myBy, myBz, myD, myE);
        myeigvals, myeigvecs = np.linalg.eigh(Fe_ham);
        #print("- eigenvectors:\n",myeigvecs.T);

        # get intensities
        Ws = get_intensities_CG(mys,myj,myeigvecs);
        if(Bval == 0.0 and False):
            print("\nGuayacq transition intensities from the ground state:\n",Ws);
            print("Normalized:\n",Ws/Ws[0]);
            steps_CG = np.zeros_like(Ws);
            for stepi in range(len(steps)):
                steps_CG[stepi] += sum( (Ws/Ws[0])[0:stepi+1]);
            print("Steps, if zero bias value is 0.01 (Guayacg Fig 2b):\n",0.01*steps);
            assert False;
        print(Ws/Ws[0]);



    

if False: # Heinrich Fe results
    #### transition intensities, a la Heinrich large anisotropy paper
    Sxop, Syop, Szop = get_spin_ops(mys);
    psi0 = eigvecs.T[0];
    W_0f = np.zeros((len(eigvals),),dtype=float);
    for findex in range(len(eigvals)):
        psif = eigvecs.T[findex];
        Sx_melement = np.dot(psif, np.dot(Sxop,psi0));
        Sy_melement = np.dot(psif, np.dot(Syop,psi0));
        Sz_melement = np.dot(psif, np.dot(Szop,psi0));
        W_0f[findex] = np.real(np.conj(Sx_melement)*Sx_melement + np.conj(Sy_melement)*Sy_melement + np.conj(Sz_melement)*Sz_melement);
        
    print("\nHeinrich transition intensities from the ground state:\n",W_0f);
    print("Normalized:\n",W_0f/W_0f[0]);
        
    
'''
####
####
#### should reformulate as C_n_me_mT
C_n_me_mT = np.zeros((2*mys+1,2,int(2*j+1)),dtype=complex);      
for nindex in range(2*mys+1): # eigenstate index
    for meindex in range(2): # electron spin
        for mTindex in range(2*j+1): # total spin
            meval = 1/2-1*meindex;
            mTval = j-1*mTindex;
            mmolval = mTval-meval;
            mmolindex = int( (j-1/2)-mmolval);
            mmol_state = np.zeros_like(eigvecs.T[0]);
            mmol_state[mmolindex] = 1.0;
            C_n_me_mT[nindex,meindex,mTindex] = np.dot(np.conj(eigvecs.T[nindex]),mmol_state);
# verify
state = np.zeros_like(eigvecs.T[0]);
for meindex in range(2): # electron spin
    for mTindex in range(2*j+1): # total spin
'''       

