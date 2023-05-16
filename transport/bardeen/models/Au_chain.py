'''
A model of a chain of Au atoms

NB Au electronic structure = [Xe] 4f14 5d10 6s1
'''

import numpy as np
import matplotlib.pyplot as plt

n_loc_dof = 1;

if True: # experimental model from https://doi.org/10.1126/science.1075242

    # # left and right leads are chains of 20 Au atoms
    Nchain = 20;
    VL = 0*0.025*np.eye(n_loc_dof)

    # get the hopping in Hartree
    m_r = 0.5; # effective mass is 1/2 electron bare mass
    a_r = 5.46; # interatomic spacing in Bohr
    # NB since 1 Ha = \hbar^2/(m_e * bohr^2), the hopping in hartree
    # = \hbar^2(m_r*m_e * (a_r * bohr)^2 ) = 1/(m_r*a_r*a_r) Ha
    tL = 1/(m_r*a_r*a_r)*np.eye(n_loc_dof);

    # in between is a vaccum, modelled as a larger inter atom distance
    Nvac=0;
    dvac = 2*a_r;
    tvac = 1/(m_r*dvac*dvac)*np.eye(n_loc_dof); # simple model of hopping through vacuum
    print(">>>", tvac);

    # construct with identical left and right leads
    Ntotal = 2*Nchain+Nvac; # total # sites
    Hsys = np.zeros((Ntotal, Ntotal, n_loc_dof, n_loc_dof));
    for sitei in range(Ntotal):
        Hsys[sitei, sitei] = VL;
    for sitei in range(Ntotal-1):
        if(sitei==Nchain-1):
            Hsys[sitei, sitei+1] = -tvac;
            Hsys[sitei+1,sitei] = -tvac;
        else:
            Hsys[sitei, sitei+1] = -tL;
            Hsys[sitei+1,sitei] = -tL;

    # save to .npy
    fname = "transport/bardeen/models/Au_chain_"+str(tvac[0,0].round(6))+".npy";
    print("Saving to "+fname);
    np.save(fname,Hsys);
        
    
