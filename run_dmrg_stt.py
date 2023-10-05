'''
Christian Bunker
M^2QM at UF
October 2023


'''

from transport import tddmrg

import numpy as np

import sys

# top level
verbose = 3;
data_file = sys.argv[1];
# if this is a .txt, then this is a json with numerical params for new data run
# if this is a .npy, it is where some data is already stored

##################################################################################
#### run for new data
if(data_file[-4:] == ".txt"):

    # dmrg info
    bdims = [700, 800, 900, 1000];
    noises = [1e-4, 1e-5, 1e-6, 0.0];

    # build up t<0 Hamiltonian
    h_base, g_base = tddmrg.Hsys_base(data_file);
    polarize_aa, polarize_bb = tddmrg.Hsys_polarizer(data_file);
    h_tlesser_aa, h_tlesser_bb = h_base+polarize_aa, h_base+polarize_bb;
    print(h_tlesser_aa);
    print(h_tlesser_bb);











