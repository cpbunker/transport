
import numpy as np
import matplotlib.pyplot as plt

# exchange between electron and localized imp, in the z basis
exchange = (1/4)*np.array([[1,0,0,0],
                     [0,-1,2,0],
                     [0,2,-1,0],
                     [0,0,0,1]]);

# matrix which gives eigenstates where electron is in the +/- x direction
# while the imp is in the +/- z direction
perp_basis_mat = np.array([[0,0,1,0],
                           [0,0,0,1],
                           [1,0,0,0],
                           [0,1,0,0]]);

_, perp_basis_states = np.linalg.eigh(perp_basis_mat);

# we want to work in a basis where the perp mat is diagonal!!
change_basis = np.copy(perp_basis_states); # column eigenvector diagonalizes
perp_basis_states = perp_basis_states.T;
perp_basis_diag = np.matmul( np.linalg.inv(change_basis), np.matmul(perp_basis_mat, change_basis));
print("eigenvectors in the Sz basis are:");
for vec in perp_basis_states: print(vec);

# interaction in this basis
exchange_perp = np.matmul( np.linalg.inv(change_basis), np.matmul(exchange, change_basis));
print("\n exchange in this basis is \n", exchange_perp.round(4))

# NB the exchange couples all these basis states to each other with
# equal strength! as a result you will always continue to get dynamics
# as you send more x-aligned electrons in. This is distinct from sending
# in z-aligned electrons, because then you can get a situation where
# the impurity is aligned with all incoming electrons and nothing happens
