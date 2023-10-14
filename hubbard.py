

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
     


L = 8
N = 8
TWOSZ = 0

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=L, n_elec=N, spin=TWOSZ)
     

t = 1
U = 2

b = driver.expr_builder()

# hopping term
b.add_term("cd", np.array([[[i, i + 1], [i + 1, i]] for i in range(L - 1)]).flatten(), -t)
b.add_term("CD", np.array([[[i, i + 1], [i + 1, i]] for i in range(L - 1)]).flatten(), -t)

# onsite term
b.add_term("cdCD", np.array([[i, ] * 4 for i in range(L)]).flatten(), U)

mpo = driver.get_mpo(b.finalize(), iprint=2)


def run_dmrg(driver, mpo):
    ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
    bond_dims = [250] * 4 + [500] * 4
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-10] * 8
    return driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=thrds, cutoff=0, iprint=1)

energies = run_dmrg(driver, mpo)
print('DMRG energy = %20.15f' % energies)
