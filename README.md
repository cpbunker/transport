# transport
Methods for calculating transport through magnetic molecules / spin impurities

## wfm
Steady state calculation of transmission coefficient of a single electron incident on the spin-interacting scattering region.

## fcdmft
Revision of DMFT code (due to Tianyu Zhu, Garnet Chan Group, Caltech) for calculation of the many body Green's function (MBGF) at a high level of quantum chemistry for an interacting region coupled to two noninteracting, block tridiagonal leads. This MBGF can then be used to obtain the linear response current due to a finite bias in the lead chemical potential( mu_L - mu_R !=  0) using the Meir Wingreen formalism.

Process as done by fcdmft/__init__.py (my code):
1. Formalism of the MBGF
- the hamiltonian can always be put in a tridiagonal form by making the blocks large enough
- assuming an orthogonal basis, we have (zI - H)G = I
- take care of coupling to leads with self energies: Sigma = V g V^\dagger where g is the unperturbed surface gf of the lead
2. Calculation of the surface GF
- Haydock recursive method for any lead hamiltonians, so long as onsite h and hopping t matrices are same for all blocks
- if h and t are just numbers (ie a nearest neighbor only tight binding model) there is an analytical solution
3. Calculation of the MBGF
- Tianyu's original fcdmft package calculates the MBGF for an interacting, periodic region at a high level
- I modify to an interacting region treated at high level coupled to two noninteracting leads treated at low level
- surface gf of leads have to be discretized into a bath
- then given a bath, second quantized hamiltonian for the interacting region, and chemical potential, can determine the thermal equilibrium MBGF for the system using desired level of quantum chemistry (fci, cc, etc)
4. Implementation of Meir Wingreen

## tdfci
Time Dependent Full Configuration Interaction

## tddmrg
Time Dependent Density Matrix Renormalization Group
