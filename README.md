# transport
Methods for calculating transport through magnetic molecules / spin impurities

## transport/wfm
Given a single electron incident on an interacting scattering region, calculate the steady state transmission coefficient. The key feature of this formalism is sensitivity to the quantum numbers of the entire system (scattered electron and scattering region), including the ability to account for changes to the quantum numbers of the scattering region during the scattering process. This accounting is implemented with the methods outlined in *Conductance calculations for quantum wires and interfaces: mode matching and Green's functions* b Khomyakov et al, 2005.

The key function is wfm.kernel() which outputs transmission and reflection information, given
- a system described by an array *h* of on site block hamiltonians, and arrays *tnn, tnnn* of nearest neighbor and next nearest neighbor hopping matrices
- an electron described by energy *E*
- a spin "boundary condition" specified by the vector *A_j\sigma*

## transport/tdfci
Time Dependent Full Configuration Interaction

## transport/tddmrg
Time Dependent Density Matrix Renormalization Group

# fcdmft
Revision of DMFT code (due to Tianyu Zhu, Garnet Chan Group, Caltech) for calculation of the many body Green's function (MBGF) at a high level of quantum chemistry for an interacting region coupled to two noninteracting, block tridiagonal leads. This MBGF can then be used to obtain the linear response current due to a finite bias in the lead chemical potential(mu_L != mu_R) using the Meir Wingreen formalism.

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
- Linear response MW: first order effects of the bias (mu_L != mu_R) are contained in the FD occupation numbers (ie, in the stat mech)
- Can just use the equilibrium (ie zero bias) value for the MBGF and its derived quantities (Gr, A, etc)
- At equilibrium, LambdaL = LambdaR and we can use MW Eq 9
- Assuming that there are no spin interactions in the lead, Lambda's are diagonal and we can use MW Eq 11
- Therefore get a spin current
