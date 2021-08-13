'''
Christian Bunker
M^2QM at UF
August 2021

scratchwork
'''

import ops
import fci_mod

import numpy as np
import matplotlib.pyplot as plt

# top level
nleads = (3,3);
Rlead_pol = 0;

# phys params, must be floats
tl = 1.0;
th = 0.1; 
Vb = -0.5;
mu = 0.0;
Vg = -1.41;
U = 2.82;

####################################################
# left lead, initially half filled, energy as we add an up electron
# left lead is a spinless system, so any electron = up electron under rotation

# left lead ham
LL_ham = ops.h_leads(tl, (nleads[0],0)); # tight binding ham
LL_ham += 0.5*Vb*np.eye(2*nleads[0]); # add bias
g2e = np.zeros( (2*nleads[0],2*nleads[0],2*nleads[0],2*nleads[0] )); # noninteracting

# increase number of electrons
nes = [nleads[0],nleads[0]+1];
LeftEs = [];
for ne in nes:

    E, _ = fci_mod.direct_FCI(LL_ham, g2e, 2*nleads[0], (ne,0) );
    LeftEs.append(E);

# plot results
LeftEs = np.array(LeftEs);
plt.plot([1,2],LeftEs, label = "Left lead gains");

####################################################
# right lead, initially half filled, energy as we remove an up electron
# initially spin polarized

# top level, physical params carried down

# right lead ham, spinfree form
RL_ham = [[0.0, -tl,0.0], [-tl,0.0,-tl], [0.0,-tl, 0.0] ]; # spin free tight binding ham
RL_ham += -0.5*Vb*np.eye(nleads[1]); # add bias
g2e = np.zeros( (nleads[0],nleads[0],nleads[0],nleads[0] )); # noninteracting

nes = [(nleads[1],0),(nleads[1]-1,0)];
RightEs = [];
for ne in nes:
    
    E, _= fci_mod.direct_FCI(RL_ham, g2e, nleads[1], ne);
    RightEs.append(E);

# plot results
RightEs = np.array(RightEs);
plt.plot([1,2], RightEs, label = "Right lead loses up");


####################################################
# add and remove an electron to the dot

dot_ham = np.array([Vg]);
dot_g2e = np.array([[[[U]]]]);

nes = [(0,1),(1,1),(1,0)]
DotEs = [];
for ne in nes:
    
    E, _= fci_mod.direct_FCI(dot_ham, dot_g2e, 1, ne);
    DotEs.append(E);

# plot results
total = -LeftEs + RightEs
print(total[1]-total[0])
DotEs = np.array(DotEs);
plt.plot([1,2], DotEs[:2], linestyle = "dashed", label = "Dot gains up");
plt.plot([1,2], DotEs[1:], linestyle = "dashed", label = "Dot loses");
plt.plot([1,2], RightEs + DotEs[:2], color = "grey", label = "RL -> dot");
plt.plot([1,2], LeftEs + DotEs[1:], color = "grey", linestyle = "dotted", label = "dot -> LL");
plt.xlabel("Initial --> Final");
plt.ylabel("Energy");
plt.legend();
plt.show();







