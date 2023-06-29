'''
'''

from utils import load_dIdV

import numpy as np
import matplotlib.pyplot as plt

muBohr = 5.788e-5;
gfactor = 2

base = "KdIdV";
metal = "Mn/";
temp = 2.5;
fields = [0,2,7];
fig, ax = plt.subplots();
for field in fields:
    V_exp, dI_exp = load_dIdV(base+"_"+"{:.0f}T".format(field)+".txt",metal,temp);
    ax.plot(V_exp, dI_exp, label = "$B = $ {:.2f} meV".format(field*gfactor*muBohr*1000));

plt.legend();
plt.tight_layout();
plt.show();
