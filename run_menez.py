'''
Christian Bunker
M^2QM at UF
September 2021

Quasi 1 body transmission through spin impurities project, part 0:
Scattering of a single electron from a spin-1/2 impurity

wfm.py
- Green's function solution to transmission of incident plane wave
- left leads, right leads infinite chain of hopping tl treated with self energy
- in the middle is a scattering region, hop on/off with th usually = tl
- in SR the spin degrees of freedom of the incoming electron and spin impurities are coupled 
'''

from transport import wfm, fci_mod, ops
from transport.wfm import utils

import numpy as np
import matplotlib.pyplot as plt

# top level
np.set_printoptions(precision = 4, suppress = True);
verbose = 5;
analytical = True; # whether to compare to menezes' calc
reflect = False; # whether to get R or T

# fig standardizing
from matplotlib.font_manager import FontProperties
myfontsize = 14;
myfont = FontProperties()
myfont.set_family('serif')
myfont.set_name('Times New Roman')
myprops = {'family':'serif','name':['Times New Roman'],
    'weight' : 'roman', 'size' : myfontsize}
mycolors = ["black","darkblue","darkgreen","darkred", "darkmagenta","darkgray","darkcyan"];
mystyles = ["solid", "dashed","dotted","dashdot"];
mylinewidth = 1.0;
plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# plotting
# panels: (a) S dot S, Delta = 0 (b) S dot S, Delta \neq 0 (c) derived

# tight binding params
tl = 1.0;
th = 1.0;
fig, axes = plt.subplots(3, sharex = True);
fig.set_size_inches(7/2,9/2);

# iter over panels a, b, b
mypanels = ["(a)","(b)","(c)"];
Deltavals = [0,0.05];
for axi in range(len(axes)):

    # panels a and b
    if(axi in range(len(Deltavals))): # axi = 0 or 1
        Delta = Deltavals[axi];
        Jeffs = [0.1,0.2,0.4];
        for Ji in range(len(Jeffs)):
            Jeff = Jeffs[Ji];
            
            # 2nd qu'd operator for S dot s
            h1e = np.zeros((4,4))
            g2e = wfm.utils.h_kondo_2e(Jeff, 0.5); # J, spin
            states_1p = [[0,1],[2,3]]; # [e up, down], [imp up, down]
            hSR = fci_mod.single_to_det(h1e, g2e, np.array([1,1]), states_1p); # to determinant form

            # zeeman splitting
            hzeeman = np.array([[Delta, 0, 0, 0],
                            [0,0, 0, 0],
                            [0, 0, Delta, 0], # spin flip gains PE delta
                            [0, 0, 0, 0]]);
            hSR += hzeeman;

            # truncate to coupled channels
            hSR = hSR[1:3,1:3];
            hzeeman = hzeeman[1:3,1:3];

            # leads
            hLL = np.copy(hzeeman);
            hRL = np.copy(hzeeman)

            # source = up electron, down impurity
            sourcei, flipi = 0,1
            source = np.zeros(np.shape(hSR)[0]);
            source[sourcei] = 1;

            # package together hamiltonian blocks
            hblocks = [hLL,hSR];
            hblocks.append(hRL);
            hblocks = np.array(hblocks);

            # hopping
            tnn = [-th*np.eye(*np.shape(hSR)),-th*np.eye(*np.shape(hSR))]; # on and off imp
            tnn = np.array(tnn);
            tnnn = np.zeros_like(tnn)[:-1];
            if(verbose and Jeff == 0.1): print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn,"\ntnnn:\n",tnnn);

            # sweep over range of energies
            # def range
            Emin, Emax = -1.99999*tl, -1.999*tl+0.2*tl;
            Evals = np.linspace(Emin, Emax, 99, dtype = float);
            Tvals, Rvals = [], [];
            for E in Evals:
                if(E in Evals[:1]): # verbose
                    Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, all_debug = not axi, verbose = verbose));
                    Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, all_debug = not axi, reflect = True));
                else:
                    Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, all_debug = not axi,));
                    Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, all_debug = not axi, reflect = True));
                    
            # plot Tvals vs E
            Tvals, Rvals = np.array(Tvals), np.array(Rvals);
            axes[axi].plot(Evals[Evals+2*tl > Delta],Tvals[:,flipi][Evals +2*tl > Delta], color = mycolors[Ji], linestyle = "dashed", linewidth = mylinewidth);
            totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
            axes[axi].plot(Evals, totals, color="red", label = "total ");

            # menezes prediction in the continuous case
            if(analytical):
                axes[axi].plot(Evals, Jeff*Jeff/(16*(Evals+2*tl)), color = mycolors[Ji], linewidth = mylinewidth);

        # format panel
        axes[axi].set_title(mypanels[axi], x=0.93, y = 0.7, fontsize = myfontsize);
        axes[axi].set_ylim(0,0.4);
        axes[axi].set_yticks([0,0.2,0.4]);
        axes[axi].set_ylabel('$T$', fontsize = myfontsize );
    # end iter over panels a and b

    # panel c
    else:

        epsilons = [-27.5,-11.3,-5.3];
        for epsi in range(len(epsilons)):
            epsilon = epsilons[epsi];

            # tight binding parameters
            tl = 1.0;
            th = 1.0;
            U2 = 100.0;
            Jeff = 2*th*th*U2/((-epsilon)*(U2+epsilon)); # better for U >> Vg
            print("Jeff = ",Jeff);

            # SR physics: site 1 is in chain, site 2 is imp with large U
            hSR = np.array([[0,-th,th,0], # up down, -
                           [-th,epsilon, 0,-th], # up, down (source)
                            [th, 0, epsilon, th], # down, up (flip)
                            [0,-th,th,U2+2*epsilon]]); # -, up down

            # source = up electron, down impurity
            source = np.zeros(np.shape(hSR)[0]);
            sourcei, flipi = 1,2;
            source[sourcei] = 1;

            # lead physics
            hLL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);
            hRL = np.diagflat([0,epsilon, epsilon, 2*epsilon]);

            # package together hamiltonian blocks
            hblocks = np.array([hLL, hSR, hRL]);
            for hb in hblocks: hb += -epsilon*np.eye(len(source));  # shift by gate voltage so source is at zero
            tnn_mat = -tl*np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]);
            tnn = np.array([np.copy(tnn_mat), np.copy(tnn_mat)]);
            tnnn = np.zeros_like(tnn)[:-1];
            #if verbose: print("\nhblocks:\n", hblocks, "\ntnn:\n", tnn, "\ntnnn:", tnnn)

            if False: # do the downfolding explicitly
                matA = np.array([[0, 0],[0,0]]);
                matB = np.array([[-th,-th],[th,th]]);
                matC = np.array([[-th,th],[-th,th]]);
                matD = np.array([[-epsilon, 0],[0,U2+epsilon]]);
                mat_downfolded = matA - np.dot(matB, np.dot(np.linalg.inv(matD), matC))  
                print("mat_df = \n",mat_downfolded);
                Jeff = 2*abs(mat_downfolded[0,0]);
                print(">>>Jeff = ",Jeff);
                mat_downfolded += np.eye(2)*Jeff/4
                print("mat_df = \n",mat_downfolded);
            
            # sweep over range of energies
            # def range
            Emin, Emax = -1.99999*tl, -2.0*tl+0.2*tl;
            Evals = np.linspace(Emin, Emax, 99, dtype = complex);
            Tvals, Rvals = [], [];
            for E in Evals:
                Tvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source));
                Rvals.append(wfm.kernel(hblocks, tnn, tnnn, tl, E, source, reflect = True));

            # plot Tvals vs E
            Tvals, Rvals = np.array(Tvals), np.array(Rvals);
            axes[axi].plot(Evals,Tvals[:,flipi], color = mycolors[epsi], linestyle = "dashed", linewidth = mylinewidth);
            if True: # check that T+R = 1
                totals = np.sum(Tvals, axis = 1) + np.sum(Rvals, axis = 1);
                axes[axi].plot(Evals, totals, color="red");
                #for Ti in range(np.shape(Tvals)[1]): ax.plot(Evals, Tvals[:,Ti], label = Ti)

            # menezes prediction in the continuous case
            if analytical:
                axes[axi].plot(Evals, Jeff*Jeff/(16*(np.real(Evals)+2*tl)), color = mycolors[epsi], linewidth = mylinewidth);

        # format panel
        axes[axi].set_title(mypanels[axi], x=0.93, y = 0.7, fontsize = myfontsize); 
        axes[axi].set_ylim(0,0.4);
        axes[axi].set_yticks([0,0.2,0.4]);
        axes[axi].set_ylabel('$T$', fontsize = myfontsize );


# format overall
axes[-1].set_xlim(-2,-1.8);
axes[-1].set_xticks([-2,-1.95,-1.9,-1.85,-1.8]);
axes[-1].set_xlabel("$E/t$", fontsize = myfontsize);
plt.tight_layout();
plt.savefig('menez_all.pdf');








