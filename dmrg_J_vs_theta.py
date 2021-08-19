'''
Christian Bunker
M^2QM at UF
August 2021

Runner file for prepping dot spin state with B field, getting current output
Now assuming polarizer btwn Rlead, dot
'''

import siam_current

import numpy as np

##################################################################################
#### prepare dot in diff spin states

# top level params from command line
verbose = int(sys.argv[1]);
nleads = (int(sys.argv[2],int(sys.argv[3]));
nelecs = (sum(nleads)+1,0); # half filling
get_data = bool(sys.argv[4]); # whether to run computations, if not data already exists
print("Command line inputs:", verbose, nelecs, get_data);

# phys params, must be floats
tl = 1.0;
th = tl/10; # can scale down and same effects are seen. Make sure to do later
Vb = -1/100*tl
mu = 10.0*tl;
Vg = mu;
U = 100*tl;
Bs = [tl*5, tl*5, tl*5,tl*5,tl*5,tl*5, tl*5, tl*5,tl*5];
thetas = np.array([0.0, np.pi/8, np.pi/4, 3*np.pi/8,np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi]);
Bs = [tl*5];
thetas = np.array([0.0]);
phi = 0.0;

#time info
dt = 0.004;
tf = 0.4;

datafs = [];
labs = [];
splots = ['Jtot','J','delta_occ','delta_Sz','Szleads']; # which subplots 
if get_data: # must actually compute data

    for i in range(len(Bs)): # iter over B, theta inputs
        B, theta = Bs[i], thetas[i];
        params = tl, th, Vb, mu, Vg, U, B, theta, phi;
        fname = siam_current.DotDataDmrg(nleads, nelecs, tf, dt, prefix = "", phys_params=params, verbose = verbose);

else:
    import plot # do here for compatibility

    for i in range(len(Bs)):
        datafs.append("fci_"+str(nleads[0])+"_1_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(Bs[i])+"_t"+str(thetas[i])[:3]+"_Vg"+str(Vg)+".npy");
        labs.append("$\\theta$ = "+str(thetas[i])[:3] );
    
    plot.CompObservables(datafs, nleads, Vg, labs, whichi = 0, splots = splots);
    plot.CompConductances(datafs, thetas, (2.0, 4.0), Vb);


    








