import numpy as np
import matplotlib.pyplot as plt

import sys
import json

def lorentzian(xarr, x0, Gamma):
    return (1/np.pi)*(0.5*Gamma)/((xarr-x0)**2 + (0.5*Gamma)**2);

def energies_to_dos(Evals, Gamma):
    dosvals = np.zeros_like(Evals);
    for E in Evals:
        dosvals += lorentzian(Evals, E, Gamma);

    return dosvals;

def direct_eigenenergies(params):
    if(params["v"]==-1.0 and params["w"]==-1.0):
        arr_in = np.array(
                [-1.9909438451461685, -1.9638573945254123, -1.918985947228994, -1.8567358660321451, -1.7776708973098465, -1.6825070656623617, -1.5721061894855748, -1.4474680762101402, -1.30972146789057, -1.160113819142396, -0.9999999999999997, -0.8308300260037728, -0.6541359266348432, -0.47151787101885423, -0.2846296765465703, -0.09516383164748421, 0.09516383164748438, 0.2846296765465699, 0.4715178710188541, 0.6541359266348434, 0.8308300260037731, 0.9999999999999997, 1.1601138191423965, 1.3097214678905704, 1.44746807621014, 1.572106189485575, 1.6825070656623617, 1.777670897309846, 1.856735866032144, 1.9189859472289945, 1.9638573945254127, 1.9909438451461687]
                 );
    elif(params["v"]==-0.95 and params["w"]==-1.0):
        arr_in = np.array(
                [-1.9411623077635933, -1.9147290470410678, -1.870938920267093, -1.810187281691921, -1.7330224177248268, -1.6401403422798115, -1.5323780888228367, -1.4107054305882416, -1.27621483525588, -1.1301091441761741, -0.9736856082116285, -0.808312289413186, -0.6353832818679812, -0.45619377750777457, -0.27133159564160964, -0.06373974285727309, 0.06373974285727317, 0.27133159564160914, 0.45619377750777435, 0.6353832818679813, 0.8083122894131854, 0.973685608211628, 1.1301091441761741, 1.27621483525588, 1.4107054305882412, 1.5323780888228358, 1.6401403422798115, 1.733022417724827, 1.8101872816919211, 1.8709389202670936, 1.914729047041068, 1.9411623077635929]
                );
    else: raise NotImplementedError;
    return arr_in;

# top level
verbose = 2; assert verbose in [1,2,3];
np.set_printoptions(precision = 4, suppress = True);
json_name = sys.argv[1];
try:
    try:
        params = json.load(open(json_name+".txt"));
    except:
        params = json.load(open(json_name));
        json_name = json_name[:-4];
    print(">>> Params = ",params);
except:
    raise Exception(json_name+" cannot be found");

# load data from json
v, w = params["v"], params["w"];
# NB u, u0 are zero
uval = 0;

# Rice Mele 
ks = np.linspace(-np.pi, np.pi, 999);
Es = np.append(-np.sqrt(v**2 + w**2 + 2*v*w*np.cos(ks)), np.sqrt(v**2 + w**2 + 2*v*w*np.cos(ks))); # dispersion
band_edges = np.array([np.sqrt(uval*uval+(w+v)*(w+v)),
                       np.sqrt(uval*uval+(w-v)*(w-v))]);

####
####
# DOS from analytical dispersion
dos_an = (2/np.pi)/abs(np.gradient(Es,np.append(ks,ks))); # handles both bands at once

# plotting
fig, (dispax,dosax) = plt.subplots(ncols=2, sharey=True);
for band in [0,1]:
    this_band = Es[band*len(ks):(band+1)*len(ks)];
    this_dos = dos_an[band*len(ks):(band+1)*len(ks)];
    dispax.plot(ks/np.pi, this_band);
    dosax.plot(this_dos,this_band);

####
####
# DOS from manual hamiltonian diagonalization
Es_direct = direct_eigenenergies(params);
dos_direct = energies_to_dos(Es_direct, Gamma=0.01);
use_dos_direct = False;

# analyze
direct_spacings = (Es_direct[1:] - Es_direct[:-1]);
myGamma = np.min(direct_spacings)/2;
print("Gamma = min(E_spacing)/2 = {:.6f}".format(myGamma));
dos_direct = energies_to_dos(Es_direct, Gamma=myGamma);
E_gap_tilde = np.min(Es_direct[len(dos_direct)//2:]) - np.max(Es_direct[:len(dos_direct)//2]);
print("E_gap_tilde = {:.6f}".format(E_gap_tilde));

# plotting
for band in [0,1]:
    this_band = direct_eigenenergies(params)[band*len(dos_direct)//2:(band+1)*len(dos_direct)//2];
    this_dos = dos_direct[band*len(dos_direct)//2:(band+1)*len(dos_direct)//2];
    print("***")
    print(this_band)
    print(this_dos)
    if use_dos_direct:
        dosax.plot(this_dos,this_band);
    else:
        for E in this_band: dosax.axhline(E, color=["tab:green","darkgreen"][band]);

# title
Egap = 2*abs(params["w"]-params["v"]*np.sign(params["w"]/params["v"]));
assert("u" not in params.keys());
title_or_label = "$t_h =${:.2f}, $V_b =${:.2f}, $v =${:.2f}, $w =${:.2f}".format(params["th"],params["Vb"],params["v"],params["w"]);
title_or_label += " $E_{gap} =$"+"{:.2f}".format(Egap);
fig.suptitle(title_or_label);

# format
dispax.set_xlim(-1.0,1.0);
dispax.set_xlabel("$ka/\pi$");
if(not use_dos_direct): dosax.set_xlim(0.0,20.0);
else: dosax.set_xlim(0.0,np.max(dos_direct));
dosax.set_xlabel("$\\rho$");
dosax.set_ylabel("$\\tilde{E}_{gap} = $"+"{:.6f}".format(E_gap_tilde));
for bedge in band_edges: 
    for edgesign in [+1,-1]:
        dispax.axhline(edgesign*bedge, color="black", linestyle="dashed");
        dosax.axhline( edgesign*bedge, color="black", linestyle="dashed");
dispax.set_ylabel("$E$");

# show
plt.show();
