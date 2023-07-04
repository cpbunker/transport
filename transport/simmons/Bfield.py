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

####################################################################
#### wrappers

def fit_Bfield_data():
    metal="Mn/"; # points to data folder
    fname = "fits/"
    stop_ats = "Bfield"

    # experimental params
    kelvin2eV =  8.617e-5;
    Tvals = np.array([2.5, 2.5, 2.5]);
    Teffs = np.aray([]);

    # lorentzian guesses
    E0_guess, G2_guess, G3_guess = 0.008, 1250, 750 # 2.5 K: 0.0105, 850,450
    Ec_guess, G1_guess = 0.013, 1500;
    E0_percent, G2_percent, G3_percent = 0.1, 0.9, 0.1;
    Ec_percent, G1_percent = 0.1, 0.1;   

    # oscillation guesses
    dI0_guess =   np.array([58.9,57.9,51.4,50.6])*1e3
    Gamma_guess = np.array([5.70,4.70,4.20,4.10])*1e-3
    EC_guess =    np.array([5.82,4.89,4.88,4.81])*1e-3
    dI0_percent = 0.4;
    Gamma_percent = 0.4;
    EC_percent = 0.2;

    # shorten
    picki = 3;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Teffs)):
        if(True):
            print("#"*60+"\nT = {:.1f} K, Teff = {:.1f} K".format(Ts[datai], Teffs[datai]));
            guesses = (E0_guess, G2_guess, G3_guess, Ec_guess, G1_guess, dI0_guess[datai], Gamma_guess[datai], EC_guess[datai]);
            percents = (E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, dI0_percent, Gamma_percent, EC_percent);

            # get fit results
            global temp_kwarg; temp_kwarg = Teffs[datai]; # very bad practice
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(metal, Ts[datai],
                guesses, percents, stop_at = stop_at, verbose=1);
            results.append(temp_results); 
            boundsT.append(temp_bounds);
    
            #save processed x and y data
            exp_fname = fname+stop_at+"stored_exp/{:.0f}".format(Ts[datai]); # <- where to get/save the plot
            np.save(exp_fname+"_x.npy", x_forfit);
            np.save(exp_fname+"_y.npy", y_forfit);

    # plot fitting results vs T
    results, boundsT = np.array(results), np.array(boundsT);
    nresults = len(results[0]);
    fig, axes = plt.subplots(nresults, sharex=True);
    if(nresults==1): axes = [axes];
    for resulti in range(nresults):
        axes[resulti].plot(Ts, results[:,resulti], color=mycolors[0],marker=mymarkers[0]);
        axes[resulti].set_ylabel(rlabels[resulti]);
        axes[resulti].plot(Ts,boundsT[:,0,resulti], color=accentcolors[0],linestyle='dashed');
        axes[resulti].plot(Ts,boundsT[:,1,resulti], color=accentcolors[0],linestyle='dashed');
        axes[resulti].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    # Amp vs T
    if(stop_at=='sin'):
        axes[1].plot(Ts, results[0,1]*5/Ts, color = 'red', label = "$A(T=5) \\times 5/T$");
        axes[1].legend();

    # save
    if True:
        print("Saving data to "+fname+stop_at);
        np.savetxt(fname+stop_at+"Ts.txt", Ts);
        np.savetxt(fname+stop_at+"Teffs.txt", Teffs);
        np.save(fname+stop_at+"results.npy", results);
        np.save(fname+stop_at+"bounds.npy", boundsT);

    # format
    axes[-1].set_xlabel("$T$ (K)");
    axes[0].set_title("Amplitude and period fitting");
    plt.tight_layout();
    plt.show();

####################################################################
#### run

if(__name__ == "__main__"):
    fit_Bfield_data();
