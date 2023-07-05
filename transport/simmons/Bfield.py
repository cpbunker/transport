'''
Describe single-electron charging effects in the presence of
an external magnetic field
'''

from utils import plot_fit, fit_wrapper, load_dIdV

import numpy as np
import matplotlib.pyplot as plt

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkorange", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

# units
kelvin2eV =  8.617e-5; # eV/K
muBohr = 5.788e-5;     # eV/T
gfactor = 2;
conductance_quantum = 7.748e-5; # amp/volt

###############################################################
#### fitting dI/dV with background and oscillations

def dIdV_imp(Vb, V0, E0, G2, G3):
    '''
    Magnetic impurity scattering
    Designed to be passed to scipy.optimize.curve_fit
    '''

    def Ffunc(E, kBT):
        # Eq 17 in XGZ's magnon paper
        numerator = np.log(1+ E0/(E+kBT));
        denominator = 1 - kBT/(E0+0.4*E) + 12*np.power(kBT/(E0+2.4*E),2);
        return numerator/denominator;
    Delta = muBohr*gfactor*bfield_kwarg;
    print(">>>> Delta = ", Delta)
    retval = G2;
    retval -= (G3/2)*Ffunc(abs(Vb-V0), kelvin2eV*temp_kwarg);
    retval -= (G3/4)*Ffunc(abs(Vb-V0+Delta), kelvin2eV*temp_kwarg);
    retval -= (G3/4)*Ffunc(abs(Vb-V0-Delta), kelvin2eV*temp_kwarg);
    return retval;

def dIdV_mag(Vb, V0, Ec, G1):
    '''
    Surface magnon scattering
    Designed to be passed to scipy.optimize.curve_fit
    '''

    def Gmag(E, kBT):
        # Eq 12 in XGZ's magnon paper
        ret = np.zeros_like(E);
        ret += -2*kBT*np.log(1-np.exp(-Ec/kBT));
        ret += (E+Ec)/( np.exp( (E+Ec)/kBT) - 1);
        ret += (E-Ec)/(-np.exp(-(E-Ec)/kBT) + 1);
        return ret
        
    return G1*Gmag(abs(Vb-V0), kelvin2eV*temp_kwarg);

def dIdV_back(Vb, V0, E0, Ec, G1, G2, G3):
    '''
    Magnetic impurity and surface magnon scattering, combined
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_imp(Vb, V0, E0, G2, G3)+dIdV_mag(Vb, V0, Ec, G1);

from landauer import dI_of_Vb, dI_of_Vb_zero

def dIdV_lorentz_zero(Vb, V0, dI0, Gamma, EC): 
    '''
    '''

    nmax = 200;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return -dI0+1e9*conductance_quantum*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, EC, 0.0, ns);

def dIdV_lorentz(Vb, V0, dI0, Gamma, EC): 
    '''
    '''

    nmax = 200;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return -dI0+1e9*conductance_quantum*dI_of_Vb(Vb-V0, mymu0, Gamma, EC, kelvin2eV*temp_kwarg, ns);

def dIdV_all_zero(Vb, V0, E0, Ec, G1, G2, G3, dI0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, E0, Ec, G1, G2, G3) + dIdV_lorentz_zero(Vb, V0, dI0, Gamma, EC);

def dIdV_all(Vb, V0, E0, Ec, G1, G2, G3, dI0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, E0, Ec, G1, G2, G3) + dIdV_lorentz(Vb, V0, dI0, Gamma, EC);

####################################################################
#### main

def fit_dIdV(metal, temp, field, nots, percents, stop_at, num_dev = 4, verbose=0):
    '''
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV_"+"{:.0f}T".format(field)+".txt",metal,temp);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-np.mean(dI_exp),2)));
    del temp, field;

    # unpack
    V0_bound = 1e-2;
    E0_not, G2_not, G3_not, Ec_not, G1_not, dI0_not, Gamma_not, EC_not = nots;
    E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, dI0_percent, Gamma_percent, EC_percent = percents
    
    #### fit background

    # initial fit to magnon + imp
    params_init_guess = np.array([0.0, E0_not, Ec_not, G1_not, G2_not, G3_not]);
    bounds_init = np.array([[-V0_bound, E0_not*(1-E0_percent), Ec_not*(1-Ec_percent), G1_not*(1-G1_percent), G2_not*(1-G2_percent), G3_not*(1-G3_percent)],
                            [ V0_bound, E0_not*(1+E0_percent), Ec_not*(1+Ec_percent), G1_not*(1+G1_percent), G2_not*(1+G2_percent), G3_not*(1+G3_percent)]]);
    params_init, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_init_guess, bounds_init, ["V0", "E0", "Ec", "G1", "G2", "G3"],
                            stop_bounds = False, verbose=verbose);
    background_init = dIdV_back(V_exp, *params_init);

    # remove outliers based on initial fit
    with_outliers = len(V_exp);
    V_exp = V_exp[abs(dI_exp-background_init) < num_dev*dI_dev];
    dI_exp = dI_exp[abs(dI_exp-background_init) < num_dev*dI_dev];
    assert(with_outliers - len(V_exp) <= with_outliers*0.05); # only remove 5%

    # fit to magnon + imp background with outliers removed
    bounds_back = np.copy(bounds_init);
    params_back, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_init, bounds_back, ["V0", "E0", "Ec", "G1", "G2", "G3"],
                            stop_bounds = False, verbose=verbose);
    background = dIdV_back(V_exp, *params_back);
    if(verbose > 4): plot_fit(V_exp, dI_exp, background, derivative=False,
                        mytitle="Impurity + magnon scattering ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'imp_mag/'): return V_exp, dI_exp, params_back, bounds_back;

    #### fit magnon + imp + oscillation
    params_zero_guess = np.zeros((len(params_back)+3,));
    params_zero_guess[:len(params_back)] = params_back; # <---
    bounds_zero = np.zeros((2,len(params_back)+3));
    bounds_zero[:,:len(params_back)] = bounds_back;

    # for oscillation
    params_zero_guess[len(params_back):] = np.array([dI0_not, Gamma_not, EC_not]);
    bounds_zero[:,len(params_back):] = np.array([ [dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                                                [ dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
    # start with zero temp oscillations to constrain
    params_zero, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                params_zero_guess, bounds_zero, ["V0", "E0", "Ec", "G1", "G2", "G3","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_zero), derivative=False,
                mytitle="$T=0$ Landauer fit ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    
    # some plotting to help with constraints
    if False:
        params_plot = np.copy(params_zero);
        params_plot[len(params_back):] = np.array([dI0_not, Gamma_not, EC_not]);
        print(params_plot)
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_plot))
        assert False

    # constrain and do finite temp fit
    params_all_guess = np.copy(params_zero);
    params_all_guess[len(params_back):] = np.array([dI0_not, Gamma_not, EC_not]);
    bounds_all = np.copy(bounds_zero);
    constrain_mask = np.array([1,1,1,0,1,0,0,0,0]); # only G1, G3, dI0, Gamma, Ec free
    bounds_all[0][constrain_mask>0] = params_all_guess[constrain_mask>0];
    bounds_all[1][constrain_mask>0] = params_all_guess[constrain_mask>0]+1e-6;
    params_all, _ = fit_wrapper(dIdV_all, V_exp, dI_exp,
                                params_all_guess, bounds_all, ["V0", "E0", "Ec", "G1", "G2", "G3","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_all), derivative=False,
                mytitle="Landauer fit ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz/'): return V_exp, dI_exp, params_all, bounds_all;
    raise NotImplementedError;

####################################################################
#### wrappers

def fit_Bfield_data():
    metal="Mn/"; # points to data folder
    fname = "fits/Bfield/"
    stop_ats = ['imp_mag/', 'lorentz/'];
    stop_at = stop_ats[1];
    if(stop_at=='imp_mag/'):
        rlabels = ["$V_0$", "$\\varepsilon_0$", "$\\varepsilon_c$", "$G_1$", "$G_2$", "$G_3$"];
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
    elif(stop_at == 'lorentz/'):
        rlabels = ["$V_0$", "$E_0$ (eV)", "$E_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"];
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
        rlabels_mask[:-3] = np.zeros((len(rlabels_mask)-3,), dtype=int)
    else: raise NotImplementedError;

    # experimental params
    Ts = np.array([2.5, 2.5, 2.5]);
    Teffs = np.array([6.5, 6.5, 6.5]);
    Bs = np.array([0.0, 2.0, 7.0]);

    # lorentzian guesses
    E0_guess, G2_guess, G3_guess = 0.008, 1250, 750 
    Ec_guess, G1_guess = 0.013, 1500;
    E0_percent, G2_percent, G3_percent = 0.1, 0.9, 0.1;
    Ec_percent, G1_percent = 0.1, 0.1;   

    # oscillation guesses
    dI0_guess =   np.array([58.9,58.9,60.3])*1e3
    Gamma_guess = np.array([5.70,5.70,5.70])*1e-3
    EC_guess =    np.array([5.82,5.81,5.70])*1e-3
    dI0_percent = 0.4;
    Gamma_percent = 0.4;
    EC_percent = 0.2;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Teffs)):
        if(True):
            print("#"*60+"\nB = {:.1f} Tesla, T = {:.1f} K, Teff = {:.1f} K".format(Bs[datai], Ts[datai], Teffs[datai]));
            guesses = (E0_guess, G2_guess, G3_guess, Ec_guess, G1_guess, dI0_guess[datai], Gamma_guess[datai], EC_guess[datai]);
            percents = (E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, dI0_percent, Gamma_percent, EC_percent);

            # get fit results
            global temp_kwarg; temp_kwarg = Teffs[datai]; # very bad practice
            global bfield_kwarg; bfield_kwarg = Bs[datai];
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(metal, Ts[datai], Bs[datai],
                guesses, percents, stop_at, verbose=1);
            results.append(temp_results); 
            boundsT.append(temp_bounds);
    
            #save processed x and y data
            exp_fname = fname+stop_at+"stored_exp/{:.0f}".format(Ts[datai]); # <- where to get/save the plot
            np.save(exp_fname+"_x.npy", x_forfit);
            np.save(exp_fname+"_y.npy", y_forfit);

    # plot fitting results vs T
    results, boundsT = np.array(results)[:,rlabels_mask>0], np.array(boundsT)[:,:,rlabels_mask>0];
    nresults = len(results[0]);
    fig, axes = plt.subplots(nresults, sharex=True);
    if(nresults==1): axes = [axes];
    for resulti in range(nresults):
        axes[resulti].plot(Bs, results[:,resulti], color=mycolors[0],marker=mymarkers[0]);
        axes[resulti].set_ylabel(rlabels[resulti]);
        axes[resulti].plot(Bs,boundsT[:,0,resulti], color=accentcolors[0],linestyle='dashed');
        axes[resulti].plot(Bs,boundsT[:,1,resulti], color=accentcolors[0],linestyle='dashed');
        axes[resulti].ticklabel_format(axis='y',style='sci',scilimits=(0,0));

    # save
    if True:
        print("Saving data to "+fname+stop_at);
        np.savetxt(fname+stop_at+"Ts.txt", Ts);
        np.savetxt(fname+stop_at+"Teffs.txt", Teffs);
        np.savetxt(fname+stop_at+"Bs.txt", Bs);
        np.save(fname+stop_at+"results.npy", results);
        np.save(fname+stop_at+"bounds.npy", boundsT);

    # format
    axes[-1].set_xlabel("$B$ (Tesla)");
    axes[0].set_title("Amplitude and period fitting");
    plt.tight_layout();
    plt.show();

####################################################################
#### run

if(__name__ == "__main__"):
    fit_Bfield_data();
