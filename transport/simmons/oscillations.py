'''
Capacitor model + landauer formula for describing evenly spaced dI/dVb oscillations
due to single electron charging effects
'''

from utils import plot_fit, load_dIdV, fit_wrapper
from landauer import dI_of_Vb, dI_of_Vb_zero

import numpy as np
import matplotlib.pyplot as plt

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkred", "darkgreen", "darkorange", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","s","^","d","*","+","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

# units
kelvin2eV =  8.617e-5;

###############################################################
#### fitting dI/dV with background and oscillations

def dIdV_imp(Vb, V0, E0, G2, G3, ohmic_heat):
    '''
    Magnetic impurity scattering
    Designed to be passed to scipy.optimize.curve_fit
    '''

    def Ffunc(E, kBT):
        # Eq 17 in XGZ's magnon paper
        numerator = np.log(1+ E0/(E+kBT));
        denominator = 1 - kBT/(E0+0.4*E) + 12*np.power(kBT/(E0+2.4*E),2);
        return numerator/denominator;
    
    retval = G2;
    retval -= (G3)*Ffunc(abs(Vb-V0), kelvin2eV*(temp_kwarg+ohmic_heat));
    return retval;

def dIdV_mag(Vb, V0, Ec, G1, ohmic_heat):
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
        
    return G1*Gmag(abs(Vb-V0), kelvin2eV*(temp_kwarg+ohmic_heat));

def dIdV_back(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat):
    '''
    Magnetic impurity and surface magnon scattering, combined
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_imp(Vb, V0, E0, G2, G3, ohmic_heat)+dIdV_mag(Vb, V0, Ec, G1, ohmic_heat);

def dIdV_sin(Vb, V0, amplitude, period, dI0):
    '''
    Sinusoidal fit function - purely mathematical, not physical
    Designed to be passed to scipy.optimize.curve_fit
    '''

    ang_freq = 2*np.pi/period
    return amplitude*(-1)*np.cos(ang_freq*(Vb-V0));

def dIdV_lorentz_zero(Vb, V0, dI0, Gamma, EC): 
    '''
    '''

    nmax = 200;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return dI0*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, EC, 0.0, ns);

def dIdV_lorentz(Vb, V0, dI0, Gamma, EC): 
    '''
    '''

    nmax = 200;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return dI0*dI_of_Vb(Vb-V0, mymu0, Gamma, EC, kelvin2eV*temp_kwarg, ns);

def dIdV_all_zero(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat, dI0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat) + dIdV_lorentz_zero(Vb, V0, dI0, Gamma, EC);

def dIdV_all(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat, dI0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat) + dIdV_lorentz(Vb, V0, dI0, Gamma, EC);

####################################################################
#### main

def fit_dIdV(metal, nots, percents, stop_at, by_hand=False, num_dev = 4, verbose=0):
    '''
    The main function for fitting the metal Pc dI/dV data
    The data is stored as metal/__dIdV.txt where __ is the temperature
    Args:
    - metal, str, the name of the metal, also gives path to data folder
    nots, a tuple of initial guesses for all params:
            - ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm"] for impurity & magnon background
            - those, plus dI0, Gamma0, EC for oscillations
    - percents: tuple of one percent for each entry in not. These are used to
        construct the upper bound =not*(1+percent) and lower bound=not*(1-percent)
    - stop_at, str telling which fit function to stop at, and return fitting
            params for. For final fit, should always = "lorentz/"
    - by_hand, bool, instead of running scipy.optimize.curve_fit, you can fit by hand
        by setting this to true, because it will just plot with dI0, Gamma0, EC
        fixed to your guesses.
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, temp_kwarg);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-np.mean(dI_exp),2)));

    # unpack
    V0_bound = 1e-2;
    E0_not, G2_not, G3_not, Ec_not, G1_not, ohm_not, dI0_not, Gamma_not, EC_not = nots;
    E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, ohm_percent, dI0_percent, Gamma_percent, EC_percent = percents
    
    #### fit background

    # initial fit to magnon + imp
    params_init_guess = np.array([0.0, E0_not, Ec_not, G1_not, G2_not, G3_not, ohm_not]);
    bounds_init = np.array([[-V0_bound, E0_not*(1-E0_percent), Ec_not*(1-Ec_percent), G1_not*(1-G1_percent), G2_not*(1-G2_percent), G3_not*(1-G3_percent), ohm_not*(1-ohm_percent)],
                            [ V0_bound, E0_not*(1+E0_percent), Ec_not*(1+Ec_percent), G1_not*(1+G1_percent), G2_not*(1+G2_percent), G3_not*(1+G3_percent), ohm_not*(1+ohm_percent)]]);
    params_init, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_init_guess, bounds_init, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm"],
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
                            params_init, bounds_back, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm"],
                            stop_bounds = False, verbose=verbose);
    background = dIdV_back(V_exp, *params_back);
    if(verbose > 4): plot_fit(V_exp, dI_exp, background, derivative=False,
                        mytitle="Background (T = {:.1f} K, T_ohm = {:.1f} K)".format(temp_kwarg,params_back[-1]), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'imp_mag/'): return V_exp, dI_exp, params_back, bounds_back;
    
    # imp, mag, lorentz_zero individually
    if(stop_at in ["imp/", "mag/", "sin/"]):
        
        # fit to magnon + imp with dropout
        params_drop, _ = fit_wrapper(dIdV_back, V_exp[dI_exp<background], dI_exp[dI_exp<background],
                                params_back, bounds_back, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm"],
                                stop_bounds = False, verbose=verbose);
        background_drop = dIdV_back(V_exp[dI_exp<background], *params_drop);
        if(verbose > 4): plot_fit(V_exp[dI_exp<background], dI_exp[dI_exp<background], background_drop, derivative=False,
                            mytitle="Background w/ dropout (T= {:.1f} K, T_ohm= {:.1f} K)".format(temp_kwarg, params_drop[-1]), myylabel="$dI/dV_b$ (nA/V)");                               

        # pretty fit to show signatures
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_back(V_exp, *params_back), derivative=True,
                            mytitle="Magnetic impurities and surface magnons ($T = ${:.1f} K,".format(temp_kwarg)+" $T_{ohm} = $"+"{:.1f} K)".format(params_drop[-1]), myylabel="$dI/dV_b$ (nA/V)");                               


        # force G1=0
        params_imp = np.copy(params_drop);
        params_imp[3] = 0;
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_back(V_exp, *params_imp), derivative=True,
                            mytitle="Magnetic impurity ($T = ${:.1f} K,".format(temp_kwarg)+" $T_{ohm} = $"+"{:.1f} K)".format(params_drop[-1]), myylabel="$dI/dV_b$ (nA/V)");
        mask_imp = np.array([1,1,0,0,1,1,1]);
        if(stop_at == 'imp/'): return V_exp, dI_exp, params_imp[mask_imp>0], bounds_back[:,mask_imp>0];

        # force G2, G3=0
        params_mag = np.copy(params_drop);
        params_mag[4] = 0;
        params_mag[5] = 0;
        if(verbose > 4): plot_fit(V_exp, dI_exp-dIdV_back(V_exp, *params_imp), dIdV_back(V_exp, *params_mag), derivative=True,
                            mytitle="Surface magnon (T= {:.1f} K, T_ohm= {:.1f} K)".format(temp_kwarg, params_drop[-1]), myylabel="$dI/dV_b$ (nA/V)");
        mask_mag = np.array([1,0,1,1,0,0,1]);
        if(stop_at == 'mag/'): return V_exp, dI_exp-dIdV_back(V_exp, *params_imp), params_mag[mask_mag>0], bounds_back[:,mask_mag>0];
        raise NotImplementedError;

    #### fit magnon + imp + oscillation
    params_zero_guess = np.zeros((len(params_back)+3,));
    params_zero_guess[:len(params_back)] = params_back; # background only results -> all guess
    bounds_zero = np.zeros((2,len(params_back)+3));
    bounds_zero[:,:len(params_back)] = bounds_back; # background only bounds -> all guess

    # for oscillation
    params_zero_guess[len(params_back):] = np.array([dI0_not, Gamma_not, EC_not]);
    bounds_zero[:,len(params_back):] = np.array([ [dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                                                [ dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
    # start with zero temp oscillations to constrain
    params_zero, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                params_zero_guess, bounds_zero, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_zero), derivative=False,
                mytitle="Landauer_zero fit (T= {:.1f} K, T_ohm= {:.1f} K)".format(temp_kwarg, params_zero[6]), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz_zero/'): return V_exp, dI_exp, params_zero, bounds_zero; 

    # some plotting to help with constraints
    if(by_hand):
        params_plot = np.copy(params_zero);
        params_plot[len(params_back):] = np.array([dI0_not, Gamma_not, EC_not]);
        print(params_plot)
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_plot))
        assert False

    # constrain and do finite temp fit
    params_all_guess = np.copy(params_zero);
    params_all_guess[len(params_back):] = np.array([dI0_not, Gamma_not, EC_not]);
    bounds_all = np.copy(bounds_zero);
    constrain_mask = np.array([1,1,1,0,1,0,1,0,0,0]); # only G1, G3, dI0, Gamma, Ec free
    bounds_all[0][constrain_mask>0] = params_all_guess[constrain_mask>0];
    bounds_all[1][constrain_mask>0] = params_all_guess[constrain_mask>0]+1e-6;
    params_all, _ = fit_wrapper(dIdV_all, V_exp, dI_exp,
                                params_all_guess, bounds_all, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm", "dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_all), derivative=False,
                mytitle="Landauer fit (T= {:.1f} K, T_ohm= {:.1f} K)".format(temp_kwarg, params_all[6]), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz/'): return V_exp, dI_exp, params_all, bounds_all;
    raise NotImplementedError;

####################################################################
#### wrappers

def fit_Mn_data(stop_at,metal="Mn/",verbose=1):
    '''
    Wrapper function for calling fit_dIdV on different temperature data sets
    and saving the results of those fits
    Args:
        - stop_at, str telling which fit function to stop at, and return fitting
            params for. For final fit, should always = "lorentz/" 
    '''
    fname = "fits/";
    stopats_2_func = {'imp/':dIdV_imp, 'mag/':dIdV_mag, 'imp_mag/':dIdV_back, 'lorentz_zero/':dIdV_all_zero, 'lorentz/':dIdV_all};

    # experimental params
    Ts = np.array([5.0,10.0,15.0,20.0,25.0,30.0]);

    # background guesses
    E0_guess, Ec_guess = 0.006, 0.006; # in eV
    E0_percent, Ec_percent = 1.0, 1.0;
    G1_guess, G2_guess, G3_guess = 1000, 1000, 1000; # in nA/V
    G1_percent, G2_percent, G3_percent = 1.5, 1.0, 1.0;
    ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin

    # oscillation guesses
    dI0_guess =   np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]); # unitless scale factor
    Gamma_guess = np.array([2.2, 2.2, 2.2, 2.2, 2.0, 2.0])*1e-3; # in eV
    EC_guess =    np.array([4.9, 4.9, 4.9, 4.9, 5.3, 5.3])*1e-3; # in eV
    dI0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        if(True):
            print("#"*60+"\nT = {:.1f} K".format(Ts[datai]));
            guesses = (E0_guess, G2_guess, G3_guess, Ec_guess, G1_guess, ohm_guess, dI0_guess[datai], Gamma_guess[datai], EC_guess[datai]);
            percents = (E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, ohm_percent, dI0_percent, Gamma_percent, EC_percent);

            # get fit results
            global temp_kwarg; temp_kwarg = Ts[datai]; # very bad practice
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(metal, guesses, percents, stop_at = stop_at, verbose=verbose);
            results.append(temp_results); 
            boundsT.append(temp_bounds);
    
            #save processed x and y data, and store plot
            if(stop_at in ["imp_mag/", "lorentz_zero/", "lorentz/"]):
                plot_fname = fname+stop_at+"stored_plots/{:.0f}".format(Ts[datai]); # <- where to save the fit plot
                y_fit = stopats_2_func[stop_at](x_forfit, *temp_results);
                mytitle="$T_{ohm} = $";
                mytitle += "{:.1f} K, $dI_0 = $ {:.0f} nA/V, $\Gamma_0 = $ {:.5f} eV, $E_C = $ {:.5f} eV".format(*temp_results[-4:])
                print("Saving plot to "+plot_fname);
                np.save(plot_fname+"_x.npy", x_forfit);
                np.save(plot_fname+"_y.npy", y_forfit);
                np.save(plot_fname+"_yfit.npy", y_fit);
                np.savetxt(plot_fname+"_title.txt", [0], header=mytitle);
                np.savetxt(plot_fname+"_results.txt", temp_results, header = str(["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm", "dI0","Gamma", "EC"]), fmt = "%.5f", delimiter=' & ');

    # save
    results, boundsT = np.array(results), np.array(boundsT);
    if(stop_at in ["imp_mag/", "lorentz_zero/", "lorentz/"]):
        print("Saving data to "+fname+stop_at);
        np.savetxt(fname+stop_at+"Ts.txt", Ts);
        np.save(fname+stop_at+"results.npy", results);
        np.save(fname+stop_at+"bounds.npy", boundsT);

def plot_saved_fit(stop_at, combined=[], verbose = 1):
    '''
    '''

    # which fit result is which
    if(stop_at=='imp/'):
        rlabels = np.array(["$V_0$", "$\\varepsilon_0$", "$G_2$", "$G_3$", "$T_{ohm}$"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
    elif(stop_at=='mag/'):
        rlabels = np.array(["$V_0$", "$\\varepsilon_c$", "$G_1$", "$T_{ohm}$"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
    elif(stop_at=='imp_mag/'):
        rlabels = np.array(["$V_0$", "$\\varepsilon_0$", "$\\varepsilon_c$", "$G_1$", "$G_2$", "$G_3$", "$T_{ohm}$"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
    elif(stop_at == 'lorentz_zero/'):
        rlabels = np.array(["$V_0$", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
        rlabels_mask[:-4] = np.zeros_like(rlabels_mask)[:-4];
    elif(stop_at == 'lorentz/'):
        rlabels = np.array(["$V_0$", "$\\varepsilon_0$ (eV)", "$\\varepsilon_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$T_{ohm}$", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
        rlabels_mask[:-4] = np.zeros_like(rlabels_mask)[:-4];
    else: raise NotImplementedError;
    
    # plot each fit
    fname = "fits/"
    Ts = np.loadtxt(fname+stop_at+"Ts.txt");
    from utils import plot_fit
    fig3, ax3 = plt.subplots();
    for Tvali, Tval in enumerate(Ts):
        plot_fname = fname+stop_at+"stored_plots/{:.0f}".format(Tval); # <- where to get/save the fit plot
        temp_results = np.loadtxt(plot_fname+"_results.txt");
        T_ohm = temp_results[6];
        x = np.load(plot_fname+"_x.npy");
        y = np.load(plot_fname+"_y.npy");
        yfit = np.load(plot_fname+"_yfit.npy");
        print("Loading fit from "+plot_fname+"_yfit.npy");

        # temp with ohmic heating
        global temp_kwarg; temp_kwarg = Ts[Tvali] + T_ohm; # very bad practice
        print(">>> Temperature = ", Tval);

        # plot
        if(combined): # plot all at once
            if(Tval in combined):
                offset=0;
                ax3.scatter(x,offset*Tvali+y, color=mycolors[Tvali], marker=mymarkers[Tvali], 
                            label="$T=$ {:.0f} K".format(Tval)+" ($T_{ohm}=$" +"{:.1f} K)".format(T_ohm));
                ax3.plot(x,offset*Tvali+yfit, color="black");
                ax3.set_xlabel("$V_b$ (V)");
                ax3.set_xlim(-0.1,0.1);
                ax3.set_ylabel("$dI/dV_b$ (nA/V)");
                #ax3.set_ylim(300,2800);
                print(temp_results);
        else:
            plot_fit(x, y, yfit, myylabel="$dI/dV_b$ (nA/V)");

    ax3.set_title("Conductance oscillations in EGaIn$|$H$_2$Pc$|$MnPc$|$NCO");
    plt.legend(loc='lower right');
    plt.show();

    # load
    print("Loading data from "+fname+stop_at);
    results = np.load(fname+stop_at+"results.npy");
    boundsT = np.load(fname+stop_at+"bounds.npy");

    # plot fitting results vs T
    nresults = sum([el for el in rlabels_mask]);
    fig, axes = plt.subplots(nresults, sharex=True);
    if(nresults==1): axes = [axes];
    axi = 0
    for resulti in range(len(rlabels)):
        if(rlabels_mask[resulti]):
            axes[axi].plot(Ts, results[:,resulti], color=mycolors[0],marker=mymarkers[0]);
            axes[axi].set_ylabel(rlabels[resulti]);
            axes[axi].plot(Ts,boundsT[:,0,resulti], color=accentcolors[0],linestyle='dashed');
            axes[axi].plot(Ts,boundsT[:,1,resulti], color=accentcolors[0],linestyle='dashed');
            axes[axi].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
            axi += 1;

    # format
    axes[-1].set_xlabel("$B$ (Tesla)");
    axes[0].set_title("Amplitude and period fitting");
    plt.tight_layout();
    plt.show();
    

    # save results in latex table format
    # recall results are [Ti, resulti]
    results_tab = np.append(np.array([[T] for T in Ts]), results, axis = 1);
    np.savetxt(fname+stop_at+"results_table.txt", results_tab, fmt = "%.5f", delimiter=' & ', newline = '\\\ \n');
    print("Saving table to "+fname+stop_at+"results_table.txt");

####################################################################
#### run

if(__name__ == "__main__"):

    stop_ats = ['imp_mag/','imp/','mag/','lorentz_zero/', 'lorentz/'];
    stop_at = stop_ats[-1];
    verbose=10

    # this one executes the fitting and stores results
    fit_Mn_data(stop_at, verbose=verbose);

    # this one plots the stored results
    # combined allows you to plot two temps side by side
    #plot_saved_fit(stop_at, verbose=verbose, combined=[]);

