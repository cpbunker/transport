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
muBohr = 5.788e-5;     # eV/T
gfactor = 2;

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

    # Eq 20 in XGZ's magnon paper
    Delta = muBohr*gfactor*bfield_kwarg;
    retval = G2;
    retval -= (G3/2)*Ffunc(abs(Vb-V0), kelvin2eV*(temp_kwarg+ohmic_heat));
    retval -= (G3/4)*Ffunc(abs(Vb-V0+Delta), kelvin2eV*(temp_kwarg+ohmic_heat));
    retval -= (G3/4)*Ffunc(abs(Vb-V0-Delta), kelvin2eV*(temp_kwarg+ohmic_heat));
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

def dIdV_sin(Vb, V0, amplitude, dV, deltaV, slope, intercept):
    '''
    Sinusoidal fit function - purely mathematical, not physical
    Designed to be passed to scipy.optimize.curve_fit
    '''

    cosines = (-1)*np.cos(2*np.pi*Vb/dV)
    cosines+= (-1)*np.cos(2*np.pi*Vb/(dV+deltaV))+(-1)*np.cos(2*np.pi*Vb/(dV-deltaV))
    return amplitude*cosines + slope*Vb + intercept;

def dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC): 
    '''
    '''

    nmax = 200;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return tau0*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, EC, 0.0, ns);

def dIdV_lorentz(Vb, V0, tau0, Gamma, EC): 
    '''
    '''

    nmax = 200;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return tau0*dI_of_Vb(Vb-V0, mymu0, Gamma, EC, kelvin2eV*temp_kwarg, ns);

def dIdV_all_zero(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat, tau0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat) + dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC);

def dIdV_all(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat, tau0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, E0, Ec, G1, G2, G3, ohmic_heat) + dIdV_lorentz(Vb, V0, tau0, Gamma, EC);

####################################################################
#### main

def fit_dIdV(metal, nots, percents, stop_at, num_dev, by_hand=True, verbose=0):
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
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal+"data/", temp_kwarg);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-np.mean(dI_exp),2)));

    # unpack
    V0_not = 0.0
    V0_bound = 0.01
    E0_not, G2_not, G3_not, Ec_not, G1_not, ohm_not, tau0_not, Gamma_not, EC_not = nots;
    E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, ohm_percent, tau0_percent, Gamma_percent, EC_percent = percents
    
    #### fit background

    # initial fit to magnon + imp
    params_init_guess = np.array([V0_not, E0_not, Ec_not, G1_not, G2_not, G3_not, ohm_not]);
    bounds_init = np.array([[V0_not-V0_bound, E0_not*(1-E0_percent), Ec_not*(1-Ec_percent), G1_not*(1-G1_percent), G2_not*(1-G2_percent), G3_not*(1-G3_percent), ohm_not*(1-ohm_percent)],
                            [V0_not+V0_bound, E0_not*(1+E0_percent), Ec_not*(1+Ec_percent), G1_not*(1+G1_percent), G2_not*(1+G2_percent), G3_not*(1+G3_percent), ohm_not*(1+ohm_percent)]]);   
    params_init, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_init_guess, bounds_init, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm"],
                            stop_bounds = False, verbose=verbose);
    background_init = dIdV_back(V_exp, *params_init);
    if(verbose > 4): plot_fit(V_exp, dI_exp, background_init);

    # remove outliers based on initial fit
    if(num_dev > 0):
        with_outliers = len(V_exp);
        V_exp = V_exp[abs(dI_exp-background_init) < num_dev*dI_dev];
        dI_exp = dI_exp[abs(dI_exp-background_init) < num_dev*dI_dev];
        assert(with_outliers - len(V_exp) <= with_outliers*0.05); # only remove 5%
    else:
        dI_exp = dI_exp[abs(V_exp) < 0.1];
        V_exp = V_exp[abs(V_exp) < 0.1];

    # fit to magnon + imp background with outliers removed
    bounds_back = np.copy(bounds_init);
    params_back, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_init, bounds_back, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm"],
                            stop_bounds = False, verbose=verbose);
    background = dIdV_back(V_exp, *params_back);
    if(verbose > 4): plot_fit(V_exp, dI_exp, background, derivative=False,
                        mytitle="Background (T = {:.1f} K, T_ohm = {:.1f} K, B = {:.1f} T)".format(temp_kwarg,params_back[-1], bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'imp_mag/'): return V_exp, dI_exp, params_back, bounds_back;
    
    # imp, mag, lorentz_zero individually
    if(stop_at in ["imp/", "mag/", "sin/"]):

        if(stop_at=="sin/"):
            dI_exp -= background;
            upthresh, downthresh = -0.0064, -0.0225;
            V_exp, dI_exp = V_exp[V_exp<upthresh], dI_exp[V_exp<upthresh];
            V_exp, dI_exp = V_exp[V_exp>downthresh], dI_exp[V_exp>downthresh];
            sin_guesses = np.array([V0_not, 4, 0.0009, 0.001, 100, 0]);
            sin_bounds = np.empty( (2,len(sin_guesses)) );
            for pi, p in enumerate(sin_guesses):
                sin_bounds[0,pi], sin_bounds[1,pi] = 0.0, 2*p;
            sin_bounds[0,0], sin_bounds[1,0] = -V0_bound, V0_bound;
            #sin_bounds[-1,-1] =
            params_sin, _ = fit_wrapper(dIdV_sin, V_exp, dI_exp, sin_guesses, sin_bounds,
                                        ["V0","amp","dV","deltaV","slope","intercept"], verbose=verbose);
            plot_fit(V_exp, dI_exp, dIdV_sin(V_exp, *params_sin));

            # fit again?
            raise NotImplementedError;
        
        # pretty fit to show signatures
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_back(V_exp, *params_back), derivative=True,
                            mytitle="Magnetic impurities and surface magnons \n $T = ${:.1f} K,".format(temp_kwarg)+" $T_{ohm} = $"+"{:.1f} K, B = {:.1f} T".format(params_back[-1], bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");                               

        # force G1=0
        params_imp = np.copy(params_back);
        params_imp[3] = 0;
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_back(V_exp, *params_imp), derivative=True,
                            mytitle="Magnetic impurity ($T = ${:.1f} K,".format(temp_kwarg)+" $T_{ohm} = $"+"{:.1f} K, B = {:.1f} T)".format(params_back[-1], bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        mask_imp = np.array([1,1,0,0,1,1,1]);
        if(stop_at == 'imp/'): return V_exp, dI_exp, params_imp[mask_imp>0], bounds_back[:,mask_imp>0];

        # force G2, G3=0
        params_mag = np.copy(params_back);
        params_mag[4] = 0;
        params_mag[5] = 0;
        if(verbose > 4): plot_fit(V_exp, dI_exp-dIdV_back(V_exp, *params_imp), dIdV_back(V_exp, *params_mag), derivative=True,
                            mytitle="Surface magnon (T= {:.1f} K, T_ohm= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, params_back[-1], bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        mask_mag = np.array([1,0,1,1,0,0,1]);
        if(stop_at == 'mag/'): return V_exp, dI_exp-dIdV_back(V_exp, *params_imp), params_mag[mask_mag>0], bounds_back[:,mask_mag>0];
        raise NotImplementedError;

    #### fit magnon + imp + oscillation
    params_zero_guess = np.zeros((len(params_back)+3,));
    params_zero_guess[:len(params_back)] = params_back; # background only results -> all guess
    bounds_zero = np.zeros((2,len(params_back)+3));
    bounds_zero[:,:len(params_back)] = bounds_back; # background only bounds -> all guess

    # for oscillation
    params_zero_guess[len(params_back):] = np.array([tau0_not, Gamma_not, EC_not]);
    bounds_zero[:,len(params_back):] = np.array([ [tau0_not*(1-tau0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                                                [ tau0_not*(1+tau0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
    # start with zero temp oscillations to constrain
    params_zero, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                params_zero_guess, bounds_zero, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm","tau0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_zero), derivative=False,
                mytitle="Landauer_zero fit (T= {:.1f} K, T_ohm= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, params_zero[6], bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz_zero/'): return V_exp, dI_exp, params_zero, bounds_zero; 

    # some plotting to help with constraints
    if(by_hand):
        params_plot = np.copy(params_zero);
        params_plot[len(params_back):] = np.array([tau0_not, Gamma_not, EC_not]);
        print(params_plot)
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_plot))
        assert False

    # constrain and do finite temp fit
    params_all_guess = np.copy(params_zero);
    params_all_guess[len(params_back):] = np.array([tau0_not, Gamma_not, EC_not]);
    bounds_all = np.copy(bounds_zero);
    constrain_mask = np.array([1,1,1,0,1,0,1,0,0,0]); # only G1, G3, tau0, Gamma, EC free
    bounds_all[0][constrain_mask>0] = params_all_guess[constrain_mask>0];
    bounds_all[1][constrain_mask>0] = params_all_guess[constrain_mask>0]+1e-6;
    params_all, _ = fit_wrapper(dIdV_all, V_exp, dI_exp,
                                params_all_guess, bounds_all, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm", "tau0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_all), derivative=False,
                mytitle="Landauer fit (T= {:.1f} K, T_ohm= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, params_all[6], bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz/'): return V_exp, dI_exp, params_all, bounds_all;
    raise NotImplementedError;

####################################################################
#### wrappers

def fit_Mn_data(stop_at, metal, verbose=1):
    '''
    Wrapper function for calling fit_dIdV on different temperature data sets
    and saving the results of those fits
    Args:
        - stop_at, str telling which fit function to stop at, and return fitting
            params for. For final fit, should always = "lorentz/" 
    '''
    stopats_2_func = {'imp/':dIdV_imp, 'mag/':dIdV_mag, 'imp_mag/':dIdV_back, 'lorentz_zero/':dIdV_all_zero, 'lorentz/':dIdV_all};

    # experimental params
    Ts = np.loadtxt(metal+"Ts.txt", ndmin=1);
    Bs = np.loadtxt(metal+"Bs.txt", ndmin=1);

    # oscillation guesses
    if(metal=="Mn/"):
        # background guesses
        E0_guess, Ec_guess = 0.005882, 0.015184; # in eV # 0.006, 0.006
        G1_guess, G2_guess, G3_guess = 502, 1019, 982; # in nA/V # 1000,1000,1000
        ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin
        E0_percent, Ec_percent = 1e-6,1e-6; G1_percent, G2_percent, G3_percent = 1e-6,1e-6,1e-6; 
        # oscillation guesses
        tau0_guess =   np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]); # unitless scale factor
        Gamma_guess = np.array([2.2, 2.2, 2.2, 2.2, 2.8, 2.8])*1e-3; # in eV
        EC_guess =    np.array([4.9, 4.9, 4.9, 4.9, 5.3, 5.3])*1e-3; # in eV
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev=0;

    ####
        
    elif(metal=="Mnv2/"):
        # background guesses
        E0_guess, Ec_guess = 0.002835, 0.019568; # in eV # 0.006, 0.006
        G1_guess, G2_guess, G3_guess = 2287, 827, 1526; # in nA/V # 1000,1000,1000
        ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin
        E0_percent, Ec_percent = 1e-6,1e-6; G1_percent, G2_percent, G3_percent = 1e-6,1e-6,1e-6; 
        # oscillation guesses
        tau0_guess =   np.array([0.01, 0.01, 0.01, 0.01, 0.01]); # unitless scale factor
        Gamma_guess = np.array([2.1, 2.2, 2.4, 2.8, 2.2])*1e-3; # in eV
        EC_guess =    np.array([5.9, 5.8, 5.6, 5.4, 5.0])*1e-3; # in eV
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev = 0;

    ####

    elif(metal=="Mn2Tesla/"):
        # background guesses
        E0_guess, Ec_guess = 0.001509, 0.019842; # in eV 
        G1_guess, G2_guess, G3_guess = 3300, 811, 1983; # in nA/V 
        ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin
        E0_percent, Ec_percent = 1e-6,1e-6; G1_percent, G2_percent, G3_percent = 1e-6,1e-6,1e-6; 
        # oscillation guesses
        tau0_guess = np.array([0.01]); Gamma_guess = np.array([2.0])*1e-3; EC_guess = np.array([5.8])*1e-3;
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev = 0;

    ####

    elif(metal=="Mn4Tesla/"):
        # background guesses
        E0_guess, Ec_guess = 0.003002, 0.000088; # in eV 
        G1_guess, G2_guess, G3_guess = 1827, 701, 1214; # in nA/V 
        ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin
        E0_percent, Ec_percent = 1e-6, 1e-6; G1_percent, G2_percent, G3_percent = 1e-6, 1e-6, 1e-6; 
        # oscillation guesses
        tau0_guess = np.array([0.01]); Gamma_guess = np.array([2.2])*1e-3; EC_guess = np.array([5.5])*1e-3;
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev = 0;
        
    ####

    elif(metal=="Mn7Tesla/"):
        # background guesses
        E0_guess, Ec_guess = 0.003871, 0.000100; # in eV 
        G1_guess, G2_guess, G3_guess = 2506, 643, 1049; # in nA/V
        ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin
        E0_percent, Ec_percent = 1e-6, 1e-6; G1_percent, G2_percent, G3_percent = 1e-6, 1e-6, 1e-6; 
        # oscillation guesses
        tau0_guess = np.array([0.01]); Gamma_guess = np.array([2.2])*1e-3; EC_guess = np.array([5.7])*1e-3;
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev = 0;

    ####

    elif(metal=="Mn-2Tesla/"):
        # background guesses
        E0_guess, Ec_guess = 0.038644, 0.002259; # in eV 
        G1_guess, G2_guess, G3_guess = 1709, 840, 264; # in nA/V
        ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin
        E0_percent, Ec_percent = 1e-6, 1e-6; G1_percent, G2_percent, G3_percent = 1e-6, 1e-6, 1e-6;
        # oscillation guesses
        tau0_guess = np.array([0.01]); Gamma_guess = np.array([2.6])*1e-3; EC_guess = np.array([5.7])*1e-3;
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev = 0;

    ####

    elif(metal=="Mn-4Tesla/"):
        # background guesses
        E0_guess, Ec_guess = 0.003002, 0.000099; # in eV 
        G1_guess, G2_guess, G3_guess = 1827, 702, 1213; # in nA/V
        ohm_guess, ohm_percent = 10.0, 1.0; # in kelvin
        E0_percent, Ec_percent = 1e-6, 1e-6; G1_percent, G2_percent, G3_percent = 1e-6,1e-6,1e-6;
        # oscillation guesses
        tau0_guess = np.array([0.01]); Gamma_guess = np.array([2.2])*1e-3; EC_guess = np.array([5.5])*1e-3;
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev = 0;
        
    ####
        
    elif(metal=="MnTrilayer/"):
        E0_guess, Ec_guess = 0.002, 0.002; # in eV
        G1_guess, G2_guess, G3_guess = 11000, 360, 60; # in nA/V
        G3_percent, E0_percent = 0.2, 0.2;
        ohm_guess = 0.1
        tau0_guess =   np.array([0.02, 0.02, 0.02, 0.02,0.02,0.02,0.02]); # unitless scale factor
        Gamma_guess = np.array([2.2, 2.2, 2.2, 2.2,2.2,2.2,2.2])*1e-3; # in eV
        EC_guess =    np.array([5.9, 5.8, 5.6, 5.4,5.4,5.4,5.4])*1e-3; # in eV
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        numdev = 1;

    ####
        
    else: raise NotImplementedError;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        if(True):
            print("#"*60+"\nT = {:.1f} K".format(Ts[datai]));
            guesses = (E0_guess, G2_guess, G3_guess, Ec_guess, G1_guess, ohm_guess, tau0_guess[datai], Gamma_guess[datai], EC_guess[datai]);
            percents = (E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, ohm_percent, tau0_percent, Gamma_percent, EC_percent);

            # get fit results
            global temp_kwarg; temp_kwarg = Ts[datai]; # very bad practice
            global bfield_kwarg; bfield_kwarg = Bs[datai];
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(metal,
                    guesses, percents, stop_at, numdev, by_hand=False, verbose=verbose);
            results.append(temp_results); 
            boundsT.append(temp_bounds);
    
            #save processed x and y data, and store plot
            if(stop_at in ["imp_mag/", "lorentz_zero/", "lorentz/"]):
                plot_fname = metal+stop_at+"stored_plots/{:.0f}".format(Ts[datai]); # <- where to save the fit plot
                y_fit = stopats_2_func[stop_at](x_forfit, *temp_results);
                mytitle="$T_{ohm} = $";
                mytitle += "{:.1f} K, $\\tau_0 = $ {:.0f} nA/V, $\Gamma = $ {:.5f} eV, $E_C = $ {:.5f} eV".format(*temp_results[-4:])
                print("Saving plot to "+plot_fname);
                np.save(plot_fname+"_x.npy", x_forfit);
                np.save(plot_fname+"_y.npy", y_forfit);
                np.save(plot_fname+"_yfit.npy", y_fit);
                np.savetxt(plot_fname+"_title.txt", [0], header=mytitle);
                np.savetxt(plot_fname+"_results.txt", temp_results, header = str(["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm", "tau0","Gamma", "EC"]), fmt = "%.5f", delimiter=' & ');

    # save
    results, boundsT = np.array(results), np.array(boundsT);
    if(stop_at in ["imp_mag/", "lorentz_zero/", "lorentz/"]):
        print("Saving data to "+metal+stop_at);
        np.save(metal+stop_at+"results.npy", results);
        np.save(metal+stop_at+"bounds.npy", boundsT);

def plot_saved_fit(stop_at, metal, combined=[], verbose = 1):
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
        rlabels = np.array(["$V_0$", "$\\varepsilon_0$ (eV)", "$\\varepsilon_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$T_{ohm}$", "$\\tau_0$", "$\Gamma$ (eV)", "$E_C$ (eV)"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
        rlabels_mask[:-3] = np.zeros_like(rlabels_mask)[:-3];
    elif(stop_at == 'lorentz/' or 'lorentz_backup/'):
        rlabels = np.array(["$V_0$", "$\\varepsilon_0$ (eV)", "$\\varepsilon_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$T_{ohm}$", "$\\tau_0$", "$\Gamma$ (eV)", "$E_C$ (eV)"]);   
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
        rlabels_mask[:-3] = np.zeros_like(rlabels_mask)[:-3];
    else: raise NotImplementedError;
    
    # plot each fit
    Ts = np.loadtxt(metal+"Ts.txt", ndmin=1);
    Bs = np.loadtxt(metal+"Bs.txt", ndmin=1);
    
    from utils import plot_fit
    fig3, ax3 = plt.subplots();
    for Tvali, Tval in enumerate(Ts):
        plot_fname = metal+stop_at+"stored_plots/{:.0f}".format(Tval); # <- where to get/save the fit plot
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
                offset = 800;
                ax3.scatter(x,offset*Tvali+y, color=mycolors[Tvali], marker=mymarkers[Tvali], 
                            label="$T=$ {:.1f} K".format(Tval)+", $T_{ohm}=$" +"{:.1f} K, B = {:.1f} T".format(T_ohm, Bs[Tvali]));
                ax3.plot(x,offset*Tvali+yfit, color="black");
                ax3.set_xlabel("$V_b$ (V)");
                ax3.set_xlim(-0.1,0.1);
                ax3.set_ylabel("$dI/dV_b$ (nA/V)");
                #ax3.set_ylim(300,2800);
                print(temp_results);
        else:
            if(verbose): plot_fit(x, y, yfit, myylabel="$dI/dV_b$ (nA/V)", mytitle="$T=$ {:.1f} K".format(Tval)+", $T_{ohm}=$" +"{:.1f} K, B = {:.1f} T".format(T_ohm, Bs[Tvali]));

    ax3.set_title("Conductance oscillations in EGaIn$|$H$_2$Pc$|$MnPc$|$NCO");
    plt.legend(loc='lower right');
    plt.show();

    # load
    print("Loading data from "+metal+stop_at);
    results = np.load(metal+stop_at+"results.npy");
    boundsT = np.load(metal+stop_at+"bounds.npy");

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
    axes[-1].set_xlabel("$T$ (K)");
    axes[0].set_title("Amplitude and period fitting");
    plt.tight_layout();
    plt.show();

    # save results in latex table format
    # recall results are [Ti, resulti]
    results_tab = np.append(np.array([[Ts[Tvali], Bs[Tvali]] for Tvali in range(len(Ts))]), results, axis = 1);
    res_header = str(["T (K)", "B (T)", "V0 (V)", "eps_0 (eV)", "eps_c (eV)", "G1 (nA/V)", "G2 (nA/V)", "G3 (nA/V)", "T_ohm (K)", "tau0 ()","Gamma (eV)", "EC (eV)"])
    np.savetxt(metal+stop_at+"results_table.txt", results_tab, header=res_header, fmt = "%.5f", delimiter=' & ', newline = '\\\ \n');
    print("Saving table to "+metal+stop_at+"results_table.txt");

    # period vs T
    if(stop_at == "lorentz/"): # <-------- !!!!
        pfig, pax = plt.subplots();
        combined_y, combined_x = [], []
        if(metal == "Mn_combined/"):
            folders = ["Mn/", "Mnv2/"];
        else:
            folders = [metal];
            
        for folder in folders:
            # load
            print("Loading data from "+folder+stop_at);
            Ts = np.loadtxt(folder+"Ts.txt");
            results = np.load(folder+stop_at+"results.npy");           
            # plot
            periods = 4*results[:,-1]*1000;
            periods, Ts = periods, Ts;
            gammas = results[:,-2]*1000;
            charges = results[:,-1]*1000;
            myx, myy = gammas, charges;
            myx, myy = Ts, periods;
            pax.scatter(myx, myy, color=mycolors[0], label="Data");
            combined_y.append(myy); combined_x.append(myx);

        # fit
        if(metal == "Mn_combined/"): myx = np.append(combined_x[0], combined_x[1]); myy = np.append(combined_y[0], combined_y[1]);
        coefs = np.polyfit(myx, myy, 1);
        myyfit = coefs[0]*myx+coefs[1];
        myrmse = np.sqrt( np.mean( np.power(myy-myyfit,2) ))/abs(np.max(myy)-np.min(myy));
        pax.plot(myx, myyfit, color=accentcolors[0], label = "Slope = {:.2f} meV/T = {:.2f}$k_B$, b = {:.2f} meV".format(coefs[0], coefs[0]/1000/kelvin2eV, coefs[1]));
        pax.plot( [np.mean(myx)], [np.mean(myy)], color='white', label = "RMSE = {:1.5f}".format(myrmse));
    
        # format
        pax.set_ylabel("");
        pax.set_xlabel("");
        pax.set_title("Dataset = "+str(metal[:-1]));
        plt.legend();
        plt.tight_layout();
        plt.show();

####################################################################
#### run

if(__name__ == "__main__"):

    metal = "Mnv2/"; # tells which experimental data to load
    stop_ats = ['imp_mag/', 'sin/', 'imp/','mag/','lorentz_zero/', 'lorentz/'];
    stop_at = stop_ats[-2];
    verbose=10;

    # this one executes the fitting and stores results
    fit_Mn_data(stop_at, metal, verbose=verbose);

    # this one plots the stored results
    # combined allows you to plot two temps side by side
    #plot_saved_fit(stop_at, metal, verbose=verbose, combined=[2.5]);

