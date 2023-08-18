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

def dIdV_imp(Vb, V0, eps0, G2, G3, T_surf):
    '''
    Magnetic impurity scattering
    Designed to be passed to scipy.optimize.curve_fit
    '''

    def Ffunc(E, kBT):
        # Eq 17 in XGZ's magnon paper
        numerator = np.log(1+ eps0/(E+kBT));
        denominator = 1 - kBT/(eps0+0.4*E) + 12*np.power(kBT/(eps0+2.4*E),2);
        return numerator/denominator;

    # Eq 20 in XGZ's magnon paper
    Delta = muBohr*gfactor*bfield_kwarg;
    retval = G2;
    retval -= (G3/2)*Ffunc(abs(Vb-V0), kelvin2eV*T_surf);
    retval -= (G3/4)*Ffunc(abs(Vb-V0+Delta), kelvin2eV*T_surf);
    retval -= (G3/4)*Ffunc(abs(Vb-V0-Delta), kelvin2eV*T_surf);
    return retval;

def dIdV_mag(Vb, V0, epsc, G1, T_surf):
    '''
    Surface magnon scattering
    Designed to be passed to scipy.optimize.curve_fit
    '''

    def Gmag(E, kBT):
        # Eq 12 in XGZ's magnon paper
        ret = np.zeros_like(E);
        ret += -2*kBT*np.log(1-np.exp(-epsc/kBT));
        ret += (E+epsc)/( np.exp( (E+epsc)/kBT) - 1);
        ret += (E-epsc)/(-np.exp(-(E-epsc)/kBT) + 1);
        return ret
        
    return G1*Gmag(abs(Vb-V0), kelvin2eV*T_surf);

def dIdV_back_phys(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, Gamma):
    '''
    Magnetic impurity and surface magnon scattering, combined
    Designed to be passed to scipy.optimize.curve_fit
    '''

    # include G dependence on Gamma
    G1, G2, G3 = Gamma*Gamma*G1*1e9, Gamma*Gamma*G2*1e9, Gamma*Gamma*G3*1e9;
    return dIdV_imp(Vb, V0, eps0, G2, G3, T_surf)+dIdV_mag(Vb, V0, epsc, G1, T_surf);

def dIdV_back(Vb, dI0, alpha_neg, alpha_pos, Gamma):
    '''
    Simple background for fitting the MnTrilayer data
    dIdV = alpha_neg abs(eVb) \theta(-eVb) + alpha_pos abs(eVb) \theta(eVb)
    Designed to be passed to scipy.optimize.curve_fit
    '''
    if( not isinstance(Vb, np.ndarray)): raise TypeError;

    # include alpha dependence on Gamma
    alpha_neg, alpha_pos = Gamma*Gamma*alpha_neg*1e9, Gamma*Gamma*alpha_pos*1e9;

    # combine Vb<0 and Vb>0 slopes
    ret = np.zeros_like(Vb);
    ret[(Vb)<0] = alpha_neg*abs(Vb[(Vb)<0]);
    ret[(Vb)>=0] = alpha_pos*abs(Vb[(Vb)>=0]);
    return ret+dI0;

def dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC): 
    '''
    '''

    nmax = 100; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    return tau0*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, EC, 0.0, ns);

def dIdV_all_zero(Vb, V0, dI0, alpha_neg, alpha_pos, tau0, Gamma, ECs):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Can be passed to scipy.optimize.curve_fit only if ECs is a float
    '''

    # background
    ret = dIdV_back(Vb-V0, dI0, alpha_neg, alpha_pos, Gamma);

    # charging
    if(isinstance(ECs,float)): # for scipy_curve_fit fitting
        ret += dIdV_lorentz_zero(Vb, V0, tau0, Gamma, ECs);
    else:
        for EC in ECs: # for brute-force
            ret += dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC);
    return ret;

def dIdV_all_zero_phys(Vb, V0, epsc, G1, T_surf, tau0, Gamma, ECs):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Can be passed to scipy.optimize.curve_fit only if ECs is a float
    '''

    # background
    eps0, G2, G3 = 0.001, 0, 0;
    ret = dIdV_back_phys(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, Gamma);

    # charging
    if(isinstance(ECs,float)): # for scipy_curve_fit fitting
        ret += dIdV_lorentz_zero(Vb, V0, tau0, Gamma, ECs);
    else:
        for EC in ECs: # for brute-force
            ret += dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC);
    return ret;

def cost_func(yexp, yfit, which="rmse"):
    if(which=="rmse"):
        return np.sqrt( np.mean( np.power(yexp-yfit,2) ))/abs(np.max(yexp)-np.min(yexp));
    elif(which==""):
        return -9999;
    else: raise NotImplementedError;


####################################################################
#### main

def fit_dIdV(metal, nots, percents, stop_at, num_dev=3, verbose=0):
    '''
    The main function for fitting the metal Pc dI/dV data
    The data is stored as metal/__dIdV.txt where __ is the temperature
    Args:
    - metal, str, the name of the metal, also gives path to data folder
    - nots, a tuple of initial guesses for all params:
            - ["V0", "eps0", "epsc", "G1", "G2", "G3", "T_surf"] for impurity & magnon background
            - those, plus dI0, Gamma0, EC, T_film for oscillations
    - percents: tuple of one percent for each entry in not. These are used to
        construct the upper bound =not*(1+percent) and lower bound=not*(1-percent)
    - num_dev, int, how many standard deviations away from initial fit to keep before
            discarding outliers
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal+"data/", temp_kwarg);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-np.mean(dI_exp),2)));

    # unpack
    V0_not, alpha_not, dI0_not, tau0_not, Gamma_not, EC_not = nots;
    V0_bound, alpha_percent, dI0_percent, tau0_percent, Gamma_percent, EC_percent = percents
    params_base = np.array([V0_not, dI0_not, alpha_not, alpha_not, tau0_not, Gamma_not, EC_not]);
    bounds_base = np.array([[V0_not-V0_bound, dI0_not*(1-dI0_percent), alpha_not*(1-alpha_percent), alpha_not*(1-alpha_percent), tau0_not*(1-tau0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                            [V0_not+V0_bound, dI0_not*(1+dI0_percent), alpha_not*(1+alpha_percent), alpha_not*(1+alpha_percent), tau0_not*(1+tau0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent)]]);  

    # initial fit
    params_init_guess = np.copy(params_base);
    bounds_init = np.copy(bounds_base);
    params_init, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                            params_init_guess, bounds_init, ["V0", "dI0", "alpha_neg", "alpha_pos", "tau0", "Gamma", "EC"],
                            stop_bounds = False, verbose=verbose);
    fit_init = dIdV_all_zero(V_exp, *params_init);
    if(verbose > 4): plot_fit(V_exp, dI_exp, fit_init, mytitle="Initial fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg));

    # remove outliers
    with_outliers = len(V_exp);
    V_exp = V_exp[abs(dI_exp-fit_init) < num_dev*dI_dev];
    dI_exp = dI_exp[abs(dI_exp-fit_init) < num_dev*dI_dev];
    assert(with_outliers - len(V_exp) <= with_outliers*0.0999); # remove no more than 10%

    # fit again
    params_zero, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                params_init, bounds_init, ["V0", "dI0", "alpha_neg", "alpha_pos", "tau0", "Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_zero), derivative=False,
                mytitle="Landauer_zero fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at=="back/"): return V_exp, dI_exp, params_zero, bounds_init;

    # physical background
    if False:
        epsc_not, G1_not, ohm_not = 0.002, 0.5, 8.0;
        epsc_percent, G1_percent, ohm_percent = 1,1,1;
        params_phys_guess = np.array([V0_not, epsc_not, G1_not, temp_kwarg+ohm_not, tau0_not, Gamma_not, EC_not]);
        bounds_phys = np.array([[V0_not-V0_bound, epsc_not*(1-epsc_percent), G1_not*(1-G1_percent), temp_kwarg+ohm_not*(1-ohm_percent), tau0_not*(1-tau0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],    
                                [V0_not+V0_bound, epsc_not*(1+epsc_percent), G1_not*(1+G1_percent), temp_kwarg+ohm_not*(1+ohm_percent), tau0_not*(1+tau0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent)]]);
        params_phys, _ = fit_wrapper(dIdV_all_zero_phys, V_exp, dI_exp,
                                      params_phys_guess, bounds_phys, ["V0", "epsc", "G1", "T_surf", "tau0", "Gamma", "EC"],
                                      stop_bounds = False, verbose=verbose);
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero_phys(V_exp, *params_phys), derivative=False,
                    mytitle="Magnon fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        if(stop_at=="back_phys/"): return V_exp, dI_exp, params_phys, bounds_phys;
        
    # oscillations only
    back_mask = np.array([0,1,1,1,0,1,0]);
    osc_mask = np.array([1,0,0,0,1,1,1]);
    params_back = np.copy(params_zero)[back_mask>0];
    dI_exp = dI_exp - dIdV_back(V_exp-params_zero[0], *params_back); # with V0
    params_osc_guess = np.copy(params_zero)[osc_mask>0];
    bounds_osc = np.copy(bounds_base)[:,osc_mask>0];
    params_osc, rmse_final = fit_wrapper(dIdV_lorentz_zero, V_exp, dI_exp,
                                    params_osc_guess, bounds_osc, ["V0", "tau0", "Gamma", "EC"],
                                    stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_lorentz_zero(V_exp, *params_osc),
                mytitle="Oscillation fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");   
    if(stop_at=="osc/"): return V_exp, dI_exp, params_osc, bounds_osc; 

def comp_with_null(xvals, yvals, yfit, conv_scale = None, noise_mult=1.0):
    '''
    Compare the best fit with the following "null hypotheses"
    - a convolution of the yvals which smoothes out features on the scale
        conv_scale, chosen to smooth out the oscillations we are trying to fit
    - a straight line at y = avg of xvals
    - previous plus gaussian noise with sigma = std dev of yvals * noise_mult
    '''
    if( yvals.shape != yfit.shape): raise ValueError;
    if(conv_scale==None): conv_scale = len(xvals)//10;

    # construct covoluted vals
    kernel = np.ones((conv_scale,)) / conv_scale;
    conv_vals = np.convolve(yvals, kernel, mode="same");

    # construct noisy data
    noise_gen = np.random.default_rng();
    noise_scale = np.std(yvals)*noise_mult;
    noise = noise_gen.normal(scale=noise_scale, size = xvals.size);
    yline = np.mean(yvals)*np.ones_like(xvals);
    
    # compare cost func
    fits = [yfit, conv_vals, yline, yline+noise];
    costfig, costaxes = plt.subplots(len(fits));
    for fiti, fit in enumerate(fits):
        costaxes[fiti].scatter(xvals, yvals, color=mycolors[0], marker=mymarkers[0]);
        costaxes[fiti].plot(xvals, fit, color=accentcolors[0], label=cost_func(yvals, fit));
        costaxes[fiti].legend();
    plt.show();

####################################################################
#### wrappers

def fit_Mn_data(stop_at, metal, verbose=1):
    '''
    '''
    stopats_2_func = {"back/" : dIdV_all_zero, "osc/" : dIdV_lorentz_zero};

    # experimental params
    Ts = np.loadtxt(metal+"Ts.txt", ndmin=1);
    Bs = np.loadtxt(metal+"Bs.txt", ndmin=1);

    # guesses
    V0_guess, dI0_guess, alpha_guess = -0.0025, 150, 0.5;
    V0_percent, dI0_percent, alpha_percent = 0.005, 0.8, 0.8;
    tau0_guess, Gamma_guess, EC_guess = 0.01, 0.005, 0.005
    tau0_percent, Gamma_percent, EC_percent = 0.8,0.8,0.8;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        if(True and datai<4):
            global temp_kwarg; temp_kwarg = Ts[datai];
            global bfield_kwarg; bfield_kwarg = Bs[datai];
            print("#"*60+"\nT = {:.1f} K".format(Ts[datai]));
            
            #### get fit results ####
            guesses = (V0_guess, alpha_guess, dI0_guess, tau0_guess, Gamma_guess, EC_guess);
            percents = (V0_percent, alpha_percent, dI0_percent, tau0_percent, Gamma_percent, EC_percent);
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(metal,
                    guesses, percents, stop_at, verbose=verbose);
            results.append(temp_results); 
            boundsT.append(temp_bounds);            
            if(stop_at in ["osc/"]):

                # compare with null
                comp_with_null(x_forfit, y_forfit, stopats_2_func[stop_at](x_forfit, *temp_results));

                #save processed x and y data, and store plot
                plot_fname = metal+stop_at+"stored_plots/{:.0f}".format(Ts[datai]); # <- where to save the fit plot
                y_fit = stopats_2_func[stop_at](x_forfit, *temp_results);  
                print("Saving plot to "+plot_fname);
                np.save(plot_fname+"_x.npy", x_forfit);
                np.save(plot_fname+"_y.npy", y_forfit);
                np.save(plot_fname+"_yfit.npy", y_fit);
                #np.savetxt(plot_fname+"_results.txt", temp_results, header = str([]), fmt = "%.5f", delimiter=' & ');
                #mytitle="$\\tau_0 = $ {:.0f} nA/V, $\Gamma = $ {:.5f} eV, $E_C = $ {:.5f} eV, T_film = "+"{:.1f} K".format(*temp_results[-4:]) 
                #np.savetxt(plot_fname+"_title.txt", [0], header=mytitle);
                
    # save
    results, boundsT = np.array(results), np.array(boundsT);
    if(stop_at in ["lorentz_zero/", "lorentz/"]):
        print("Saving data to "+metal+stop_at);
        np.save(metal+stop_at+"results.npy", results);
        np.save(metal+stop_at+"bounds.npy", boundsT);
    

####################################################################
#### run

if(__name__ == "__main__"):

    metal = "MnTrilayer/"; # tells which experimental data to load
    stop_ats = ["back/", "back_phys/", "osc/"];
    stop_at = stop_ats[-1];
    verbose=10;

    # this one executes the fitting and stores results
    fit_Mn_data(stop_at, metal, verbose=verbose);

    # this one plots the stored results
    # combined allows you to plot two temps side by side
    #plot_saved_fit(stop_at, metal, verbose=verbose, combined=[]);

