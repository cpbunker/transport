'''
Capacitor model + landauer formula for describing evenly spaced dI/dVb oscillations
due to single electron charging effects
'''

from utils import plot_fit, load_dIdV, fit_wrapper
from landauer import dI_of_Vb, dI_of_Vb_zero

import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(1016)

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
        return ret;

    # split Vb pos and Vb neg branches
    #Vb_pos, Vb_neg = np.zeros_like(Vb), np.zeros_like(Vb);
    #Vb_pos[Vb>=0] = Vb[Vb>=0];
    #Vb_neg[Vb <0] = Vb[Vb <0];
        
    return G1*Gmag(abs(Vb-V0), kelvin2eV*T_surf);

def dIdV_back(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, Gamma):
    '''
    Magnetic impurity and surface magnon scattering, combined
    Designed to be passed to scipy.optimize.curve_fit
    '''

    # include G dependence on Gamma
    G1, G2, G3 = Gamma*Gamma*G1*1e9, Gamma*Gamma*G2*1e9, Gamma*Gamma*G3*1e9;
    return dIdV_imp(Vb, V0, eps0, G2, G3, T_surf)+dIdV_mag(Vb, V0, epsc, G1, T_surf);

from utils import error_func, comp_with_null

def make_EC_list(EC, d=0.1):
    '''
    '''
    assert(num_EC_kwarg==2);
    return np.array([EC, EC*(1-d)])

def make_EC_dist(EC_mu):
    '''
    '''
    assert(num_EC_kwarg==2);
    ECs = [];
    for _ in range(num_EC_kwarg):
        ECs.append( EC_mu*random.lognormvariate(0.1, 0.4) );
    return np.array(ECs);

def dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC): 
    '''
    '''

    nmax = 100; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    ret = np.zeros_like(Vb);
    for ECval in make_EC_list(EC):
        ret += tau0*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, ECval, 0.0, ns);
    return ret;

def search_space_lorentz_zero(V_exp, dI_exp, dI_back, V0, tau0, Gamma, EC_mu, which_error = "rmse", num_trials = 1000, verbose=0):
    '''
    '''

    # trials
    trial_errors = np.zeros((num_trials,),dtype=float);
    trial_ECs = np.zeros((num_trials, num_EC_kwarg),dtype=float);
    trial_fits = np.zeros((num_trials, len(V_exp)),dtype=float);
    for trial in range(num_trials):

        # oscillation terms on top of background
        dI_osc = np.zeros_like(dI_back);
        trial_ECs[trial] = make_EC_dist(EC_mu);
        for EC in trial_ECs[trial]:
            dI_osc += dIdV_lorentz_zero(V_exp, V0, tau0, Gamma, EC);
        if(trial % 100 == 0): print("Trial #",trial, " ECs = ",trial_ECs[trial]);
        
        # get fit and cost function
        dI_fit = dI_back + dI_osc;
        trial_errors[trial] = error_func(dI_exp, dI_fit, which=which_error);
        trial_fits[trial] = dI_fit;

    # compare across trials
    besti = np.argmin(trial_errors);
    return trial_fits[besti], trial_ECs[besti];

def dIdV_lorentz_fine(Vb, V0, EC1, EC2, EC3):
    '''
    Like dIdV_lorentz_zero, but doesn't fit tau0 or Gamma, so faster
    '''
    assert(num_EC_kwarg==3);
    
    nmax = 100; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    ret = np.zeros_like(Vb);
    for ECval in [EC1, EC2, EC3]:
        ret += tau0_kwarg*dI_of_Vb_zero(Vb-V0, mymu0, Gamma_kwarg, ECval, 0.0, ns);
    return ret;

def dIdV_all_zero(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, tau0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, Gamma) + dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC);

def dIdV_all_fine(Vb, V0, eps0, epsc, G1, G2, G3, EC1, EC2, EC3):
    '''
    For refining multiple EC
    '''

    return dIdV_back(Vb, V0, eps0, epsc, G1, G2, G3, temp_kwarg, Gamma_kwarg) + dIdV_lorentz_fine(Vb, V0, EC1, EC2, EC3); 

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

    if False:
        from utils import remove_jumps
        locs = np.array([-0.0021,-0.00206,-0.0201,-0.0074,0.00243]);
        ds = np.array([50,28,100,89,73]);
        ts = np.array(["<","<","<","<",">"]);
        remove_jumps(V_exp, dI_exp, locs, ds, ts);

    #################
    dropout_l, dropout_g = 0.0, 0.01;
    dI_exp = dI_exp[V_exp > dropout_l];
    dI_exp = dI_exp[V_exp < dropout_g];
    V_exp = V_exp[V_exp > dropout_l];
    V_exp = V_exp[V_exp < dropout_g];
    # fourier transform
    raise NotImplementedError

    # unpack
    V0index = np.argmin( abs(V_exp));
    V0_bound = abs(V_exp[V0index] - V_exp[3+V0index]);
    V0_not, eps0_not, epsc_not, G1_not, G2_not, G3_not, ohm_not, tau0_not, Gamma_not, EC_not = nots;
    V0_percent, eps0_percent, epsc_percent, G1_percent, G2_percent, G3_percent, ohm_percent, tau0_percent, Gamma_percent, EC_percent = percents
    params_base = np.array([V0_not, eps0_not, epsc_not, G1_not, G2_not, G3_not, temp_kwarg+ohm_not, tau0_not, Gamma_not, EC_not]);
    bounds_base = np.array([[V0_not-V0_bound, eps0_not*(1-eps0_percent), epsc_not*(1-epsc_percent), G1_not*(1-G1_percent), G2_not*(1-G2_percent), G3_not*(1-G3_percent), temp_kwarg+ohm_not*(1-ohm_percent), tau0_not*(1-tau0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent) ],
                            [V0_not+V0_bound, eps0_not*(1+eps0_percent), epsc_not*(1+epsc_percent), G1_not*(1+G1_percent), G2_not*(1+G2_percent), G3_not*(1+G3_percent), temp_kwarg+ohm_not*(1+ohm_percent), tau0_not*(1+tau0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);

    #### initial fit ####
    # all data present
    # lorentzians turned off
    # Gamma frozen
    back_mask = np.array([1,1,1,1,1,1,1,0,1,0]); # turn off lorentzian
    params_init_guess = np.copy(params_base)[back_mask>0];
    bounds_init = np.copy(bounds_base)[:,back_mask>0];
    bounds_init[0,-1] = params_init_guess[-1]; # freeze gamma
    bounds_init[1,-1] = params_init_guess[-1]+1e-12;
    params_init, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_init_guess, bounds_init, ["V0", "eps_0", "eps_c", "G1", "G2", "G3", "T_surf", "Gamma"],
                            stop_bounds = False, verbose=verbose);
    fit_init = dIdV_back(V_exp, *params_init);
    if(verbose > 4): plot_fit(V_exp, dI_exp, fit_init, mytitle="Initial fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");

    # remove outliers
    with_outliers = len(V_exp);
    V_exp = V_exp[abs(dI_exp-fit_init) < num_dev*dI_dev];
    dI_exp = dI_exp[abs(dI_exp-fit_init) < num_dev*dI_dev];
    assert(with_outliers - len(V_exp) <= with_outliers*0.05); # remove no more than 5%

    #### Background fit ####
    # outliers removed
    # lorentzians still turned off
    # Gamma still frozen
    V0_bound = V0_percent;  # reset V0 now that outliers are removed
    params_init[0] = V0_not;bounds_init[0,0] = V0_not-V0_bound;bounds_init[1,0] = V0_not+V0_bound;
    params_back, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                                params_init, bounds_init, ["V0", "eps_0", "eps_c", "G1", "G2", "G3", "T_surf", "Gamma"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_back(V_exp, *params_back), derivative=False,
                mytitle="Background fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at=="back/"): return V_exp, dI_exp, params_back, bounds_init;

    #### Lorentz_zero fit ####
    # outliers removed
    # lorentzians turned on
    # Gamma unfrozen
    params_zero_guess = np.copy(params_base);
    params_zero_guess[back_mask>0] = params_back;
    bounds_zero = np.copy(bounds_base);
    bounds_zero[:,0] = bounds_init[:,0]; # update V0 bounds
    params_zero, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                 params_zero_guess, bounds_zero, ["V0", "eps_0", "eps_c", "G1", "G2", "G3", "T_surf", "tau0", "Gamma", "EC"],
                                 stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_zero), derivative = False,
                              mytitle="Lorentz_zero fit (T= {:.1f} K, B = {:.1f} T, N = {:.0f})".format(temp_kwarg, bfield_kwarg, num_EC_kwarg)+"\nEC = "+str(np.round(make_EC_list(params_zero[-1])*1000, decimals=2))+" meV",   
                              myylabel="$dI/dV_b$ (nA/V)");
    # return osc only
    back_mask_zero = np.array([1,1,1,1,1,1,1,0,1,0]);
    osc_mask_zero = np.array([1,0,0,0,0,0,0,1,1,1]);
    dI_back_zero = dIdV_back(V_exp, *params_zero[back_mask_zero>0]);
    if(stop_at=="lorentz_zero/"): return V_exp, dI_exp-dI_back_zero, params_zero[osc_mask_zero>0], bounds_zero[:,osc_mask_zero>0];

    #### fine tune the lorentz_zero fit ####
    # outliers removed
    # lorentzians turned on
    # Gamma frozen
    fine_mask = np.array([1,1,1,1,1,1,0,0,0,0]);
    params_fine_guess = params_zero[fine_mask>0];
    EClist = make_EC_list(params_zero[-1]);
    params_fine_guess = np.append(params_fine_guess, EClist);
    global Gamma_kwarg; Gamma_kwarg = params_zero[-2];
    global tau0_kwarg; tau0_kwarg = params_zero[-3];
    bounds_fine = bounds_base[:,fine_mask>0];
    bounds_fine[:,0] = bounds_init[:,0]; # update V0 bounds
    bounds_fine = np.append(bounds_fine, np.array([[EClist[0]*(1-EC_percent),EClist[1]*(1-EC_percent),EClist[2]*(1-EC_percent)],
                                                   [EClist[0]*(1+EC_percent),EClist[1]*(1+EC_percent),EClist[2]*(1+EC_percent)]]), axis = 1);
    params_fine, _ = fit_wrapper(dIdV_all_fine, V_exp, dI_exp,
                                 params_fine_guess, bounds_fine, ["V0", "eps_0", "eps_c", "G1", "G2", "G3", "EC1", "EC2"],
                                 stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_fine(V_exp, *params_fine), derivative = False,
                              mytitle="Lorentz_fine fit (T= {:.1f} K, B = {:.1f} T, N = {:.0f})".format(temp_kwarg, bfield_kwarg, num_EC_kwarg)+"\nEC = "+str(np.round(params_fine[-num_EC_kwarg:]*1000, decimals=2))+" meV",   
                              myylabel="$dI/dV_b$ (nA/V)");
    # return osc only
    osc_mask_fine = np.array([1,0,0,0,0,0,1,1,1]);
    params_fine_back = np.array([params_fine[0], params_fine[1], params_fine[2], params_fine[3], params_fine[4], params_fine[5], temp_kwarg, Gamma_kwarg]);
    dI_back_fine = dIdV_back(V_exp, *params_fine_back);
    if(stop_at=="lorentz_fine/"): return V_exp, dI_exp-dI_back_fine, params_fine[osc_mask_fine>0], bounds_fine[:,osc_mask_fine>0];

    #### try a bunch of different combinations ####
    fit_best, EC_best = search_space_lorentz_zero(V_exp, dI_exp, dI_back_fine,
                        params_fine[0], tau0_kwarg, Gamma_kwarg, np.average(params_fine[-3:]) );
    if(verbose > 4): plot_fit(V_exp, dI_exp, fit_best, derivative = False,
                              mytitle="Best fit from search (T= {:.1f} K, B = {:.1f} T, N = {:.0f})".format(temp_kwarg, bfield_kwarg, num_EC_kwarg)+"\nEC = "+str(np.round(EC_best*1000, decimals=2))+" meV",
                              myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at=="trial/"): return;



####################################################################
#### wrappers

def fit_Mn_data(stop_at, metal, num_islands = 2, verbose=1):
    '''
    '''
    stopats_2_func = {"back/" : dIdV_all_zero, "lorentz_zero/" : dIdV_lorentz_zero, "lorentz_fine/" : dIdV_lorentz_fine, "osc/" : dIdV_lorentz_zero};

    # experimental params
    Ts = np.loadtxt(metal+"Ts.txt", ndmin=1);
    Bs = np.loadtxt(metal+"Bs.txt", ndmin=1);
    from utils import show_raw_data
    show_raw_data(metal, Ts);

    #### guesses ####
    # surface magnons
    epsc_guess, epsc_percent = 0.002, 1;
    G1_guess, G1_percent = 50.0, 0.5;
    Gamma_guess, Gamma_percent = 0.0006, 0.5;
    # magnetic impurities
    eps0_guess, eps0_percent = 0.001, 1e-12;
    G2_guess, G2_percent = 1.0, 0.5;
    G3_guess, G3_percent = 1e-12, 0.5;
    # other
    ohm_guess, ohm_percent = 1e-12, 1.0;
    tau0_guess, tau0_percent = 0.002, 0.5;
    EC_guess, EC_percent = 0.0012, 0.5;
    V0_guesses = np.array([-0.002413,-0.0035,-0.002089,-0.002226,-0.0026048,-0.001825,-0.001418]);
    V0_percent = 1e-12;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        if(True and datai in [1]): 
            global temp_kwarg; temp_kwarg = Ts[datai];
            global bfield_kwarg; bfield_kwarg = Bs[datai];
            global num_EC_kwarg; num_EC_kwarg = num_islands;
            print("#"*60+"\nT = {:.1f} K".format(Ts[datai]));
            
            #### get fit results ####
            guesses = (V0_guesses[datai], eps0_guess, epsc_guess, G1_guess, G2_guess, G3_guess, ohm_guess, tau0_guess, Gamma_guess, EC_guess);
            percents = (V0_percent, eps0_percent, epsc_percent, G1_percent, G2_percent, G3_percent, ohm_percent, tau0_percent, Gamma_percent, EC_percent);    
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(metal,
                    guesses, percents, stop_at, verbose=verbose);
            results.append(temp_results); 
            boundsT.append(temp_bounds);            
            if(stop_at in ["lorentz_zero/", "lorentz_fine/"]):
                # compare with null
                y_fit = stopats_2_func[stop_at](x_forfit, *temp_results);  
                comp_with_null(x_forfit, y_forfit, y_fit);

                #save processed x and y data, and store plot
                if False:
                    plot_fname = metal+stop_at+"stored_plots/{:.0f}".format(Ts[datai]); # <- where to save the fit plot
                    print("Saving plot to "+plot_fname);
                    np.save(plot_fname+"_x.npy", x_forfit);
                    np.save(plot_fname+"_y.npy", y_forfit);
                    np.save(plot_fname+"_yfit.npy", y_fit);
                    np.savetxt(plot_fname+"_results.txt", temp_results, header = str([]), fmt = "%.5f", delimiter=' & ');
                   
    # save
    results, boundsT = np.array(results), np.array(boundsT);
    if(stop_at in []):
        print("Saving data to "+metal+stop_at);
        np.save(metal+stop_at+"results.npy", results);
        np.save(metal+stop_at+"bounds.npy", boundsT);   

####################################################################
#### run

if(__name__ == "__main__"):

    metal = "MnTrilayer/"; # tells which experimental data to load
    stop_ats = ["back/", "lorentz_zero/", "lorentz_fine/", "trial/", "osc/"];
    stop_at = stop_ats[0];
    verbose=10;

    # this one executes the fitting and stores results
    fit_Mn_data(stop_at, metal, verbose=verbose);

    # this one plots the stored results
    # combined allows you to plot two temps side by side
    #plot_saved_fit(stop_at, metal, verbose=verbose, combined=[]);

