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

def dIdV_back(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, Gamma):
    '''
    Magnetic impurity and surface magnon scattering, combined
    Designed to be passed to scipy.optimize.curve_fit
    '''

    # include G dependence on Gamma
    G1, G2, G3 = Gamma*Gamma*G1*1e9, Gamma*Gamma*G2*1e9, Gamma*Gamma*G3*1e9;
    return dIdV_imp(Vb, V0, eps0, G2, G3, T_surf)+dIdV_mag(Vb, V0, epsc, G1, T_surf);

def dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC): 
    '''
    '''

    nmax = 20; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    return tau0*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, EC, 0.0, ns);

def dIdV_lorentz(Vb, V0, tau0, Gamma, EC): 
    '''
    '''

    nmax = 20; # <- has to be increased with increasing Gamma
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # grounded
    return tau0*dI_of_Vb(Vb-V0, mymu0, Gamma, EC, kelvin2eV*temp_kwarg, ns);

def dIdV_all_zero(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, tau0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, Gamma) + dIdV_lorentz_zero(Vb, V0, tau0, Gamma, EC);

def dIdV_all(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, tau0, Gamma, EC):
    '''
    Magnetic impurity surface magnon scattering, and T=0 lorentzian all together
    Designed to be passed to scipy.optimize.curve_fit
    '''

    return dIdV_back(Vb, V0, eps0, epsc, G1, G2, G3, T_surf, Gamma) + dIdV_lorentz(Vb, V0, tau0, Gamma, EC);

####################################################################
#### main

def fit_dIdV(metal, nots, percents, stop_at, num_dev=5, freeze_back=False, verbose=0):
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
    - stop_at, str telling which fit function to stop at, and return fitting
            params for. For final fit, should always be "lorentz/"
    - num_dev, int, how many standard deviations away from initial fit to keep before
            discarding outliers
    - freeze_back, bool, whether to freeze background params (eps0, epsc, G1, G2, G3)
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal+"data/", temp_kwarg);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-np.mean(dI_exp),2)));

    # unpack
    V0_not = 0.0;
    V0_bound = np.max(V_exp)/10;
    eps0_not, epsc_not, G1_not, G2_not, G3_not, ohm_not, tau0_not, Gamma_not, EC_not = nots;
    eps0_percent, epsc_percent, G1_percent, G2_percent, G3_percent, ohm_percent, tau0_percent, Gamma_percent, EC_percent = percents
    params_base = np.array([V0_not, eps0_not, epsc_not, G1_not, G2_not, G3_not, temp_kwarg+ohm_not, tau0_not, Gamma_not, EC_not]);
    bounds_base = np.array([[V0_not-V0_bound, eps0_not*(1-eps0_percent), epsc_not*(1-epsc_percent), G1_not*(1-G1_percent), G2_not*(1-G2_percent), G3_not*(1-G3_percent), temp_kwarg+ohm_not*(1-ohm_percent), tau0_not*(1-tau0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent) ],
                            [V0_not+V0_bound, eps0_not*(1+eps0_percent), epsc_not*(1+epsc_percent), G1_not*(1+G1_percent), G2_not*(1+G2_percent), G3_not*(1+G3_percent), temp_kwarg+ohm_not*(1+ohm_percent), tau0_not*(1+tau0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);  

    # initial fit
    params_init_guess = np.copy(params_base);
    bounds_init = np.copy(bounds_base);
    if(freeze_back): # freeze eps0, epsc, G1, G2, G3, AND Gamma
        freeze_mask_phys = np.array([0,1,1,1,1,1,0,0,1,0]); 
    else: # freeze nothing 
        freeze_mask_phys = np.array([0,0,0,0,0,0,0,0,0,0]);
    bounds_init[0][freeze_mask_phys>0] = params_init_guess[freeze_mask_phys>0];
    bounds_init[1][freeze_mask_phys>0] = params_init_guess[freeze_mask_phys>0]+1e-12;  
    params_init, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                            params_init_guess, bounds_init, ["V0", "eps_0", "eps_c", "G1", "G2", "G3", "T_surf", "tau0", "Gamma", "EC"],
                            stop_bounds = False, verbose=verbose);
    fit_init = dIdV_all_zero(V_exp, *params_init);
    if(verbose > 4): plot_fit(V_exp, dI_exp, fit_init, mytitle="Initial fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg));

    #### Step 2: remove outliers
    with_outliers = len(V_exp);
    V_exp = V_exp[abs(dI_exp-fit_init) < num_dev*dI_dev];
    dI_exp = dI_exp[abs(dI_exp-fit_init) < num_dev*dI_dev];
    assert(with_outliers - len(V_exp) <= with_outliers*0.0999); # remove no more than 10%

    #### Step 3: freeze experimental background parameters with lorentz_zero
    params_zero, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                params_init, bounds_init, ["V0", "eps_0", "eps_c", "G1", "G2", "G3", "T_surf", "tau0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_zero), derivative=False,
                mytitle="Landauer_zero fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz_zero/'): return V_exp, dI_exp, params_zero, bounds_init; 

    # pretty fit to show signatures
    if(stop_at in ["imp/", "mag/", "sin/"]): 
        background_only = dIdV_back(V_exp, *params_zero[:-4], params_zero[-3]);
        if(verbose > 4): plot_fit(V_exp, dI_exp, background_only, derivative=True,
                            mytitle="Magnetic impurities and surface magnons \n $T = ${:.1f} K".format(temp_kwarg)+", B = {:.1f} T".format(bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");                               
        return V_exp, dI_exp-background_only, params_zero, bounds_init;

    #### Step 4: Fit oscillation parameters with lorentz
    params_all_guess = np.copy(params_zero);
    bounds_all = np.copy(bounds_base); # reset pre-freezing
    if(freeze_back): # only tau0, EC free
        freeze_mask_back = np.array([1,1,1,1,1,1,1,0,1,0]);
    else: # freeze V0 and T_surf
        freeze_mask_back = np.array([1,0,0,0,0,0,1,0,0,0]); 
    bounds_all[0][freeze_mask_back>0] = params_all_guess[freeze_mask_back>0];
    bounds_all[1][freeze_mask_back>0] = params_all_guess[freeze_mask_back>0]+1e-12;
    params_all, _ = fit_wrapper(dIdV_all, V_exp, dI_exp,
                                params_all_guess, bounds_all, ["V0", "eps_0", "eps_c", "G1", "G2", "G3", "T_surf", "tau0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_all), derivative=False,
                mytitle="Landauer fit (T= {:.1f} K, B = {:.1f} T)".format(temp_kwarg, bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz/'): return V_exp, dI_exp, params_all, bounds_all;
    raise NotImplementedError; # invalid stop_at value

####################################################################
#### wrappers

def fit_Mn_data(stop_at, metal, freeze_back, verbose=1):
    '''
    Wrapper function for calling fit_dIdV on different temperature data sets
    and saving the results of those fits
    Args:
        - stop_at, str telling which fit function to stop at, and return fitting
            params for. For final fit, should always = "lorentz/"
        - metal, path to folder of datset(s) at fixed B, where dIdV data is stored
        - freeze_back, bool, whether to freeze the physical background
          params in the fitting
    '''
    stopats_2_func = {'imp/':dIdV_imp, 'mag/':dIdV_mag, 'imp_mag/':dIdV_back, 'lorentz_zero/':dIdV_all_zero, 'lorentz/':dIdV_all};

    # experimental params
    Ts = np.loadtxt(metal+"Ts.txt", ndmin=1);
    Bs = np.loadtxt(metal+"Bs.txt", ndmin=1);
        
    if(metal=="Mnv2/"):            
        if freeze_back: # all by high
            eps0_guess = 0.007150; 
            G2_guess, G3_guess = 0.1047, 0.1177;
            epsc_guess, G1_guess = 0.000032, 0.07179; 
            Gamma_guess = 0.002785;
            # looks good for unfrozen run, pursue further
        else:
            G2_guess, G3_guess = 0.2, 0.2; # impurity, det'd by low temp
            eps0_guess = 0.0065; # cutoff, det'd by HIGH temp
            epsc_guess, G1_guess = 0.005, 0.05; # magnon, det'd by low temp 
            Gamma_guess = 0.0022; # lead coupling, det'd by low temp 
        eps0_percent, epsc_percent = 1e-1,1; G1_percent, G2_percent, G3_percent = 1,1,1;
        
        # experimental background params
        ohm_guess, ohm_percent = 4.0, 1.0; # in kelvin # also V0, but that is set by data
        # oscillation guesses # <- change these after background is fixed
        tau0_guess =   0.01; # unitless scale factor
        EC_guess =    np.array([5.9, 5.7, 5.6, 5.4, 5.1])*1e-3; # in eV, sometimes needs to be tuned for convergence
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;

    ####

    elif(metal=="Mnv4/"):
        # physical background params
        eps0_percent, epsc_percent = 0.2,1; G1_percent, G2_percent, G3_percent = 1,1,1;
        
        # experimental background params
        ohm_guess, ohm_percent = 8.0, 0.4; # in kelvin # also V0, but that is set by data
        # oscillation guesses # <- change these after background is fixed
        tau0_guess =   0.01 # unitless scale factor
        EC_guess =    np.array([4.9,4.9,4.7,4.6,5.7,5.7])*1e-3; # in eV, sometimes needs to be tuned for convergence
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;

    ####

    elif(metal=="Mn7Tesla/"):
        # physical background params
        eps0_guess, epsc_guess = 0.00433, 0.00540 # in eV # 0.008, 0.010
        G1_guess, G2_guess, G3_guess = 0.371, 0.0972, 0.120; # in A/V/eV^2
        Gamma_guess = 0.00269; # in eV
        eps0_percent, epsc_percent = 0.2,1; G1_percent, G2_percent, G3_percent = 1,1,1;
        # experimental background params
        ohm_guess, ohm_percent = 8.0, 0.4; # in kelvin # also V0, but that is set by data
        # oscillation guesses # <- change these after background is fixed
        tau0_guess =   0.01 # unitless scale factor
        EC_guess =    np.array([5.9])*1e-3; # in eV, sometimes needs to be tuned for convergence
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        freeze_back = False; # whether to freeze the physical background params in the fitting

    ####

    elif(metal=="Mn4Tesla/"):
        # physical background params
        eps0_guess, epsc_guess = 0.00432, 0.000027; # in eV # 0.008, 0.010
        G1_guess, G2_guess, G3_guess = 0.333, 0.159, 0.194; # in A/V/eV^2
        Gamma_guess = 0.00225; # in eV
        eps0_percent, epsc_percent = 0.2,1; G1_percent, G2_percent, G3_percent = 1,1,1;
        # experimental background params
        ohm_guess, ohm_percent = 8.0, 0.4; # in kelvin # also V0, but that is set by data
        # oscillation guesses # <- change these after background is fixed
        tau0_guess =   0.01 # unitless scale factor
        EC_guess =    np.array([5.9])*1e-3; # in eV, sometimes needs to be tuned for convergence
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        freeze_back = False; # whether to freeze the physical background params in the fitting

    ####

    elif(metal=="Mn2Tesla/"):
        # physical background params
        eps0_guess, epsc_guess = 0.00432, 0.00487; # in eV # 0.008, 0.010
        G1_guess, G2_guess, G3_guess = 0.525, 0.143, 0.161; # in A/V/eV^2
        Gamma_guess = 0.00228; # in eV
        eps0_percent, epsc_percent = 0.2,1; G1_percent, G2_percent, G3_percent = 1,1,1;
        # experimental background params
        ohm_guess, ohm_percent = 8.0, 0.4; # in kelvin # also V0, but that is set by data
        # oscillation guesses # <- change these after background is fixed
        tau0_guess =   0.01 # unitless scale factor
        EC_guess =    np.array([5.9])*1e-3; # in eV, sometimes needs to be tuned for convergence
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        freeze_back = False; # whether to freeze the physical background params in the fitting

    ####

    elif(metal=="Mn-2Tesla/"):
        # physical background params
        eps0_guess, epsc_guess = 0.0119, 0.000; # in eV # 0.008, 0.010
        G1_guess, G2_guess, G3_guess = 0.374, 0.117, 0.074; # in A/V/eV^2
        Gamma_guess = 0.00242; # in eV
        eps0_percent, epsc_percent = 0.2,1; G1_percent, G2_percent, G3_percent = 1,1,1;
        # experimental background params
        ohm_guess, ohm_percent = 8.0, 0.4; # in kelvin # also V0, but that is set by data
        # oscillation guesses # <- change these after background is fixed
        tau0_guess =   0.01 # unitless scale factor
        EC_guess =    np.array([5.9])*1e-3; # in eV, sometimes needs to be tuned for convergence
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        freeze_back = False; # whether to freeze the physical background params in the fitting

    ####

    elif(metal=="Mn-4Tesla/"):
        # physical background params
        eps0_guess, epsc_guess = 0.00432, 0.000055; # in eV # 0.008, 0.010
        G1_guess, G2_guess, G3_guess = 0.333, 0.159, 0.194; # in A/V/eV^2
        Gamma_guess = 0.00225; # in eV
        eps0_percent, epsc_percent = 0.2,1; G1_percent, G2_percent, G3_percent = 1,1,1;
        # experimental background params
        ohm_guess, ohm_percent = 8.0, 0.4; # in kelvin # also V0, but that is set by data
        # oscillation guesses # <- change these after background is fixed
        tau0_guess =   0.01 # unitless scale factor
        EC_guess =    np.array([5.9])*1e-3; # in eV, sometimes needs to be tuned for convergence
        tau0_percent, Gamma_percent, EC_percent = 0.4, 0.4, 0.4;
        freeze_back = False; # whether to freeze the physical background params in the fitting

    ####
        
    else: raise NotImplementedError;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        if(True and datai in [0,1,2,3]):
            global temp_kwarg; temp_kwarg = Ts[datai];
            global bfield_kwarg; bfield_kwarg = Bs[datai];
            print("#"*60+"\nT = {:.1f} K".format(Ts[datai]));
            
            #### get fit results ####
            guesses = (eps0_guess, epsc_guess, G1_guess, G2_guess, G3_guess, ohm_guess, tau0_guess, Gamma_guess, EC_guess[datai]);
            percents = (eps0_percent, epsc_percent, G1_percent, G2_percent, G3_percent, ohm_percent, tau0_percent, Gamma_percent, EC_percent);   
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(
                    metal, guesses, percents, stop_at,
                    freeze_back = freeze_back, verbose=verbose);
            results.append(temp_results); 
            boundsT.append(temp_bounds);
    
            #save processed x and y data, and store plot
            if(stop_at in ["lorentz/"]):
                plot_fname = metal+stop_at+"stored_plots/{:.0f}".format(Ts[datai]); # <- where to save the fit plot
                y_fit = stopats_2_func[stop_at](x_forfit, *temp_results);
                mytitle="$\\tau_0 = $ {:.0f} nA/V, $\Gamma = $ {:.5f} eV, $E_C = $ {:.5f} eV, T_film = "+"{:.1f} K".format(*temp_results[-4:])
                print("Saving plot to "+plot_fname);
                np.save(plot_fname+"_x.npy", x_forfit);
                np.save(plot_fname+"_y.npy", y_forfit);
                np.save(plot_fname+"_yfit.npy", y_fit);
                np.savetxt(plot_fname+"_title.txt", [0], header=mytitle);
                np.savetxt(plot_fname+"_results.txt", temp_results, header = str(["V0", "E0", "Ec", "G1", "G2", "G3", "T_surf", "tau0","Gamma", "EC", "T_film"]), fmt = "%.5f", delimiter=' & ');

    # save
    results, boundsT = np.array(results), np.array(boundsT);
    if(stop_at in ["lorentz_zero/", "lorentz/"]):
        print("Saving data to "+metal+stop_at);
        np.save(metal+stop_at+"results.npy", results);
        np.save(metal+stop_at+"bounds.npy", boundsT);

def plot_saved_fit(stop_at, metal, combined=[], verbose = 1):
    '''
    '''

    # which fit result is which
    rlabels = np.array(["$V_0$", "$\\varepsilon_0$ (eV)", "$\\varepsilon_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$T_{surf}$", "$\\tau_0$", "$\Gamma$ (eV)", "$E_C$ (eV)", "$T_{film}$"]);  
    if(stop_at=='mag/'):
        rlabels_mask = np.array([1,1,0,0,1,1,0,1,1,1,1]);
    elif(stop_at == 'lorentz_zero/' or stop_at == 'lorentz/'): 
        rlabels_mask = np.array([1,1,1,1,1,1,1,0,0,0,0]);
    else: raise NotImplementedError;
    
    # plot each fit
    Ts = np.loadtxt(metal+"Ts.txt", ndmin=1);
    Bs = np.loadtxt(metal+"Bs.txt", ndmin=1);
    
    from utils import plot_fit
    fig3, ax3 = plt.subplots();
    for Tvali, Tval in enumerate(Ts):
        plot_fname = metal+stop_at+"stored_plots/{:.0f}".format(Tval); # <- where to get/save the fit plot
        temp_results = np.loadtxt(plot_fname+"_results.txt");
        x = np.load(plot_fname+"_x.npy");
        y = np.load(plot_fname+"_y.npy");
        yfit = np.load(plot_fname+"_yfit.npy");
        print("Loading fit from "+plot_fname+"_yfit.npy");

        # plot
        if(combined): # plot all at once
            if(Tval in combined):
                offset = 800;
                ax3.scatter(x,offset*Tvali+y, color=mycolors[Tvali], marker=mymarkers[Tvali], 
                            label="$T=$ {:.1f} K".format(Tval)+", B = {:.1f} T".format(Bs[Tvali]));
                ax3.plot(x,offset*Tvali+yfit, color="black");
                ax3.set_xlabel("$V_b$ (V)");
                ax3.set_xlim(-0.1,0.1);
                ax3.set_ylabel("$dI/dV_b$ (nA/V)");
                #ax3.set_ylim(300,2800);
                print(temp_results);
        else:
            if(verbose): plot_fit(x, y, yfit, myylabel="$dI/dV_b$ (nA/V)", mytitle="$T=$ {:.1f} K".format(Tval)+", B = {:.1f} T".format(Bs[Tvali]));

    ax3.set_title("Conductance oscillations in EGaIn$|$H$_2$Pc$|$MnPc$|$NCO");
    plt.legend(loc='lower right');
    plt.show();

    # load
    print("Loading data from "+metal+stop_at);
    results = np.load(metal+stop_at+"results.npy");
    boundsT = np.load(metal+stop_at+"bounds.npy");

    # plot fitting results vs T
    nresults = len(rlabels_mask[rlabels_mask<1]);
    fig, axes = plt.subplots(nresults, sharex=True);
    if(nresults==1): axes = [axes];
    axi = 0
    for resulti in range(len(rlabels)):
        if(rlabels_mask[resulti]==0):
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

    # plot one set of params vs another, across dataset
    if(stop_at == "lorentz/"): # <-------- !!!!        
        # plot
        pfig, pax = plt.subplots();  
        periods = 4*results[:,-2]*1000;
        periods, Ts = periods, Ts;
        gammas = results[:,-3]*1000;
        charges = results[:,-2]*1000;
        Tfilms = results[:,-1];
        myx, myy = gammas, charges;
        myx, myy = Ts, periods;
        pax.scatter(myx, myy, color=mycolors[0], label="Data");
        # fit
        coefs = np.polyfit(myx, myy, 1);
        myyfit = coefs[0]*myx+coefs[1];
        myrmse = np.sqrt( np.mean( np.power(myy-myyfit,2) ))/abs(np.max(myy)-np.min(myy));
        pax.plot(myx, myyfit, color=accentcolors[0], label = "Slope = {:.2f} meV/T = {:.2f}$k_B$, b = {:.2f} meV".format(coefs[0], coefs[0]/1000/kelvin2eV, coefs[1]));
        pax.plot( [np.mean(myx)], [np.mean(myy)], color='white', label = "RMSE = {:1.5f}".format(myrmse));
        # format
        pax.set_title("Dataset = "+str(metal[:-1]));
        plt.legend();
        plt.tight_layout();
        plt.show();

####################################################################
#### run

if(__name__ == "__main__"):

    metal = "Mnv2/"; # tells which experimental data to load
    stop_ats = ['mag/', 'lorentz_zero/', 'lorentz/'];
    stop_at = stop_ats[2];
    freeze_back = True;
    verbose=1;

    # this one executes the fitting and stores results
    fit_Mn_data(stop_at, metal, freeze_back, verbose=verbose);

    # this one plots the stored results
    # combined allows you to plot two temps side by side
    #plot_saved_fit(stop_at, metal, verbose=verbose, combined=[2.5]);

