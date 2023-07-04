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
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkorange", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mymarkevery = (40, 40);
mylinewidth = 1.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

# units
kelvin2eV =  8.617e-5;
conductance_quantum = 7.748e-5; # units amp/volt

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
    Delta = 0.0;
    retval = G2;
    retval -= (G3)*Ffunc(abs(Vb-V0), kelvin2eV*temp_kwarg);
    #retval -= (G3/4)*Ffunc(abs(Vb-V0+Delta), kelvin2eV*temp_kwarg);
    #retval -= (G3/4)*Ffunc(abs(Vb-V0-Delta), kelvin2eV*temp_kwarg);
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

def dIdV_sin(Vb, V0, amplitude, period, dI0):
    '''
    Sinusoidal fit function - purely mathematical, not physical
    Designed to be passed to scipy.optimize.curve_fit
    '''

    ang_freq = 2*np.pi/period
    return dI0+amplitude+amplitude*(-1)*np.cos(ang_freq*(Vb-V0));

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

def fit_dIdV(metal, temp, nots, percents, stop_at, num_dev = 4, verbose=0):
    '''
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, temp);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-np.mean(dI_exp),2)));
    del temp

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
    
    # imp, mag, lorentz_zero individually
    if(stop_at in ["imp/", "mag/", "sin/", "lorentz_zero/"]):
        
        # fit to magnon + imp with dropout
        params_drop, _ = fit_wrapper(dIdV_back, V_exp[dI_exp<background], dI_exp[dI_exp<background],
                                params_back, bounds_back, ["V0", "E0", "Ec", "G1", "G2", "G3"],
                                stop_bounds = False, verbose=verbose);
        background_drop = dIdV_back(V_exp[dI_exp<background], *params_drop);
        if(verbose > 4): plot_fit(V_exp[dI_exp<background], dI_exp[dI_exp<background], background_drop, derivative=False,
                            mytitle="Impurity + magnon scattering ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");                               

        # force G1=0
        params_imp = np.copy(params_drop);
        params_imp[3] = 0;
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_back(V_exp, *params_imp), derivative=True,
                            mytitle="Magnetic impurity scattering ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        mask_imp = np.array([1,1,0,0,1,1]);
        if(stop_at == 'imp/'): return V_exp, dI_exp, params_imp[mask_imp>0], bounds_back[:,mask_imp>0];

        # force G2, G3=0
        params_mag = np.copy(params_drop);
        params_mag[4] = 0;
        params_mag[5] = 0;
        if(verbose > 4): plot_fit(V_exp, dI_exp-dIdV_back(V_exp, *params_imp), dIdV_back(V_exp, *params_mag), derivative=True,
                            smooth=True, mytitle="Surface magnon scattering ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        mask_mag = np.array([1,0,1,1,0,0]);
        if(stop_at == 'mag/'): return V_exp, dI_exp-dIdV_back(V_exp, *params_imp), params_mag[mask_mag>0], bounds_back[:,mask_mag>0];

        # fit remaining oscillations
        dI_osc = dI_exp - dIdV_back(V_exp, *params_drop);
        params_zero_guess = np.array([params_drop[0], dI0_not, Gamma_not, EC_not]);
        bounds_zero = np.array([ [-V0_bound, dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                                 [ V0_bound, dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
        params_zero, _ = fit_wrapper(dIdV_lorentz_zero, V_exp, dI_osc,
                                    params_zero_guess, bounds_zero, ["V0","dI0","Gamma", "EC"],
                                    stop_bounds = False, verbose=verbose);
        if(verbose > 4): plot_fit(V_exp, dI_osc, dIdV_lorentz_zero(V_exp, *params_zero), derivative=False,
                            mytitle="$T=0$ Lorentzians ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        if(stop_at == 'lorentz_zero/'): return V_exp, dI_osc, params_zero, bounds_zero;
        raise NotImplementedError;

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

def fit_Mn_data():
    metal="Mn/"; # points to data folder
    fname = "fits/"
    stop_ats = ['imp/','mag/','imp_mag/', 'sin/', 'lorentz_zero/', 'lorentz/'];
    stop_at = stop_ats[5];
    if(stop_at=='imp/'):
        rlabels = ["$V_0$", "$\\varepsilon_0$", "$G_2$", "$G_3$"];
    elif(stop_at=='mag/'):
        rlabels = ["$V_0$", "$\\varepsilon_c$", "$G_1$"];
    elif(stop_at=='imp_mag/'):
        rlabels = ["$V_0$", "$\\varepsilon_0$", "$\\varepsilon_c$", "$G_1$", "$G_2$", "$G_3$"];
    elif(stop_at=='sin/'):
        rlabels = ["$V_0$ (V)", "$A$ (nA/V)", "$\Delta V_b$ (V)", "$dI_0$ (nA/V)"];
    elif(stop_at == 'lorentz_zero/'):
        rlabels = ["$V_0$", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"];
    elif(stop_at == 'lorentz/'):
        rlabels = ["$V_0$", "$E_0$ (eV)", "$E_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"];
        rlabel_mask = np.ones(np.shape(rlabels), dtype=int);
        rlabel_mask[:-3] = np.zeros((len(rlabel_mask)-3,), dtype=int)
    else: raise NotImplementedError;

    # experimental params
    kelvin2eV =  8.617e-5;
    Ts = np.array([5.0,10.0,15.0,20.0,25.0,30.0]);
    Ts = np.array([5.0,10.0,15.0]);
    # sample temp shifted due to ohmic heating
    Teffs = np.array([8.5,10.0,14.0,19.0]); # <------

    # lorentzian guesses
    E0_guess, G2_guess, G3_guess = 0.008, 1250, 750 # 2.5 K: 0.0105, 850,450
    Ec_guess, G1_guess = 0.013, 1500;
    E0_percent, G2_percent, G3_percent = 0.1, 0.9, 0.1;
    Ec_percent, G1_percent = 0.1, 0.1;   

    # oscillation guesses
    dI0_guess =   np.array([57.9,51.4,50.6])*1e3
    Gamma_guess = np.array([4.70,4.20,4.10])*1e-3
    EC_guess =    np.array([4.89,4.88,4.81])*1e-3
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

def plot_saved_fit():
    '''
    '''
    verbose=10;
    metal="Mn/"; # points to data folder
    stop_ats = ['imp_mag/', 'lorentz_zero/', 'lorentz/'];
    stopats_2_func = {'imp/':dIdV_imp, 'mag/':dIdV_mag, 'imp_mag/':dIdV_back, 'lorentz_zero/':dIdV_all_zero, 'lorentz/':dIdV_all};
    stop_at = stop_ats[-1];
    stored_plots = False;

    # load
    fname = "fits/"
    print("Loading data from "+fname+stop_at);
    Ts = np.loadtxt(fname+stop_at+"Ts.txt");
    results = np.load(fname+stop_at+"results.npy");
    boundsT = np.load(fname+stop_at+"bounds.npy");
    Teffs = np.loadtxt(fname+stop_at+"Teffs.txt");

    # save results in latex table format
    # recall results are [Ti, resulti]
    results_tab = np.append(np.array([[Teff] for Teff in Teffs]), results, axis = 1);
    np.savetxt(fname+stop_at+"results_table.txt", results_tab, fmt = "%.5f", delimiter='&', newline = '\\\ \n');
    print("Saving table to "+fname+stop_at+"results_table.txt");
    
    # plot each fit
    from utils import plot_fit
    fig3, ax3 = plt.subplots();
    for Tvali, Tval in enumerate(Ts):
        global temp_kwarg; temp_kwarg = Teffs[Tvali]; # very bad practice
        print(">>> Effective temperature = ", temp_kwarg);
        plot_fname = fname+stop_at+"stored_plots/{:.0f}".format(Tval); # <- where to get/save the fit plot
        exp_fname = fname+stop_at+"stored_exp/{:.0f}".format(Tval); # <- where to get raw data

        if(stored_plots): # fit already stored
            x = np.load(plot_fname+"_x.npy");
            y = np.load(plot_fname+"_y.npy");
            yfit = np.load(plot_fname+"_yfit.npy");
            try:
                mytxt = open(plot_fname+"_title.txt", "r");
                mytitle = mytxt.readline()[1:];
            finally:
                mytxt.close();
            
            if False:
                plot_fit(x, y, yfit, mytitle=mytitle, myylabel="$dI/dV_b$");
            else: # plot 3 at once
                if(Tval in [5,15]):
                    offset=400;
                    print(30*"#", Tval, ":");
                    for parami, _ in enumerate(results[Tvali]):
                        print(results[Tvali,parami], boundsT[Tvali, :, parami])
                    ax3.scatter(x,offset*Tvali+y, color=mycolors[Tvali], marker=mymarkers[Tvali], 
                                label="$T=$ {:.0f} K".format(Tval)+" ($T_{eff}=$" +"{:.0f} K)".format(Teffs[Tvali]));
                    ax3.plot(x,offset*Tvali+yfit, color="black");
                    ax3.set_xlabel("$V_b$ (V)");
                    ax3.set_xlim(-0.1,0.1);
                    ax3.set_ylabel("$dI/dV_b$ (nA/V)");
                    ax3.set_ylim(300,2800);
        
        else: # need to generate fit

            # raw data w/out outliers
            V_exp = np.load(exp_fname+"_x.npy");
            dI_exp = np.load(exp_fname+"_y.npy");

            # evaluate at fit results and plot
            dI_fit = stopats_2_func[stop_at](V_exp, *results[Tvali]);
            mytitle="$T_{eff} = $";
            mytitle += "{:.1f} K, $\Gamma_0 = $ {:.5f} eV, $E_C = $ {:.5f} eV".format(Teffs[Tvali], *results[Tvali,-2:])
            if(verbose > 4): plot_fit(V_exp, dI_exp, dI_fit, mytitle=mytitle, myylabel="$dI/dV_b$"); 
        
            # save V_exp, dI_exp, dI_fit for easy access
            print("Saving plot to "+plot_fname);
            np.save(plot_fname+"_x.npy", V_exp);
            np.save(plot_fname+"_y.npy", dI_exp);
            np.save(plot_fname+"_yfit.npy", dI_fit);
            np.savetxt(plot_fname+"_title.txt", [0], header=mytitle);

    ax3.set_title("Conductance oscillations in EGaIn$|$H$_2$Pc$|$MnPc$|$NCO");
    plt.legend(loc='lower right');
    plt.show()

####################################################################
#### run

if(__name__ == "__main__"):
    #fit_Mn_data();
    plot_saved_fit();

