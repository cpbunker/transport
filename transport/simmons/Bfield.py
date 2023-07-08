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

def fit_dIdV(metal, nots, percents, stop_at, num_dev = 4, rescale=1, verbose=0):
    '''
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV_"+"{:.0f}T".format(bfield_kwarg)+".txt",metal,temp_kwarg);
    dI_exp = dI_exp*rescale; # <--- rescale !
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-np.mean(dI_exp),2)));

    # unpack
    V0_bound = 1e-2;
    E0_not, G2_not, G3_not, Ec_not, G1_not, dI0_not, Gamma_not, EC_not = nots;
    E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, dI0_percent, Gamma_percent, EC_percent = percents
    ohm_not, ohm_max = 0.0, 5.0; # ohmic heating lims in kelvin
    G1_not, G2_not, G3_not = rescale*G1_not, rescale*G2_not, rescale*G3_not; # <--- rescale !

    #### fit background

    # initial fit to magnon + imp
    params_init_guess = np.array([0.0, E0_not, Ec_not, G1_not, G2_not, G3_not, ohm_not]);
    bounds_init = np.array([[-V0_bound, E0_not*(1-E0_percent), Ec_not*(1-Ec_percent), G1_not*(1-G1_percent), G2_not*(1-G2_percent), G3_not*(1-G3_percent), ohm_not],
                            [ V0_bound, E0_not*(1+E0_percent), Ec_not*(1+Ec_percent), G1_not*(1+G1_percent), G2_not*(1+G2_percent), G3_not*(1+G3_percent), ohm_max]]);
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
                        mytitle="Background (B = {:.1f} T)".format(bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'imp_mag/'): return V_exp, dI_exp, params_back, bounds_back;

    #### fit magnon + imp + oscillation
    params_zero_guess = np.zeros((len(params_back)+3,));
    params_zero_guess[:len(params_back)] = params_back; # background only results -> all guess
    bounds_zero = np.zeros((2,len(params_back)+3));
    bounds_zero[:,:len(params_back)] = bounds_back;  # background only bounds -> all guess

    # for oscillation
    params_zero_guess[len(params_back):] = np.array([dI0_not, Gamma_not, EC_not]);
    bounds_zero[:,len(params_back):] = np.array([ [dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                                                [ dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
    # start with zero temp oscillations to constrain
    params_zero, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                params_zero_guess, bounds_zero, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_zero), derivative=False,
                mytitle="Landauer_zero fit (B= {:.1f} T)".format(bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz_zero/'): return V_exp, dI_exp, params_zero, bounds_zero; 

    # some plotting to help with constraints
    if False:
        params_plot = np.copy(params_zero);
        # <- notice that dI0, Gamma, EC also inherit from params_zero since thermal effects are so small
        print(params_plot)
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_plot))
        assert False

    # constrain and do finite temp fit
    params_all_guess = np.copy(params_zero);
    # <- notice that dI0, Gamma, EC also inherit from params_zero since thermal effects are so small
    bounds_all = np.copy(bounds_zero);
    constrain_mask = np.array([1,1,1,0,1,0,1,0,0,0]); # only G1, G3, dI0, Gamma, Ec free
    bounds_all[0][constrain_mask>0] = params_all_guess[constrain_mask>0];
    bounds_all[1][constrain_mask>0] = params_all_guess[constrain_mask>0]+1e-6;
    params_all, _ = fit_wrapper(dIdV_all, V_exp, dI_exp,
                                params_all_guess, bounds_all, ["V0", "E0", "Ec", "G1", "G2", "G3", "T_ohm","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all(V_exp, *params_all), derivative=False,
                mytitle="Landauer fit  (B= {:.1f} T)".format(bfield_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz/'): return V_exp, dI_exp, params_all, bounds_all;
    raise NotImplementedError;

####################################################################
#### wrappers

def fit_Bfield_data(stop_at,metal="Mn/",verbose=1):
    fname = "fits/Bfield/";
    stopats_2_func = {'imp/':dIdV_imp, 'mag/':dIdV_mag, 'imp_mag/':dIdV_back, 'lorentz_zero/':dIdV_all_zero, 'lorentz/':dIdV_all};

    # experimental params
    Ts = np.array([2.5, 2.5, 2.5]);
    Bs = np.array([0.0, 2.0, 7.0]);

    # lorentzian guesses
    E0_guess, G2_guess, G3_guess = 0.008, 1.00*1e3, 0.70*1e3 # <_ G2 and G3 get rescaled
    Ec_guess, G1_guess = 0.013, 1.9*1e3; # <- G1 gets rescaled
    E0_percent, G2_percent, G3_percent = 0.1, 0.9, 0.1;
    Ec_percent, G1_percent = 0.1, 0.1;   

    # oscillation guesses
    dI0_guess =   np.array([20.0,20.0,20.0])*1e3
    Gamma_guess = np.array([3.00,3.00,3.00])*1e-3
    EC_guess =    np.array([5.81,5.80,5.70])*1e-3
    dI0_percent, Gamma_percent, EC_percent = 0.2, 0.2, 0.2;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Bs)):
        if(True):
            print("#"*60+"\nB = {:.1f} Tesla, T = {:.1f} K".format(Bs[datai], Ts[datai]));
            guesses = (E0_guess, G2_guess, G3_guess, Ec_guess, G1_guess, dI0_guess[datai], Gamma_guess[datai], EC_guess[datai]);
            percents = (E0_percent, G2_percent, G3_percent, Ec_percent, G1_percent, dI0_percent, Gamma_percent, EC_percent);

            # get fit results
            global temp_kwarg; temp_kwarg = Ts[datai]; # very bad practice
            global bfield_kwarg; bfield_kwarg = Bs[datai];
            x_forfit, y_forfit, temp_results, temp_bounds = fit_dIdV(metal, guesses, percents, stop_at,
                                            rescale = 10, verbose=verbose);
            results.append(temp_results); 
            boundsT.append(temp_bounds);
    
            #save processed x and y data, and store plot
            if(stop_at == "lorentz_zero/" or stop_at == "lorentz/"):
                plot_fname = fname+stop_at+"stored_plots/{:.0f}".format(Bs[datai]); # <- where to save the plot
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
    if(stop_at == "lorentz_zero/" or stop_at == "lorentz/"):
        print("Saving data to "+fname+stop_at);
        np.savetxt(fname+stop_at+"Ts.txt", Ts);
        np.savetxt(fname+stop_at+"Bs.txt", Bs);
        np.save(fname+stop_at+"results.npy", results);
        np.save(fname+stop_at+"bounds.npy", boundsT);

def plot_saved_fit(stop_at, verbose = 1, combined=False):
    '''
    '''
    stopats_2_func = {'imp/':dIdV_imp, 'mag/':dIdV_mag, 'imp_mag/':dIdV_back, 'lorentz_zero/':dIdV_all_zero, 'lorentz/':dIdV_all};

    # which fitting param is which
    if(stop_at=='imp_mag/'):
        rlabels = np.array(["$eV_0$ (eV)", "$\\varepsilon_0$ (eV)", "$\\varepsilon_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$T_{ohm}$ (K)"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
    elif(stop_at == "lorentz_zero/" or stop_at == "lorentz/"):
        rlabels = np.array(["$eV_0$ (eV)", "$\\varepsilon_0$ (eV)", "$\\varepsilon_c$ (eV)", "$G_1$ (nA/V)","$G_2$ (nA/V)","$G_3$ (nA/V)", "$T_{ohm}$ (K)", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"]);
        rlabels_mask = np.ones(np.shape(rlabels), dtype=int);
        rlabels_mask[1:-3] = np.zeros_like(rlabels_mask)[1:-3];

    # load
    fname = "fits/Bfield/"
    print("Loading data from "+fname+stop_at);
    Ts = np.loadtxt(fname+stop_at+"Ts.txt");
    Bs = np.loadtxt(fname+stop_at+"Bs.txt");
    results = np.load(fname+stop_at+"results.npy");
    boundsT = np.load(fname+stop_at+"bounds.npy");

    # save results in latex table format
    # recall results are [Ti, resulti]
    results_tab = np.append(np.array([[B] for B in Bs]), results, axis = 1);
    np.savetxt(fname+stop_at+"results_table.txt", results_tab, fmt = "%.5f", delimiter=' & ', newline = '\\\ \n');
    print("Saving table to "+fname+stop_at+"results_table.txt");

    # plot fitting results vs T
    nresults = sum([el for el in rlabels_mask]);
    fig, axes = plt.subplots(nresults, sharex=True);
    if(nresults==1): axes = [axes];
    axi = 0
    for resulti in range(len(rlabels)):
        if(rlabels_mask[resulti]):
            axes[axi].plot(Bs, results[:,resulti], color=mycolors[0],marker=mymarkers[0]);
            axes[axi].set_ylabel(rlabels[resulti]);
            axes[axi].plot(Bs,boundsT[:,0,resulti], color=accentcolors[0],linestyle='dashed');
            axes[axi].plot(Bs,boundsT[:,1,resulti], color=accentcolors[0],linestyle='dashed');
            axes[axi].ticklabel_format(axis='y',style='sci',scilimits=(0,0));
            axi += 1;

    # format
    axes[-1].set_xlabel("$B$ (Tesla)");
    axes[0].set_title("Amplitude and period fitting");
    plt.tight_layout();
    plt.show();

    # plot each fit
    from utils import plot_fit
    fig3, ax3 = plt.subplots();
    for Bvali, Bval in enumerate(Bs):
        global temp_kwarg; temp_kwarg = Ts[Bvali]; # very bad practice
        global bfield_kwarg; bfield_kwarg = Bs[Bvali];
        print(">>> Temperature = ", temp_kwarg);
        print(">>> External B field = ", bfield_kwarg);
        plot_fname = fname+stop_at+"stored_plots/{:.0f}".format(Bval); # <- where to get/save the fit plot
        x = np.load(plot_fname+"_x.npy");
        y = np.load(plot_fname+"_y.npy");
        yfit = np.load(plot_fname+"_yfit.npy");
        print("Loading fit from "+plot_fname+"_yfit.npy");
        if(not combined):
            plot_fit(x, y, yfit, myylabel="$dI/dV_b$");
        else: # plot all at once
            if(True):
                offset=8000;
                ax3.scatter(x,offset*Bvali+y, color=mycolors[Bvali], marker=mymarkers[Bvali], label="$B=$ {:.0f} T".format(Bval));
                ax3.plot(x,offset*Bvali+yfit, color="black");
                ax3.set_xlabel("$V_b$ (V)");
                ax3.set_xlim(-0.1,0.1);
                ax3.set_ylabel("$dI/dV_b$ (nA/V)");
                #ax3.set_ylim(300,2800);
        
    ax3.set_title("Conductance oscillations in EGaIn$|$H$_2$Pc$|$MnPc$|$NCO");
    plt.legend(loc='lower right');
    plt.show();

def show_raw_data():
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
#### run

if(__name__ == "__main__"):
    stop_ats = ['imp_mag/', 'lorentz_zero/', 'lorentz/'];
    stop_at = stop_ats[-2];
    verbose=1;
    
    #show_raw_data();
    #fit_Bfield_data(stop_at,verbose=verbose);
    plot_saved_fit(stop_at,verbose=verbose, combined=True);
