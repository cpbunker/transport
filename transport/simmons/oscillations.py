'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
'''

from utils import plot_fit, load_dIdV, fit_wrapper

import numpy as np
import matplotlib.pyplot as plt

# fig standardizing
myxvals = 199;
myfontsize = 14;
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
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

    return G2 - G3*Ffunc(abs(Vb-V0), kelvin2eV*temp_kwarg);

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

def dIdV_sin(Vb, alpha, amplitude, period):
    '''
    Sinusoidal fit function - purely mathematical, not physical
    Designed to be passed to scipy.optimize.curve_fit
    '''

    ang_freq = 2*np.pi/period
    return amplitude*np.sin(ang_freq*Vb-alpha);

def dIdV_lorentz_zero(Vb, V0, dI0, Gamma, EC): 
    '''
    '''
    from landauer import dI_of_Vb_zero

    nmax = 40;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return -dI0+1e9*conductance_quantum*dI_of_Vb_zero(Vb-V0, mymu0, Gamma, EC, 0.0, ns);

def dIdV_lorentz(Vb, V0, dI0, Gamma, EC): 
    '''
    '''
    from landauer import dI_of_Vb

    nmax = 40;
    ns = np.arange(-nmax, nmax+1);
    mymu0 = 0.0; # otherwise breaks equal spacing
    return -dI0+1e9*conductance_quantum*dI_of_Vb(Vb-V0, mymu0, Gamma, EC, kelvin2eV*temp_kwarg, ns);

####################################################################
#### main

def fit_dIdV(metal, V0_not, dI0_not, Gamma_not, EC_not,
             dI0_percent, Gamma_percent, EC_percent, stop_at='sin', num_dev = 4, verbose=0):
    '''
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, temp_kwarg);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_mu = np.mean(dI_exp);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-dI_mu,2)));
    
    #### fit background to impurity scattering

    # first fit
    E0_guess, G2_guess, G3_guess = 0.0075, 1250, 750;
    V0_bound = 1e-2;
    params_imp_guess = np.array([0.0, E0_guess, G2_guess, G3_guess]);
    bounds_imp = np.array([[-V0_bound, E0_guess*0.99, G2_guess*0.99, G3_guess*0.99], # forced to be temp independent
                           [ V0_bound, E0_guess*1.01, G2_guess*1.01, G3_guess*1.01]]);
    params_imp, _ = fit_wrapper(dIdV_imp, V_exp, dI_exp,
                            params_imp_guess, bounds_imp, ["V0", "E0", "G2", "G3"],
                            stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_imp(V_exp, *params_imp), derivative=False,
                              mytitle="Magnetic impurity scattering ($T=$ {:.0f})".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");

    # remove outliers based on impurity background
    background_imp = dIdV_imp(V_exp, *params_imp);
    pre_dropout = len(V_exp);
    V_exp = V_exp[abs(dI_exp-background_imp) < num_dev*dI_dev];
    dI_exp = dI_exp[abs(dI_exp-background_imp) < num_dev*dI_dev];
    assert(pre_dropout - len(V_exp) <= pre_dropout*0.05); # only remove 5%

    # re-fit impurity background without outliers
    params_imp, _ = fit_wrapper(dIdV_imp, V_exp, dI_exp,
                    params_imp_guess, bounds_imp, ["V0", "E0", "G2", "G3"],
                    stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_imp(V_exp, *params_imp), derivative=True,
                              mytitle="Magnetic impurity scattering ($T=$ {:.0f})".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'imp/'): return params_imp, bounds_imp;

    #### fit background to magnon scattering

    # subtract impurity background
    background_imp = dIdV_imp(V_exp, *params_imp);
    dI_exp = dI_exp - background_imp;

    # fit magnon
    Ec_guess, G1_guess = 0.015, 1200;
    params_mag_guess = np.array([params_imp[0], Ec_guess, G1_guess]);
    bounds_mag = np.array([[params_imp[0], Ec_guess*0.8, G1_guess*0.8],
                            [ params_imp[0]+1e-6, Ec_guess*1.2, G1_guess*1.2]]);
    params_mag, _ = fit_wrapper(dIdV_mag, V_exp, dI_exp,
                            params_mag_guess, bounds_mag, ["V0", "Ec", "G1"],
                            stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_mag(V_exp, *params_mag), smooth = True, derivative=True,
                        mytitle="Surface magnon scattering ($T=$ {:.0f})".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'mag/'): return params_mag, bounds_mag;
    
    #### fit oscillations to sin

    # subtract magnon background
    background_mag = dIdV_mag(V_exp, *params_mag);
    dI_exp = dI_exp - background_mag;
    dI_mu = np.mean(dI_exp);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-dI_mu,2)));

    # fit sin
    params_sin_guess = np.array([np.pi/2, dI_dev, 4*EC_not]);
    bounds_sin = [[0,0.0*dI_dev, 4*EC_not*0.9],
                    [2*np.pi, 2.0*dI_dev, 4*EC_not*1.1]];
    bounds_sin = np.array(bounds_sin);
    params_sin, _ = fit_wrapper(dIdV_sin, V_exp, dI_exp,
                    params_sin_guess, bounds_sin, ["alpha","amp","per"],
                    stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_sin(V_exp, *params_sin), derivative=True,
                        mytitle="Sinusoidal fit ($T=$ {:.0f})".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'sin/'): return params_sin, bounds_sin;

    #### fit oscillations to T=0 landauer

    # all params get fit here
    params_zero_guess = np.array([V0_not, dI0_not, Gamma_not, EC_not]);
    bounds_zero = np.array([ [-V0_bound, dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                [ V0_bound, dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
    params_zero, _ = fit_wrapper(dIdV_lorentz_zero, V_exp, dI_exp,
                                params_zero_guess, bounds_zero, ["V0","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_lorentz_zero(V_exp, *params_zero), derivative=False,
                mytitle="$T=0$ Landauer fit ($T=$ {:.0f})".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz_zero/'): return params_zero, bounds_zero;

    #### fit oscillations to T!=0 landauer

    # only fit dI0 and Gamma here. NB they should both go down
    bounds_final = np.array([[params_zero[0], params_zero[1]*(1-dI0_percent), params_zero[2]*(1-Gamma_percent), params_zero[3]],
                        [params_zero[0]+1e-6, params_zero[1]*(1+0.1*dI0_percent), params_zero[2]*(1+0.1*Gamma_percent), params_zero[3]+1e-6]]);
    params_final, _ = fit_wrapper(dIdV_lorentz, V_exp, dI_exp,
                                params_zero, bounds_final, ["V0","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    return params_final, bounds_final;

####################################################################
#### wrappers

def fit_Mn_data():
    metal="Mn/"; # points to data folder
    stop_ats = ['imp/','mag/','sin/', 'lorentz_zero/', 'lorentz/'];
    stop_at = stop_ats[3];
    if(stop_at=='imp/'):
        rlabels = ["$V_0$", "$\\varepsilon_0$", "$G_2$", "$G_3$"];
    elif(stop_at=='mag/'):
        rlabels = ["$V_0$", "$\\varepsilon_c$", "$G_1$"];
    elif(stop_at=='sin/'):
        rlabels = ["$\\alpha$", "$A$ (nA/V)", "$\Delta V_b$ (V)"];
    elif(stop_at == 'lorentz_zero/' or stop_at =='lorentz/'):
        rlabels = ["$V_0$", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"];
    else: raise NotImplementedError;

    # experimental params
    kelvin2eV =  8.617e-5;
    Ts = np.array([5.0,10.0,15.0,20.0,25.0,30.0]);
    Ts = [25.0];

    # guesses
    V0_guess = -0.0044*np.ones_like(Ts);
    dI0_guess =      np.array([65452, 66574, 68000, 68000, 70000, 70000]);
    Gamma_guess = np.array([0.0054, 0.0054, 0.0059, 0.0062, 0.0065, 0.0067]);
    EC_guess = (0.0196/4)*np.ones_like(Ts);
    dI0_percent = 0.4;
    Gamma_percent = 0.4;
    EC_percent = 0.1;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("#"*60+"\nT = {:.1f} K ({:.4f} eV)".format(Ts[datai], Ts[datai]*kelvin2eV));

        # get fit results
        global temp_kwarg; temp_kwarg = Ts[datai]; # very bad practice
        temp_results, temp_bounds = fit_dIdV(metal,
            V0_guess[datai], dI0_guess[datai], Gamma_guess[datai], EC_guess[datai],
            dI0_percent, Gamma_percent, EC_percent, stop_at = stop_at, verbose=10);
        results.append(temp_results); 
        boundsT.append(temp_bounds);

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
        fname = "fits/"
        print("Saving data to "+fname+stop_at);
        np.savetxt(fname+stop_at+"Ts.txt", Ts);
        np.save(fname+stop_at+"results.npy", results);
        np.save(fname+stop_at+"bounds.npy", boundsT);

    # format
    axes[-1].set_xlabel("$T$ (K)");
    axes[0].set_title("Amplitude and period fitting");
    plt.show();

def comp_backgrounds():
    '''
    '''
    verbose=10;
    metal="Mn/"; # points to data folder

    # load
    fname = "fits/"
    print("Loading data from "+fname+"imp/");
    Ts = np.loadtxt(fname+"imp/"+"Ts.txt");
    results_imp = np.load(fname+"imp/"+"results.npy");
    results_mag = np.load(fname+"mag/"+"results.npy");

    # fit is from only one temp !
    refi = 4;
    ref_imp = results_imp[refi];
    ref_mag = results_mag[refi];
    
    # plot each fit
    fig3, ax3 = plt.subplots();
    for Tvali, Tval in enumerate([5,15,25]):
        global temp_kwarg; temp_kwarg = Tval; # very bad practice

        # raw data
        V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, Tval);

        # fit
        V_fine = np.linspace(np.min(V_exp), np.max(V_exp), len(V_exp)*5);
        dI_fit_imp = dIdV_imp(V_fine, *ref_imp);
        dI_fit_mag = dIdV_mag(V_fine, *ref_mag);
        dI_fit = dI_fit_imp + dI_fit_mag;

        # plot
        offset = 1500;
        ax3.scatter(V_exp, Tvali*offset+dI_exp, label="$T=$ {:.0f} K".format(Tval), color=mycolors[Tvali], marker=mymarkers[Tvali]);
        ax3.plot(V_fine, Tvali*offset+dI_fit, color="black");

    # format
    ax3.set_title("Background fit, $\\varepsilon_c = $ {:.2f} meV, $\\varepsilon_0 = $ {:.2f} meV".format(1000*ref_mag[1], 1000*ref_imp[1]));
    ax3.set_xlabel("$V_b$ (V)");
    ax3.set_xlim(-0.1,0.1);
    ax3.set_ylabel("$dI/dV_b$ (nA/V)");
    ax3.set_ylim(-1000,5000);
    plt.legend(loc='upper right');
    plt.show()

def plot_saved_fit():
    '''
    '''
    verbose=10;
    metal="Mn/"; # points to data folder
    stop_ats = ['imp/','mag/','sin/', 'lorentz_zero/', 'lorentz/'];
    stopats_2_func = {'imp/':dIdV_imp, 'mag/':dIdV_mag, 'sin/':dIdV_sin, 'lorentz_zero/':dIdV_lorentz_zero, 'lorentz/':dIdV_lorentz};
    stop_at = stop_ats[-1];
    stored_plots = True;

    # load
    fname = "fits/"
    print("Loading data from "+fname+stop_at);
    Ts = np.loadtxt(fname+stop_at+"Ts.txt");
    results = np.load(fname+stop_at+"results.npy");
    boundsT = np.load(fname+stop_at+"bounds.npy"); 

    # save results in latex table format
    # recall results are [Ti, resulti]
    results_tab = np.append(np.array([[Tval] for Tval in Ts]), results, axis = 1);
    np.savetxt(fname+stop_at+"results_table.txt", results_tab, fmt = "%.5f", delimiter='&', newline = '\\\ \n');
    print("Saving table to "+fname+stop_at+"results_table.txt");
    
    # plot each fit
    from utils import plot_fit
    fig3, ax3 = plt.subplots();
    for Tvali, Tval in enumerate(Ts):
        global temp_kwarg; temp_kwarg = Tval; # very bad practice
        plot_fname = fname+stop_at+"stored_plots/{:.0f}".format(Tval); # <- where to get/save the plot

        if(stored_plots): # fit already stored
            x = np.load(plot_fname+"_x.npy");
            y = np.load(plot_fname+"_y.npy");
            yfit = np.load(plot_fname+"_yfit.npy");
            try:
                mytxt = open(plot_fname+"_title.txt", "r");
                mytitle = mytxt.readline()[1:];
            finally:
                mytxt.close();
            #plot_fit(x, y, yfit, mytitle=mytitle, myylabel="$dI/dV_b$");

            # plot 3 at once
            if(Tval in [5,10,15,25]):
                offset=800;
                print(30*"#", Tval, ":");
                for parami, _ in enumerate(results[Tvali]):
                    print(results[Tvali,parami], boundsT[Tvali, :, parami])
                ax3.scatter(x,offset*Tvali+y, color=mycolors[Tvali], marker=mymarkers[Tvali], label="$T=$ {:.0f} K".format(Tval));
                ax3.plot(x,offset*Tvali+yfit, color="black");
        
        else: # need to generate fit

            # raw data
            V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, Tval);

            if(stop_at not in stop_ats[:2]): # remove background
                for back in stop_ats[:2]:
                    back_params = np.load(fname+back+"results.npy")[Tvali];
                    back_fit = stopats_2_func[back](V_exp, *back_params);
                    if(verbose > 4): plot_fit(V_exp, dI_exp, back_fit, mytitle=back, myylabel="$dI/dV_b$"); 
                    dI_exp = dI_exp - back_fit

            # evaluate at fit results and plot
            dI_fit = stopats_2_func[stop_at](V_exp, *results[Tvali]);
            mytitle="$T = $ {:.1f} K, $\Gamma_0 = $ {:.5f} eV, $E_C = $ {:.5f} eV".format(Tval, *results[Tvali,2:])
            if(verbose > 4): plot_fit(V_exp, dI_exp, dI_fit, mytitle=mytitle, myylabel="$dI/dV_b$"); 
        
            # save V_exp, dI_exp, dI_fit for easy access
            print("Saving plot to "+plot_fname);
            np.save(plot_fname+"_x.npy", V_exp);
            np.save(plot_fname+"_y.npy", dI_exp);
            np.save(plot_fname+"_yfit.npy", dI_fit);
            np.savetxt(plot_fname+"_title.txt", [0], header=mytitle);

    ax3.set_title("Conductance oscillations in EGaIn$|$H$_2$Pc$|$MnPc|NCO");
    ax3.set_xlabel("$V_b$ (V)");
    ax3.set_xlim(-0.1,0.1);
    ax3.set_ylabel("$dI/dV_b$ (nA/V)");
    ax3.set_ylim(-400,2000);
    plt.legend(loc='upper right');
    plt.show()

####################################################################
#### run

if(__name__ == "__main__"):
    fit_Mn_data();
    #plot_saved_fit();
    #comp_backgrounds();
