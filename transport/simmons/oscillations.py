'''
Simmons formula description of tunneling through a tunnel junction,
under different physical scenarios
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
    retval -= (G3/2)*Ffunc(abs(Vb-V0), kelvin2eV*temp_kwarg);
    retval -= (G3/4)*Ffunc(abs(Vb-V0+Delta), kelvin2eV*temp_kwarg);
    retval -= (G3/4)*Ffunc(abs(Vb-V0-Delta), kelvin2eV*temp_kwarg);
    return retval

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
    return dI0+amplitude*(-1)*np.cos(ang_freq*(Vb-V0));

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

####################################################################
#### main

def fit_dIdV(metal, temp, dI0_not, Gamma_not, EC_not,
             dI0_percent, Gamma_percent, EC_percent, stop_at='sin', num_dev = 4, verbose=0):
    '''
    '''

    # load data
    V_exp, dI_exp = load_dIdV("KdIdV.txt",metal, temp);
    Vlim = min([abs(np.min(V_exp)), abs(np.max(V_exp))]);
    dI_mu = np.mean(dI_exp);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-dI_mu,2)));
    del temp
    
    #### fit background
    E0_guess, G2_guess, G3_guess = 0.0105, 1340,815 # 2.5 K: 0.0105, 850,450
    Ec_guess, G1_guess = 0.013, 2000;
    Edown, Eup, Gdown, Gup = 0.9, 1.1, 0.1, 1.9;
    V0_bound = 1e-2;

    # fit to magnon and imp together
    params_back_guess = np.array([0.0, E0_guess, Ec_guess, G1_guess, G2_guess, G3_guess]);
    bounds_back = np.array([[-V0_bound, E0_guess*Edown, Ec_guess*Edown, G1_guess*Gdown, 0.1*G2_guess, G3_guess*Gdown],
                            [ V0_bound, E0_guess*Eup, Ec_guess*Eup, G1_guess*Gup, 1.9*G2_guess, G3_guess*Gup]]);
    params_back, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_back_guess, bounds_back, ["V0", "E0", "Ec", "G1", "G2", "G3"],
                            stop_bounds = False, verbose=verbose);
    background = dIdV_back(V_exp, *params_back);

    # remove outliers based on background
    pre_dropout = len(V_exp);
    V_exp = V_exp[abs(dI_exp-background) < num_dev*dI_dev];
    dI_exp = dI_exp[abs(dI_exp-background) < num_dev*dI_dev];
    assert(pre_dropout - len(V_exp) <= pre_dropout*0.05); # only remove 5%

    # imp or mag individually
    if(stop_at == "imp/" or stop_at == "mag/" or True):
        # force G1=0
        params_imp_guess = np.copy(params_back_guess);
        params_imp_guess[3] = 0;
        bounds_imp = np.array([[-V0_bound, E0_guess, Ec_guess, 0.0, G2_guess, G3_guess],
                                [ V0_bound, E0_guess+1e-6, Ec_guess+1e-6, 1e-6, G2_guess+1e-6, G3_guess+1e-6]]);
        params_imp, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                        params_imp_guess, bounds_imp, ["V0", "E0", "Ec", "G1", "G2", "G3"],
                        stop_bounds = False, verbose=verbose);
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_back(V_exp, *params_imp), derivative=True,
                            mytitle="Impurity scattering ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        mask_imp = np.array([1,1,0,0,1,1]);
        if(stop_at == 'imp/'): return params_imp[mask_imp>0], bounds_imp[:,mask_imp>0];

        # force G2, G3=0
        params_mag_guess = np.copy(params_back_guess);
        params_mag_guess[4] = 0;
        params_mag_guess[5] = 0;
        bounds_mag = np.array([[-V0_bound, E0_guess, Ec_guess, G1_guess, 0,0],
                                [ V0_bound, E0_guess+1e-6, Ec_guess+1e-6, G1_guess+1e-6, 1e-6,1e-6]]);
        params_mag, _ = fit_wrapper(dIdV_back, V_exp, dI_exp-dIdV_back(V_exp, *params_imp),
                        params_mag_guess, bounds_mag, ["V0", "E0", "Ec", "G1", "G2", "G3"],
                        stop_bounds = False, verbose=verbose);
        if(verbose > 4): plot_fit(V_exp, dI_exp-dIdV_back(V_exp, *params_imp), dIdV_back(V_exp, *params_mag), derivative=True,
                            smooth=False, mytitle="Magnon scattering ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        mask_mag = np.array([1,0,1,1,0,0]);
        if(stop_at == 'mag/'): return params_mag[mask_mag>0], bounds_mag[:,mask_mag>0];

    # fit to magnon and imp again
    params_back, _ = fit_wrapper(dIdV_back, V_exp, dI_exp,
                            params_back, bounds_back, ["V0", "E0", "Ec", "G1", "G2", "G3"],
                            stop_bounds = False, verbose=verbose);
    background = dIdV_back(V_exp, *params_back);
    if(verbose > 4): plot_fit(V_exp, dI_exp, background, derivative=False,
                        mytitle="Impurity + magnon scattering ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'imp_mag/'): return params_back, bounds_back;

    #### fit magnon + imp + oscillation
    params_all_guess = np.zeros((len(params_back)+3,));
    params_all_guess[:len(params_back)] = params_back;
    bounds_all = np.zeros((2,len(params_all_guess)));
    bounds_all[:,:len(params_back)] = bounds_back;

    # for oscillation
    params_all_guess[len(params_back_guess):] = np.array([dI0_not, Gamma_not, EC_not]);
    bounds_all[:,len(params_back_guess):] = np.array([ [dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                                                [ dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
    params_all, _ = fit_wrapper(dIdV_all_zero, V_exp, dI_exp,
                                params_all_guess, bounds_all, ["V0", "E0", "Ec", "G1", "G2", "G3","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_all_zero(V_exp, *params_all), derivative=False,
                mytitle="$T=0$ Landauer fit ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
    if(stop_at == 'lorentz_zero/'): return params_all, bounds_all;   

    # subtract combined background
    dI_exp = dI_exp - background; 
    dI_mu = np.mean(dI_exp);
    dI_dev = np.sqrt( np.median(np.power(dI_exp-dI_mu,2)));

    #### fit oscillations to sin

    # fit sin
    if(stop_at == 'sin/'):
        Vdroplims = 0.0, 0.1;
        dI_exp = dI_exp[abs(V_exp) > Vdroplims[0]];
        V_exp = V_exp[abs(V_exp) > Vdroplims[0]];
        dI_exp = dI_exp[abs(V_exp) < Vdroplims[1]];
        V_exp = V_exp[abs(V_exp) < Vdroplims[1]];
        params_sin_guess = np.array([params_back[0], dI_dev, 4*EC_not,0.0]);
        bounds_sin = [[params_back[0],dI_dev*1.0, 4*EC_not*0.9, 0.0],
                        [params_back[0]+1e-6, dI_dev*2.0, 4*EC_not*1.1, dI_dev]];
        bounds_sin = np.array(bounds_sin);
        params_sin, _ = fit_wrapper(dIdV_sin, V_exp, dI_exp,
                        params_sin_guess, bounds_sin, ["V0","amp","per","dI0"],
                        stop_bounds = False, verbose=verbose);
        if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_sin(V_exp, *params_sin), derivative=False,
                            mytitle="Sinusoidal fit ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
        return params_sin, bounds_sin;

    #### fit oscillations to T=0 landauer

    # all params get fit here
    params_zero_guess = np.array([params_all[0], dI0_not, Gamma_not, EC_not]);

    import time
    start = time.time();
    dIdV_lorentz(V_exp, *params_zero_guess);
    stop = time.time();
    print("Integration time = ", stop-start);
    assert False
    
    bounds_zero = np.array([ [-V0_bound, dI0_not*(1-dI0_percent), Gamma_not*(1-Gamma_percent), EC_not*(1-EC_percent)],
                [ V0_bound, dI0_not*(1+dI0_percent), Gamma_not*(1+Gamma_percent), EC_not*(1+EC_percent) ]]);
    params_zero, _ = fit_wrapper(dIdV_lorentz_zero, V_exp, dI_exp,
                                params_zero_guess, bounds_zero, ["V0","dI0","Gamma", "EC"],
                                stop_bounds = False, verbose=verbose);
    if(verbose > 4): plot_fit(V_exp, dI_exp, dIdV_lorentz_zero(V_exp, *params_zero), derivative=False,
                mytitle="$T=0$ Landauer fit ($T=$ {:.1f} K)".format(temp_kwarg), myylabel="$dI/dV_b$ (nA/V)");
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
    elif(stop_at == 'lorentz_zero/' or stop_at =='lorentz/'):
        rlabels = ["$V_0$", "$dI_0$ (nA/V)", "$\Gamma_0$ (eV)", "$E_C$ (eV)"];
    else: raise NotImplementedError;

    # experimental params
    kelvin2eV =  8.617e-5;
    Ts = np.array([2.5, 5.0,10.0,15.0,20.0,25.0,30.0]);
    Ts = np.array([25.0]);
    # sample temp shifted due to ohmic heating
    ohmic_Ts = np.array([4]);
    assert(Ts.shape == ohmic_Ts.shape);

    # guesses
    dI0_guess = np.array([63000,65452, 66574, 68000, 68000, 70000, 70000]);
    Gamma_guess = np.array([0.0045,0.0054, 0.0054, 0.0059, 0.0062, 0.0065, 0.0067]);
    EC_guess = (0.023/4)*np.ones_like(Ts);
    dI0_percent = 0.4;
    Gamma_percent = 0.4;
    EC_percent = 0.1;

    #fitting results
    results = [];
    boundsT = [];
    for datai in range(len(Ts)):
        print("#"*60+"\nT = {:.1f} K ({:.4f} eV)".format(Ts[datai], Ts[datai]*kelvin2eV));

        # get fit results
        global temp_kwarg; temp_kwarg = Ts[datai]+ohmic_Ts[datai]; # very bad practice
        temp_results, temp_bounds = fit_dIdV(metal, Ts[datai],
            dI0_guess[datai], Gamma_guess[datai], EC_guess[datai],
            dI0_percent, Gamma_percent, EC_percent,
            stop_at = stop_at, verbose=10);
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
    if False:
        fname = "fits/"
        print("Saving data to "+fname+stop_at);
        np.savetxt(fname+stop_at+"Ts.txt", Ts);
        np.save(fname+stop_at+"results.npy", results);
        np.save(fname+stop_at+"bounds.npy", boundsT);

    # format
    axes[-1].set_xlabel("$T$ (K)");
    axes[0].set_title("Amplitude and period fitting");
    plt.show();

def plot_saved_fit():
    '''
    '''
    verbose=10;
    metal="Mn/"; # points to data folder
    stop_ats = ['imp/','mag/','imp_mag/', 'sin/', 'lorentz_zero/', 'lorentz/'];
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
            
            if True:
                plot_fit(x, y, yfit, mytitle=mytitle, myylabel="$dI/dV_b$");
            else: # plot 3 at once
                if(Tval in [5,15,25]):
                    offset=400;
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
            mytitle="$T = $ {:.1f} K, $\Gamma_0 = $ {:.5f} eV, $E_C = $ {:.5f} eV".format(Ts[Tvali], *results[Tvali,2:])
            if(verbose > 4): plot_fit(V_exp, dI_exp, dI_fit, mytitle=mytitle, myylabel="$dI/dV_b$"); 
        
            # save V_exp, dI_exp, dI_fit for easy access
            print("Saving plot to "+plot_fname);
            np.save(plot_fname+"_x.npy", V_exp);
            np.save(plot_fname+"_y.npy", dI_exp);
            np.save(plot_fname+"_yfit.npy", dI_fit);
            np.savetxt(plot_fname+"_title.txt", [0], header=mytitle);

    ax3.set_title("Conductance oscillations in EGaIn$|$H$_2$Pc$|$MnPc$|$NCO");
    ax3.set_xlabel("$V_b$ (V)");
    ax3.set_xlim(-0.1,0.1);
    ax3.set_ylabel("$dI/dV_b$ (nA/V)");
    ax3.set_ylim(-500,2500);
    plt.legend(loc='upper right');
    plt.show()

####################################################################
#### run

if(__name__ == "__main__"):
    fit_Mn_data();
    #plot_saved_fit();

