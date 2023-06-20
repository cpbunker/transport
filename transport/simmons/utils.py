'''
Simmons formula utils
'''

import numpy as np
import matplotlib.pyplot as plt

###############################################################
#### loading data

def load_IVb(base,folder,temp):
    '''
    Get I vs V data at a certain temp

    returns:
    V in volts, I in nano amps
    '''

    fname = "{:.0f}".format(temp) + base;
    print("Loading data from "+folder+fname);
    IV = np.loadtxt(folder+fname);
    Vs = IV[:, 0];
    Is = IV[:, 1];
    if( len(np.shape(Vs)) != 1 or np.shape(Vs) != np.shape(Is) ): raise TypeError;

    return Vs, 1e9*Is;

def load_dIdV(base,folder,temp):
    '''
    Get dIdV vs V data at a certain temp

    returns:
    V in volts, I in nano amps
    '''

    fname = "{:.0f}".format(temp) + base;
    print("Loading data from "+folder+fname);
    IV = np.loadtxt(folder+fname);
    Vs = IV[:, 0];
    dIs = IV[:, 1];
    if( len(np.shape(Vs)) != 1 or np.shape(Vs) != np.shape(dIs) ): raise TypeError;

    # sort by ascending Vs
    inds = np.argsort(Vs);
    Vs = np.take_along_axis(Vs, inds, 0);
    dIs = np.take_along_axis(dIs, inds, 0);

    return Vs, 1e9*dIs;

def load_dIdV_tocurrent(base,folder,temp,Vcutoff):
    '''
    Get dIdV vs V data at a certain temp

    returns:
    V in volts, I in nano amps
    '''

    fname = "{:.0f}".format(temp) + base;
    print("Loading data from "+folder+fname);
    IV = np.loadtxt(folder+fname);
    Vs = IV[:, 0];
    dIs = IV[:, 1];
    if( len(np.shape(Vs)) != 1 or np.shape(Vs) != np.shape(dIs) ): raise TypeError;

    # truncate
    if(Vcutoff != None):
        dIs = dIs[abs(Vs)<Vcutoff];
        Vs = Vs[abs(Vs)<Vcutoff];

    # trim outliers
    sigma = np.std(dIs);
    mu = np.mean(dIs);
    Vs = Vs[ abs(dIs-mu) < 6*sigma];
    dIs = dIs[ abs(dIs-mu) < 6*sigma];

    # sort by ascending Vs
    inds = np.argsort(Vs);
    Vs = np.take_along_axis(Vs, inds, 0);
    dIs = np.take_along_axis(dIs, inds, 0);

    # antideriv of dI is I
    # antideriv of dIdV is I
    Is = np.empty_like(dIs);
    for eli in range(len(Is)):
        Is[eli] = np.trapz(dIs[:eli],Vs[:eli]);

    if(False):
        # compare I and \int dI dV
        if temp==5.0: V_exp, I_exp = load_IVb("KExp.txt",folder,temp); # in volts, nano amps
        else: V_exp, I_exp = np.copy(Vs), np.copy(1e9*dIs);
        compfig, compax = plt.subplots();
        compax.plot(V_exp, I_exp);
        compax.plot(Vs,1e9*Is-114 )
        plt.show();
        assert False

    return Vs, 1e9*Is; # in nano amps

###############################################################
#### plotting functions

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

def plot_fit(V_exp, I_exp, I_fit, mytitle = "", myylabel=""):
    '''
    '''
    if( np.shape(I_exp) != np.shape(I_fit) ): raise TypeError;
    fig, ax = plt.subplots();
   
    # plot
    ax.scatter(V_exp, I_exp, color=mycolors[0], label = "Exp", linewidth = mylinewidth);
    ax.plot(V_exp, I_fit, color=accentcolors[0], label = "Fit", linewidth = mylinewidth);

    # error
    error = np.sqrt( np.mean( np.power(I_fit - I_exp,2) ))/abs(np.max(I_exp)-np.min(I_exp));
    ax.plot( [0.0], [0.0], color='white', label = "RMSE = {:1.5f}".format(error));
    
    # format
    ax.set_xlabel("$V_b$ (V)");
    ax.set_ylabel(myylabel);
    plt.legend();
    plt.title(mytitle, fontsize = myfontsize);
    plt.show();

###############################################################
#### misc

def fit_wrapper(fit_func, xvals, yvals, p0, bounds, p_names, verbose=0, max_nfev = 2000, myylabel="$I$ (nA)"):
    '''
    Wrapper for
    calling scipy.optimize.curve_fit and getting fitting params
    calculating the rmse
    Printing the fitting params and rmse results
    Plotting the fitting
    '''
    from scipy.optimize import curve_fit as scipy_curve_fit
    from scipy.optimize import Bounds as scipy_Bounds
    if( np.shape(xvals) != np.shape(yvals)): raise ValueError;
    if( (p0 is not None) and np.shape(p0) != np.shape(p_names)): raise ValueError;

    # use scipy to get fitting params
    ls_verbose = min(2, verbose+1);
    if(p0 is not None and bounds is not None): # fit with guesses, bounds
        fit_params, _ = scipy_curve_fit(fit_func, xvals, yvals,
                                p0=p0,bounds=bounds, loss='linear', max_nfev = max_nfev, verbose=ls_verbose);
    elif(False and p0 is not None and bounds is None): # fit with guesses but without bounds
        fit_params, _ = scipy_curve_fit(fit_func, xvals, yvals,p0=p0, verbose=ls_verbose);
    elif(False and p0 is None and bounds is None): # fit without guesses, bounds
        fit_params, _ = scipy_curve_fit(fit_func, xvals, yval, verbose=ls_verboses);
    else: raise NotImplementedError;

    # fit and error
    yfit = fit_func(xvals, *fit_params);
    rmse = np.sqrt( np.mean( np.power(yvals-yfit,2) ))/abs(np.max(yvals)-np.min(yvals));

    # visualize fitting
    if(verbose):
        namelength = 6;
        numdigits = 4
        print_str = "\n"+str(fit_func)+" fitting results:\n";
        for pi in range(len(fit_params)):
            print_str += "\t"+p_names[pi]+(" "*(namelength-len(p_names[pi])));
            print_str += "= {:6."+str(numdigits)+"f},";
            if(bounds is not None): print_str += "bounds = "+str(bounds[:,pi]);
            print_str += "\n";
        if("phi1" in p_names and "phi2" in p_names and "m_r" in p_names):
            phibar = fit_params[1]/2+fit_params[2]/2;
            m_r = fit_params[3];
            print_str += "\tphi*mr= "+str((phibar*m_r).round(numdigits))+"\n";
        elif("phibar" in p_names and "m_r" in p_names):
            phibar = fit_params[1];
            m_r = fit_params[2];
            print_str += "\tphi*mr= "+str((phibar*m_r).round(numdigits))+"\n";
        print_str += "\tRMSE  = "+str(rmse.round(numdigits));
        print(print_str.format(*fit_params),"\n");
    if(verbose > 4): plot_fit(xvals, yvals, yfit, mytitle = str(fit_func), myylabel=myylabel);
    return fit_params, rmse;

def my_curve_fit(fx, xvals, fxvals, init_params, bounds, focus_i, focus_meshpts=10, verbose=0):
    '''
    On top of scipy_curve_fit, build extra resolution to the
    dependence of the error on params[focus_i]
    '''
    if(not isinstance(bounds, np.ndarray)): raise TypeError;
    if(focus_i >= len(init_params)): raise ValueError;

    # mesh search over focus param vals
    focus_lims = np.linspace(bounds[0][focus_i],bounds[1][focus_i],focus_meshpts);
    focus_opts = np.empty((len(focus_lims)-1,));
    focus_errors = np.empty((len(focus_lims)-1,));    
    for fvali in range(len(focus_lims)-1):

        # update guess
        init_fparams = np.copy(init_params);
        init_fparams[focus_i] = (focus_lims[fvali] + focus_lims[fvali+1])/2;

        # truncate focus bounds
        fbounds = np.copy(bounds);
        fbounds[0][focus_i], fbounds[1][focus_i] = focus_lims[fvali], focus_lims[fvali+1];

        # fit within this narrow range
        nano, convert_J2I = 1e9, 1;
        #plot_guess(myT, myA, 0.0, 1e-10/convert_J2I, init_params[0], ):
        fparams, pcov = scipy_curve_fit(fx, xvals, fxvals,
                                 p0 = init_fparams, bounds = fbounds, max_nfev = 1e6);
        fxvals_fit = fx(xvals, *fparams);
        ferror = np.sqrt( np.mean( (fxvals-fxvals_fit)*(fxvals-fxvals_fit) ))/np.mean(fxvals);

        # update error and optimum
        focus_opts[fvali] = fparams[focus_i];
        focus_errors[fvali] = ferror;

        # visualize
        if(verbose > 2):
            print("my_curve_fit fitting results, focus_i = ",focus_i);
            print_str = "            d = {:6.4f} "+str((fbounds[0][0],fbounds[1][0]))+" nm\n\
            phi = {:6.4f} "+str((fbounds[0][1],fbounds[1][1]))+" eV\n\
            m_r = {:6.4f} "+str((fbounds[0][2],fbounds[1][2]))+"\n\
            err = {:6.4f}";
            print(print_str.format(*fparams, ferror));
        if(verbose > 4): plot_fit(xvals, fxvals*convert_J2I*nano, fxvals_fit*convert_J2I*nano,
                                  mytitle = "d = {:1.2f} nm, phi = {:1.2f} eV, m_r = {:1.2f} ".format(*fparams));

    # show the results of the mesh search
    if verbose:
        fig, ax = plt.subplots();
        ax.plot(focus_opts, focus_errors, color=accentcolors[0]);

        # format
        ax.set_xlabel("Params["+str(focus_i)+"]");
        ax.set_ylabel("Norm. RMS Error");
        #ax.set_yscale('log');
        plt.show();
    return focus_opts, focus_errors;

