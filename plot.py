'''
Plotting module for quick methods of making matplotlib plots in pyscf context
'''

import siam_current

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines

# format matplotlib globally

###############################################################################
#### plotting from txt file

def PlotTxt2D(fname, show = False, handles=[""], styles = [""], labels=["x","y",""]):
    '''
    Take 2D np array stored in txt file and plot x v y
    '''

    # for debugging
    cfname = "PlotTxt2D";

    # unpack data
    dat = np.loadtxt(fname);
    x, y = dat[0], dat[1];

    # check inputs
    if( type(x) != type(np.zeros(1) ) ): # check that x is an np array
        raise PlotTypeError(cfname+" 1st arg must be np array.\n");
    legend=True; # decide whether to implement legend
    if(handles==[""]): # no handles for legend provided
        legend = False;
    if(styles==[]): # no plot style kwargs provided
        pass;
        
    # construct axes
    fig, ax = plt.subplots();
    
    plt.plot(x, y, styles[0], label = handles[0]);

    # format and show
    ax.set(xlabel = labels[0], ylabel = labels[1], title=labels[2]);
    if(legend):
        ax.legend();
        
    if show: plt.show();

    return x, y; # end plot text 2d



###############################################################################
#### plotting directly

def GenericPlot(x,y, handles=[], styles = [], labels=["x","y",""]):
    '''
    Quick x vs y plot
    y can be > 1d and will plot seperate lines
    '''
    
    # for debugging
    cfname = "GenericPlot";
    
    # screen depth of y
    depth = np.shape(y)[0];
    if( len(np.shape(y)) <= 1): # bypass for 1d array inputs
        depth = 1;
        y = np.array([y]);
    
    # check inputs
    if( type(x) != type(np.zeros(1) ) ): # check that x is an np array
        raise PlotTypeError(cfname+" 1st arg must be np array.\n");
    legend=True;
    if(handles==[]): # no handles for legend provided
        handles = np.full(depth, "");
        legend = False;
    if(styles==[]): # no plot style kwargs provided
        styles = np.full(depth, "");
        
    # construct axes
    fig, ax = plt.subplots();

    #iter over y val sets
    for yi in range(depth):
    
        plt.plot(x, y[yi], styles[yi], label = handles[yi]);

    # format and show
    ax.set(xlabel = labels[0], ylabel = labels[1], title=labels[2]);
    if(legend):
        ax.legend();
    plt.show();

    return; # end generic plot


###############################################################################
#### very specific plot functions


def PlotObservables(dataf, nleads = (0,0), thyb = (1e-5,0.4), splots = ['Jtot','occ','Sz']):
    '''
    plot observables from td fci run
    Supported observables: J(up, down sep) Jtot(=Jup+Jdown), occ, change in
    occ, Sz, energy

    Args:
    - nleads, tuple of left lead sites, right lead sites
    - dataf, string of .npy filename where data array is stored
    - splots, list of strings which tell which subplots to make

    TODO: get nleads and thyb from input txt file
    '''

    # top level inputs
    numplots = len(splots);
    fig, axes = plt.subplots(numplots, sharex = True);
    ax_counter = 0; # update every time an ax is plotted

    # unpack
    print("Loading data from ",dataf);
    observables = np.load(dataf);
    t, E, Jup, Jdown, occL, occD, occR, SzL, SzD, SzR = tuple(observables); # scatter
    J = Jup + Jdown;
    mytitle = "Dot impurity:\n"+str(nleads[0])+" left sites, "+str(nleads[1])+" right sites, $t_{hyb}$ = "+str(thyb[0])+" -> "+str(thyb[1]);
    myxlabel = "time (dt = "+str(np.real(t[1]))+")"

    # plot current vs time, total and spin pol versions
    if 'J' in splots:
        axes[ax_counter].plot(t, J, color="black", linestyle = "dashed", label = "J"); # current
        axes[ax_counter].plot(t, Jup, color="red", label = "$J_{up}$");
        axes[ax_counter].plot(t, Jdown, color="yellow", label = "$J_{down}$");
        axes[ax_counter].set_ylabel("Current");
        axes[ax_counter].legend();
        ax_counter += 1;

    # just total current vs time
    if 'Jtot' in splots:
        axes[ax_counter].plot(t, J, color="black", linestyle = "dashed", label = "J"); # current
        axes[ax_counter].set_ylabel("Current");
        axes[ax_counter].set_title("Dot impurity, "+str(nleads[0])+" left sites, "+str(nleads[1])+" right sites");
        ax_counter += 1;
        
    # plot occupancy vs time
    if 'occ' in splots:
        axes[ax_counter].plot(t, occL, label = "Left lead"); # occupancy
        axes[ax_counter].plot(t, occD, label = "dot");
        axes[ax_counter].plot(t, occR, label = "Right lead");
        axes[ax_counter].set_ylabel("Occupancy")
        axes[ax_counter].legend();
        ax_counter += 1;

    # change in occupancy vs time
    if 'delta_occ' in splots:
        axes[ax_counter].plot(t, occL - occL[0], label = "Left lead");
        axes[ax_counter].plot(t, occD - occD[0], label = "dot");
        axes[ax_counter].plot(t, occR - occR[0], label = "Right lead");
        axes[ax_counter].set_ylabel(r"$\Delta$ Occupancy");
        ax_counter += 1;

    # z spin vs time
    if 'Sz' in splots: 
        axes[ax_counter].plot(t,SzL); 
        axes[ax_counter].plot(t,SzD);
        axes[ax_counter].plot(t,SzR);
        axes[ax_counter].set_ylabel("$<S_z>$");
        axes[ax_counter].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False)
        ax_counter += 1;

    # energy vs time
    if 'E' in splots: # energy
        axes[ax_counter].plot(t, E); # energy
        axes[ax_counter].set_ylabel("E");
        ax_counter += 1;

    # configure all axes, show
    for axi in range(len(axes) ): # customize axes
        if axi == 0: axes[axi].set_title(mytitle);
        if axi == len(axes)-1: axes[axi].set_xlabel(myxlabel);
        axes[axi].minorticks_on();
        axes[axi].grid(which='major', color='#DDDDDD', linewidth=0.8);
        axes[axi].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();
    return; # end plot observables


def CompObservablesB(dats, nleads, Bs, ts, Vg, splots = ['Jtot','Sz'] ):
    '''
    Compare current etc for different init spin states of dot
    due to different B fields
    '''

    # check params
    assert(len(Bs) == len(ts) == len(dats) );

    # top level inputs
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]
    mytitle = "Initial spin state comparison, $V_{g} =$"+str(Vg)+":\n"+str(nleads[0])+" left lead sites, "+str(nleads[1])+" right lead sites."
    numplots = len(splots);
    if 'Freq' in splots:
        fig, axes = plt.subplots(numplots); # dont share x axis
        numplots += -1; # this should force xlabel to last vs t plot, instead of freq plot
    else:
        fig, axes = plt.subplots(numplots, sharex = True);

    # parse B, t into strings
    for i in range(len(Bs)):
        try: # get 2 dec places if they are there
            Bs[i] = str(Bs[i])[:4];
            ts[i] = str(ts[i])[:4];
        except: # always at least 1 decimal
            Bs[i] = str(Bs[i])[:3];
            ts[i] = str(ts[i])[:3];

    for dati in range(len(dats)): # iter over data sets
        observables = np.load(dats[dati]);
        print("Loading data from "+dats[dati]);
        t, E, Jup, Jdown, occL, occD, occR, SzL, SzD, SzR = tuple(observables); # scatter
        J = Jup + Jdown;
        dt = np.real(t[1]);
        myxlabel = "time (dt = "+str(dt)+")"
        axcounter = 0;

        # plot current vs time
        if 'Jtot' in splots:
            lab = r"B = "+str(Bs[dati])+", $\\theta$ = "+str(ts[dati]);
            axes[axcounter].plot(t,J,label = lab);
            axes[axcounter].set_ylabel("Current");
            axcounter += 1

        if 'J' in splots:
            axes[axcounter].plot(t, Jup, color=colors[dati], linestyle = "dashed", label = "$J_{up}$"); # current
            axes[axcounter].plot(t, Jdown, color=colors[dati], linewidth = 3, linestyle = "dotted", label = "$J_{down}$");
            axes[axcounter].set_ylabel("Current");
            dashline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dashed');
            dotline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dotted');
            axes[axcounter].legend(handles=[dashline, dotline],labels=['$J_{up}$','$J_{down}$']);           
            axcounter += 1;

        # plot occupancy vs time
        if 'occ' in splots:
            axes[axcounter].plot(t, occL+0.1*dati, label = "Left lead"); # occupancy
            axes[axcounter].plot(t, occD+0.1*dati, label = "dot");
            axes[axcounter].plot(t, occR+0.1*dati, label = "Right lead");
            axes[axcounter].set_ylabel("Occupancy")
            axes[axcounter].legend();
            axcounter += 1;

        # plot Sz of dot vs time
        if 'Sz' in splots:
            axes[axcounter].plot(t, SzD);
            axes[axcounter].set_ylabel("Dot $S_z$");
            axcounter += 1;

        # plot Sz of leads vs time
        if 'Szleads' in splots:
            axes[axcounter].plot(t, SzL, color = colors[dati], linestyle='dashed', label = "left")
            axes[axcounter].plot(t, SzR, color = colors[dati], linestyle = "dotted", label = "right")
            axes[axcounter].set_ylabel("Lead $S_z$");
            dashline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dashed');
            dotline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dotted');
            axes[axcounter].legend(handles=[dashline, dotline],labels=['Left lead','Right lead']);  
            axcounter += 1;

        # plot energy vs time
        if 'E' in splots:
            axes[axcounter].plot(t, E);
            axes[axcounter].set_ylabel("Energy");
            axes[axcounter].get_shared_x_axes().join(axes[0],axes[3]);
            axcounter += 1

        # plot freq modes of current
        if 'Freq' in splots:
            Fnorm, freq = siam_current.Fourier(np.real(J), np.real(1/dt), angular = True);
            axes[axcounter].plot(freq, Fnorm);
            axes[axcounter].set_ylabel("Amplitude");
            axes[axcounter].set_xlabel(r"$\omega$");
            axes[axcounter].set_xlim(0,3);
            axcounter += 1;

    # format and show
    axes[0].legend();
    for axi in range(len(axes) ): # customize axes
        if axi == 0: axes[axi].set_title(mytitle);
        if axi == numplots-1: axes[axi].set_xlabel(myxlabel);
        axes[axi].minorticks_on();
        axes[axi].grid(which='major', color='#DDDDDD', linewidth=0.8);
        axes[axi].grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.show();

    return;
    
    
def PlotdtdE():

    # system inputs
    nleads = (4,4);
    nimp = 1;
    nelecs = (nleads[0] + nleads[1] + 1,0); # half filling
    mu = 0;
    Vg = -1.0;

    # time step is variable
    tf = 1.0;
    dts = [0.2, 0.167, 0.1, 0.02, 0.0167, 0.01]

    # delta E vs dt data
    dEvals = np.zeros(len(dts));
    dtvals = np.array(dts);

    # start the file name string
    folderstring = "txtdat/DotCurrentData/";

    # unpack each _E.txt file
    for i in range(len(dts)):

        dt = dts[i];

        # get arr from the txt file
        fstring = folderstring+"dt"+str(dt)+"_"+ str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_e"+str(nelecs[0])+"_mu"+str(mu)+"_Vg"+str(Vg);
        dtdE_arr = np.loadtxt(fstring+"_E.txt");
        
        # what we eant is Ef-Ei
        dEvals[i] = dtdE_arr[1,-1] - dtdE_arr[1,0];
        
    # fit to quadratic
    quad = np.polyfit(dtvals, dEvals, 2);
    tspace = np.linspace(dtvals[0], dtvals[-1], 100);
    quadvals = tspace*tspace*quad[0] + tspace*quad[1] + quad[2];

    # fit to exp
    def e_x(x,a,b,c):
        return a*np.exp(b*x) + c;
    fit = scipy.optimize.curve_fit(e_x, dtvals, dEvals);
    fitvals = e_x(tspace, *fit[0]);
    
    # plot results
    plt.plot(dtvals, dEvals, label = "data");
    plt.plot(tspace, quadvals, label ="Quadratic fit: $y = ax^2 + bx + c$");
    plt.plot(tspace, fitvals, label ="Exponential fit: $y= ae^{bx} + c$");
    plt.xlabel("time step");
    plt.ylabel("E(t=1.0) - E(t=0.0)");
    plt.title("$\Delta E$ vs dt, 4 leads each side");
    plt.legend();
    plt.show();
        
    return # end plot dt dE


def PlotFiniteSize():

    #### 1st plot: time period vs num sites

    # top level params and return vals
    nimp = 1; # dot model so 1 imp site
    tf = 12.0
    dt = 0.01
    mu = [0.0]
    Vg = [0.0]
    chainlengths = np.array([1,2,3,4]);
    TimePeriods = np.zeros(len(chainlengths) );

    # prep plot
    fig, (ax01, ax02) = plt.subplots(2);
    
    # how dominant freq depends on length of chain, for dot identical to lead site
    for chaini in range(len(chainlengths) ):

        chainlength = chainlengths[chaini];
        print(chainlength);
        nleads = chainlength, chainlength;
        nelecs = (2*chainlength+nimp,0); # half filling

        # plot J data for diff chain lengths
        folder = "dat/DotCurrentData/chain/"
        x, J, dummy, dummy = siam_current.UnpackDotData(folder, nleads, nimp, nelecs, mu, Vg);
        x, J = x[0], J[0] ; # unpack lists
        ax01.plot(x,J, label = str(nelecs[0])+" sites");

        # get time period data
        for xi in range(len(x)): # iter over time
            if( J[xi] < 0 and J[xi+1] > 0): # indicates full period
                TimePeriods[chaini] = x[xi];
                break;

    # format J vs t plot (ax1)
    ax01.legend();
    ax01.set_xlim(0,12);
    ax01.axhline(color = "grey", linestyle = "dashed")
    ax01.set_xlabel("time (dt = "+str(dt)+" s)");
    ax01.set_ylabel("$J*\pi/|V_{bias}|$");
    ax01.set_title("Finite size effects");

    # second plot: time period vs chain length
    numsites = 2*chainlengths + nimp;
    numsites = np.insert(numsites, 0,0); # 0, 0 should be a point too
    TimePeriods = np.insert(TimePeriods, 0,0)
    print(TimePeriods);
    ax02.plot(numsites, TimePeriods, label = "Data", color = "black");
    linear = np.polyfit(numsites, TimePeriods, 1); # plot linear fit
    linearvals = numsites*linear[0] + linear[1];
    ax02.plot(numsites, linearvals, label = "Linear fit, m = "+str(linear[0])[:6], color = "grey", linestyle = "dashed");
    
    ax02.legend();
    ax02.set_xlabel("Number of sites")
    ax02.set_ylabel("Time Period (s)");

    #### 2nd plot: fourier analyze 1 chain length at time
    mychainlength = 9; # can change
    dt = 0.01 # have to change manually as well
    Energies = [[0,1,2], [-4.464, -3.732, -3.464, -2.732, -2.464, -2.00],
        [-8.055, -7.289, -6.640, -6.524,-6.207, -5.875],
        [-10.009, -9.452, -9.391, -9.009, -8.834, -8.773, -8.725, -8.391, -8.276, -8.216, -8.107]];

    # show
    plt.show();
    return; # end plot finite size

def CurrentPlot(folder, nleads, nimp, nelecs, Vgs, B, theta, splots = ['Jtot','J','Sz'], mytitle = "", verbose = 0):
    '''
    Plot current and energy against time for dot impurity
    Fourier analyze and plot modes

    Designed for multiple data sets at a time, e.g. sweeps of Vg or mu

    DO NOT modify mu or Vg before calling Unpack to get data
    '''

    # confirm Vgs is iterable
    assert( isinstance(Vgs, list) or isinstance(Vgs, np.ndarray) );

    # control layout of plots
    numplots = len(splots);
    fig, axes = plt.subplots(numplots, sharex=True);
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple"]

    # plot data across Vg sweep
    for i in range(len(Vgs)):

        # get data from txt
        fname = folder+str(nleads[0])+"_"+str(nimp)+"_"+str(nleads[1])+"_e"+str(sum(nelecs))+"_B"+str(B)[:3]+"_t"+str(theta)[:3]+"_Vg"+str(Vgs[i])+".npy";
        observables = np.load(fname);
        t, E, Jup, Jdown, occL, occD, occR, SzL, SzD, SzR = tuple(observables); # scatter
        J = Jup + Jdown;
        dt = np.real(t[1]);
        axcounter = 0;
        print("Loading data from "+fname);

        # plot J vs t on top
        if 'Jtot' in splots:
            axes[axcounter].plot(t, J, label = "$V_g$ = "+str(Vgs[i]) );
            axes[axcounter].set_title(mytitle);
            axes[axcounter].set_xlabel("time (dt = "+str(dt)+" s)");
            axes[axcounter].set_ylabel("Current");
            axes[axcounter].legend();
            axcounter += 1;

        if 'J' in splots:
            axes[axcounter].plot(t, Jup, color=colors[i], linestyle = "dashed", label = "$J_{up}$"); # current
            axes[axcounter].plot(t, Jdown, color=colors[i], linestyle = "dotted", label = "$J_{down}$");
            axes[axcounter].set_ylabel("Current");
            dashline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dashed');
            dotline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dotted');
            axes[axcounter].legend(handles=[dashline, dotline],labels=['$J_{up}$','$J_{down}$']);           
            axcounter += 1;

        # plot occupancy vs time
        if 'occ' in splots:
            axes[axcounter].plot(t, occL, label = "Left lead"); # occupancy
            axes[axcounter].plot(t, occD, label = "dot");
            axes[axcounter].plot(t, occR, label = "Right lead");
            axes[axcounter].set_ylabel("Occupancy")
            axes[axcounter].legend();
            axcounter += 1;

        # plot Sz of dot vs time
        if 'Sz' in splots:
            axes[axcounter].plot(t, SzD, label = "$V_g$ = "+str(Vgs[i]) );
            axes[axcounter].set_ylabel("Dot $S_z$");
            axcounter += 1;

        # plot Sz of leads vs time
        if 'Szleads' in splots:
            axes[axcounter].plot(t, SzL, linestyle='dashed', color = colors[i])
            axes[axcounter].plot(t, SzR, linestyle = "dotted", color = colors[i])
            axes[axcounter].set_ylabel("Lead $S_z$");
            dashline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dashed');
            dotline = matplotlib.lines.Line2D([],[],color = 'black', linestyle = 'dotted');
            axes[axcounter].legend(handles=[dashline, dotline],labels=['Left lead','Right lead']);
            axcounter += 1;

        # plot E vs t middle
        if 'E' in splots:
            axes[axcounter].plot(t,E);
            axes[axcounter].set_xlabel("time (dt = "+str(dt)+" s)");
            axes[axcounter].set_ylabel("Energy");
            axcounter += 1;

        # plot frequencies below
        if False:
            Fnorm, freq = siam_current.Fourier(J, 1/dt, angular = True); # gives dominant frequencies
            axes[axcounter].plot(freq, Fnorm);
            axes[axcounter].set_xlabel("$\omega$ (2$\pi$/s)")
            axes[axcounter].set_ylabel("Amplitude")
            axes[axcounter].set_xlim(0,3);
            axcounter += 1;

    # config and show
    for axi in axes:
        axi.minorticks_on();
        axi.grid(which='major', color='#DDDDDD', linewidth=0.8);
        axi.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5);
    plt.tight_layout();
    plt.show();

    return; # end dot current plot


def FourierEnergyPlot(folder, nleads, nimp, nelecs, mu, Vg, Energies, mytitle = ""):
    '''
    From J data, plot discrete FT freqs, delta E freqs
    Compare actual freqs to those expected by energy differences

    Needs work

    Args:
    - folder, string, where to get data from
    - nleads, tuple of ints, num sites on each side
    - nimp, int, num impurity sites
    - nelecs, tuple of total num electrons, 0 due to ASU formalism
    - mu, float, chem potential of leads
    - Vg, float, chem potential of dot
    - Energies, list or array of energy spectrum of system
    '''

    assert(len(mu) == 1); # should both be lists of 1, b/c these plots dont compare across vals
    assert(len(Vg) == 1);
    
    # control layout of plots
    ax1 = plt.subplot2grid((3, 3), (1, 0), rowspan = 2)         # energy spectrum
    ax2 = plt.subplot2grid((3, 3), (1, 1), colspan = 2)         # freqs
    ax3 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)# J vs t
    ax4 = plt.subplot2grid((3, 3), (2, 1), colspan=2, sharex = ax2)           # expected freqs

    # get data fromtxt file
    xJ, yJ, xE, yE = siam_current.UnpackDotData(folder, nleads, nimp, nelecs, mu, Vg);
    xJ, yJ, xE, yE = xJ[0], yJ[0], xE[0], yE[0]; # undo lists
    dt = xJ[1]; # since xJ[0] = 0

    # get time period of J
    TimePeriod = 0;
    for xi in range(len(xJ)): # iter over time
        if( yJ[xi] < 0 and yJ[xi+1] > 0): # indicates full period has passed
            TimePeriod = xJ[xi];
            break;

    # e spectrum must be done manually
    xElevels, yElevels = ESpectrumPlot(Energies);
    for E in yElevels: # each energy level is sep line
        ax1.plot(xElevels, E, color = "black");
        ax1.set_ylabel("Energy (a.u.)")
    ax1.grid(which = 'both')
    ax1.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False);

    # convert e spectrum to expected frequencies
    omega = np.zeros(len(Energies));
    for Ei in range(len(Energies)):
        omega[Ei] = Energies[Ei] - Energies[0];
    ax4.hist(omega[1:], 100, color = "black");
    ax4.set_xlabel("$\Delta E$ (2$\pi$/s)")
    ax4.set_xlim(0,3);
    ax4.grid();

    # plot actual frequencies
    Fnorm, freq = siam_current.Fourier(yJ, 1/dt, angular = True); # gives dominant frequencies
    ax2.plot(freq, Fnorm, label = "Fourier");
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("$\omega$ ($2\pi/s$)");
    ax2.set_xlim(0,3);
    ax2.grid();

    # basic expectations for freqs
    ax2.axvline(x = 2*np.pi/(sum(nleads)+nimp), color = "grey", linestyle = "dashed", label = "$2\pi$/num. sites");
    ax2.axvline(x=2*np.pi/TimePeriod, color = "navy", linestyle = "dashed", label = "$2\pi$/T");
    ax2.legend();

    # plot J vs t on bottom 
    ax3.plot(xJ, yJ, label = "$V_g$ = "+str(Vg[0])+", $\mu$ = "+str(mu[0])  );
    ax3.set_title(mytitle);
    ax3.set_xlabel("time (dt = "+str(dt)+" s)");
    ax3.set_ylabel("$J*\pi/V_{bias}$");
    ax3.axhline(color = "grey", linestyle = "dashed");
    ax3.legend();

    # config and show
    plt.tight_layout();
    plt.show();
    return; # end fourier enrgy plot

    


###################################################################################
#### exec code

if __name__ == "__main__":

    pass;
