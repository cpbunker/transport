'''
Christian Bunker
M^2QM at UF
November 2022

Scattering from two spin-s MSQs
Want to make a SWAP gate
solved in time-independent QM using wfm method in transport/wfm
'''

from transport import wfm

from run_wfm_gate import get_hblocks, get_U_gate, get_Fval, get_suptitle, get_indep_vals;

import numpy as np
import matplotlib.pyplot as plt

import sys
          
#########################################################
#### barrier in right lead for total reflection

#### top level
np.set_printoptions(precision = 2, suppress = True);
verbose = 1;
case = sys.argv[2];
final_plots = int(sys.argv[3]);
#if final_plots: plt.rcParams.update({"text.usetex": True,"font.family": "Times"})

# fig standardizing
myxvals = 29; myfigsize = (5/1.2,3/1.2); myfontsize = 14;
if(final_plots): myxvals = 199; 
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["+","o","^","s","d","*","X","+"];
mymarkevery = (myxvals//3, myxvals//3);
mypanels = ["(a)","(b)","(c)","(d)"];
ylabels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
    "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"];
 
# tight binding params
tl = 1.0;
myTwoS = 1;
n_mol_dof = (myTwoS+1)*(myTwoS+1); 
n_loc_dof = 2*n_mol_dof; # electron is always spin-1/2
Jval = float(sys.argv[1]); # sd exchange
Vq = 0.0#Jval/2; # onsite energy on qubits
VBar = 0.000*tl; # on site energy in barrier region
Vend = 5.0*tl; # wide band gap at far right end, never changes
the_ticks = [0.0,1.0]; # always positive since it is fidelity

#### plot already existing data
if(final_plots == 10):

    # open corresponding itle file
    if(case in ["NB","kNB"]): 
        which_color = "K";
    elif(case in ["onsite_NB500","VB_NB500"]): 
        which_color = "x";
    elif(case in ["gates_lambda", "gates_K", "conc_lambda", "conc_K", "roots_lambda", "roots_K", "dimensionless_energy"]): 
        which_color = "NB";
    whichval = int(sys.argv[4]);
    title_and_colors = ("data/gate/"+case+"/ALL_J{:.2f}_"+which_color+"{:.0f}_title.txt").format(Jval,whichval);
    suptitle = open(title_and_colors,"r").read().splitlines()[0][1:];
    
    # array of numerical values corresponding to physical quantities eg NB,
    # which we are representing on plots as different colored lines
    colorvals = np.loadtxt(title_and_colors,ndmin=1); 
    if(case in ["NB","kNB"]):
        which_color_list = np.arange(len(colorvals));
    elif(case in ["onsite_NB500","VB_NB500"]):
        XVAL_INT_CONVERSION = 100;
        which_color_list = 1*colorvals.astype(int); 
        colorvals = (1/XVAL_INT_CONVERSION)*colorvals;
    elif(case in ["gates_lambda", "gates_K", "conc_lambda", "conc_K", "roots_lambda", "roots_K"]):
        colorvals = colorvals.astype(int);
        which_color_list = 1*colorvals;
        
    if(case in ["dimensionless_energy"]):
        print(colorvals);
        del Jval;
        gates = sys.argv[5:];
        nrows, ncols = len(gates), 1;
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
        if(nrows==1): axes = [axes];
        fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
        for gatevali in range(len(gates)):

            # load and plot Fidelity
            for colori in range(len(colorvals)):
                # load data
                yvals = np.load(("data/gate/gates_lambda/"+gates[gatevali]+"_J{:.2f}_"+which_color+"{:.0f}_y.npy").format(colorvals[colori,0], colorvals[colori,1]));
                xvals = np.load(("data/gate/gates_lambda/"+gates[gatevali]+"_J{:.2f}_"+which_color+"{:.0f}_x.npy").format(colorvals[colori,0], colorvals[colori,1]));
                mymarkevery = (len(xvals)//3, len(xvals)//3);
                
                # maxima
                indep_star = xvals[np.argmax(yvals)];
                if(verbose):
                    indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(yvals));
                    print("\n"+gates[gatevali]+"\n",indep_comment);
                
                # determine label
                mylabel = "$J =${:.4f}, $N_B = ${:.0f}".format(colorvals[colori,0], colorvals[colori,1]);
                    
                # plot
                dl_energy_vals = xvals*(2*np.pi/colorvals[colori,1])*abs(1/colorvals[colori,0]);
                axes[gatevali].plot(dl_energy_vals,yvals, label = mylabel, color=mycolors[colori], marker=mymarkers[1+colori], markevery=mymarkevery);
                
            #### end loop over colors (here fixed K or NB vals)

            # plot formatting
            axes[-1].set_xlabel("$\\frac{2\pi a}{\lambda_i} \\frac{t}{|J|} = \sqrt{\\frac{E+2t}{|J|}} \sqrt{\\frac{t}{|J|}}$",fontsize=myfontsize);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks:                     
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[gatevali].annotate("$\mathbf{U}_{"+gates[gatevali]+"}$", (xvals[1],1.01),fontsize=myfontsize);
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{R}, \mathbf{U})$",fontsize=myfontsize);
        #### end loop over gates
        
    # iter over gates
    elif(case not in ["roots_lambda", "roots_K","onsite_NB500","VB_NB500"]): 
        gates = sys.argv[5:];
        nrows, ncols = len(gates), 1;
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
        if(nrows==1): axes = [axes];
        fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
        for gatevali in range(len(gates)):

            # load and plot Fidelity
            for colori in range(len(colorvals)):
                # load data
                yvals = np.load(("data/gate/"+case+"/"+gates[gatevali]+"_J{:.2f}_"+which_color+"{:.0f}_y.npy").format(Jval, which_color_list[colori]));
                xvals = np.load(("data/gate/"+case+"/"+gates[gatevali]+"_J{:.2f}_"+which_color+"{:.0f}_x.npy").format(Jval, which_color_list[colori]));
                mymarkevery = (len(xvals)//3, len(xvals)//3);
                
                # maxima
                indep_star = xvals[np.argmax(yvals)];
                if(verbose):
                    indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(yvals));
                    print("\n"+gates[gatevali]+"\n",indep_comment);
                
                # determine label
                if(case in ["NB","kNB"]):
                    correct_Kpower = False;
                    for Kpower in range(-9,-1):
                        if(abs(colorvals[colori] - 10.0**Kpower) < 1e-10): correct_Kpower = 1*Kpower;
                    if(correct_Kpower is not False): mylabel = "$k_i^2 a^2 = 10^{"+str(correct_Kpower)+"}$";
                    else: mylabel = "$k_i^2 a^2 = {:.6f} $".format(colorvals[colori]);
                elif(case in ["gates_lambda", "gates_K", "conc_lambda", "conc_K"]):
                    mylabel = "$N_B = ${:.0f}".format(colorvals[colori]);
                    
                # plot
                axes[gatevali].plot(xvals,yvals, label = mylabel, color=mycolors[colori], marker=mymarkers[1+colori], markevery=mymarkevery);
                
            #### end loop over colors (here fixed K or NB vals)

            # plot formatting
            
            # x axis
            if(case=="NB"):
                axes[-1].set_xlabel("$N_B$",fontsize=myfontsize);
            elif(case in ["gates_K", "conc_K"]):
                axes[-1].set_xlabel("$k_i^2 a^2$",fontsize=myfontsize);
                axes[-1].set_xscale("log", subs = []);
            elif(case in ["gates_lambda", "conc_lambda"]):
                axes[-1].set_xlabel("$N_B a/\lambda_i $",fontsize=myfontsize);
                axes[gatevali].set_xlim(0,np.max(xvals));
            else: raise NotImplementedError;
            
            # y axis
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            if(gates[gatevali]=="conc"):
                axes[gatevali].set_ylabel("$C$");
            elif(gates[gatevali]=="overlap"):
                axes[gatevali].set_ylabel("$P_{swap}$");
            elif(gates[gatevali]=="overlap_sf"):
                axes[gatevali].set_ylabel("$P_{sf}$");
            else:
                axes[gatevali].annotate("$\mathbf{U}_{"+gates[gatevali]+"}$", (xvals[1],1.01),fontsize=myfontsize);
                axes[gatevali].set_ylabel("$F_{avg}(\mathbf{R}, \mathbf{U})$",fontsize=myfontsize);
        #### end loop over gates

    else: # data structure is different
        if(case=="roots_K"): raise NotImplementedError;
        roots = sys.argv[5:]; # these are actually the colors, not colorvals
        if(roots[-1] == "SeS12"): mycolors[len(roots)-1] = "black";
        nrows, ncols = np.shape(colorvals)[0], 1;
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
        if(nrows==1): axes = [axes];
        fig.set_size_inches(ncols*myfigsize[0], nrows*myfigsize[1]);
        for colori in range(len(colorvals)): # this iters over axis rows!

            # iter over SWAP roots, which are colors
            for rootvali in range(len(roots)):
                # load data
                yvals = np.load(("data/gate/"+case+"/"+roots[rootvali]+"_J{:.2f}_"+which_color+"{:.0f}_y.npy").format(Jval, which_color_list[colori]));
                xvals = np.load(("data/gate/"+case+"/"+roots[rootvali]+"_J{:.2f}_"+which_color+"{:.0f}_x.npy").format(Jval, which_color_list[colori]));
                
                # determine label
                mymarkevery = (len(xvals)//3, len(xvals)//3);
                if(case in ["roots_lambda", "roots_K"]):
                    mylabel = "$n = $"+roots[rootvali];
                    annotate_string = "$N_B =${:.0f}";
                    my_ylabel = "$F_{avg}[\mathbf{R}, (\mathbf{U}_{SWAP})^{1/n}]$";
                    my_xlabel = "$N_B a /\lambda_i$";
                elif(case in ["onsite_NB500","VB_NB500"]):
                    mylabel = roots[rootvali];
                    annotate_string = "$N_B a/\lambda_i =${:.2f}";
                    my_ylabel = "";
                    if("onsite" in case): my_xlabel = "$V_q$";
                    elif("VB" in case): my_xlabel = "$V_B$";
                
                # plot
                axes[colori].plot(xvals,yvals, label = mylabel, color=mycolors[rootvali],marker=mymarkers[1+rootvali],markevery=mymarkevery);
                
                # maxima
                indep_star = xvals[np.argmax(yvals)];
                if(verbose):
                    indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(yvals));
                    print("\nU^1/"+roots[rootvali]+"\n",indep_comment);
            #### end loop over colors (root vals)
                
             # plot formatting

            # x axis
            axes[-1].set_xlabel(my_xlabel,fontsize=myfontsize);
            #axes[gatevali].set_xlim(0,np.max(xvals));
            
            # y axis
            #axes[colori].set_yticks(the_ticks);
            axes[colori].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[colori].axhline(tick,color='lightgray',linestyle='dashed');
            axes[colori].annotate(annotate_string.format(colorvals[colori]), (xvals[len(xvals)*1//4],1.01),fontsize=myfontsize);
            axes[colori].set_ylabel(my_ylabel,fontsize=myfontsize);
        #### end loop over fixed NB vals
            
    # show
    fig.suptitle(suptitle);
    plt.tight_layout();
    axes[-1].legend(loc = "lower right");
    plt.show();

######################################################################################
#### generate data
elif(case in ["NB","kNB"]): # fidelities at fixed Ki, as a function of NB
    if(case=="NB"): NB_indep = True; # whether to put NB, alternatively wavenumber*NB
    elif(case=="kNB"): NB_indep = False;
    raise NotImplementedError;

    # axes
    gates = ["SeS12","SQRT","SWAP","I"];
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);

    # iter over incident wavenumber (colors)
    Energies = Kvals - 2*tl; # -2t < Energy < 2t and is the argument of self energies, Green's functions etc
    rhatvals = np.empty((n_loc_dof,n_loc_dof,len(Kvals),myxvals),dtype=complex); # by  init spin, final spin, energy, NB
    Fvals_min = np.empty((len(Kvals), myxvals, len(gates)),dtype=float); # avg fidelity
    for Kvali in range(len(Kvals)):
               
        # iter over barrier distance (x axis)
        kNBmax = 1.0*np.pi;
        NBmax = np.ceil(kNBmax/knumbers[Kvali]);
        if(NB_indep): NBmax = 150;
        NBvals = np.linspace(1,NBmax,myxvals,dtype=int);
        if(verbose): print("\nk^2 = {:6f}, NBmax = {:.0f}".format(knumbers[Kvali]**2, NBmax)); 
        
        for NBvali in range(len(NBvals)):
            NBval = NBvals[NBvali];

            # construct hblocks from cicc-type spin ham
            hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, Vq, VBar, Vend, NBval);

            # define source, although it doesn't function as a b.c. since we return Rhat matrix
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source, False, is_Rhat = True, all_debug = False);

            # iter over gates to get fidelity for each one
            for gatevali in range(len(gates)):
                U_gate, dummy_ticks = get_U_gate(gates[gatevali],myTwoS);
                Fvals_min[Kvali, NBvali, gatevali] = get_Fval(gates[gatevali], myTwoS, U_gate[:,:], rhatvals[:,:,Kvali,NBvali]);                
        #### end loop over NB

        # plotting considerations
        if(NB_indep): indep_vals = 1*NBvals; 
        else: indep_vals = 2*knumbers[Kvali]*NBvals/np.pi;
        if(verbose): print(">>> indep_vals = ",indep_vals);
        for gatevali in range(len(gates)):
            # determine fidelity and kNB*, ie x val where the SWAP happens
            indep_argmax = np.argmax(Fvals_min[Kvali,:,gatevali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, max F = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[Kvali,:,gatevali]));
                print("\nU = "+gates[gatevali]+"\n",indep_comment);

            # formatting
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks:
                axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[gatevali].set_xlim(0,np.max(indep_vals));
            if(NB_indep): axes[-1].set_xlabel('$N_B$',fontsize=myfontsize);
            else: axes[-1].set_xlabel('$2 k_i a N_B/\pi $',fontsize=myfontsize);
            if(Kvali==0): axes[gatevali].annotate("$\mathbf{U}_{"+gates[gatevali]+"}$", (indep_vals[0],1.01),fontsize=myfontsize);
            axes[gatevali].set_ylabel("$F_{avg}(\mathbf{R}, \mathbf{U})$",fontsize=myfontsize);

            # get and store label info
            if(abs(knumbers[Kvali] - np.sqrt(10.0**Kpowers[Kvali])) < 1e-10):
                mylabel = "$k_i^2 a^2 = 10^{"+str(Kpowers[Kvali])+"}$"
            else: 
                mylabel = "$k_i^2 a^2 = {:.6f} $".format(knumbers[Kvali]**2);
                
            # plot fidelity, starred SWAP locations, as a function of NB
            axes[gatevali].plot(indep_vals,Fvals_min[Kvali,:,gatevali], label = mylabel,color=mycolors[Kvali],marker=mymarkers[1+Kvali],markevery=mymarkevery);     

            # save Fvals to .npy
            if(final_plots>1):
                np.save("data/gate/"+case+"/"+gates[gatevali]+"_J{:.2f}_K{:.0f}_y.npy".format(Jval, Kvali),Fvals_min[Kvali,:,gatevali]);
                np.save("data/gate/"+case+"/"+gates[gatevali]+"_J{:.2f}_K{:.0f}_x.npy".format(Jval, Kvali),indep_vals);
        #### end loop over gates
            
    #### end loop over Ki
            
    # show
    fig.suptitle(get_suptitle(myTwoS, Jval, Vq, VBar));
    plt.tight_layout();
    if(final_plots > 1): # save fig and legend
        # title and color values
        np.savetxt("data/gate/"+case+"_ALL_J{:.2f}_K{:.0f}_title.txt".format(Jval,len(Kvals)),knumbers*knumbers,header=get_suptitle(myTwoS, Jval, Vq, VBar));
    else:
        axes[-1].legend();
        plt.show();

elif(case in ["gates_lambda","gates_K","conc_lambda","conc_K"]): # at fixed NB, as a function of Ki,
    if("lambda" in case): K_indep = False;
    else: K_indep = True; # whether to put ki^2 on x axis, alternatively NBa/\lambda

    # axes
    gates = ["SQRT", "SWAP","I", "SeS12"];
    if("conc" in case): gates = ["overlap","conc", "overlap_sf"];
    nrows, ncols = len(gates), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    
    # cases / other options
    if("_lambda" in case): K_indep = False; # whether to plot (ki*a)^2 or NBa/lambda on the x axis
    else: K_indep = True;

    # iter over barrier distance (colors)
    NBvals = np.array([1400]);
    #NBvals = np.array([1000,1400,1800]); assert(Jval==-0.02);
    Fvals_min = np.empty((myxvals, len(NBvals),len(gates)),dtype=float); # avg fidelity
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):
        NBval = NBvals[NBvali];
        if(verbose): print("NB = ",NBval); 

        # iter over incident wavenumber (x axis)
        xmax = 3.0;
        Kvals, Energies, indep_vals = get_indep_vals(True, K_indep, myxvals, xmax, NBval, tl,
            the_xmin = 0.0);
        print(indep_vals);
        # -2t < Energy < 2t, the argument of self energies, Green's funcs, etc
        for Kvali in range(len(Kvals)):

            # construct hblocks from cicc-type spin ham
            hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, Vq, VBar, Vend, NBval, verbose=0);

            # define source, although it doesn't function as a b.c. since we return Rhat matrix
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] =wfm.kernel(hblocks,tnn,tnnn,tl,Energies[Kvali], source,
                                    is_psi_jsigma = False, is_Rhat = True, all_debug = False);

             # iter over gates to get fidelity for each one
            for gatevali in range(len(gates)):
                U_gate, dummy_ticks = get_U_gate(gates[gatevali],myTwoS);
                Fvals_min[Kvali, NBvali, gatevali] = get_Fval(gates[gatevali], myTwoS, U_gate, rhatvals[:,:,Kvali,NBvali], the_espin=0);                
        #### end loop over Ki

        # plotting considerations
        for gatevali in range(len(gates)):

            # determine fidelity and kNB*, ie x val where the SWAP is best
            indep_argmax = np.argmax(Fvals_min[:,NBvali,gatevali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, fidelity(indep_star) = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[:,NBvali,gatevali]));
                print("\nU = "+gates[gatevali]+"\n",indep_comment);

            # plot formatting

            # x axis
            if(K_indep):
                axes[-1].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
                axes[-1].set_xscale('log', subs = []);
            else:
                axes[-1].set_xlabel('$N_B a/\lambda_i$',fontsize=myfontsize);
                axes[-1].set_xlim(0.0, np.max(indep_vals));
            
            # y axis
            #axes[gatevali].set_yticks(the_ticks);
            axes[gatevali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[gatevali].axhline(tick,color='lightgray',linestyle='dashed');
            if(gates[gatevali]=="conc"):
                axes[gatevali].set_ylabel("$C$");
            elif(gates[gatevali]=="overlap"):
                axes[gatevali].set_ylabel("$P_{swap}$");
            elif(gates[gatevali]=="overlap_sf"):
                axes[gatevali].set_ylabel("$P_{sf}$");
            else:
                axes[gatevali].annotate("$\mathbf{U}_{"+gates[gatevali]+"}$", (indep_vals[1],1.01),fontsize=myfontsize);
                axes[gatevali].set_ylabel("$F_{avg}(\mathbf{R},\mathbf{U})$",fontsize=myfontsize);
                
            # plot fidelity, starred SWAP locations
            axes[gatevali].plot(indep_vals, Fvals_min[:,NBvali,gatevali], label = "$N_B = ${:.0f}".format(NBvals[NBvali]),color=mycolors[NBvali]);

            # save Fvals to .npy
            if(final_plots > 1):
                xy_savename = "data/gate/"+case+"/"+gates[gatevali]+"_J{:.2f}_NB{:.0f}".format(Jval, NBval);
                np.save(xy_savename + "_y.npy", Fvals_min[:,NBvali,gatevali]);
                np.save(xy_savename + "_x.npy",indep_vals);
        #### end loop over gates

    #### end loop over NB
            
    # show
    fig.suptitle(get_suptitle(myTwoS, Jval, Vq, VBar), fontsize=myfontsize);
    plt.tight_layout();
    if(final_plots > 1): # save fig and legend
        # title and color values
        np.savetxt("data/gate/"+case+"/ALL_J{:.2f}_NB{:.0f}_title.txt".format(Jval,NBvals[-1]),NBvals,header=get_suptitle(myTwoS, Jval, Vq, VBar));
    else:
        axes[-1].legend(loc="lower right");
        plt.show();
        
# compare different roots of swap at fixed NB, vs NBa/\lambda_i
elif(case in ["roots_lambda", "roots_K"]):

    # different axes are different NB
    NBvals = np.array([100]) #,140,200,500]); # for the J=-0.2 case
    #NBvals = np.array([30, 38, 60, 200]); assert(Jval == -0.4); # for the J=-0.4 case
    nrows, ncols = len(NBvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0], nrows*myfigsize[1]);
    
    # cases / other options
    if(case=="roots_lambda"): K_indep = False; # whether to plot (ki*a)^2 or NBa/lambda on the x axis
    else: K_indep = True;

    # iter over roots
    # roots are functionally the color (replace NBval) and NBs are axes (replace gates)
    # but still order axes as Kvals, NBvals, roots
    roots = np.array(["4","2","1","SeS12"]); 
    if(roots[-1] in ["SeS12", "I"]): mycolors[len(roots)-1] = "black";
    Fvals_min = np.empty((myxvals, len(NBvals),len(roots)),dtype=float); # avg fidelity 
    rhatvals = np.empty((n_loc_dof,n_loc_dof,myxvals,len(NBvals)),dtype=complex); # by  init spin, final spin, energy, NB
    for NBvali in range(len(NBvals)):
        NBval = NBvals[NBvali];
        if(verbose): print("NB = ",NBval); 

        # iter over incident wavenumber (x axis)
        xmax = 1.5;
        Kvals, Energies, indep_vals = get_indep_vals(True, K_indep, myxvals, xmax, NBval, tl);
        # -2t < Energy < 2t, the argument of self energies, Green's funcs, etc
        for Kvali in range(len(Kvals)):

            # construct hblocks from cicc-type spin ham
            hblocks, tnn, tnnn = get_hblocks(myTwoS, tl, Jval, Vq, VBar, Vend, NBval);

            # define source, although it doesn't function as a b.c. since we return Rhat matrix
            source = np.zeros((n_loc_dof,));
            source[-1] = 1;
                    
            # get reflection operator
            rhatvals[:,:,Kvali,NBvali] = wfm.kernel(hblocks, tnn, tnnn, tl, Energies[Kvali], source,
                                         is_psi_jsigma = False, is_Rhat = True, all_debug = False);

             # iter over gates to get fidelity for each one
            for rootvali in range(len(roots)):
                gatestr = "RZ"+roots[rootvali];
                U_gate, dummy_ticks = get_U_gate(gatestr,myTwoS);
                Fvals_min[Kvali, NBvali, rootvali] = get_Fval(gatestr, myTwoS, U_gate, rhatvals[:,:,Kvali,NBvali], the_espin=0); 
                       
        #### end loop over Ki

        # plotting considerations
        for rootvali in range(len(roots)):

            # determine fidelity and kNB*, ie x val where the SWAP is best
            indep_argmax = np.argmax(Fvals_min[:,NBvali,rootvali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, fidelity(indep_star) = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[:,NBvali,rootvali]));
                print("\nU^1/"+roots[rootvali]+"\n",indep_comment);

            # plot formatting

            # x axis
            if(K_indep):
                axes[-1].set_xlabel('$k_i^2 a^2$',fontsize=myfontsize);
                axes[-1].set_xscale('log', subs = []);
            else:
                axes[-1].set_xlabel('$N_B a/\lambda_i$',fontsize=myfontsize);
                axes[-1].set_xlim(0.0, np.max(indep_vals));
              
            # y axis  
            #axes[NBvali].set_yticks(the_ticks);
            axes[NBvali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[NBvali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[NBvali].annotate("$N_B = {"+str(NBvals[NBvali])+"}$", (indep_vals[1],1.01),fontsize=myfontsize);
            axes[NBvali].set_ylabel("$F_{avg}[\mathbf{R}, (\mathbf{U}_{SWAP})^{1/n}]$",fontsize=myfontsize);
                
            # plot fidelity, starred SWAP locations
            axes[NBvali].plot(indep_vals,Fvals_min[:,NBvali,rootvali], label = "$n = "+roots[rootvali]+"$",color=mycolors[rootvali]);

            # save Fvals to .npy
            if(final_plots > 1):
                xy_savename = "data/gate/"+case+"/"+roots[rootvali]+"_J{:.2f}_NB{:.0f}".format(Jval, NBval);
                np.save(xy_savename + "_y.npy", Fvals_min[:,NBvali,rootvali]);
                np.save(xy_savename + "_x.npy",indep_vals);

        #### end loop over roots

    #### end loop over NB
            
    # show
    fig.suptitle(get_suptitle(myTwoS, Jval, Vq, VBar), fontsize=myfontsize);
    plt.tight_layout();
    if(final_plots > 1): # save title, don't show
        np.savetxt("data/gate/"+case+"/ALL_J{:.2f}_NB{:.0f}_title.txt".format(Jval,NBvals[-1]),NBvals, header=get_suptitle(myTwoS, Jval, Vq, VBar));
    else:
        axes[-1].legend(loc="lower right");
        plt.show();

####################################################################################
#### time dependent data      
elif(case in ["direct","directJ"]):
    from scipy.linalg import expm as scipy_expm

    # override existing axes
    plt.close();
    del gates, fig, axes;
    zvals = np.array([0.0]); # different values of \delta J/J_H (J_H not J !!)
    nrows, ncols = len(zvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="directJ"): time_indep=False;
    else: time_indep = True;
    del V0, VB;
    extend = False; # more multiples of 2kiaNB/pi
    if(extend):
        myxvals = 5*myxvals;
        mymarkevery = (myxvals//3, myxvals//3);
    J_on_JH_off = False;

    # physical params;
    suptitle = "$s=${:.1f}, $J =${:.2f}, $J_H =${:.2f}".format(0.5*myTwoS, 0.0, Jval);

    # iter over roots
    # roots are functionally the color (replace NBval) and NBs are axes (replace gates)
    # but still order axes as Kvals, NBvals, roots
    roots = np.array(["4","2","1"]); 
    if(roots[-1] == "SeS12"): mycolors[len(roots)-1] = "black";
    Fvals_min = np.empty((myxvals, len(zvals),len(roots)),dtype=float); # avg fidelity 
    rhatvals = np.zeros((n_loc_dof,n_loc_dof,myxvals,len(zvals)),dtype=complex); 
    
    for axvali in range(len(zvals)): # perturbation strengths (non-identical J1,J2)

        # iter over time (x axis)
        xmax = 4.0*np.pi;
        if(extend): xmax = 10.0*np.pi;
        assert(Jval<0);
        timevals = np.linspace(0,xmax/(-Jval),myxvals,dtype=float);
        xvals = np.linspace(0,xmax,myxvals,dtype=float);
        # fill in rhat 
        if(J_on_JH_off):
            raise NotImplementedError;
            Hexch =(-Jval/4)*np.array([[2, 0, 0, 0, 0, 0, 0, 0], # -J Se.(S1+S2)
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0,-2, 0, 2, 2, 0],
                                  [0, 2, 2, 0,-2, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 2]
                                  ],dtype=complex);
            Hexch +=(Jval*zvals[axvali]/4)*np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 2, 0, 0,-2, 0, 0, 0], # \delta J Se.(S1-S2)
                                       [0, 0,-2, 0, 2, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 2,-2, 0],
                                       [0,-2, 2, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 2, 0,-2, 0, 0],
                                       [0, 0, 0,-2, 0, 0, 2, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0]],
                                       dtype=float);
        else:
            assert(zvals[axvali]==0); # z = \delta J/J_H
            Hexch = (Jval/4)*np.array([[-1, 0, 0, 0, 0, 0, 0, 0], # -J_H S1.S2 + \delta J Se.(S1-S2)
                                       [0,1+2*zvals[axvali],  -2,0,-2*zvals[axvali], 0, 0,0],
                                       [0,-2,1-2*zvals[axvali],0,   2*zvals[axvali], 0, 0,0],
                                       [0,0, 0, -1,        0, 2*zvals[axvali],-2*zvals[axvali],0],
                                       [0,-2*zvals[axvali],2*zvals[axvali],0,-1, 0, 0,0],
                                       [0,0, 0, 2*zvals[axvali], 0, 1-2*zvals[axvali],-2,0],
                                       [0,0, 0, -2*zvals[axvali],     0, -2, 1+2*zvals[axvali],0],
                                       [0, 0, 0, 0, 0, 0, 0, -1]],
                                       dtype=float);
        # end if else statement
        assert( np.all(abs(np.transpose(Hexch) - Hexch) < 1e-10));
        for timevali in range(len(timevals)):
            U_coupled = scipy_expm(complex(0,-timevals[timevali])*Hexch);
            assert(np.all(abs(np.matmul(np.conj(np.transpose(U_coupled)), U_coupled)-np.eye(len(U_coupled))) < 1e-10));
            rhatvals[:,:,timevali,axvali] = 1*U_coupled;
        
        # iter over gates to get fidelity for each one
        for rootvali in range(len(roots)):
            gatestr = "RZ"+roots[rootvali];
            U_gate, dummy_ticks = get_U_gate(gatestr,myTwoS);
            for xvali in range(len(xvals)):
                Fvals_min[xvali, axvali, rootvali] = get_Fval(gatestr, myTwoS, 
                           U_gate[:,:], rhatvals[:,:,xvali,axvali]); 
               
        # plotting considerations
        if(time_indep): indep_vals = timevals;
        else: indep_vals = xvals/np.pi;
        if(verbose):
            print(">>> indep_vals =\n",indep_vals);
        for rootvali in range(len(roots)):
            # determine fidelity and kNB*, ie x val where the SWAP is best
            indep_argmax = np.argmax(Fvals_min[:,axvali,rootvali]);
            indep_star = indep_vals[indep_argmax];
            if(verbose):
                indep_comment = "indep_star, fidelity(Kstar) = {:.8f}, {:.4f}".format(indep_star, np.max(Fvals_min[:,axvali,rootvali]));
                print("\nU^1/"+roots[rootvali]+"\n",indep_comment);

            # plot formatting
            #axes[axvali].set_yticks(the_ticks);
            axes[axvali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[axvali].axhline(tick,color='lightgray',linestyle='dashed');
            if(time_indep):
                axes[-1].set_xlabel('$\\tau$ ($\hbar$/Energy)',fontsize=myfontsize);
            else:
                axes[-1].set_xlabel('$|J_H|\\tau/\pi \hbar$',fontsize=myfontsize);
            axes[-1].set_xlim(np.floor(indep_vals[0]), np.ceil(indep_vals[-1]));
            if(len(axes)>1): axes[axvali].annotate("$\delta J/J_H = {:.2f}$".format(zvals[axvali]), (indep_vals[1],1.01),fontsize=myfontsize);
            axes[axvali].set_ylabel("$F_{avg}[\mathbf{U}(\\tau),(\mathbf{U}_{SWAP})^{1/n}]$",fontsize=myfontsize);
                
            # plot fidelity, starred SWAP locations
            axes[axvali].plot(indep_vals,Fvals_min[:,axvali,rootvali], label = "$n = "+roots[rootvali]+"$",color=mycolors[rootvali],marker=mymarkers[1+rootvali],markevery=mymarkevery);

            # save Fvals to .npy

        #### end loop over roots

    #### end loop over axvals
            
    # show
    fig.suptitle(suptitle, fontsize=myfontsize);
    plt.tight_layout();
    axes[0].legend(loc="lower right");
    plt.show();  
        
elif(case in ["med","medJ"]): # just time evolve initial state
    from scipy.linalg import expm as scipy_expm

    # override existing axes
    plt.close();
    del gates, fig, axes;
    zvals = np.array([0.00]);
    nrows, ncols = 1+len(zvals), 1;
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True);
    if(nrows==1): axes = [axes];
    fig.set_size_inches(ncols*myfigsize[0],nrows*myfigsize[1]);
    if(case=="medJ"): time_indep=False;
    else: time_indep = True;
    extend = True; # more multiples of 2kiaNB/pi
    if(extend):
        myxvals = 5*myxvals;
        mymarkevery = (myxvals//3, myxvals//3);
    
    # define time-dep observables
    which_states = np.array([1,4,2]);
    state_labels = ["\\uparrow_e \\uparrow_1 \\uparrow_2","\\uparrow_e \\uparrow_1 \downarrow_2","\\uparrow_e \downarrow_1 \\uparrow_2","\\uparrow_e \downarrow_1 \downarrow_2",
        "\downarrow_e \\uparrow_1 \\uparrow_2","\downarrow_e \\uparrow_1 \downarrow_2","\downarrow_e \downarrow_1 \\uparrow_2","\downarrow_e \downarrow_1 \downarrow_2"]; 
    observables = np.empty((len(which_states), myxvals),dtype=float);
    psi0 = np.zeros((n_loc_dof,),dtype=complex);
    psi0[which_states[0]] = 1.0; # |up up down>
    deltaJval = 0.0;
    
    # iter over roots
    # roots are functionally the color (replace NBval) and NBs are axes (replace gates)
    # but still order axes as Kvals, NBvals, roots
    roots = np.array(["2", "SeS12", "1"]); 
    if(roots[-1] == "SeS12"): mycolors[len(roots)-1] = "black";
    if(roots[-2] == "SeS12"): mycolors[len(roots)-2] = "black";
    Fvals_min = np.empty((myxvals, len(zvals), len(roots)),dtype=float); # avg fidelity 
    rhatvals = np.zeros((n_loc_dof,n_loc_dof,myxvals,len(zvals)),dtype=complex); 
    
    for axvali in range(len(zvals)): # perturbation strengths (direct S1-S2 exchange)


        # Hamiltonian and psi0
        Hexch =(-Jval/4)*np.array([[2, 0, 0, 0, 0, 0, 0, 0], # -J Se.(S1+S2)
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0, 0, 2, 0, 0, 0],
                                  [0, 0, 0,-2, 0, 2, 2, 0],
                                  [0, 2, 2, 0,-2, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 2]
                                  ],dtype=complex);
        Hexch+=(-Jval*zvals[axvali]/4)*np.array(
                                 [[1, 0, 0, 0, 0, 0, 0, 0], # -JH S1.S2
                                  [0,-1, 2, 0, 0, 0, 0, 0],
                                  [0, 2,-1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0,-1, 2, 0],
                                  [0, 0, 0, 0, 0, 2,-1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1]
                                  ],dtype=complex);
        assert( np.all(abs(np.transpose(Hexch) - Hexch) < 1e-10));
        assert(deltaJval == 0.0);
                                              
        # time evolution
        xmax = 4.0*np.pi;
        if(extend): xmax = 10.0*np.pi;
        assert(Jval<0);
        timevals = np.linspace(0,xmax/abs(Jval),myxvals,dtype=float);
        xvals = np.linspace(0,xmax,myxvals,dtype=float);
    
        # iter over time
        for timevali in range(len(timevals)):
    
            # U(time)
            U_coupled = scipy_expm(complex(0,-timevals[timevali])*Hexch);
            assert(np.all(abs(np.matmul(np.conj(np.transpose(U_coupled)), U_coupled)-np.eye(len(U_coupled))) < 1e-10));
            rhatvals[:,:,timevali,axvali] = 1*U_coupled;

            # time dependent state
            psit = np.matmul(U_coupled, psi0);
            # get observables
            for whichi in range(len(which_states)):
                statei = which_states[whichi];
                bra = np.zeros_like(psi0);
                bra[statei] = 1.0;
                overlap = np.dot(np.conj(bra), psit);
                if( abs(np.imag(np.conj(overlap)*overlap)) > 1e-10): assert False;
                observables[whichi,timevali] = np.real(np.conj(overlap)*overlap);
            
        # iter over gates to get fidelity for each one
        for rootvali in range(len(roots)):
            gatestr = "RZ"+roots[rootvali];
            U_gate, dummy_ticks = get_U_gate(gatestr,myTwoS);
            for timevali in range(len(timevals)):
                Fvals_min[timevali, axvali, rootvali] = get_Fval(gatestr, myTwoS, 
                           U_gate[:,:], rhatvals[:,:,timevali,axvali]);  
            
        # plotting time evol
        if(time_indep): indep_vals = timevals;
        else: indep_vals = xvals/np.pi;
        if(axvali==0):
            for whichi in range(len(which_states)):
                axes[axvali].plot(indep_vals, observables[whichi], label = "$|\langle\psi(\\tau)|"+state_labels[which_states[whichi]]+"\\rangle|^2$", color=mycolors[whichi], marker = mymarkers[1+whichi], markevery = mymarkevery);
            #axes[axvali].plot(indep_vals, np.sum(observables, axis=0), color="red");
            axes[axvali].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
            for tick in the_ticks: axes[axvali].axhline(tick,color='lightgray',linestyle='dashed');
            axes[axvali].legend(loc="lower right");
            axes[axvali].annotate("$J_H/J = {:.2f}$".format(zvals[axvali]), (indep_vals[1],1.01),fontsize=myfontsize);
                
        # plot fidelity, starred SWAP locations
        for rootvali in range(len(roots)):
            axes[1+axvali].plot(indep_vals,Fvals_min[:,axvali,rootvali], label = "$n = "+roots[rootvali]+"$",color=mycolors[rootvali],marker=mymarkers[1+rootvali],markevery=mymarkevery);
            # save Fvals to .npy

        #### end loop over roots
        
        # format
        #axes[axvali+1].set_yticks(the_ticks);
        axes[axvali+1].set_ylim(-0.1+the_ticks[0],0.1+the_ticks[-1]);
        for tick in the_ticks: axes[axvali+1].axhline(tick,color='lightgray',linestyle='dashed');
        axes[axvali+1].legend(loc="lower right");
        axes[axvali+1].annotate("$J_H/J = {:.2f}$".format(zvals[axvali]), (indep_vals[1],1.01),fontsize=myfontsize);
        axes[axvali+1].set_ylabel("$F_{avg}[\mathbf{U}(\\tau),(\mathbf{U}_{SWAP})^{1/n}]$",fontsize=myfontsize);
    #### end loop over axvals    
    
    # show
    if(time_indep):
        axes[-1].set_xlabel('$\\tau$ ($\hbar$/Energy)',fontsize=myfontsize);
    else:
        axes[-1].set_xlabel('$|J|\\tau/\pi \hbar$',fontsize=myfontsize);
    axes[-1].set_xlim(np.floor(indep_vals[0]), np.ceil(indep_vals[-1]));
    suptitle = "$s=${:.1f}, $J=${:.2f}, $\delta J =${:.2f}".format(0.5*myTwoS, Jval, deltaJval);
    fig.suptitle(suptitle, fontsize=myfontsize);
    plt.tight_layout();
    plt.show();
    
        
        
