'''
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

import sys
import json

########################################################################
#### functions

def get_fname(the_path, the_obs, the_xy, the_time):
    '''
    '''
    return the_path+"_arrays/"+the_obs+the_xy+"js_time{:.2f}.npy".format(the_time)

def get_ylabel(the_obs, the_factor, dstring="d", ddt=False):
    '''
    '''
    if(isinstance(dstring, int)): 
        dstringp1 = "{:.0f}".format(dstring+1);
        dstring = "{:.0f}".format(dstring);
    else:
        dstringp1 = dstring+" + 1";
        
    if(the_obs=="Sdz_"): ret = "{:.0f}".format(the_factor)+"$\langle S_"+dstring+"^z \\rangle /\hbar$";
    elif(the_obs=="occ_"): ret = "${:.0f}\langle n_j \\rangle$".format(the_factor);
    elif(the_obs=="sz_"): ret = "${:.0f}\langle s_j^z \\rangle /\hbar$".format(the_factor);
    elif(the_obs=="pur_"): ret = "{:.0f}".format(the_factor)+"$|\mathbf{S}_"+dstring+"|$";
    elif(the_obs== "conc_"): ret = "{:.0f}".format(the_factor)+"$ C_{"+dstring+", "+dstring+"+1}$";
    elif(the_obs=="S2_"): ret = "{:.1f}".format(the_factor)+"$\langle (\mathbf{S}_"+dstring+" + \mathbf{S}_{"+dstringp1+"})^2 \\rangle/\hbar^2$";
    elif(the_obs=="MI_"): ret = "$\\frac{1}{\ln(2)}MI["+dstring+", "+dstringp1+"]$";
    else: print(the_obs); raise NotImplementedError;
    
    # time derivatives
    if(ddt): ret = "$|\\frac{d}{dt}$"+ret+"$|$";
    
    # return
    print(the_obs,"-->",ret);
    return ret;

########################################################################
#### run code

# top level
case = int(sys.argv[1]);
update0 = int(sys.argv[2]);  # time to start at, in units of update interval
datafiles = sys.argv[3:];

# plotting
obs1, factor1, color1, mark1, = "Sdz_", 2,"darkred", "s";
ticks1, linewidth1, fontsize1 =  (-1.0,-0.5,0.0,0.5,1.0), 3.0, 16;
obs2, factor2, color2, mark2 = "occ_", 1, "cornflowerblue", "o";
obs3, factor3, color3, mark3 = "sz_", 2, "darkblue", "o";
obs4, factor4, color4 = "MI_", 1/np.log(2), "black";
num_xticks = 4;
datamarkers = ["s","^","d","*"];

if(case in [0]): # standard charge density vs site snapshot
    from transport import tddmrg
    from transport.tddmrg import plot
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));
    print("\nUpdate time = {:.2f}".format(params["tupdate"]));
    tddmrg.plot.snapshot_fromdata(datafile, update0*params["tupdate"], "STT")

if(case in [1,2]): # observables vs time
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));
    if(case in [2]): plot_S2 = True;
    else: plot_S2 = False; 

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    ax.set_yticks(ticks1);
    ax.set_xlabel("Time $(\hbar/t_l)$", fontsize = fontsize1);
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);

    # time evolution params
    Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
    times = np.zeros((Nupdates+1,),dtype=float);
    print("\nUpdate time = {:.2f}".format(params["tupdate"]));
    for ti in range(len(times)):
        times[ti] = (update0 + ti)*tupdate;

    # impurity spin vs time
    Nsites = params["NL"]+params["NFM"]+params["NR"]; # number of j sites
    which_imp = 0;
    yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
    for ti in range(len(times)):
        yds_vs_time[ti] = np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
    ax.plot(times,factor1*yds_vs_time[:,which_imp],color=color1);
    ax.set_ylabel(get_ylabel(obs1, factor1, dstring=which_imp), color=color1, fontsize=fontsize1);

    # AVERAGE electron spin vs time
    Ne = params["Ne"]; # number deloc electrons
    factor3 = factor3/Ne; # sum normalization
    label3 = "$\\frac{1}{N_e} \sum_j  2\langle s_j^z \\rangle /\hbar $";
    print(obs3,"-->",label3);
    yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
    for ti in range(len(times)):
        yjs_vs_time[ti] = np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
    yjsum_vs_time = np.sum(yjs_vs_time, axis=1);
    ax.plot(times, factor3*yjsum_vs_time,color=color3);
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.15));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([]);
    ax3.set_ylabel(label3, color=color3, fontsize=fontsize1);

    if(plot_S2): # plot S^2
        obs4, factor4, color4 = "S2_", 0.5, "black";
        label4 = get_ylabel(obs4, factor4, dstring=which_imp);
        S2_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            S2_vs_time[ti] = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.plot(times, factor4*S2_vs_time[:,which_imp],color=color4);
    else: # plot mutual info
        label4 = get_ylabel(obs4, factor4, dstring=which_imp);
        S2_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            S2_vs_time[ti] = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.plot(times, factor4*S2_vs_time[:,which_imp],color=color4);
    ax4 = ax.twinx();
    ax4.yaxis.set_label_position("right");
    ax4.spines.right.set_position(("axes", 1.0));
    ax4.spines.right.set(alpha=1.0);
    ax4.set_yticks([])
    ax4.set_ylabel(label4, color=color4, fontsize=fontsize1);

    # show
    plt.tight_layout();
    plt.show();

if(case in [3,4]): # observables vs time, for two data sets side by side

    # axes
    fig, ax = plt.subplots();
    params = json.load(open(datafiles[0]+".txt"));
    if(case in [4]): plot_S2 = True;
    else: plot_S2 = False;     

    #### iter over triplet/singlet
    for dfile in datafiles:
        if("singlet" in dfile): mylinestyle = "dashed"; ticks1 = (0.0,0.5,1.0);
        elif("triplet" in dfile): mylinestyle = "solid"; ticks1 = (0.0,0.5,1.0);
        elif("nosd" in dfile): mylinestyle = "dotted"; ticks1 = (0.0,0.5,1.0);
        else: mylinestyle = "dashdot"; ticks1 = (-1.0,-0.5,0.0,0.5,1.0);
        print("\n>>>",mylinestyle,"=",dfile);
        
        # time evolution params
        params = json.load(open(dfile+".txt"));
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("\nUpdate time = {:.2f}".format(params["tupdate"]));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;

        # which imps to get data for
        which_imp = 0;
        assert(which_imp == 0);
        assert(params["NFM"] == 2); # number of d sites
        Nsites = params["NL"]+params["NFM"]+params["NR"]; # number of j sites

        # COMBINED impurity z spin vs time
        obs1, factor1, color1 = "Sdz_", 1, "darkred";
        label1 = "$ \langle S_1^z + S_2^z \\rangle /\hbar$";
        print(obs1,"-->",label1);
        yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            yds_vs_time[ti] = np.load(dfile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
        yds_summed = np.sum(yds_vs_time, axis=1);
        ax.plot(times,factor1*yds_summed,color=color1, linestyle=mylinestyle);
        ax.set_ylabel(label1, color=color1, fontsize=fontsize1);

        # COMBINED electron spin vs time
        Ne = params["Ne"]; # number deloc electrons
        factor3 = 2/Ne; # sum normalization
        label3 = "$\\frac{1}{N_e} \sum_j  2\langle s_j^z \\rangle /\hbar $";
        print(obs3,"-->",label3);
        yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
        yjs_summed = np.sum(yjs_vs_time, axis=1);
        ax.plot(times, factor3*yjs_summed,color=color3, linestyle=mylinestyle);
    
        # (S1 + S2)^2
        if(plot_S2):
            obs4, factor4, color4 = "S2_", 0.5, "black";
            label4 = get_ylabel(obs4, factor4, dstring=which_imp);
            S2_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
            for ti in range(len(times)):
                S2_vs_time[ti] = np.load(dfile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
            Line2D_obs4 = ax.plot(times, factor4*S2_vs_time[:,which_imp],color=color4, linestyle=mylinestyle);
        else: # plot mutual info
            label4 = get_ylabel(obs4, factor4, dstring=which_imp);
            S2_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
            for ti in range(len(times)):
                S2_vs_time[ti] = np.load(dfile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
            Line2D_obs4 = ax.plot(times, factor4*S2_vs_time[:,which_imp],color=color4, linestyle=mylinestyle);
        Line2D_data = Line2D_obs4[0].get_xydata();
        for dati in range(len(times)):
            print("  TIME = {:.0f}, MI = {:.6f}".format(Line2D_data[dati,0], Line2D_data[dati,1]));
    # formatting
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.2));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    # label later -> on right side
    ax4 = ax.twinx();
    ax4.yaxis.set_label_position("right");
    ax4.spines.right.set_position(("axes", 1.0));
    ax4.spines.right.set(alpha=1.0);
    ax4.set_yticks([])
    ax3.set_ylabel(label4, color=color4, fontsize=fontsize1); # labels (S1+S2)^2 on left
    ax4.set_ylabel(label3, color=color3, fontsize=fontsize1); # labels s_j^z on right
    ax.set_xlabel("Time $(\hbar/t_l)$", fontsize = fontsize1);
    ax.set_title( open(datafiles[-1]+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
   
    # show
    plt.tight_layout();
    plt.show();
    
if(case in [5,6,7,8,9]): # observables RATES OF CHANGE vs time, for two data sets side by side

    def do_gradient(yarr, xarr, do=True):
        if(do): # actually take gradient
            return abs(np.gradient(yarr, xarr));
        else:
            return yarr;
        
    # axes
    fig, ax = plt.subplots();
    params = json.load(open(datafiles[0]+".txt"));
    if(case in [6,8,9]): plot_occ = True; # chooses observables to be plotted as occupancies
    else: plot_occ = False; 
    norm_to_Jflux = True;
    if(case in [5,6]): norm_to_Jflux=False;
    take_gradient = 1*norm_to_Jflux; 
    if(case in [9]): 
        use_Jobs = True; # particle current (<J>) replaces d/dt of occupancy 
        assert("nosd" in datafiles[0]); # normalize to nosd case ONLY 
    else: use_Jobs = False; 

    #### iter over triplet/singlet
    for dfile in datafiles:
        if("singlet" in dfile): mylinestyle = "dashed";
        elif("triplet" in dfile): mylinestyle = "solid";
        elif("nosd" in dfile): mylinestyle = "dotted";
        else: mylinestyle = "dashdot";
        print("\n>>>",mylinestyle,"=",dfile);
        
        # time evolution params
        params = json.load(open(dfile+".txt"));
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("\nUpdate time = {:.2f}".format(params["tupdate"]));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
        time_window_limits = (10,30);
        time_window_mask = np.logical_and(np.array(times>time_window_limits[0]), np.array(times<time_window_limits[1]));

        # which imps to get data for
        which_imp = 0;
        assert(which_imp == 0);
        assert(params["NFM"] == 2); # number of d sites
        Nconf = params["Nconf"];
        Nsites = params["NL"]+params["NFM"]+params["NR"]; # number of j sites

        if(not plot_occ): # plot spin observables: S1^z + S2^z, MI[1,2]
        
            # normalize by max incoming electron flux
            # Jflux = particle current through site Nconf 
            #if(norm_to_Jflux): raise notImplementedError #Jflux = max(abs(yjs_vs_time[:,0]));
            #else: Jflux = 1.0; # no normalization
            
            # incoming electron flux = dn_conf / dt
            if(take_gradient): label2 = "$ \left| \\frac{d}{dt} n_{conf} \\right|$";
            else: label2 = "$ \left|n_{conf} \\right|$";
            if(norm_to_Jflux): label2 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
            print(obs2,"-->",label2);
            yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
            for ti in range(len(times)):
                yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
        
            # normalize by max incoming electron flux
            yjC_vs_time = np.sum(yjs_vs_time[:,:Nconf], axis=1);
            # Jflux = particle current through site Nconf, here expressed as dn/dt
            if(norm_to_Jflux): Jflux = max(abs(np.gradient(factor2*yjC_vs_time, times)));
            else: Jflux = 1.0; # no normalization
            ax.plot(times, abs(do_gradient(yjC_vs_time,times,do=take_gradient))/Jflux, 
                    color=color2, linestyle=mylinestyle);
     
            # COMBINED impurity z spin vs time, time scale normalized by Jflux
            obs1, factor1, color1 = "Sdz_", 1, "darkred";
            if(take_gradient): label1 = "$ \left|\\frac{d}{dt} \langle S_1^z + S_2^z \\rangle \\right|$";
            else: label1 = "$\langle S_1^z + S_2^z \\rangle /\hbar$";
            if(norm_to_Jflux): label1 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
            print(obs1,"-->",label1);
            yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
            for ti in range(len(times)):
                yds_vs_time[ti] = np.load(dfile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
            yds_summed = np.sum(yds_vs_time, axis=1);
            ax.plot(times, do_gradient(factor1*yds_summed,times,do=take_gradient)/Jflux,
                    color=color1, linestyle=mylinestyle);
    
            # mutual info between the two impurities, time scale normalized by Jflux   
            label4 = get_ylabel(obs4, factor4, dstring=which_imp, ddt=take_gradient);
            if(norm_to_Jflux): label4 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
            MI_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
            for ti in range(len(times)):
                MI_vs_time[ti] = np.load(dfile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
            ax.plot(times, do_gradient(factor4*MI_vs_time[:,which_imp],times,do=take_gradient)/Jflux, 
                    color=color4, linestyle=mylinestyle);
            
        else: # plot rate of change of occupancy of different regions
        
            if(use_Jobs): # occ rate of change expressed with particle current (<J>)
                assert(take_gradient);
                
                obs2, label2 = "J_", "$J_{flux}$";
                if(norm_to_Jflux): label2 += "$/$max$\left( J_{flux}^{nosd} \\right)$";
                print(obs2,"-->",label2);
                yjs_vs_time = np.zeros((len(times),3),dtype=float);
                for ti in range(len(times)):
                    yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));

                # normalize by max incoming electron flux
                # Jflux = particle current through site Nconf
                if("nosd" in dfile and dfile==datafiles[0]): # <--- Jflux only gets determined from nosd
                    if(norm_to_Jflux): Jflux = max(abs(yjs_vs_time[:,0]));
                    else: Jflux = 1.0; # no normalization
                    print(">>> setting Jflux = {:.6f}".format(Jflux));
                print("Jflux = {:.6f}".format(Jflux));
             
                # plot particle currents -> NO FACTORS
                ax.plot(times, yjs_vs_time[:,0]/Jflux, 
                        color=color2,linestyle=mylinestyle); # current through site Nconf
                ax.plot(times, yjs_vs_time[:,1]/Jflux, 
                        color=color1,linestyle=mylinestyle); # current through site NL
                ax.plot(times, yjs_vs_time[:,2]/Jflux, 
                        color=color4,linestyle=mylinestyle); # current through site NR
                # current through NR, plateau-averaged
                plateau = np.mean((yjs_vs_time[:,2]/Jflux)[time_window_mask]);
                print("plateau = {:.6f}".format(plateau))

                # labels
                label1 = "$J_{L}(t)$";
                label4 = "$J_{R}(t)$";
                if(norm_to_Jflux):
                    label1 += "$/$max$\left( J_{flux}^{nosd} \\right)$";
                    label4 += "$/$max$\left( J_{flux}^{nosd} \\right)$";
                    
            else: # occ rate of change expressed with dn/dt
            
                # incoming electron flux = dn_conf / dt
                if(take_gradient): label2 = "$ \left| \\frac{d}{dt} n_{conf} \\right|$";
                else: label2 = "$ \left|n_{conf} \\right|$";
                if(norm_to_Jflux): label2 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
                print(obs2,"-->",label2);
                yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
                for ti in range(len(times)):
                    yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
        
                # normalize by max incoming electron flux
                yjC_vs_time = np.sum(yjs_vs_time[:,:Nconf], axis=1);
                # Jflux = particle current through site Nconf, here expressed as dn/dt
                if(norm_to_Jflux): Jflux = max(abs(np.gradient(factor2*yjC_vs_time, times)));
                else: Jflux = 1.0; # no normalization
                ax.plot(times, abs(do_gradient(yjC_vs_time,times,do=take_gradient))/Jflux, 
                        color=color2, linestyle=mylinestyle);
             
                # occ vs time
                NL, Nconf, NFM, NR = params["NL"], params["Nconf"], params["NFM"], params["NR"];
                Nsites = NL+NFM+NR; 
                yjL_vs_time = np.sum(yjs_vs_time[:,:NL], axis=1);
                yjR_vs_time = np.sum(yjs_vs_time[:,NL+NFM:], axis=1);
             
                # plot occupancies -> NO FACTORS
                ax.plot(times, do_gradient(yjL_vs_time,times,do=take_gradient)/Jflux, 
                        color=color1,linestyle=mylinestyle);
                ax.plot(times, do_gradient(yjR_vs_time,times,do=take_gradient)/Jflux, 
                        color=color4,linestyle=mylinestyle);
            
                # labels
                label1 = "$n_{L}(t)$";
                label4 = "$n_{R}(t)$";
                if(take_gradient):
                    label1 = "$\left|\\frac{d}{dt}n_{L}(t)\\right|$";
                    label4 = "$\left|\\frac{d}{dt}n_{R}(t)\\right|$";
                if(norm_to_Jflux):
                    label1 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
                    label4 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
        
    # formatting
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.2));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    # label later -> on right side
    ax4 = ax.twinx();
    ax4.yaxis.set_label_position("right");
    ax4.spines.right.set_position(("axes", 1.0));
    ax4.spines.right.set(alpha=1.0);
    ax4.set_yticks([]);
    if(plot_occ and norm_to_Jflux): ticks1 = [0.0, 1.0];
    else: ticks1 = [];
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    
    # labels
    ax.set_ylabel( label1, color=color1, fontsize=fontsize1); # observable rate of change on left
    ax3.set_ylabel(label4, color=color4, fontsize=fontsize1); # observable rate of change on left
    ax4.set_ylabel(label2, color=color2, fontsize=fontsize1); # labels dn/dt normalizing quantity on right
    ax.set_xlabel("Time $(\hbar/t_l)$", fontsize = fontsize1);
    ax.set_title( open(datafiles[-1]+"_arrays/occ_title.txt","r").read().splitlines()[0][1:]);

    # show
    plt.tight_layout();
    plt.show();

if(case in [10,11]): # animate time evol
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));
    if(case in [11]): plot_S2 = True;
    else: plot_S2 = False; 
    
    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    ax.set_yticks(ticks1);
    ax.set_xlabel("$j(d)$", fontsize=fontsize1);
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);
    
    # time evolution params
    Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
    print("\nUpdate time = {:.2f}".format(params["tupdate"]));
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = (update0 + ti)*tupdate;

    # set up impurity spin animation
    xds = np.load(datafile+"_arrays/"+obs1+"xjs_time{:.2f}.npy".format(update0*tupdate));
    yds = np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(update0*tupdate));
    impurity_sz, = ax.plot(xds, factor1*yds, marker=mark1, color=color1, markersize=linewidth1**2);
    ax.set_ylabel(get_ylabel(obs1, factor1), color=color1, fontsize=fontsize1);
    time_annotation = ax.annotate("Time = {:.2f}".format(update0*tupdate), (0.0,-0.96),fontsize=fontsize1);

    # set up charge density animation
    xjs = np.load(datafile+"_arrays/"+obs2+"xjs_time{:.2f}.npy".format(update0*tupdate));
    yjs = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(update0*tupdate));
    charge_density = ax.fill_between(xjs, factor2*yjs, color=color2);
    ax2 = ax.twinx();
    ax2.set_yticks([]);
    ax2.set_ylabel(get_ylabel(obs2, factor2), color=color2, fontsize=fontsize1);

    # set up spin density animation
    xjs_3 = np.load(datafile+"_arrays/"+obs3+"xjs_time{:.2f}.npy".format(update0*tupdate));
    yjs_3 = np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(update0*tupdate));
    spin_density, = ax.plot(xjs_3, factor3*yjs_3, marker=mark3, color=color3);
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.15));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    ax3.set_ylabel(get_ylabel(obs3, factor3), color=color3, fontsize=fontsize1);
    ax.set_ylim([0.0,0.25]);

    # pairwise observable
    if(params["NFM"]>1):
        if(plot_S2): # plot (S1+S2)^2 /2
            obs4, factor4, color4, mark4 = "S2_", 0.5, "black", "^";
            xds_4 = np.load(datafile+"_arrays/"+obs4+"xjs_time{:.2f}.npy".format(update0*tupdate));
            yds_4 = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(update0*tupdate));
        else: # plot mutual information
            obs4, factor4, color4, mark4 = "MI_", 1/np.log(2), "black", "^";
            xds_4 = np.load(datafile+"_arrays/"+obs4+"xjs_time{:.2f}.npy".format(update0*tupdate));
            yds_4 = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(update0*tupdate));
        S2, = ax.plot(xds_4, factor4*yds_4,marker=mark4,color=color4);
        ax4 = ax.twinx();
        ax4.yaxis.set_label_position("right");
        ax4.spines.right.set_position(("axes", 1.05));
        ax4.spines.right.set(alpha=0.0);
        ax4.set_yticks([])
        ax4.set_ylabel(get_ylabel(obs4, factor4), color=color4, fontsize=fontsize1);

    # time evolve observables
    plt.tight_layout();
    def time_evolution(time):
        # impurity spin
        yds_t = np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(time));
        impurity_sz.set_ydata(factor1*yds_t);
        time_annotation.set_text("Time = {:.2f}".format(time));
        # charge density
        yjs_t = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(time));
        ax.collections.clear();
        charge_density_update = ax.fill_between(xjs, factor2*yjs_t, color=color2)
        charge_density.update_from(charge_density_update);
        # spin density
        yjs_3_t = np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(time));
        spin_density.set_ydata(factor3*yjs_3_t);
        # (S1+S2)^2 / 2
        yds_4_t = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(time));
        S2.set_ydata(factor4*yds_4_t);
        ax.set_ylim([0.0,0.25]);

    # animate
    if(Nupdates > 0): interval = 1000*(10/Nupdates); # so total animation time is 10 sec
    elif(params["time_step"]==1.0): interval = 400;
    elif(params["time_step"]==0.5): interval = 200;
    else: interval = 500;
    ani = animation.FuncAnimation(fig, time_evolution,
                                  frames = times, interval=interval,
                                  repeat=True, blit=False);

    plt.show()
    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)


