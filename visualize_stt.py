'''
'''

import visualize_rm
from transport import wfm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

import sys
import os
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
    elif(the_obs== "J_"): ret = "max$(J_{"+dstring+"})$";
    elif(the_obs=="S2_"): ret = "{:.1f}".format(the_factor)+"$\langle (\mathbf{S}_"+dstring+" + \mathbf{S}_{"+dstringp1+"})^2 \\rangle/\hbar^2$";
    elif(the_obs=="MI_"): ret = "$\\frac{1}{\ln(2)}MI["+dstring+", "+dstringp1+"]$";
    else: print(the_obs); raise NotImplementedError;
    
    # time derivatives
    if(ddt): ret = "$|\\frac{d}{dt}$"+ret+"$|$";
    
    # return
    print(the_obs,"-->",ret);
    return ret;
    
def get_title(f, to_exclude=[]):
    '''
    '''
    
    in_title = open(f+"_arrays/occ_title.txt","r").read().splitlines()[0][1:];
    in_title = in_title.split(",");
    title_mask = np.ones(np.shape(in_title),dtype=int);
    for i in range(len(in_title)):
        for ex in to_exclude:
            if(ex in in_title[i]): title_mask[i] = 0;
    out_title = [];  
    for i in range(len(in_title)):
        if(title_mask[i]): out_title.append(in_title[i]); 
    out_title = ",".join(out_title);
    return out_title;
    
########################################################################
#### run code

if(__name__=="__main__"):

    # top level
    case = int(sys.argv[1]);
    update0 = int(sys.argv[2]);  # time to start at, in units of update interval
    datafiles = sys.argv[3:];

    # plotting
    myfontsize=14;
    obs1, factor1, color1, mark1, = "Sdz_", 2,"darkred", "s";
    ticks1, linewidth1, fontsize1 =  (-1.0,-0.5,0.0,0.5,1.0), 3.0, 16;
    obs2, factor2, color2, mark2 = "occ_", 1, "cornflowerblue", "o";
    obs3, factor3, color3, mark3 = "sz_", 2, "darkblue", "o";
    obs4, factor4, color4 = "MI_", 1/np.log(2), "black";
    num_xticks = 4;
    datamarkers = ["o","^","s","*"];
    plt.rcParams.update({"font.family": "serif"});
    plt.rcParams.update({"text.usetex": True});
    UniversalFigRatios = [4.5/1.25,5.5/1.25/1.25];
    from transport.wfm import UniversalColors, UniversalAccents, ColorsMarkers, AccentsMarkers, UniversalMarkevery, UniversalPanels;

if(case in [0]): # standard charge density vs site snapshot
    from transport import tddmrg
    from transport.tddmrg import plot
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));
    print("\nUpdate time = {:.2f}".format(params["tupdate"]));
    tddmrg.plot.snapshot_fromdata(datafile, update0*params["tupdate"], "STT")

if(case in [1,2]): # observables vs time
    datafile = datafiles[0];
    if(case in [2]): plot_S2 = True;
    else: plot_S2 = False; 

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    ax.set_yticks(ticks1);
    ax.set_xlabel("Time $(\hbar/t_l)$", fontsize = fontsize1);
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);

    # time evolution params
    params = json.load(open(datafile+".txt"));
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
    '''
    5 -- plot MI, S_1^z + S_2^z vs time
    6 -- plot nL, nR, nconf vs time
    8 -- plot time derivatives of nL, nR, nconf vs time. Normalize w/r/t incoming pcurrent in Jsd=0 case
    '''

    def do_gradient(yarr, xarr, do=True):
        if(do): # actually take gradient
            return abs(np.gradient(yarr, xarr));
        else:
            return yarr;
        
    # axes
    fig, ax = plt.subplots();

    # chooses observables to be plotted
    # True means we plot occupancies, False means we describe MSQ quantum state (MI, Sz, etc)
    if(case in [6,8,9]): plot_occ = True; 
    else: plot_occ = False;     
    
    # choose whether or not to normalize plotted quantities
    norm_to_Jconf = True; # as written, the code always normalizes for d/dt and J quantities
    if(case in [5,6]): norm_to_Jconf=False;
    take_gradient = 1*norm_to_Jconf; # whether to plot time derivatives of quantities
    if(case in [9]): 
        norm_to_Jconf = False;
        use_Jobs = True; # particle current (<J>) replaces d/dt of occupancy 
    else: use_Jobs = False; 
    assert(norm_to_Jconf == False);

    #### iter over triplet/singlet
    for dfile in datafiles:
        if("singlet" in dfile): mylinestyle = "dashed";
        elif("triplet" in dfile): mylinestyle = "solid";
        elif("nosd" in dfile): mylinestyle = "dotted" #"None" #to turn off
        else: mylinestyle = "dashdot";
        print("\n>>>",mylinestyle,"=",dfile);
        
        # time evolution params
        params = json.load(open(dfile+".txt"));
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("    Update time = {:.2f}, Stop time = {:.2f}, Nupdates = {:.0f}".format(tupdate,tupdate*Nupdates, Nupdates));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
        time_window_limits = (times[-1]//4, 2*times[-1]//4);
        time_window_mask = np.logical_and(np.array(times>time_window_limits[0]), np.array(times<time_window_limits[1]));
        # ^^ time window mask is used to extract plateau values!!

        # which imps to get data for
        which_imp = 0;
        assert(which_imp == 0);
        if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): 
            block2site = 2;
        else: 
            block2site = 1;
            assert(params["NFM"] == 2); # number of d sites
        Nconf = params["Nconf"];
        Nsites = params["NL"]+params["NFM"]+params["NR"]; # number of j sites

        if(not plot_occ): # plot spin observables: S1^z + S2^z, MI[1,2]
        
            # normalize by max incoming electron pcurrent
            # Jconf = particle current through site Nconf 
            #if(norm_to_Jconf): raise notImplementedError #Jconf = max(abs(yjs_vs_time[:,0]));
            #else: Jconf = 1.0; # no normalization
            
            # incoming electron pcurrent = dn_conf / dt
            if(take_gradient): label2 = "$ \left| \\frac{d}{dt} n_{conf} \\right|$";
            else: label2 = "$ n_{conf}$";
            if(norm_to_Jconf): label2 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
            print(obs2,"-->",label2);
            yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
            for ti in range(len(times)):
                yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
        
            # normalize by max incoming electron pcurrent
            yjC_vs_time = np.sum(yjs_vs_time[:,:Nconf], axis=1);
            # Jconf = particle current through site Nconf, here expressed as dn/dt
            if(norm_to_Jconf): Jconf = max(abs(np.gradient(factor2*yjC_vs_time, times)));
            else: Jconf = 1.0; # no normalization
            ax.plot(times, abs(do_gradient(yjC_vs_time,times,do=take_gradient))/Jconf, 
                    color=color2, linestyle=mylinestyle);
     
            # COMBINED impurity z spin vs time, time scale normalized by Jconf
            obs1, factor1, color1 = "Sdz_", 1, "darkred";
            if(take_gradient): label1 = "$ \left|\\frac{d}{dt} \langle S_1^z + S_2^z \\rangle \\right|$";
            else: label1 = "$\langle S_1^z + S_2^z \\rangle /\hbar$";
            if(norm_to_Jconf): label1 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
            print(obs1,"-->",label1);
            yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
            for ti in range(len(times)):
                yds_vs_time[ti] = np.load(dfile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
            yds_summed = np.sum(yds_vs_time, axis=1);
            ax.plot(times, do_gradient(factor1*yds_summed,times,do=take_gradient)/Jconf,
                    color=color1, linestyle=mylinestyle);
    
            # mutual info between the two impurities, time scale normalized by Jconf   
            label4 = get_ylabel(obs4, factor4, dstring=which_imp, ddt=take_gradient);
            if(norm_to_Jconf): label4 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
            MI_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
            for ti in range(len(times)):
                MI_vs_time[ti] = np.load(dfile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
            ax.plot(times, do_gradient(factor4*MI_vs_time[:,which_imp],times,do=take_gradient)/Jconf, 
                    color=color4, linestyle=mylinestyle);
            
        else: # plot rate of change of occupancy of different regions
        
            if(use_Jobs): # occ rate of change expressed with particle current (<J>)
                assert(take_gradient);
                
                obs2, label2 = "J_", "$J_{conf}$";
                normalizer_label = " (normalized)"
                if(norm_to_Jconf): label2 += normalizer_label;
                print(obs2,"-->",label2);
                yjs_vs_time = np.zeros((len(times),3*block2site),dtype=float);
                xjs_vs_time = np.zeros((len(times),3*block2site),dtype=float);
                for ti in range(len(times)):
                    yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
                    xjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"xjs_time{:.2f}.npy".format(times[ti]));
                print(obs2+" site indices = ",xjs_vs_time[0].astype(int));

                # normalize by max incoming electron pcurrent
                # Jconf = particle current through site Nconf
                if(norm_to_Jconf and dfile==datafiles[0]): # <--- Jconf must come from Jsd=0 case
                    # set Jconf
                    Jconf = max(abs(yjs_vs_time[:,0]));
                    print(">>> setting Jconf = {:.6f}".format(Jconf));
                    assert("nosd" in dfile); # must always normalize to nosd case 
                else: Jconf = 1.0; # no normalization
                print("Jconf = {:.6f}".format(Jconf));
             
                # plot particle currents -> NO FACTORS
                current_index = np.shape(yjs_vs_time)[1] - 1*block2site; # ensures inter-cell if diatomic
                #ax.plot(times, yjs_vs_time[:,0*block2site]/Jconf, 
                #        color=color2,linestyle=mylinestyle); # current through site Nconf
                #ax.plot(times, yjs_vs_time[:,1*block2site]/Jconf, 
                #        color=color1,linestyle=mylinestyle); # current through site NL
                ax.plot(times, yjs_vs_time[:,current_index]/Jconf, 
                        color=color4,linestyle=mylinestyle); # current through site NR
                # current through NR, plateau-averaged
                plateau = np.mean((yjs_vs_time[:,current_index]/Jconf)[time_window_mask]);
                print("plateau avg = {:.6f}".format(plateau), "plateau limits = ",time_window_limits);
                print("current max = {:.6f}".format(np.max(yjs_vs_time[:,current_index])));

                # labels
                label1 = "$J_{L}(t)$";
                label4 = "$J_{R}(t)$";
                if(norm_to_Jconf):
                    label1 += normalizer_label;
                    label4 += normalizer_label;
                    
            else: # occ rate of change expressed with dn/dt
            
                # incoming electron pcurrent = dn_conf / dt
                if(take_gradient): label2 = "$ \left| \\frac{d}{dt} n_{conf} \\right|$";
                else: label2 = "$ n_{conf}$";
                if(norm_to_Jconf): label2 += "$/max \left( \left| \\frac{d}{dt} n_{conf} \\right| \\right)$";
                print(obs2,"-->",label2);
                yjs_vs_time = np.zeros((len(times),block2site*Nsites),dtype=float);
                for ti in range(len(times)):
                    yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
        
                # normalize by max incoming electron pcurrent
                Jconf = 1.0; # no normalization

             
                # occ vs time
                NL, Nconf, NFM, NR = params["NL"], params["Nconf"], params["NFM"], params["NR"];
                Ntotal = NL+NFM+NR; 
                yjL_vs_time = np.sum(yjs_vs_time[:,:block2site*NL], axis=1);
                yjR_vs_time = np.sum(yjs_vs_time[:,block2site*(NL+NFM):], axis=1);
             
                # plot occupancies -> NO FACTORS
                ax.plot(times, do_gradient(yjL_vs_time,times,do=take_gradient)/Jconf, 
                        color=color1,linestyle=mylinestyle);
                ax.plot(times, do_gradient(yjR_vs_time,times,do=take_gradient)/Jconf, 
                        color=color4,linestyle=mylinestyle);
                        
                # specific unit cell occupancy
                if(True):
                    specific_last = Ntotal - 1 # last unit cell on right
                    yj_specific = np.sum(yjs_vs_time[:,block2site*specific_last:], axis=1);
                    ax.plot(times, do_gradient(yj_specific,times,do=take_gradient)/Jconf, 
                        color="hotpink", linestyle=mylinestyle);
                    ax.plot(times, do_gradient(yj_specific,times,do=True)/np.max(do_gradient(yj_specific,times,do=True)), 
                        color="violet", linestyle=mylinestyle);                        
            
                # labels
                label1 = "$n_{L}(t)$";
                label4 = "$n_{R}(t)$";
                if(take_gradient):
                    label1 = "$\left|\\frac{d}{dt}n_{L}(t)\\right|$";
                    label4 = "$\left|\\frac{d}{dt}n_{R}(t)\\right|$";
        
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
    
    # labels
    ax.set_ylabel( label1, color=color1, fontsize=fontsize1); # observable rate of change on left
    ax3.set_ylabel(label4, color=color4, fontsize=fontsize1); # observable rate of change on left
    ax4.set_ylabel(label2, color=color2, fontsize=fontsize1); # labels dn/dt normalizing quantity on right
    ax.set_xlabel("Time $(\hbar/t_l)$", fontsize = fontsize1);
    ax.set_title( get_title(datafiles[-1]), fontsize = fontsize1);

    # show
    plt.tight_layout();
    plt.show();

elif(case in [10,11,12,13]): # time-independent transport metric
                       # vs band structure metric

    # axes
    fignrows, figncols = 1, 1
    change_ratios = {};
    fig, axes = plt.subplots(ncols=figncols,nrows=fignrows, gridspec_kw=change_ratios);
    metricax = axes;
    fig.set_size_inches(UniversalFigRatios[0]*figncols, UniversalFigRatios[1]*fignrows)
    normalize = False;
    if(case in [10,11]): convert_wvals = True;
    else: convert_wvals = False;
    if(case in [10]): savefig = True;
    else: savefig = False;

    #### iter over triplet/singlet
    qubit_labels = ["Qubits Removed",
                    "$|T_0\\rangle$",
                    "$|S  \\rangle$"]; # must all be same number characters
    # here we label triplet/singlet but plot against w on the x axis
    myaxlabels = np.full((len(datafiles),), " "*len(qubit_labels[0]));
    wvals = np.full((len(datafiles),),np.nan);

    # transport metric
    threshold_charge = 4;
    ref_time_inds = np.full((len(datafiles),),1e10,dtype=int);
    nRvals = np.full((len(datafiles),),np.nan); # metric plotted against w
    yjs_observable = "occ_";
    yjs_label = "$n_R$";
    
    # iter over input files to get reference times (to eval transport metric at)
    assert(len(datafiles) % len(qubit_labels) == 0);
    for di, dfile in enumerate(datafiles):
        if("nosd" in dfile):      myaxlabels[di] = qubit_labels[0];
        elif("triplet" in dfile): myaxlabels[di] = qubit_labels[1];
        elif("singlet" in dfile): myaxlabels[di] = qubit_labels[2];
        else: raise NotImplementedError;

        params = json.load(open(dfile+".txt"));
        assert(params["Ne"] > threshold_charge);
        wvals[di] = params["w"];
        print("\nLoading "+dfile+"_arrays/"+yjs_observable+"yjs_time0.00.npy");

        # time evolution params
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("    Update time = {:.2f}, Stop time = {:.2f}, Nupdates = {:.0f}".format(tupdate,tupdate*Nupdates, Nupdates));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
            
        # rice-mele ?
        if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): block2site = 2;
        else: block2site = 1;
        
        # get occ vs time vs site
        Ntotal = params["NL"] + params["NFM"] + params["NR"];
        yjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        xjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"yjs_time{:.2f}.npy".format(times[ti]));
            xjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"xjs_time{:.2f}.npy".format(times[ti]));

        # sum the occupancy in the right lead for all times
        yjs_RL = np.sum(yjs_vs_time[:,block2site*(params["NL"]+params["NFM"]):], axis=1);
        if("nosd" in dfile):
            ref_time_inds[di] = np.argmin(abs(yjs_RL - threshold_charge));
            print("ref time = {:.0f}".format(times[ref_time_inds[di]]));
            print("nR (reftime) = {:.2f}".format(yjs_RL[ref_time_inds[di]]));
        else: pass; # leave fake value in ref_time inds

    # place nosd ref_time_inds in triplet and singlet based on wvals
    for wvali in range(len(wvals)):
        if("nosd" not in datafiles[wvali]):
            # find matching nosd
            for wvalj in range(len(wvals)):
                if("nosd" in datafiles[wvalj]):
                    if(wvals[wvali]==wvals[wvalj]):
                        #  overwrite
                        ref_time_inds[wvali] = 1*ref_time_inds[wvalj];

    # reference times
    print(">>> reference times = ");
    for di in range(len(datafiles)):
        print(datafiles[di], wvals[di], ref_time_inds[di]);

    # reload input files to get transport metric
    assert(len(datafiles) % len(qubit_labels) == 0);
    for di, dfile in enumerate(datafiles):
        if("nosd" in dfile):      myaxlabels[di] = qubit_labels[0];
        elif("triplet" in dfile): myaxlabels[di] = qubit_labels[1];
        elif("singlet" in dfile): myaxlabels[di] = qubit_labels[2];
        else: raise NotImplementedError;

        params = json.load(open(dfile+".txt"));
        # time evolution params
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("    Update time = {:.2f}, Stop time = {:.2f}, Nupdates = {:.0f}".format(tupdate,tupdate*Nupdates, Nupdates));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
            
        # rice-mele ?
        if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): block2site = 2;
        else: block2site = 1;
        
        # get occ vs time vs site
        Ntotal = params["NL"] + params["NFM"] + params["NR"];
        yjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        xjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"yjs_time{:.2f}.npy".format(times[ti]));
            xjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"xjs_time{:.2f}.npy".format(times[ti]));

        # sum the occupancy in the right lead for all times
        yjs_RL = np.sum(yjs_vs_time[:,block2site*(params["NL"]+params["NFM"]):], axis=1);
        nRvals[di] = yjs_RL[ref_time_inds[di]];
        

    # set aside qubits decoupled maxima for normalization
    for qubitstate_formask in qubit_labels[:1]:
        metric_normalizers = nRvals[np.isin(myaxlabels, [qubitstate_formask])];
        # ^ len of this will = len(wvals)
    if(not normalize):
        metric_normalizers = np.ones_like(metric_normalizers);
    else:
        yjs_label += " (norm.)";

    if(convert_wvals):
        indep_vals = visualize_rm.wvals_to_rhoEF(wvals, json.load(open(datafiles[0]+".txt")));
        indep_label = "$\\rho(E_F)$";
    else:
        indep_vals = 1*wvals;
        indep_label = "$w/|v|$";

    # plot
    for colori, qubitstate_formask in enumerate(myaxlabels[:len(qubit_labels)]):
        print(qubitstate_formask)
        print(wvals)
        label_mask = np.isin(myaxlabels, [qubitstate_formask]);
        metricax.plot(indep_vals[label_mask], nRvals[label_mask]/metric_normalizers, label=qubitstate_formask,color=UniversalColors[colori],marker=ColorsMarkers[colori]);
        print(  "x >>> ",wvals[label_mask], 
              "\ny >>> ",nRvals[label_mask]);
    # format
    metricax.set_title( get_title(datafiles[-1], to_exclude=["w"]), fontsize = myfontsize);
    metricax.set_ylabel(yjs_label, fontsize = myfontsize);
    metricax.set_xlabel(indep_label, fontsize = myfontsize);
    
    
    # show
    plt.legend();
    plt.tight_layout();
    plt.show();
    
elif(case in [20,21,22,23]): # charge accumulation vs phient, with wval as color
    # axes
    fignrows, figncols = 1, 1
    change_ratios = {};
    fig, axes = plt.subplots(ncols=figncols,nrows=fignrows, gridspec_kw=change_ratios);
    metricax = axes;
    fig.set_size_inches(UniversalFigRatios[0]*figncols, UniversalFigRatios[1]*fignrows)
    normalize = False;
    if(case in [20]): savefig = True;
    else: savefig = False;

    #### iter over triplet/singlet
    qubit_labels = ["Qubits Removed",
                    "$|T_0\\rangle$",
                    "$|S  \\rangle$"]; # must all be same number characters
    # here we label triplet/singlet but plot against w on the x axis
    myaxlabels = np.full((len(datafiles),), " "*len(qubit_labels[0]));
    wvals = np.full((len(datafiles),),np.nan);
    wvals_hist = {};
    phivals = np.full((len(datafiles),),np.nan);
    

    # transport metric
    threshold_charge = 4;
    ref_time_inds = np.full((len(datafiles),),1e10,dtype=int);
    nRvals = np.full((len(datafiles),),np.nan); # metric plotted against w
    yjs_observable = "occ_";
    yjs_label = "$n_R$";
    
    # iter over input files to get reference times (to eval transport metric at)
    assert(len(datafiles) % len(qubit_labels) == 0);
    for di, dfile in enumerate(datafiles):

        params = json.load(open(dfile+".txt"));
        assert(params["Ne"] > threshold_charge);
        print("\nLoading "+dfile+"_arrays/"+yjs_observable+"yjs_time0.00.npy");
        
        # fill entries for w and phient
        wvals[di] = params["w"];
        if(wvals[di] not in wvals_hist.keys()): wvals_hist[wvals[di]]=1;
        phivals[di] = params["phient"];

        # time evolution params
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("    Update time = {:.2f}, Stop time = {:.2f}, Nupdates = {:.0f}".format(tupdate,tupdate*Nupdates, Nupdates));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
            
        # rice-mele ?
        if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): block2site = 2;
        else: block2site = 1;
        
        # get occ vs time vs site
        Ntotal = params["NL"] + params["NFM"] + params["NR"];
        yjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        xjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"yjs_time{:.2f}.npy".format(times[ti]));
            xjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"xjs_time{:.2f}.npy".format(times[ti]));

        # sum the occupancy in the right lead for all times
        yjs_RL = np.sum(yjs_vs_time[:,block2site*(params["NL"]+params["NFM"]):], axis=1);
        if("nosd" in dfile):
            ref_time_inds[di] = np.argmin(abs(yjs_RL - threshold_charge));
            print("ref time = {:.0f}".format(times[ref_time_inds[di]]));
            print("nR (reftime) = {:.2f}".format(yjs_RL[ref_time_inds[di]]));
        else: pass; # leave fake value in ref_time inds

    # place nosd ref_time_inds in triplet and singlet based on wvals
    for wvali in range(len(wvals)):
        if("nosd" not in datafiles[wvali]):
            # find matching nosd
            for wvalj in range(len(wvals)):
                if("nosd" in datafiles[wvalj]):
                    if(wvals[wvali]==wvals[wvalj]):
                        #  overwrite
                        ref_time_inds[wvali] = 1*ref_time_inds[wvalj];

    # reference times
    print(">>> reference times = ");
    for di in range(len(datafiles)):
        print(datafiles[di], wvals[di], ref_time_inds[di]);    
    
    
    # reload input files to get transport metric
    assert(len(datafiles) % len(qubit_labels) == 0);
    for di, dfile in enumerate(datafiles):

        params = json.load(open(dfile+".txt"));    
        # time evolution params
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("    Update time = {:.2f}, Stop time = {:.2f}, Nupdates = {:.0f}".format(tupdate,tupdate*Nupdates, Nupdates));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
            
        # rice-mele ?
        if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): block2site = 2;
        else: block2site = 1;
        
        # get occ vs time vs site
        Ntotal = params["NL"] + params["NFM"] + params["NR"];
        yjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        xjs_vs_time = np.zeros((len(times),Ntotal*block2site),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"yjs_time{:.2f}.npy".format(times[ti]));
            xjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"xjs_time{:.2f}.npy".format(times[ti]));

        # sum the occupancy in the right lead for all times
        yjs_RL = np.sum(yjs_vs_time[:,block2site*(params["NL"]+params["NFM"]):], axis=1);
        nRvals[di] = yjs_RL[ref_time_inds[di]];
        
    # get unique rhoEF vals
    wvals_unique = np.array(list(wvals_hist.keys()));
    ref_params = json.load(open(datafiles[0]+".txt"))
    rhoEFvals_unique = visualize_rm.wvals_to_rhoEF(wvals_unique, ref_params); 
    # ^ this fails if geometry is not same for all input files
    ref_Ne, ref_Nconf =   ref_params["Ne"], ref_params["Nconf"];
    for dfile in datafiles:
        params = json.load(open(dfile+".txt"));
        assert(params["Ne"] == ref_Ne);
        assert(params["Nconf"] == ref_Nconf);
        
    # screen out nosd nRvals -- they were just to get reference times, not to plot
    for di, dfile in enumerate(datafiles):
        if("nosd" in dfile):
            nRvals[di] = np.nan;
  
    # plot
    assert(len(datafiles) % len(wvals_hist.keys()) == 0);
    print("wvals = ",wvals);
    print("phivals = ",phivals);
    for colori in range(len(wvals_unique)):
        label_mask = np.isin(wvals, [wvals_unique[colori]]);
        label = "$\\rho(E_F) = {:.2f}$".format(rhoEFvals_unique[colori])+" ($w={:.2f}$)".format(wvals_unique[colori])
        metricax.plot(phivals[label_mask]/np.pi, nRvals[label_mask], label=label,color=UniversalColors[colori],marker=ColorsMarkers[colori]);
        print(  "x >>> ",phivals[label_mask], 
              "\ny >>> ",nRvals[label_mask]);
    # format
    metricax.set_title( get_title(datafiles[-1], to_exclude=["w"]), fontsize = myfontsize);
    metricax.set_ylabel(yjs_label, fontsize = myfontsize);
    metricax.set_xlabel("$\phi_{ent}/\pi$", fontsize = myfontsize);
    
    # show
    plt.legend();
    plt.tight_layout();
    plt.show();
    
    
elif(case in [20,21]): # transport vs phient, colors are w vals

    # axes
    fignrows, figncols = 1, 1
    change_ratios = {};
    fig, axes = plt.subplots(ncols=figncols,nrows=fignrows, gridspec_kw=change_ratios);
    metricax = axes;
    fig.set_size_inches(UniversalFigRatios[0]*figncols, UniversalFigRatios[1]*fignrows)

    # here we label w but plot against phient
    wvals_hist = {};
    maxJvals = np.full((len(datafiles),),np.nan); # metric plotted against w
    wvals = np.empty_like(maxJvals);
    phivals = np.empty_like(maxJvals)
    
    # iter over input files
    for di, dfile in enumerate(datafiles):

        params = json.load(open(dfile+".txt"));
        wvals[di] = params["w"];
        if(wvals[di] not in wvals_hist.keys()): wvals_hist[wvals[di]]=1;
        phivals[di] = params["phient"];
        yjs_observable = "J_";
        print("\nLoading "+dfile+"_arrays/"+yjs_observable+"yjs_time0.00.npy");

        # time evolution params
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("    Update time = {:.2f}, Stop time = {:.2f}, Nupdates = {:.0f}".format(tupdate,tupdate*Nupdates, Nupdates));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
            
        # rice-mele ?
        if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): block2site = 2;
        else: block2site = 1;
        
        # get current scattering region -> right lead
        # NB J_ is measured at boundary of Nconf, NL, NR (ie in 3 unit cells)
        yjs_vs_time = np.zeros((len(times),3*block2site),dtype=float);
        xjs_vs_time = np.zeros((len(times),3*block2site),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"yjs_time{:.2f}.npy".format(times[ti]));
            xjs_vs_time[ti] = np.load(dfile+"_arrays/"+yjs_observable+"xjs_time{:.2f}.npy".format(times[ti]));
        print("J_ site index in ",xjs_vs_time[0].astype(int));
        current_index = np.shape(yjs_vs_time)[1] - 1*block2site; # ensures inter-cell if diatomic
        yjs_label = get_ylabel("J_", None, dstring = int(xjs_vs_time[0,current_index]));
        maxJvals[di] = np.max(yjs_vs_time[:,current_index]);
        print("time(maxJ) = {:.0f} ({:.0f}%)".format(times[np.argmax(yjs_vs_time[:,current_index])],100*np.argmax(yjs_vs_time[:,current_index])/Nupdates));
        
    # plot
    assert(len(datafiles) % len(wvals_hist.keys()) == 0);
    print(wvals);
    print(phivals);
    for colori, wstring in enumerate(wvals_hist.keys()):
        label_mask = np.isin(wvals, [wstring]);
        metricax.plot(phivals[label_mask]/np.pi, maxJvals[label_mask], label="$w={:.2f}$".format(wstring),color=UniversalColors[colori],marker=ColorsMarkers[colori]);
        print(  "x >>> ",phivals[label_mask], 
              "\ny >>> ",maxJvals[label_mask]);
    # format
    metricax.set_title( get_title(datafiles[-1], to_exclude=["w"]), fontsize = myfontsize);
    metricax.set_ylabel(yjs_label, fontsize = myfontsize);
    metricax.set_xlabel("$\phi_{ent}/\pi$", fontsize = myfontsize);
    
    # show
    plt.legend();
    plt.tight_layout();
    plt.show();

elif(case in [24,25]): # Jmax vs Jsd, colors are S/T0
    pass;
    
elif(case in [30,31]): # single dataset heatmap, decorated by time-zero density profile and <k_m|k_n> distro
    assert(len(datafiles)==1);
    dfile = datafiles[0];
    params = json.load(open(dfile+".txt"));

    if(False):
        myfig, myax = plt.subplots();
        rm_occs = visualize_rm.get_occs_Ne_TwoSz(params["Ne"], params["TwoSz"]);

        # distribution over E_n states
        rm_Em, rm_km, rm_En, rm_kn, rm_distro, rm_charge, _ = visualize_rm.get_overlaps(params, rm_occs, plotwfs=False);
        myax.scatter(rm_kn/np.pi, rm_En)
        myax.scatter(rm_km/np.pi, rm_Em)
        plt.show();
        assert False

    
    # axes
    fig = plt.figure(layout="constrained");
    change_ratios = {"width_ratios":[0.5,1.0,0.5,0.00001]};
    fignrows, figncols = 1, len(change_ratios["width_ratios"])
    densityax, heatmapax, Emax, Enax = fig.subplots(ncols=figncols,nrows=fignrows,sharey=True, gridspec_kw=change_ratios);
    fig.set_size_inches(UniversalFigRatios[0]*np.sum(change_ratios["width_ratios"]),UniversalFigRatios[1])

    #### iter over triplet/singlet
    qubit_labels = ["Qubits Removed",
                    "$|T_0\\rangle$",
                    "$|S  \\rangle$"]; # must all be same number characters
    if("nosd" in dfile):      myaxlab = qubit_labels[0];
    elif("triplet" in dfile): myaxlab = qubit_labels[1];
    elif("singlet" in dfile): myaxlab = qubit_labels[2];
    else: raise NotImplementedError;

    # time evolution params
    Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
    print("\n    Nupdates = {:.0f}, Update time = {:.2f}".format(Nupdates,tupdate));
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = (update0 + ti)*tupdate;
            
    # rice-mele ?
    if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): 
        block2site = 2;
        h00 = np.array([[params["u"], params["v"]], [params["v"],-params["u"]]]);
        h01 = np.array([[0.0, 0.0],[params["w"], 0.0]]);
        band_edges = wfm.bandedges_RiceMele(h00, h01);
    else: 
        block2site = 1;
        band_edges = [];

    # get heatmap data
    Ntotal = block2site*(params["NL"]+params["NFM"]+params["NR"]); # number of j sites
    yjs_vs_time = np.zeros((len(times),Ntotal),dtype=float);
    for ti in range(len(times)):
        yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
    
    #### heatmap        
    # mesh grids for time (x) and site (y) plotting heatmap
    timex, sitey = np.mgrid[0:int(times[-1])+tupdate:tupdate, 0:Ntotal];
        
    # plot heatmap
    heatmapax.set_facecolor("steelblue");
    heatmap=heatmapax.pcolormesh(timex, sitey, yjs_vs_time, cmap='bwr', vmin=0.0, vmax=np.max(yjs_vs_time));
    print("    np.max(yjs_vs_time) = {:.2f}".format(np.max(yjs_vs_time))); #colorbar limits depend on Ne, Nconf
        
    # format heatmap
    heatmapax.text(0.05,0.9,myaxlab, color="white",
            transform=heatmapax.transAxes,fontsize=myfontsize);
    heatmapax.set_xlabel("Time $(\hbar/|v|)$",fontsize=myfontsize);
    cbar = fig.colorbar(heatmap,ax=heatmapax, location='top',pad=0.01);
    if(True):
        cbar_ticks = cbar.ax.get_xticks() #.tolist();
        cbar_labels = np.copy(cbar_ticks).round(1).astype(str);
        cbar_labels[0] = "$\langle n_{j\mu} \\rangle $";
        print("\n",cbar_ticks,"\n",cbar_labels);
        cbar.ax.set_xticks(cbar_ticks, labels=cbar_labels);           
        cbar.ax.set_xlim(0.0,np.max(yjs_vs_time))
        
    ####  
    #### E_n state occupation distribution and time-zero charge density profile
    if(True): # use visualize_rm get_overlaps function to get time-zero charge density, energy distro
        
        # occupation of |km>
        rm_occs = visualize_rm.get_occs_Ne_TwoSz(params["Ne"], params["TwoSz"]);
        m_Fermi = len(rm_occs) - 1; # index of Fermi energy, among energy arrs

        # distribution over E_n states
        rm_Em, rm_km, rm_En, rm_kn, rm_distro, rm_charge, _ = visualize_rm.get_overlaps(params, rm_occs, plotwfs=False);

        # occupation of |km>
        t0_distro = np.zeros((len(rm_Em),), dtype=int);
        t0_distro[:len(rm_occs)] = rm_occs[:];

        # format occupation of |km>
        Emax.set_xlabel("$n(E_m) \\rho(E_m) (|v|^{-1} a^{-1})$", fontsize=myfontsize);
        fixed_rho_points = (0,10.0);
        Emax.set_xlim(fixed_rho_points);
        Emax = Emax.twinx();
        Emax.set_ylabel("$E_m \, (|v|$)",fontsize=myfontsize);
        Emax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"));
        for edge in band_edges:
            Emax.axhline(edge, color="gray", linestyle="dashed");
        Emax.set_ylim(np.min(rm_En)*1.05, np.max(rm_En)*1.05);
        # ^^ makes sure we don't plot non-confined energies

        # format distribution over E_n states
        Enax.set_xlabel("$n(E_n) \\rho(E_n) (|v|^{-1} a^{-1})$", fontsize=myfontsize);
        Enax.set_xlim(fixed_rho_points);
        Enax, Enax_pretwin = Enax.twinx(), Enax;
        Enax.set_ylabel("$E_n \, (|v|$)",fontsize=myfontsize); 
        Enax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"));
        for edge in band_edges:
            Enax.axhline(edge, color="gray", linestyle="dashed");
        Enax.set_ylim(np.min(rm_En)*1.05, np.max(rm_En)*1.05);
                    
        # Em density of states
        rm_Em_banded = np.array([rm_Em[:len(rm_Em)//2],rm_Em[len(rm_Em)//2:]]);
        Em_gradient = np.array([np.gradient(rm_Em[:len(rm_Em)//2],rm_km[:len(rm_Em)//2]),
                                np.gradient(rm_Em[len(rm_Em)//2:],rm_km[len(rm_Em)//2:])]); 
        # ^ handles both bands at once
        rm_dos_factor = 2/np.pi 
        for bandi in range(len(Em_gradient)):
            Emax.plot(rm_dos_factor*abs(1/Em_gradient[bandi]),rm_Em_banded[bandi], color="black",linestyle="dotted");
            
        #  En density of states
        rm_En_banded = np.array([rm_En[:len(rm_En)//2],rm_En[len(rm_En)//2:]]);
        En_gradient = np.array([np.gradient(rm_En[:len(rm_En)//2],rm_kn[:len(rm_En)//2]),
                                np.gradient(rm_En[len(rm_En)//2:],rm_kn[len(rm_En)//2:])]); 
        # ^ handles both bands at once
        for bandi in range(len(En_gradient)):
            Enax.plot(rm_dos_factor*abs(1/En_gradient[bandi]),rm_En_banded[bandi], color="black",linestyle="dotted");

        # plot n(Em) * rho(Em)
        # since m's are 2's, normalize to 1
        Emax.plot(0.5*t0_distro[:len(rm_Em_banded[0])]*rm_dos_factor*abs(1/Em_gradient[0]), rm_Em_banded[0], color = "black");

        # density of states at Fermi energy
        rho_Fermi = rm_dos_factor*abs(1/Em_gradient[0,m_Fermi]);
        Emax.text(0, 0, "$\\rho(E_F) = {:.2f}$".format(rho_Fermi), fontsize=myfontsize);
          
        ####
        # plot and format time-zero charge density
        densityax.plot(rm_charge, np.arange(0,Ntotal), color = matplotlib.colormaps['bwr'](np.linspace(0,1,10))[9]); 
        densityax.set_xlabel("$\langle n_{j\mu} \\rangle$",fontsize=myfontsize);
        densityax.set_ylabel("Site",fontsize=myfontsize); 

    #### get 
    #### E_m density of states and time-zero charge density profile           
    else:  # direct from stored observables  
        # plot time-zero charge density profile
        densityax.plot(yjs_vs_time[0], np.arange(0,Ntotal), color="black",linestyle="dashed");                    
        # plot En distribution
        nB_vs_time = np.zeros((len(times),Ntotal),dtype=float);
        for ti in [0]: # only need initial
            nB_vs_time[ti] = np.load(dfile+"_arrays/nB_yjs_time{:.2f}.npy".format(times[ti]));           
        distroax.plot(nB_vs_time[0], np.arange(0,Ntotal),color="black",linestyle="dashed");
        
    # format
    fig.suptitle(get_title(dfile,["N_{conf}"]),fontsize=myfontsize);
    #for axi, ax in enumerate([densityax, heatmapax, distroax]):ax.text(0.7,0.9,UniversalPanels[axi],fontsize=myfontsize,transform=ax.transAxes);

    # show
    fig.delaxes(Enax); fig.delaxes(Enax_pretwin);
    folder = datafiles[-1].split("_")[0];
    savename = "/home/cpbunker/Desktop/FIGS_Cicc_with_DMRG/"+folder+"init.pdf";
    if(case in [31]): 
        print("Saving to "+savename);
        plt.savefig(savename);
    else: plt.show();

    
elif(case in [90,91]): # occupancy vs orbital vs time heatmap

    # axes
    horizontal = True; # puts heatmaps side by side, else stack
    if(horizontal): fignrows, figncols = 1, len(datafiles)+1;
    else: fignrows, figncols = len(datafiles)+1, 1;
    add_MI = True;
    change_ratios = {};
    if(add_MI):
        fignrows += 1; assert(horizontal);
        change_ratios["height_ratios"]=[1.0,0.25];
    else:
        change_ratios["height_ratios"]=[1.0];
    fig, axes = plt.subplots(ncols=figncols,nrows=fignrows, sharey="row",sharex="col",
                             gridspec_kw=change_ratios);
    if(add_MI): axes, MIaxes = axes[0], axes[1];
    if((horizontal and figncols==1) or (not horizontal and fignrows==1)): axes = [axes];
    fig.set_size_inches(UniversalFigRatios[0]*figncols, UniversalFigRatios[1]*np.sum(change_ratios["height_ratios"]))

    #### iter over triplet/singlet
    myaxlabels = [];
    for axi, dfile in enumerate(datafiles):
        if("nosd" in dfile):      myaxlab = " Qubits Removed";
        elif("triplet" in dfile): myaxlab = " $|T_0\\rangle$";
        elif("singlet" in dfile): myaxlab = " $|S  \\rangle$";
        elif("nofield" in dfile): myaxlab = " Field Removed";
        else: myaxlab = dfile.split("/")[-1].split("_")[0];

        params = json.load(open(dfile+".txt"));
        if("MSQ_spacer" in params.keys()): myaxlab += ", $d =${:.0f}".format(params["NFM"]-1);
        if(isinstance(myaxlabels,str)):
            arrow_labels = ["$\swarrow$","$\downarrow$","$\searrow$"];
            if(axi==2): myaxlabels += myaxlab+arrow_labels[axi];
            else: myaxlabels += arrow_labels[axi] + myaxlab;
            print("\n>>>",myaxlabels.split(" ")[-1],"=",dfile);
        else: 
            myaxlabels.append(myaxlab);
            print("\n>>>",myaxlabels[-1],"=",dfile);
        
        # time evolution params
        Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
        print("\n    Nupdates = {:.0f}, Update time = {:.2f}".format(Nupdates,tupdate));
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = (update0 + ti)*tupdate;
            
        # rice-mele ?
        if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): block2site = 2;
        else: block2site = 1;

        # get heatmap data
        Ntotal = block2site*(params["NL"]+params["NFM"]+params["NR"]); # number of j sites
        yjs_vs_time = np.zeros((len(times),Ntotal),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
        
        # mesh grids for time (x) and site (y) plotting heatmap
        timex, sitey = np.mgrid[0:int(times[-1])+tupdate:tupdate, 0:Ntotal];
        
        # plot heatmap
        axes[axi].set_facecolor("steelblue");
        heatmap=axes[axi].pcolormesh(timex, sitey, yjs_vs_time, cmap='bwr', vmin=0.0, vmax=np.max(yjs_vs_time));
        print("    np.max(yjs_vs_time) = {:.2f}".format(np.max(yjs_vs_time))); #colorbar limits depend on Ne, Nconf
        
        # plot mutual info  
        which_imp = 0; # d=N_L 
        #label4 = get_ylabel(obs4, factor4, dstring=which_imp, ddt=take_gradient);
        if("MSQ_spacer" in params.keys()): NFM_numsites = 2; assert("_RM" not in params_dict["sys_type"]);
        else: NFM_numsites = block2site*params["NFM"];
        MI_vs_time = np.zeros((len(times),NFM_numsites),dtype=float);
        if(add_MI):
            print("Loading "+dfile+"_arrays/"+obs4+"yjs_time0.00.npy");
            for ti in range(len(times)):
                MI_vs_time[ti] = np.load(dfile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
            # plot
            MIaxes[axi].plot(times, factor4*MI_vs_time[:,which_imp], color=color4, linewidth=2);
        
        # format colorbar
        if(axi==len(axes)-2): 
            cbar = fig.colorbar(heatmap,ax=axes[-1], pad=0.35,location='left');
            axes[-1].text(-0.90,1.03,"$\langle n_{j\mu} \\rangle $",
            transform=axes[-1].transAxes, fontsize=myfontsize);
            axes[-1].axis('off');
            if(add_MI): MIaxes[-1].axis('off');
        
    # format axes
    axes[0].set_ylabel("Site",fontsize=myfontsize); 
    for axi in range(len(datafiles)): 
        axes[axi].text(0.05,0.9,myaxlabels[axi], color="white",
            transform=axes[axi].transAxes,fontsize=myfontsize);
        if(len(datafiles)%2 != 0): axes[len(datafiles)//2].set_title( get_title(datafiles[-1]),fontsize=myfontsize);
        else: axes[axi].set_title(get_title(datafiles[axi],["N_{conf}"]),loc="left",fontsize=myfontsize);

    # format MIaxes
    if(add_MI):
        MIaxes[0].set_ylabel("$I({:.0f},{:.0f})/\ln(2)$".format(block2site*params["NL"], block2site*(params["NL"]+params["NFM"])-1),fontsize=myfontsize);
        MIaxes[0].set_ylabel("$I/\ln(2)$",fontsize=myfontsize);
        for axi in range(len(datafiles)):
            MIaxes[axi].set_ylim(-0.1,1.1);
            MIticks = [0,0.5,1.0];
            for tick in MIticks: MIaxes[axi].axhline(tick, color="gray",linestyle="dashed");
            MIaxes[axi].set_yticks([0,1]);
            MIaxes[axi].set_xlabel("Time $(\hbar/|v|)$",fontsize=myfontsize);
    else:
        for axi in range(len(datafiles)):
            axes[axi].set_xlabel("Time $(\hbar/|v|)$",fontsize=myfontsize);
        
    #### iter over triplet/singlet to mark 
    for axi, dfile in enumerate(datafiles):
        # qubit positions
        params = json.load(open(dfile+".txt"));
        qubit_sites = block2site*params["NL"], block2site*(params["NL"]+params["NFM"]) -1
        if(params["Jsd"] != 0.0):
            rightax = axes[axi].twinx();
            rightax.set_ylim(axes[axi].get_ylim());
            rightax.set_yticks(qubit_sites,labels=[]);
            rightax.tick_params(axis='y', left=True,right=False, color='black', length=12, width=2, grid_color='none');

    # show
    fig.tight_layout();
    folder = datafiles[-1].split("_")[-1];
    savename = "/home/cpbunker/Desktop/FIGS_Cicc_with_DMRG/"+folder+".pdf"
    if(case in [91]): 
        print("Saving to "+savename);
        plt.savefig(savename);
    else: plt.show();   

elif(case in [100,101]): # animate time evol
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));
    if(case in [101]): plot_S2 = True;
    else: plot_S2 = False; 
    
    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    ax.set_yticks(ticks1);
    ax.set_xlabel("Site", fontsize=fontsize1);
    ax.set_title(get_title(datafile));

    # rice-mele ?
    if(params["sys_type"] in ["STT_RM","SIETS_RM","SIAM_RM"]): block2site = 2;
    else: block2site = 1;
    
    # time evolution params
    Nupdates, tupdate = params["Nupdates"]-update0, params["tupdate"];
    print("\nUpdate time = {:.2f}".format(params["tupdate"]));
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = (update0 + ti)*tupdate;
    time_coords = (0.0,-0.96);
    if(params["Ne"]==1): time_coords = (0.0,0.20);

    # set up impurity spin animation
    xds = np.load(datafile+"_arrays/"+obs1+"xjs_time{:.2f}.npy".format(update0*tupdate));
    yds = np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(update0*tupdate));
    impurity_sz, = ax.plot(xds, factor1*yds, marker=mark1, color=color1, markersize=linewidth1**2);
    ax.set_ylabel(get_ylabel(obs1, factor1), color=color1, fontsize=fontsize1);
    time_annotation = ax.annotate("Time = {:.2f}".format(update0*tupdate), time_coords, fontsize=fontsize1);

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
    if(params["Ne"]==1):
        ax.set_ylim([0.0,0.25]);
        ax.set_yticks([0.0,0.20]);

    # pairwise observable
    if(block2site*params["NFM"]>1):
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

else: raise Exception("case = {:.0f} not supported".format(case));
