'''
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

import sys
import json

# top level
case = int(sys.argv[1]);
time0 = int(sys.argv[2]);
Nupdates = int(sys.argv[3]);
datafiles = sys.argv[4:];

# plotting
obs1, color1, ticks1, linewidth1, fontsize1 = "Sdz_", "darkred", (-1.0,-0.5,0.0,0.5,1.0), 3.0, 16;
obs2, color2 = "occ_", "cornflowerblue";
obs3, color3 = "sz_", "darkblue";
obs4, color4, ticks4 = "pur_", "gray", (0.0,0.5,1.0);
num_xticks = 4;
datamarkers = ["s","^","d","*"];

if(case in [0]): # standard charge density vs site snapshot
    from transport import tddmrg
    from transport.tddmrg import plot
    tddmrg.plot.snapshot_fromdata(datafiles[0], time0, "STT")

if(case in [1,2]): # observable as a function of time
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));
    if(case in [2]): plot_purity = True;
    else: plot_purity = False; # plot (pseudo) concurrence instead

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    #ax.set_yticks(ticks1);
    ax.set_xlabel("Time $(\hbar/t_l)$", fontsize = fontsize1);
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);

    # time evolution params
    tupdate = params["tupdate"];
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = time0 + ti*tupdate;
    time_ticks = np.arange(times[0], times[-1], times[-1]//(num_xticks-1))
    ax.set_xticks(time_ticks);
    ax.set_xlim((times[0], times[-1]));

    # impurity spin vs time
    Nbuffer = 0;
    if("Nbuffer" in params.keys()): Nbuffer = params["Nbuffer"];
    Nsites = Nbuffer + params["NL"]+params["NFM"]+params["NR"]; # number of d sites
    which_imp = 0;
    yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
    for ti in range(len(times)):
        yds_vs_time[ti] = 2*np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
    ax.plot(times,yds_vs_time[:,which_imp],color=color1);
    ax.set_ylabel("$2 \langle S_{"+str(which_imp)+"}^z \\rangle /\hbar$", color=color1, fontsize=fontsize1);

    # AVERAGE electron spin vs time
    Ne = params["Ne"]; # number deloc electrons
    yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
    for ti in range(len(times)):
        yjs_vs_time[ti] = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
    yjavg_vs_time = np.sum(yjs_vs_time, axis=1)/Ne;
    ax.plot(times, yjavg_vs_time,color=color3);
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.15));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    ax3.set_ylabel("$2 \overline{ \langle s_j^z \\rangle} /\hbar$", color=color3, fontsize=fontsize1);

    # impurity purity vs time
    if plot_purity: # plot purity
        label4 = "$||$";
        purds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            purds_vs_time[ti] = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.fill_between(times, (-2)*purds_vs_time[:,which_imp],color=color4);
    else: # plot (pseudo) concurrence
        obs4, color4 = "pconc_", "black";
        label4 = "$\langle pC_{d,d+1} \\rangle$";
        normalizer4 = 1;
        purds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            purds_vs_time[ti] = normalizer4*np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.plot(times, purds_vs_time[:,which_imp],color=color4);
    ax4 = ax.twinx();
    ax4.yaxis.set_label_position("right");
    ax4.spines.right.set_position(("axes", 1.0));
    ax4.spines.right.set(alpha=1.0);
    ax4.set_yticks([])
    ax4.set_ylabel(label4, color=color4, fontsize=fontsize1);

    # show
    plt.tight_layout();
    plt.show();

if(case in [3,4]): # observable as a function of time

    # axes
    fig, ax = plt.subplots();
    ax.set_xlabel("Time $(\hbar/t_l)$", fontsize = fontsize1);
    ax.set_title( open(datafiles[0]+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);
    for tick in ticks4: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    params = json.load(open(datafiles[0]+".txt"));
    if(len(datafiles) == 2): assert("triplet" in datafiles[0] and "singlet" in datafiles[1]);
    
    # time evolution params
    tupdate = params["tupdate"];
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = time0 + ti*tupdate;
    time_ticks = np.arange(times[0], times[-1], times[-1]//(num_xticks-1));
    ax.set_xticks(time_ticks);
    ax.set_xlim((times[0], times[-1]));

    #### iter over triplet/singlet
    for dfile in datafiles:
        if("triplet" in dfile): mylinestyle = "solid";
        else: mylinestyle = "dashed";
        print(">>>",dfile)

        # COMBINED impurity spins vs time
        assert(params["NFM"] == 2);
        Nbuffer = 0;
        if("Nbuffer" in params.keys()): Nbuffer = params["Nbuffer"];
        Nsites = Nbuffer + params["NL"]+params["NFM"]+params["NR"]; # number of d sites
        yds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            yds_vs_time[ti] = np.load(dfile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
        yds_summed = np.sum(yds_vs_time, axis=1);
        ax.plot(times,yds_summed,color=color1, linestyle=mylinestyle);
        ax.set_ylabel("$ \langle S_1^z + S_2^z \\rangle /\hbar$", color=color1, fontsize=fontsize1);

        # COMBINED electron spin vs time
        Ne = params["Ne"]; # number deloc electrons
        label3 = "$\\frac{1}{N_e} \sum_j  2\langle s_j^z \\rangle /\hbar $";
        normalizer3 = 2/Ne;
        yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(dfile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(times[ti]));
        yjs_summed = np.sum(yjs_vs_time, axis=1);
        ax.plot(times, normalizer3*yjs_summed,color=color3, linestyle=mylinestyle);
    
        # determine S1 + S2 eigenstate
        assert(params["NFM"] == 2);
        obs4, color4 = "S2_", "black";
        label4 = "$\langle (\mathbf{S}_1 + \mathbf{S}_2)^2 \\rangle/2 \hbar$";
        normalizer4 = 1/2;
        purds_vs_time = np.zeros((len(times),params["NFM"]),dtype=float);
        for ti in range(len(times)):
            purds_vs_time[ti] = normalizer4*np.load(dfile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.plot(times, purds_vs_time[:,0],color=color4, linestyle=mylinestyle);

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
   
    # show
    plt.tight_layout();
    plt.show();
    
elif(case in [5,6]): # left lead, SR, right lead occupancy as a function of time
    if(case in [6]): difference=True;
    else: difference = False;

    # axes
    fig, ax = plt.subplots();
    ax.set_xlabel("Time $(\hbar/t_l)$");

    # plot observables for EACH datafile
    for datai in range(len(datafiles)):
        params = json.load(open(datafiles[datai]+".txt"));
        title_or_label = open(datafiles[datai]+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:];
        if(len(datafiles)==1): the_title = title_or_label[:]; the_label = "";
        else: the_title = ""; the_label = title_or_label[:];

        # time evolution params
        tupdate = params["tupdate"];
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = time0 + ti*tupdate;
            
        # occ vs time
        NL, NFM, NR = params["NL"], params["NFM"], params["NR"];
        Nsites = NL+NFM+NR; 
        yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));

        # break up occupancies
        yjL_vs_time = np.sum(yjs_vs_time[:,:NL], axis=1);
        yjSR_vs_time = np.sum(yjs_vs_time[:,NL:NL+NFM], axis=1);
        yjR_vs_time = np.sum(yjs_vs_time[:,NL+NFM:], axis=1);
        if(difference): # only plot change in occupancy
            print("LL n(0) = {:.4f}".format(yjL_vs_time[0]));
            yjL_vs_time = yjL_vs_time - yjL_vs_time[0];
            print("SR n(0) = {:.4f}".format(yjSR_vs_time[0]));
            yjSR_vs_time = yjSR_vs_time - yjSR_vs_time[0];
            print("RL n(0) = {:.4f}".format(yjR_vs_time[0]));
            yjR_vs_time = yjR_vs_time - yjR_vs_time[0];
        
        # plot occupancies
        ax.plot(times, yjL_vs_time,color=color1,marker=datamarkers[datai]);
        ax.plot(times, yjSR_vs_time,color=color2,marker=datamarkers[datai],label=the_label);
        ax.plot(times, yjR_vs_time,color=color3,marker=datamarkers[datai]);
        
    # formatting
    if(difference): ax.set_ylabel("$\Delta n_{SR}(t)$", color=color2, fontsize=fontsize1);
    else: ax.set_ylabel("$n_{SR}(t)$", color=color2, fontsize=fontsize1);
    ax.set_title(the_title);
    time_ticks = np.arange(times[0], times[-1], times[-1]//(num_xticks-1))
    ax.set_xticks(time_ticks);
    ax.set_xlim((times[0], times[-1]));

    # show
    if(len(datafiles) > 1): ax.legend();
    plt.tight_layout();
    plt.show();

if(case in [10]): # animate time evol
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    ax.set_yticks(ticks1);
    ax.set_xlabel("$j(d)$", fontsize=fontsize1);
    ax.set_title( open(datafile+"_arrays/"+obs2+"title.txt","r").read().splitlines()[0][1:]);
    
    # time evolution params
    tupdate = params["tupdate"];
    times = np.zeros((Nupdates+1,),dtype=float);
    for ti in range(len(times)):
        times[ti] = time0 + ti*tupdate;

    # set up impurity spin animation
    xds = np.load(datafile+"_arrays/"+obs1+"xjs_time{:.2f}.npy".format(time0));
    yds = 2*np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(time0));
    impurity_sz, = ax.plot(xds, yds, marker="s", color=color1, markersize=linewidth1**2);
    ax.set_ylabel("$2 \langle S_d^z \\rangle /\hbar$", color=color1, fontsize=fontsize1);
    time_annotation = ax.annotate("Time = {:.2f}".format(time0), (0.0,-0.96),fontsize=fontsize1);

    # set up charge density animation
    xjs = np.load(datafile+"_arrays/"+obs2+"xjs_time{:.2f}.npy".format(time0));
    yjs = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(time0));
    charge_density = ax.fill_between(xjs, yjs, color=color2);
    ax2 = ax.twinx();
    ax2.set_yticks([]);
    ax2.set_ylabel("$\langle n_j \\rangle$", color=color2, fontsize=fontsize1);

    # set up deloc spin animation
    xjs_3 = np.load(datafile+"_arrays/"+obs3+"xjs_time{:.2f}.npy".format(time0));
    yjs_3 = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(time0));
    spin_density, = ax.plot(xjs_3, yjs_3, marker="o", color=color3);
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.15));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    ax3.set_ylabel("$2 \langle s_j^z \\rangle /\hbar$", color=color3, fontsize=fontsize1);

    # plot (S1+S2)^2 /2
    obs4, color4, label4 = "S2_", "black", "$\langle (\mathbf{S}_d + \mathbf{S}_{d+1})^2 \\rangle$";
    xds_4 = np.load(datafile+"_arrays/"+obs4+"xjs_time{:.2f}.npy".format(time0));
    yds_4 = (1/2)*np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(time0));
    S2, = ax.plot(xds_4,yds_4,marker="^",color=color4);
    ax4 = ax.twinx();
    ax4.yaxis.set_label_position("right");
    ax4.spines.right.set_position(("axes", 1.05));
    ax4.spines.right.set(alpha=0.0);
    ax4.set_yticks([])
    ax4.set_ylabel(label4, color=color4, fontsize=fontsize1);

    # time evolve observables
    plt.tight_layout();
    def time_evolution(time):
        # impurity spin
        yds_t = 2*np.load(datafile+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(time));
        impurity_sz.set_ydata(yds_t);
        time_annotation.set_text("Time = {:.2f}".format(time));
        # charge density
        yjs_t = np.load(datafile+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(time));
        ax.collections.clear();
        charge_density_update = ax.fill_between(xjs, yjs_t, color=color2)
        charge_density.update_from(charge_density_update);
        # spin density
        yjs_3_t = 2*np.load(datafile+"_arrays/"+obs3+"yjs_time{:.2f}.npy".format(time));
        spin_density.set_ydata(yjs_3_t);
        # (S1+S2)^2 / 2
        yds_4_t = (1/2)*np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(time));
        S2.set_ydata(yds_4_t);

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


