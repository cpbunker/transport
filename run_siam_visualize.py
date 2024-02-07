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
obs1, color1, ticks1, linewidth1, fontsize1 = "occ_", "cornflowerblue", (-1.0,-0.5,0.0,0.5,1.0), 3.0, 16;
obs2, color2 = "G_", "darkred";
num_xticks = 4;
datamarkers = ["s","^","d","*"]

if(case in [0]): # standard charge density vs site snapshot
    from transport import tddmrg
    from transport.tddmrg import plot
    tddmrg.plot.snapshot_fromdata(datafiles[0], time0, "SIAM")

elif(case in [1,2]): # observable as a function of time

    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    #ax.set_yticks(ticks1);
    ax.set_xlabel("Time $(\hbar/t_l)$");

    # plot observables for EACH datafile
    for datai in range(len(datafiles)):
        params = json.load(open(datafiles[datai]+".txt"));
        title_or_label = open(datafiles[datai]+"_arrays/"+obs1+"title.txt","r").read().splitlines()[0][1:];
        if(len(datafiles)==1): the_title = title_or_label[:]; the_label = "";
        else: the_title = ""; the_label = title_or_label[:];

        # time evolution params
        tupdate = params["tupdate"];
        times = np.zeros((Nupdates+1,),dtype=float);
        for ti in range(len(times)):
            times[ti] = time0 + ti*tupdate;

        # current vs time
        which_imp = 0;
        Nsites = params["NL"]+1+params["NR"]; # number of d sites
        yds_vs_time = np.zeros((len(times),1),dtype=float);
        for ti in range(len(times)):
            yds_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs2+"yjs_time{:.2f}.npy".format(times[ti]));
        ax.plot(times,yds_vs_time[:,which_imp],color=color2, marker=datamarkers[datai],label=the_label);

        # CHANGE IN left lead occ vs time
        yjs_vs_time = np.zeros((len(times),Nsites),dtype=float);
        for ti in range(len(times)):
            yjs_vs_time[ti] = np.load(datafiles[datai]+"_arrays/"+obs1+"yjs_time{:.2f}.npy".format(times[ti]));
        yjdelta_vs_time = np.sum(yjs_vs_time[:,:params["NL"]], axis=1) - np.sum(yjs_vs_time[:,:params["NL"]], axis=1)[0];
        #ax.plot(times, yjdelta_vs_time,color=color1,marker=datamarkers[datai]);

    # formatting
    ax.set_ylabel("$\pi \langle J_{Imp}^z \\rangle /V_b$", color=color2, fontsize=fontsize1);
    ax.set_title(the_title);
    time_ticks = np.arange(times[0], times[-1], times[-1]//(num_xticks-1))
    ax.set_xticks(time_ticks);
    ax.set_xlim((times[0], times[-1]));
    ax3 = ax.twinx();
    ax3.yaxis.set_label_position("left");
    ax3.spines.left.set_position(("axes", -0.2));
    ax3.spines.left.set(alpha=0.0);
    ax3.set_yticks([])
    ax3.set_ylabel("$n_L(t) - n_L(0)$", color=color1, fontsize=fontsize1);
    
    # show
    if(len(datafiles) > 1): ax.legend();
    plt.tight_layout();
    plt.show();

elif(case in [10]):
    # axes
    fig, ax = plt.subplots();
    for tick in ticks1: ax.axhline(tick,linestyle=(0,(5,5)),color="gray");
    ax.set_yticks(ticks1);
    ax.set_xlabel("$j(d)$", fontsize=fontsize1);
    datafile = datafiles[0];
    params = json.load(open(datafile+".txt"));
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

    # plot concurrence
    obs4, color4 = "conc_", "black";
    xds_4 = np.load(datafile+"_arrays/"+obs4+"xjs_time{:.2f}.npy".format(time0));
    yds_4 = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(time0));
    conc, = ax.plot(xds_4,yds_4,marker="^",color=color4);
    plt.tight_layout();

    # time evolve observables
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
        # concurrence
        yds_4_t = np.load(datafile+"_arrays/"+obs4+"yjs_time{:.2f}.npy".format(time));
        conc.set_ydata(10*yds_4_t);

    # animate
    if(params["time_step"]==1.0): interval = 400;
    elif(params["time_step"]==0.5): interval = 200;
    else: interval = 500;
    interval = 1000*(10/Nupdates); # so total animation time is 10 sec
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


