'''
'''

import numpy as np
import matplotlib.pyplot as plt
from transport import tdfci, tddmrg

# fig standardizing
mycolors = ["cornflowerblue", "darkgreen", "darkred", "darkcyan", "darkmagenta","darkgray"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mylinewidth = 3.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def vs_site(psi,eris_or_driver,block,which_obs):
    '''
    '''
    
    obs_funcs = {"occ_":tddmrg.get_occ, "sz_":tddmrg.get_sz, "sx01_":tddmrg.get_sx01, "sx10_":tddmrg.get_sx10}
    if(which_obs not in obs_funcs.keys()): raise ValueError;

    # site array
    if(block): Nspinorbs = eris_or_driver.n_sites*2;
    else: Nspinorbs = len(eris_or_driver.h1e[0]);
    js = np.arange(Nspinorbs//2);
    vals = np.empty_like(js,dtype=float)
    for j in js:
        op = obs_funcs[which_obs](Nspinorbs,eris_or_driver,j,block);
        if(block):
            vals[j] = np.real(tddmrg.compute_obs(psi, op, eris_or_driver));
        else:
            vals[j] = np.real(tdfci.compute_obs(psi, op));

    return js, vals;

def snapshot_bench(psi_ci, psi_mps, eris_inst, driver_inst, params_dict, savename,
                   time = 0.0, plot_sx = False, draw_arrow=False):
    '''
    '''

    # unpack
    concur_sites = params_dict["ex_sites"];
    Jsd, Jx, Jz = params_dict["Jsd"], params_dict["Jx"], params_dict["Jz"];
    NL, NFM, NR, Ne = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Ne"];
    Ndofs = NL+2*NFM+NR;
    central_sites = [j for j in range(NL,Ndofs-NR)  if j%2==0];
    loc_spins = [sitei for sitei in range(NL,Ndofs-NR)  if sitei%2==1];

    # plot charge and spin vs site
    obs_strs = ["occ_","sz_"];
    ylabels = ["$\langle n_j \\rangle $","$ \langle s_j^{z} \\rangle $"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);

    if(psi_ci is not None): # with fci
        C_ci = tddmrg.concurrence_wrapper(psi_ci, eris_inst, concur_sites, False);
        for obsi in range(len(obs_strs)):
            x, y = vs_site(psi_ci,eris_inst,False,obs_strs[obsi]);
            y_js = y[np.isin(x,loc_spins,invert=True)];# on chain sites
            y_ds = y[np.isin(x,loc_spins)];# off chain impurities
            x_js = np.array(range(len(y_js)));
            x_ds = np.array(range(central_sites[0],central_sites[0]+len(central_sites)));
            # delocalized spins
            axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',
                            label = ("FCI ($C"+str(concur_sites)+"=${:.2f})").format(C_ci),linewidth=mylinewidth);
            # localized spins
            axes[obsi].scatter(x_ds, y_ds, color=mycolors[1], marker="^", s=(3*mylinewidth)**2);

    if(psi_mps is not None): # with dmrg
        C_dmrg = tddmrg.concurrence_wrapper(psi_mps, driver_inst, concur_sites, True);
        for obsi in range(len(obs_strs)):
            x, y = vs_site(psi_mps,driver_inst,True,obs_strs[obsi]);
            y_js = y[np.isin(x,loc_spins,invert=True)];# on chain sites
            y_ds = y[np.isin(x,loc_spins)];# off chain impurities
            x_js = np.array(range(len(y_js)));
            x_ds = np.array(range(central_sites[0],central_sites[0]+len(central_sites)));
            # delocalized spins
            axes[obsi].scatter(x_js,y_js,marker=mymarkers[0], edgecolors=accentcolors[1],
                               s=(3*mylinewidth)**2, facecolors='none',label = ("DMRG ($C"+str(concur_sites)+"=${:.2f})").format(C_dmrg));
            # localized spins
            if(draw_arrow and obs_strs[obsi] != "occ_"):
                for di in range(len(central_sites)):
                    axes[obsi].arrow(x_ds[di],0,0,y_ds[di],color=mycolors[1],
                                     width=0.01*mylinewidth,length_includes_head=True);
            else:
                axes[obsi].scatter(x_ds, y_ds, marker="^", edgecolors=accentcolors[1],
                               s=(3*mylinewidth)**2, facecolors='none');

            # save DMRG data
            arrs = [x_js, y_js, x_ds, y_ds];
            arr_names = ["xjs","yjs","xds","yds"];
            for arri in range(len(arr_names)):
                np.save(savename[:-4]+"_arrays/"+obs_strs[obsi]+arr_names[arri]+"_time{:.2f}".format(time), arrs[arri]);

        # plot <sx> from DMRG
        if(plot_sx):
            x, sx01 = vs_site(psi_mps,driver_inst,False,"sx01_");
            x, sx10 = vs_site(psi_mps,driver_inst,False,"sx10_");
            sx = sx01+sx10;
            sx_js = sx[np.isin(x,loc_spins,invert=True)];# on chain sites
            axes[-1].plot(np.array(range(len(sx_js))), sx_js, color="purple",marker='s', linewidth=mylinewidth);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j$");
    axes[-1].set_xlim(np.min(x_js), np.max(x_js));
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    title_str = "$J_{sd} = $"+"{:.4f}$t_l$".format(Jsd)+", $J_x = ${:.4f}$t_l$, $J_z = ${:.4f}$t_l$, $N_e = ${:.0f}".format(Jx, Jz, Ne);
    axes[0].set_title(title_str);
    np.savetxt(savename[:-4]+"_arrays/"+obs_strs[0]+"title.txt",[0.0], header=title_str);
    plt.tight_layout();
    plt.savefig(savename[:-4]+"_time{:.2f}.pdf".format(time));

def snapshot_fromdata(loadname, time):
    '''
    '''
    
    # plot charge and spin vs site
    obs_strs = ["occ_","sz_"];
    ylabels = ["$\langle n_j \\rangle $","$ \langle s_j^{z} \\rangle $"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);
    for obsi in range(len(obs_strs)):
        y_js = np.load(loadname[:-4]+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}.npy".format(time));
        y_ds = np.load(loadname[:-4]+"_arrays/"+obs_strs[obsi]+"yds_time{:.2f}.npy".format(time))
        x_js = np.load(loadname[:-4]+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}.npy".format(time))
        x_ds = np.load(loadname[:-4]+"_arrays/"+obs_strs[obsi]+"xds_time{:.2f}.npy".format(time))
        # delocalized spins
        axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',linewidth=mylinewidth);
        # localized spins
        axes[obsi].scatter(x_ds, y_ds, color=mycolors[1], marker="^", s=(3*mylinewidth)**2);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j$");
    axes[-1].set_xlim(np.min(x_js), np.max(x_js));
    axes[-1].legend(title = "td-DMRG | Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title( open(loadname[:-4]+"_arrays/"+obs_strs[0]+"title.txt","r").read().splitlines()[0][1:]);
    plt.tight_layout();
    plt.show();

