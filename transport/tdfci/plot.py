'''
'''

import numpy as np
import matplotlib.pyplot as plt
from transport import tdfci, tddmrg

# fig standardizing
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mylinewidth = 3.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def vs_site(js,psi,eris_or_driver,block,which_obs):
    '''
    '''
    
    obs_funcs = {"occ_":tddmrg.get_occ, "sz_":tddmrg.get_sz, 
            "conc_":tddmrg.concurrence_wrapper, "pur_":tddmrg.purity_wrapper,
            "sx01_":tddmrg.get_sx01, "sx10_":tddmrg.get_sx10}
    if(which_obs not in obs_funcs.keys()): raise ValueError;

    # site array
    if(block): Nspinorbs = eris_or_driver.n_sites*2;
    else: Nspinorbs = len(eris_or_driver.h1e[0]);
    vals = np.zeros_like(js,dtype=float)
    for ji in range(len(js)):
        if(which_obs in ["conc_","pur_"]): # since concur = C(d,d+1), we cannot compute C for the final d site. Just leave it as 0.0
            if(ji!=len(js)-1): # for simplicity, also apply this to the purity. even though we could compute it, Feiguin never does so we don't need to
                vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,[js[ji],js[ji+1]],block);
        else:
            op = obs_funcs[which_obs](Nspinorbs,eris_or_driver,js[ji],block);
            if(block):
                vals[ji] = np.real(tddmrg.compute_obs(psi, op, eris_or_driver));
            else:
                vals[ji] = np.real(tdfci.compute_obs(psi, op));

    return js, vals;

def snapshot_bench(psi_ci, psi_mps, eris_inst, driver_inst, params_dict, savename,
                   time = 0.0, plot_fig=False):
    '''
    '''
    if(psi_ci is None and psi_mps is None): return;

    # unpack
    concur_sites = params_dict["ex_sites"];
    concur_dj = [concur_sites[0]//2, concur_sites[1]//2];
    concur_strs = [["j","d"][concur_sites[0]%2],["j","d"][concur_sites[1]%2]];
    Jsd, Jx, Jz = params_dict["Jsd"], params_dict["Jx"], params_dict["Jz"];
    NL, NFM, NR, Ne = params_dict["NL"], params_dict["NFM"], params_dict["NR"], params_dict["Ne"];
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    js_all = np.arange(2*Nsites);
    central_sites = np.array([j for j in range(2*NL,2*(NL+NFM) ) if j%2==0]);
    loc_spins = np.array([d for d in range(2*NL,2*(NL+NFM))  if d%2==1]);
    j_sites = np.array([j for j in range(2*Nsites) if j%2==0]); # on chain only!

    # plot charge and spin vs site
    obs_strs = ["occ_","sz_","conc_","pur_"];
    ylabels = ["$\langle n_{j(d)} \\rangle $","$ \langle s_{j(d)}^{z} \\rangle $","$C_{d,d+1}$","$|\mathbf{S}_d|$"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[1.0,0.0],[0.5,0.0]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);

    if(psi_ci is not None): # with fci
        C_ci = tddmrg.concurrence_wrapper(psi_ci, eris_inst, concur_sites, False);
        for obsi in range(len(obs_strs)):
            if(obs_strs[obsi] not in ["conc_","pur_"]): js_pass = js_all;
            else: js_pass = loc_spins;
            x, y = vs_site(js_pass,psi_ci,eris_inst,False,obs_strs[obsi]);
            y_js = y[np.isin(x,j_sites)];# on chain sites
            y_ds = y[np.isin(x,j_sites,invert=True)];# off chain impurities
            x_js = x[np.isin(x,j_sites)]//2;
            x_ds = x[np.isin(x,j_sites,invert=True)]//2;
            # delocalized spins
            axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',
                            label = ("FCI ($C"+str(concur_sites)+"=${:.2f})").format(C_ci),linewidth=mylinewidth);
            # localized spins
            axes[obsi].scatter(x_ds, y_ds, color=mycolors[1], marker="s", s=(3*mylinewidth)**2);
            if(obs_strs[obsi] == "sz_"): print("Total Sz (FCI) = {:.6f}".format(sum(y)));

    if(psi_mps is not None): # with dmrg
        C_dmrg = tddmrg.concurrence_wrapper(psi_mps, driver_inst, concur_sites, True);
        for obsi in range(len(obs_strs)):
            if(obs_strs[obsi] not in ["conc_","pur_"]): js_pass = js_all;
            else: js_pass = loc_spins;
            x, y = vs_site(js_pass,psi_mps,driver_inst,True,obs_strs[obsi]);
            y_js = y[np.isin(x,j_sites)];# on chain sites
            y_ds = y[np.isin(x,j_sites,invert=True)];# off chain impurities
            x_js = x[np.isin(x,j_sites)]//2;
            x_ds = x[np.isin(x,j_sites,invert=True)]//2;
            # delocalized spins
            axes[obsi].scatter(x_js,y_js,marker=mymarkers[0], edgecolors=accentcolors[1],
                               s=(3*mylinewidth)**2, facecolors='none',label = ("DMRG ($C"+str(concur_sites)+"=${:.2f})").format(C_dmrg));
            # localized spins
            axes[obsi].scatter(x_ds, y_ds, marker="s", edgecolors=accentcolors[1],
                               s=(3*mylinewidth)**2, facecolors='none');
            
            # save DMRG data
            if(not plot_fig):
                arrs = [x_js, y_js, x_ds, y_ds];
                arr_names = ["xjs","yjs","xds","yds"];
                for arri in range(len(arr_names)):
                    np.save(savename[:-4]+"_arrays/"+obs_strs[obsi]+arr_names[arri]+"_time{:.2f}".format(time), arrs[arri]);

        # plot <sx> from DMRG
        if(False):
            x, sx01 = vs_site(psi_mps,driver_inst,False,"sx01_");
            x, sx10 = vs_site(psi_mps,driver_inst,False,"sx10_");
            sx = sx01+sx10;
            sx_js = sx[np.isin(x,loc_spins,invert=True)];# on chain sites
            axes[-1].plot(np.array(range(len(sx_js))), sx_js, color="purple",marker='^', linewidth=mylinewidth);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j (d)$");
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    title_str = "$J_{sd} = $"+"{:.4f}$t_l$".format(Jsd)+", $J_x = ${:.4f}$t_l$, $J_z = ${:.4f}$t_l$, $N_e = ${:.0f}".format(Jx, Jz, Ne);
    axes[0].set_title(title_str);
    plt.tight_layout();
    if(plot_fig): plt.show();
    else:
        np.savetxt(savename[:-4]+"_arrays/"+obs_strs[0]+"title.txt",[0.0], header=title_str);
        plt.savefig(savename[:-4]+"_time{:.2f}.pdf".format(time));
    plt.close(); # keeps figure from being stored in memory

def snapshot_fromdata(loadname, time):
    '''
    '''
    
    # plot charge and spin vs site
    obs_strs = ["occ_","sz_","conc_","pur_"];
    ylabels = ["$\langle n_{j(d)} \\rangle $","$ \langle s_{j(d)}^{z} \\rangle $","$C_{d,d+1}$","$|\mathbf{S}_d|$"];
    axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[1.0,0.0],[0.5,0.0]];
    fig, axes = plt.subplots(len(obs_strs),sharex=True);
    for obsi in range(len(obs_strs)):
        y_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}.npy".format(time));
        y_ds = np.load(loadname+"_arrays/"+obs_strs[obsi]+"yds_time{:.2f}.npy".format(time))
        x_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}.npy".format(time))
        x_ds = np.load(loadname+"_arrays/"+obs_strs[obsi]+"xds_time{:.2f}.npy".format(time))
        # delocalized spins
        axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',linewidth=mylinewidth);
        # localized spins
        axes[obsi].scatter(x_ds, y_ds, color=mycolors[1], marker="s", s=(3*mylinewidth)**2);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j(d)$");
    #axes[-1].set_xlim(np.min(x_js), np.max(x_js));
    axes[-1].legend(title = "td-DMRG | Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title( open(loadname+"_arrays/"+obs_strs[0]+"title.txt","r").read().splitlines()[0][1:]);
    plt.tight_layout();
    plt.show();

