'''
'''

import numpy as np
import matplotlib.pyplot as plt
from transport import tddmrg, tdfci

# fig standardizing
mycolors = ["darkblue", "darkred", "darkorange", "darkcyan", "darkgray","hotpink", "saddlebrown"];
accentcolors = ["black","red"];
mymarkers = ["o","+","^","s","d","*","X"];
mylinewidth = 3.0;
mypanels = ["(a)","(b)","(c)","(d)"];
#plt.rcParams.update({"text.usetex": True,"font.family": "Times"});

def vs_site(js,psi,eris_or_driver,which_obs, block, prefactor):
    '''
    '''
    if(not isinstance(prefactor, float)): raise TypeError;
    obs_funcs = {"occ_":tddmrg.get_occ, "sz_":tddmrg.get_sz,"Sdz_":tddmrg.get_Sd_mu, 
            "conc_":tddmrg.concurrence_wrapper, "pur_":tddmrg.purity_wrapper,
            "G_":tddmrg.conductance_wrapper};
    if(block): compute_func = tddmrg.compute_obs;
    else: compute_func = tdfci.compute_obs;


    # site array
    vals = np.zeros_like(js,dtype=float)
    for ji in range(len(js)):
        if(which_obs in ["conc_"]): 
            if(ji!=len(js)-1): vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,[js[ji],js[ji+1]],block);
            else: vals[ji] = np.nan; # since concur = C(d,d+1), we cannot compute C for the final d site.
        elif(which_obs in ["pur_","G_"]):
            vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,js[ji],block);
        else:
            op = obs_funcs[which_obs](eris_or_driver,js[ji],block);
            ret = compute_func(psi, op, eris_or_driver);
            if(np.imag(ret) > 1e-12): print(ret); raise ValueError;
            vals[ji] = np.real(ret);

    return js, prefactor*vals;

def snapshot_bench(psi_mps, driver_inst, params_dict, savename, time, block=True):
    '''
    '''
    assert(isinstance(block, bool));

    # unpack
    sys_type = params_dict["sys_type"];
    plot_fig = params_dict["plot"];
    NL, NR = params_dict["NL"], params_dict["NR"];
    Nbuffer, NFM = 0, 1;
    if(sys_type=="STT"):
        Jsd, Jx, Jz = params_dict["Jsd"], params_dict["Jx"], params_dict["Jz"];
        NFM,  Ne = params_dict["NFM"], params_dict["Ne"];
        if("Nbuffer" in params_dict.keys()): Nbuffer = params_dict["Nbuffer"];
        title_str = "$J_{sd} = $"+"{:.4f}$t_l$".format(Jsd)+", $J_x = ${:.4f}$t_l$, $J_z = ${:.4f}$t_l$, $N_e = ${:.0f}".format(Jx, Jz, Ne);
        # plot charge and spin vs site
        obs_strs = ["occ_","sz_","Sdz_","pur_","conc_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $","$ \langle S_{d}^{z} \\rangle $","$|\mathbf{S}_d|$","$C_{d,d+1}$"];
        axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[0.5,0.0,-0.5],[0.5,0.0],[1.0,0.0]];
    elif(sys_type=="SIAM"):
        th, Vg, U, Vb = params_dict["th"], params_dict["Vg"], params_dict["U"], params_dict["Vb"];
        NFM, Ne = 1, (NL+1+NR)//2;
        title_str = "$t_h =$ {:.4f}$t_l, V_g =${:.4f}$t_l, U =${:.4f}$t_l, V_b =${:.4f}$t_l$".format(th, Vg, U, Vb);
        obs_strs = ["occ_", "sz_", "G_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $", "$\pi \langle J_{"+str(NL)+"} \\rangle/V_b$"];
        axlines = [ [1.2,1.0,0.8],[0.1,0.0,-0.1],[1.0,0.0]];
    elif(sys_type=="SIETS"):
        th, Delta, Vb = params_dict["th"], params_dict["Delta"], params_dict["Vb"];
        NFM, Ne = params_dict["NFM"], (NL+params_dict["NFM"]+NR)//2;
        title_str = "$t_h =$ {:.4f}$t_l, \Delta =${:.4f}$t_l, V_b =${:.4f}$t_l$".format(th, Delta, Vb);
        obs_strs = ["occ_", "sz_", "Sdz_", "G_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $","$ \langle S_{d}^{z} \\rangle $", "$\pi \langle J_{"+str(NL)+"} \\rangle/V_b$"];
        axlines = [ [1.2,1.0,0.8],[0.1,0.0,-0.1],[0.5,0.0,-0.5],[1.0,0.0]];
    else:
        raise Exception("System type = "+sys_type+" not supported");
    Nsites = Nbuffer+NL+NFM+NR; # number of j sites in 1D chain
    central_sites = np.array([j for j in range(Nbuffer+NL,Nbuffer+NL+NFM)]);
    all_sites = np.array([j for j in range(Nsites)]);

    # plot
    fig, axes = plt.subplots(len(obs_strs));
    if(psi_mps is not None): # with dmrg
        for obsi in range(len(obs_strs)):
            if(obs_strs[obsi] not in ["Sdz_","conc_","pur_", "G_"]): js_pass = all_sites;
            else: js_pass = central_sites;
            if(obs_strs[obsi] in ["G_"]): prefactor = np.pi*params_dict["th"]/params_dict["Vb"];
            else: prefactor = 1.0;
            x_js, y_js = vs_site(js_pass,psi_mps,driver_inst,obs_strs[obsi],block,prefactor);
            axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',linewidth=mylinewidth,
                               label = ("DMRG (te_type = "+str(params_dict["te_type"])+", dt= "+str(params_dict["time_step"])));
            print("Total <"+obs_strs[obsi]+"> = {:.6f}".format(np.sum(y_js)));            

            # save DMRG data
            if(block and (not plot_fig)):
                np.save(savename[:-4]+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}".format(time), x_js);
                np.save(savename[:-4]+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}".format(time), y_js);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j(d)$");
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title(title_str);
    plt.tight_layout();
    if(plot_fig): pass #plt.show();
    else:
        np.savetxt(savename[:-4]+"_arrays/"+obs_strs[0]+"title.txt",[0.0], header=title_str);
        #plt.savefig(savename[:-4]+"_arrays/time{:.2f}.pdf".format(time));
    plt.close(); # keeps figure from being stored in memory

def snapshot_fromdata(loadname, time, sys_type):
    '''
    '''
    
    # plot charge and spin vs site
    if(sys_type == "STT"):
        obs_strs = ["occ_","sz_","Sdz_","pur_","conc_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $","$ \langle S_{d}^{z} \\rangle $","$|\mathbf{S}_d|$","$C_{d,d+1}$"];
        axlines = [ [1.0,0.0],[0.5,0.0,-0.5],[0.5,0.0,-0.5],[0.5,0.0],[1.0,0.0]];
    elif(sys_type == "SIAM"):
        obs_strs = ["occ_", "sz_", "G_"];
        ylabels = ["$\langle n_{j} \\rangle $","$ \langle s_{j}^{z} \\rangle $", "$\pi \langle J_{Imp} \\rangle/V_b$"];
        axlines = [ [1.2,1.0,0.8],[0.1,0.0,-0.1],[1.0,0.0]];
    else:
        raise Exception("System type = "+sys_type+" not supported");

    # plot
    fig, axes = plt.subplots(len(obs_strs),sharex=True);
    for obsi in range(len(obs_strs)):
        y_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}.npy".format(time));
        x_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}.npy".format(time));
        axes[obsi].plot(x_js,y_js,color=mycolors[0],marker='o',linewidth=mylinewidth);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(ylabels[obsi]);
        for lineval in axlines[obsi]:
            axes[obsi].axhline(lineval,color="gray",linestyle="dashed");
    axes[-1].set_xlabel("$j(d)$");
    axes[-1].legend(title = "Time = {:.2f}$\hbar/t_l$".format(time));
    axes[0].set_title( open(loadname+"_arrays/"+obs_strs[0]+"title.txt","r").read().splitlines()[0][1:]);
    plt.tight_layout();
    plt.show();

