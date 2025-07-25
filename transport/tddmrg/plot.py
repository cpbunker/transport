'''
'''

import numpy as np
import matplotlib.pyplot as plt
from transport import tddmrg, tdfci

# fig standardizing
mylinewidth = 3.0;

def vs_site(params_dict, js,psi,eris_or_driver,which_obs, is_impurity, block, prefactor):
    '''
    '''
    if(not isinstance(prefactor, float)): raise TypeError;
    obs_funcs = {"occ_":tddmrg.get_occ, "sz_":tddmrg.get_sz,"Sdz_":tddmrg.get_Sd_mu, 
                 "pur_":tddmrg.purity_wrapper,
                 "G_":tddmrg.conductance_wrapper, "J_":tddmrg.pcurrent_wrapper,
                 "S2_":tddmrg.S2_wrapper, "MI_":tddmrg.mutual_info_wrapper,
                 "nB_":tddmrg.band_wrapper};
    if(block): compute_func = tddmrg.compute_obs;
    else: compute_func = tdfci.compute_obs;

    # site array
    vals = np.zeros_like(js,dtype=float)
    for ji in range(len(js)):
        if(which_obs in ["S2_", "MI_"]): # WRAPPED operators on MULTIPLE sites -> all should have impurity/not impurity functionality
            if(ji!=len(js)-1):
                vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,[js[ji],js[ji+1]],is_impurity,block);
            else: vals[ji] = np.nan; # since op = op(d,d+1), we cannot compute for the final site.
        elif(which_obs in ["pur_","G_","J_"]): # WRAPPED operators
            vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,js[ji],block);
        elif(which_obs in ["nB_"]): # WRAPPED operator w/ unique call signature
            vals[ji] = obs_funcs[which_obs](psi,eris_or_driver,js[ji],params_dict, block);
        else: # simple operators
            op = obs_funcs[which_obs](eris_or_driver,js[ji],block);
            ret = compute_func(psi, op, eris_or_driver);
            if(np.imag(ret) > 1e-10): print(ret); raise ValueError;
            vals[ji] = np.real(ret);

    return js, prefactor*vals;

def snapshot_bench(psi_mps, driver_inst, params_dict, savename, time, block=True):
    '''
    '''
    if(not isinstance(block, bool)): raise TypeError;

    # unpack
    sys_type = params_dict["sys_type"];
    NL, NR = params_dict["NL"], params_dict["NR"];

    # is Rice-Mele ?
    if(params_dict["sys_type"] in ["STT_RM","SIAM_RM", "SIETS_RM"]):
        assert("v" in params_dict.keys());
        is_RM = True;
        tlstring = "|v|"
    else:
        is_RM = False;
        tlstring = "t_l";

    # Sys_type tells what observables we want to compute and how to compute them
    if(sys_type in ["STT", "STT_RM"]):
        is_impurity = True; # bool that tells us whether custom operators (Z, P, M) defining localized spins are defined
        NFM,  Ne = params_dict["NFM"], params_dict["Ne"];
        title_str = "$J_{sd} = "+"{:.2f}".format(params_dict["Jsd"])+tlstring+"$, $N_e = {:.0f}$".format(Ne)+", $N_\mathrm{conf} ="+"{:.0f}$".format(params_dict["Nconf"]);
        if(is_RM): title_str = "$w ={:.2f}".format(params_dict["w"])+tlstring+"$, "+title_str;
        # plot charge and spin vs site
        obs_strs = ["occ_","sz_","Sdz_", "J_", "MI_"];
        if(not is_RM and params_dict["NFM"]< 2): # not enough impurities to do S2_ or MI_
            raise NotImplementedError;
    elif(sys_type in ["SIAM", "SIAM_RM"]):
        is_impurity = False;
        NFM, Ne = 1, NL+1+NR;
        if("Ne_override" in params_dict.keys()): Ne = params_dict["Ne_override"];
        title_str = "$U =${:.2f}$, t_h =$ {:.2f}$, V_g =${:.2f}$, V_b =${:.2f}".format(params_dict["U"], params_dict["th"], params_dict["Vg"], params_dict["Vb"]);
        obs_strs = ["occ_", "sz_", "G_"];
    elif(sys_type in ["SIETS", "SIETS_RM"]):
        is_impurity = True; # bool that tells us whether custom operators (Z, P, M) defining localized spins are defined
        NFM, Ne = params_dict["NFM"], params_dict["Ne"];
        title_str = "$J_{sd} = $"+"{:.2f}, ".format(params_dict["Jsd"])+"$t_h = ${:.2f}, $V_g =${:.2f}$, V_b =${:.2f}".format(params_dict["th"], params_dict["Vg"], params_dict["Vb"]);
        obs_strs = ["occ_", "sz_", "Sdz_", "G_", "MI_"];
    else:
        raise Exception("System type = "+sys_type+" not supported");

    # system geometry
    Nsites = NL+NFM+NR; # number of j sites in 1D chain
    if("MSQ_spacer" in params_dict.keys()): # NFM has MSQs on either end only
        centrals = np.array([NL,NL+NFM-1]);
    else: # NFM full of MSQs
        centrals = np.arange(NL,NL+NFM);
    alls = np.arange(Nsites);
    
    # plot
    fig, axes = plt.subplots(len(obs_strs));
    if(psi_mps is not None): # with dmrg
        for obsi in range(len(obs_strs)):

            # what sites to find <operator> over
            if(obs_strs[obsi] in ["Sdz_","pur_","S2_","MI_"]): 
                js_pass = centrals; # operators on impurity sites only
            #elif(obs_strs[obsi] in ["nB_"]):
            #js_pass = np.arange(Nsites//2);
            else: js_pass = alls;

            # what prefactors to apply to <operator>
            if(obs_strs[obsi] in ["G_"]): 
                assert(sys_type in ["SIAM", "SIAM_RM","SIETS","SIETS_RM"]);
                prefactor = np.pi*params_dict["th"]/params_dict["Vb"]; # prefactor needed for G/G0
                js_pass = np.append(centrals, [centrals[-1]+1]); # one extra
            elif(obs_strs[obsi] in ["J_"]):
                js_pass = [params_dict["Nconf"], NL, NL+NFM]; 
                if(sys_type in ["SIAM", "SIAM_RM", "SIETS", "SIETS_RM"]):
                    prefactor = params_dict["th"]; # prefactor needed for J
                elif(sys_type in ["STT_RM"]):
                    prefactor = abs(params_dict["w"]);
                elif(sys_type in ["STT"]): 
                    prefactor = params_dict["tl"];
                else: raise NotImplementedError("sys_type =",sys_type);
            else: prefactor = 1.0;

            # find <operator> vs sites
            if(is_RM):
                new_js_pass = [];
                for j in js_pass:
                    new_js_pass.append(2*j);
                    new_js_pass.append(2*j+1);
                js_pass = new_js_pass;
            print(obs_strs[obsi], "\njs = ", js_pass)

            # calculate expectation values
            x_js, y_js = vs_site(params_dict,js_pass,psi_mps,driver_inst,obs_strs[obsi],is_impurity,block,prefactor);
            axes[obsi].plot(x_js,y_js,label="DMRG (te_type="+str(params_dict["te_type"])+", dt= "+str(params_dict["time_step"])+")");
            print("Total <"+obs_strs[obsi]+"> = {:.6f}".format(np.sum(y_js)));            

            # save DMRG data
            if(not params_dict["plot"]):
                np.save(savename+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}".format(time), x_js);
                np.save(savename+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}".format(time), y_js);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(obs_strs[obsi]);
    axes[-1].set_xlabel("$j(d)$");
    axes[-1].legend(title = "Time = {:.2f}".format(time));
    axes[0].set_title(title_str);
    plt.tight_layout();
    if(params_dict["plot"]):
        plt.show();
    else:
        np.savetxt(savename+"_arrays/"+obs_strs[0]+"title.txt",[0.0], header=title_str);
    plt.close(); # keeps figure from being stored in memory

def snapshot_fromdata(loadname, time, sys_type):
    '''
    '''
    
    # plot charge and spin vs site
    if(sys_type in ["STT", "STT_RM"]):
        obs_strs = ["occ_", "sz_", "Sdz_", "MI_"];
    elif(sys_type in ["SIAM", "SIAM_RM"]):
        obs_strs = ["occ_", "sz_", "G_"];
    elif(sys_type in ["SIETS", "SIETS_RM"]):
        obs_strs = ["occ_", "sz_", "Sdz_", "G_", "MI_"];
    else:
        raise Exception("System type = "+sys_type+" not supported");

    # plot
    fig, axes = plt.subplots(len(obs_strs),sharex=True);
    for obsi in range(len(obs_strs)):
        y_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"yjs_time{:.2f}.npy".format(time));
        x_js = np.load(loadname+"_arrays/"+obs_strs[obsi]+"xjs_time{:.2f}.npy".format(time));
        axes[obsi].plot(x_js,y_js);

    #format
    for obsi in range(len(obs_strs)):
        axes[obsi].set_ylabel(obs_strs[obsi]);
    axes[-1].set_xlabel("$j(d)$");
    axes[-1].legend(title = "Time = {:.2f}".format(time));
    axes[0].set_title( open(loadname+"_arrays/"+obs_strs[0]+"title.txt","r").read().splitlines()[0][1:]);
    plt.tight_layout();
    plt.show();

