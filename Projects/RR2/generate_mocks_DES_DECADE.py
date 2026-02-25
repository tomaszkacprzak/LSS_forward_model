import LSS_forward_model
from LSS_forward_model.cosmology import *
from LSS_forward_model.lensing import *
from LSS_forward_model.maps import *
from LSS_forward_model.halos import *
from LSS_forward_model.tsz import *
from LSS_forward_model.theory import *
import os
import pandas as pd
import numpy as np
import healpy as hp
from cosmology import Cosmology
import astropy.io.fits as fits
import copy
import glass
import pyccl as ccl
from mpi4py import MPI
import BaryonForge as bfn
from pathlib import Path
from LSS_forward_model.theory import LimberTheory


def run(path_simulation, rots, delta_rots ,noise_rels):

    sims_parameters, cosmo_bundle = read_sims_params(path_simulation)

    
    
    
    

    shells_info = recover_shell_info(path_simulation+'/z_values.txt', max_z=49)
    print (noise_rels)
    for noise_rel in noise_rels:
        
        if run_type == 'normal':
            baryons = {
            "enabled": True,
            "max_z_halo_catalog": 1,
            "mass_cut": 13.2,
            "do_tSZ": False,
            "base_params_path": "../../Data/Baryonification_default_parameters.npy",
            "filename_new_params": "sys_baryo_{0}.npy".format(noise_rel)}

            baryon_priors =   {"M_c": (12.5, 15.5, "log10"),   "theta_ej": (3.0, 10.0, "lin"),    "eta": (-2.0, -0.1, "log10")}

        if baryons['enabled']:
            baryons[ "values_to_update"] = draw_params_from_specs(baryon_priors)
            bpar, sys = load_or_save_updated_params(path_simulation,baryons['base_params_path'],baryons['filename_new_params'],baryons['values_to_update'], overwrite = False)
            sims_parameters.update(sys)

            tsz_path = os.path.join(path_simulation, f"tsz_{nside_maps}_{noise_rel}.npy")
            dens_path = os.path.join(path_simulation, f"delta_b_{nside_maps}_{noise_rel}.npy")
        else:
            dens_path = os.path.join(path_simulation, f"delta_{nside_maps}.npy")


    
        # shells --------------------------------------------------------------------
        shells, steps, zeff_array = build_shells(shells_info)
 
        # density field ---------------------
        print ('computing density fields')
        density, label_baryonification = load_and_baryonify_gower_st_shells(
            path_simulation,
            sims_parameters,
            cosmo_bundle,
            baryons,
            nside_maps,
            shells_info,
            shells,
            overwrite_baryonified_shells = False,
            dens_path = dens_path,
            tsz_path = tsz_path
        )


    if do_maps:
        print ('computing lensing fields')
        # shear field ------------------------------------------------------------------------------------------------
        fields = compute_lensing_fields(density, shells, cosmo_bundle['pars_camb'], nside_maps, do_kappa=True, do_shear=True, do_IA=True)
        fields['density'] = density

        '''
        print ('computing theory')
        # Test theory and save power spectrum (optional) ------------------------------------------------------------
        theory = LimberTheory(cosmo_bundle['pars_camb'], lmax=4000, nonlinear="mead")  # "euclidemu" | "mead" | "halofit"
        theory.set_Wshear(np.vstack([nz_RR2['z_rebinned'],nz_shifted]).T)
        Cgg = theory.cl_gg(nonlinear=True)
        kappa_tomo = integrate_field(ngal_glass, fields["kappa"])
        Cls = np.array([(hp.anafast(kappa_tomo[tomo,:])) for tomo in range(len(ngal_glass))])
        ratio = [Cls[tomo, :2000]/(Cgg[tomo, tomo, :2000] * (hp.pixwin(nside_maps)[:2000]**2)) for tomo in tomo_bins ]
        np.save(path_simulation+'theory_checks.npy',{'ratio':ratio,'Cls':Cls[:, :2000],'theory':Cgg[:, :, :2000]})
        '''

        
        
        
        
        for rot in rots:  
            for delta_rot in delta_rots:
        
                # setup IA parameters ---------------------------------------------------------------------------
                IA_parameters_path = os.path.join(path_simulation, f"IA_params_{nside_maps}_{rot}_{delta_rot}.npy")
                if os.path.exists(IA_parameters_path):
                    mute_params = np.load(IA_parameters_path,allow_pickle=True).item()
                    sims_parameters['A_IA'] = copy.deepcopy(mute_params['A_IA'])
                    sims_parameters['eta_IA'] = copy.deepcopy(mute_params['eta_IA'])  
                else:
                    A_IA = np.random.uniform(A0_interval[0],A0_interval[1])
                    eta_IA = np.random.uniform(eta_interval[0],eta_interval[1])
                    sims_parameters['A_IA'] = A_IA 
                    sims_parameters['eta_IA'] = eta_IA  
                    np.save(IA_parameters_path,{'A_IA':A_IA,'eta_IA':eta_IA})


                for experiment in experiments:
                            
                    if experiment == 'DESY6':
                        dz_mean = [0,0,0,0]
                        dz_spread = [0.,0.,0.,0.]
                        dm_mean = 1.+np.array([-0.00343755,  0.0064513 ,  0.01591432,  0.00162992])
                        dm_spread = [0.00296,0.00421,0.00428,0.00462]


                    if experiment == 'SGC':
                        dz_mean = [0,0,0,0]
                        dz_spread = [0.016,0.014,0.010,0.0116]
                        dm_mean = 1.+np.array([-1.33,-2.26,-3.67,-5.72])*0.01
                        dm_spread = [0.00472,0.004657,0.00697,0.00804]

                    if experiment == 'NGC':
                        dz_mean = [0,0,0,0]
                        dz_spread = [0.016,0.0139,0.0101,0.0117]
                        dm_mean = 1.+np.array([-0.92,-1.9,4.0,-3.73])*0.01
                        dm_spread = [0.00296,0.00421,0.00428,0.00462]
                    else:
                        dz_mean = [0.]
                        dz_spread = [0]
                        dm_mean = [1.]
                        dm_spread = [0.]
       
        
        
        
                    # nuisance parameters --------------------------------------------------------
                    dz = np.random.normal(dz_mean,dz_spread)
                    dm = np.random.normal(dm_mean,dm_spread)    
                    bias_sc = [np.random.uniform(bias_SC_interval[0],bias_SC_interval[1]) for tomo in range(len(dz_mean))]


                    sims_parameters['dz'] = dz
                    sims_parameters['dm'] = dm
                    sims_parameters['bias_sc'] = bias_sc
                            
                           
                    if experiment =='DESY6':
                        SC_corrections = np.load('../../Data/SC_DES.npy',allow_pickle =True).item()
                        if (run_type == 'covariance') or  (run_type == 'derivatives'):
                            nz_RR2 = np.load('/global/cfs/cdirs/m5099/DESY3/nz_DESY6.npy',allow_pickle=True).item()
                        else:
                            m = np.load('/global/cfs/cdirs/m5099/DESY3/20k_m_nz_realizations.npz')
                            idx_rel = np.random.randint(0,20000,1)[0]
                            sims_parameters['dm'] = 1.+m['ms'][:,idx_rel]
                            nz_RR2 = dict()
                            nz_RR2['z_rebinned'] = np.arange(0,3,0.01)
                            nz_RR2['nz_rebinned'] = m['nzs'][idx_rel]
                            nz_RR2['nz_rebinned'][:,-1] = 0.
                            sims_parameters['idx_rel_nz'] = idx_rel

                    elif experiment =='NGC':
                        SC_corrections = np.load('../../Data/SC_NGC.npy',allow_pickle =True).item()
                        nz_RR2 = np.load('/global/cfs/cdirs/m5099/DESY3/nz_NGC.npy',allow_pickle=True).item()

                    elif experiment =='SGC':
                        SC_corrections = np.load('../../Data/SC_SGC.npy',allow_pickle =True).item()
                        nz_RR2 = np.load('/global/cfs/cdirs/m5099/DESY3/nz_SGC.npy',allow_pickle=True).item()
                    else:
                        print ('no actual data selected')
                        nz_RR2 = dict()
                        nz_RR2['z_rebinned'] =np.linspace(0,6,300)
                        nz_RR2['nz_rebinned'] = np.array([np.ones(300)]) 


                    
                    nz_shifted, shells, steps, zeff_glass, ngal_glass = apply_nz_shifts_and_build_shells(
                        z_rebinned=nz_RR2['z_rebinned'],
                        nz_all=nz_RR2['nz_rebinned'],
                        dz_values=sims_parameters["dz"],
                        shells_info=shells_info,
                    )


                    # ------------------------------------------------------------------------------------------------   
                    if baryons['enabled']:
                        label_baryonification = 'baryonified_{0}'.format(noise_rel)
                        path_maps_gower = str(path_simulation)+'/maps_Gower_baryonified_{0}_{1}_{2}_{3}.npy'.format(rot,delta_rot,noise_rel,experiment)
                    else:
                        label_baryonification = 'normal'
                        path_maps_gower = str(path_simulation)+'/maps_Gower_{0}_{1}_{2}_{3}.npy'.format(rot,delta_rot,noise_rel,experiment)


                    if not os.path.exists(path_maps_gower):
                        sims_parameters['rot'] = copy.deepcopy(rot)
                        sims_parameters['delta_rot'] = copy.deepcopy(delta_rot)
                        print ('making maps - ',path_maps_gower)
                        # make RR2 mocks


                        if experiment == 'NGC':
                            path_data_cats ='/global/cfs/cdirs/m5099/DESY3/DECADE_NGC.npy'
                        elif experiment == 'SGC':
                            path_data_cats='/global/cfs/cdirs/m5099/DESY3/DECADE_SGC.npy'
                        elif experiment == 'DESY6':
                            path_data_cats='/global/cfs/cdirs/m5099/DESY3/DESY6.npy'



                        cats_Euclid  = np.load(path_data_cats,allow_pickle=True).item()
                        maps_Gower_WL,_ = make_WL_sample(ngal_glass, zeff_glass, cosmo_bundle, sims_parameters, nside_maps, fields, cats_Euclid, SC_corrections = SC_corrections, do_catalog = False, include_SC = True,compact_savings = True)


                        # save mock
                        maps_Gower_WL['sims_parameters'] = copy.deepcopy(sims_parameters)
                        np.save(path_maps_gower,maps_Gower_WL)
                print ('DONE making maps - ',path_maps_gower)



if __name__ == '__main__':
    run_type = 'normal'
    do_maps = True
    nside_maps = 1024
    tomo_bins = [0,1,2,3]
    
    experiments = ['DESY6','NGC', 'SGC']

    ########################################################################################################################################
    # covariance run ------------------------------------------------------------------------------------------------------------------------

    if run_type == 'covariance':


        delta_rot_ = [0]
        dz_spread = [0,0,0,0]
        dm_spread = [0,0,0,0]
        A0_interval  = [0,0]
        eta_interval = [0,0]
        bias_SC_interval = [1,1]
        
        baryons = {
                "enabled": False,
                "max_z_halo_catalog": 1.5,
                "mass_cut": 13,
                "do_tSZ": False,
                "base_params_path": "../Data/Baryonification_wl_tsz_flamingo_parameters.npy",
                "filename_new_params": "sys_baryo_0.npy",
                "values_to_update":  None, # or: {'Mc': 10**13,'theta_ej' : 4.} or draw_params_from_specs( {"M_c": (12.5, 15.5, "log10"),   "theta_ej": (3.0, 10.0, "lin"),    "eta": (-2.0, -0.1, "log10")} )
        }
    
        
        BASE = Path("/global/cfs/cdirs/m5099/GowerSt2/Fiducial/")
        TARGET = "particles_100_4096.parquet"
        have = sorted(p for p in BASE.glob("*_big/") if (p / TARGET).is_file())
        missing = sorted(set(BASE.glob("*_big/")) - set(have))




        done = 0 
        runs = []

        rots = [0,1,2,3]
        noise_rels =  [0]
        delta_rots = [0]
        
        
        for path in have:
            missing_any = False
            delta_exist = True
            for rot in rots:
                for noise_rel in noise_rels:
                    for delta_rot in delta_rots:
                    
                        if baryons['enabled']:
                            label_baryonification = 'baryonified_{0}'.format(noise_rel)
                            path_maps_gower = str(path)+'/maps_Gower_baryonified_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                        else:
                            label_baryonification = 'normal'
                            path_maps_gower = str(path)+'/maps_Gower_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                        if not os.path.exists(path_maps_gower):
                            if  os.path.exists(str(path)+'/delta_1024.npy'):
                                runs.append([str(path)+'/',rots,delta_rots, noise_rels])
               
               
                                missing_any = True
                                break
                            else:
                                delta_exist = False
                    if missing_any: break
                if missing_any: break
        
        
                                
            if not missing_any:
                if delta_exist:
                    done+=1    
    
        print ('RUNSTODO: ',len(runs),'RUNS DONE: ',done)
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    
        # Start with run_count = 0, but each process handles tasks based on rank
        run_count = rank 
    
        while run_count < len(runs):
       #     try:
                path, rots, delta_rots, noise_rels = runs[run_count]
                run(path, rots, delta_rots, noise_rels)
                run_count += size
       #     except:
        #        pass
        comm.Barrier()



        




    elif run_type == 'derivatives':
        
        
        delta_rot_ = [0]
        dz_spread = [0,0,0,0]
        dm_spread = [0,0,0,0]
        A0_interval  = [0,0]
        eta_interval = [0,0]
        bias_SC_interval = [1,1]

        
        
        baryons = {
                "enabled": False,
                "max_z_halo_catalog": 1.5,
                "mass_cut": 13,
                "do_tSZ": False,
                "base_params_path": "../Data/Baryonification_wl_tsz_flamingo_parameters.npy",
                "filename_new_params": "sys_baryo_0.npy",
                "values_to_update":  None, # or: {'Mc': 10**13,'theta_ej' : 4.} or draw_params_from_specs( {"M_c": (12.5, 15.5, "log10"),   "theta_ej": (3.0, 10.0, "lin"),    "eta": (-2.0, -0.1, "log10")} )
        }
    

        done = 0 
        runs = []
        import glob
        ff = glob.glob("/global/cfs/cdirs/m5099/GowerSt2/derivatives/*")
        for ff_ in ff:
            BASE = Path(ff_+'/')
            TARGET = "particles_100_4096.parquet"
            have = sorted(p for p in BASE.glob("*/") if (p / TARGET).is_file())
            missing = sorted(set(BASE.glob("*/")) - set(have))
        
            do_maps = False
            for path in have:
                path_maps_gower = str(path)+'/maps_Gower_{0}_{1}_{2}.npy'.format(0,0,0)
                if not os.path.exists(str(path)+'/delta_1024.npy'):
                   runs.append([str(path)+'/',0,0, 0, path_maps_gower])
                else:
                    done+=1
    
        print ('')
        print ('runs done [derivatives] : ',done, 'TODO: ',len(runs))
        print ('')
    
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    
        # Start with run_count = 0, but each process handles tasks based on rank
        run_count = rank 
    
        while run_count < len(runs):
            path, rot, delta_rot, noise_rel, path_maps_gower = runs[run_count]
            run(path, rot, delta_rot, noise_rel, path_maps_gower)
            run_count += size
        comm.Barrier()



        # make paired mocks -----------------------------
        dp = ['runsdom_p','runsdom_p2','runsdsigma8_p','runsdsigma8_p2']
        dm = ['runsdom_m','runsdom_m2','runsdsigma8_m','runsdsigma8_m2']
        n_runs = 20
        
        
        
        do_maps = True
        
        
        rots = [0,1,2,3]
        noise_rels =  [0]
        delta_rots = [0]
        
        done = 0
        
        runs = []       
        for i in range(len(dp)):
            for run_ in range(n_runs):
                missing_any = False
        
                for rot in rots:
                    for delta_rot in delta_rots:
                        for noise_rel in noise_rels:
                        
                            path_p = '/global/cfs/cdirs/m5099/GowerSt2/derivatives/{0}/run{1:03d}'.format(dp[i],run_)
                            path_m = '/global/cfs/cdirs/m5099/GowerSt2/derivatives/{0}/run{1:03d}'.format(dm[i],run_)
                            
        
        
                            if baryons['enabled']:
                                label_baryonification = 'baryonified_{0}'.format(noise_rel)
                                path_maps_gowerp = str(path_p)+'/maps_Gower_baryonified_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                                path_maps_gowerm = str(path_m)+'/maps_Gower_baryonified_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                            else:
                                label_baryonification = 'normal'
                                path_maps_gowerp = str(path_p)+'/maps_Gower_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                                path_maps_gowerm = str(path_m)+'/maps_Gower_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                            if not os.path.exists(path_maps_gowerm):
                        
                                runs.append([str(path_p)+'/',str(path_m)+'/',rots,delta_rots, noise_rels])
        
           
                                missing_any = True
                                break
                            else:
                                delta_exist = False
                        if missing_any: break
                    if missing_any: break
              
        
                if not missing_any:
                    done+=1    
            
        print ('')
        print ('runs done [derivatives] : ',done, 'TODO: ',len(runs))
        print ('')    

        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    
        # Start with run_count = 0, but each process handles tasks based on rank
        run_count = rank 
    
        while run_count < len(runs):

            pathp,pathm, rots, delta_rots, noise_rels, = runs[run_count]

            print('doing ---',pathp)
            print('doing ---',pathm)
            run(pathp, rots, delta_rots, noise_rels)
            run(pathm, rots, delta_rots, noise_rels)

            # adjust shape noise (copy from + )
            for rot in rots:
                for noise_rel in noise_rels:
                    for delta_rot in delta_rots:
                        if baryons['enabled']:    
                            label_baryonification = 'baryonified_{0}'.format(noise_rel)
                            path_maps_gowerp = str(pathp)+'/maps_Gower_baryonified_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                            path_maps_gowerm = str(pathm)+'/maps_Gower_baryonified_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                        else:
                            label_baryonification = 'normal'
                            path_maps_gowerp = str(pathp)+'/maps_Gower_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                            path_maps_gowerm = str(pathm)+'/maps_Gower_{0}_{1}_{2}.npy'.format(rot,delta_rot,noise_rel)
                        
                        pp = np.load(path_maps_gowerp,allow_pickle=True).item()
                        mm = np.load(path_maps_gowerm,allow_pickle=True).item()
                        print (path_maps_gowerp)
                        for k in pp.keys():
                            if k!='sims_parameters':
                                print (pp[k].keys())
                                
                                n1 = pp[k]['e1'] - pp[k]['g1_map']
                                n2 = pp[k]['e2'] - pp[k]['g2_map']
                                
                                mm[k]['e1'] = mm[k]['g1_map'] + n1
                                mm[k]['e2'] = mm[k]['g2_map'] + n2
                                
                                mm[k]['e1n'] = pp[k]['e1n']
                                mm[k]['e2n'] = pp[k]['e2n']
            
                        np.save(path_maps_gowerm,mm)

            run_count += size
        comm.Barrier()

    else:
        
        ########################################################################################################################################
        # general run ------------------------------------------------------------------------------------------------------------------------

        A0_interval  = [-2.5,2.5]
        eta_interval = [-2.5,2.5]
        
        bias_SC_interval = [0.5,1.5]
        
     
        
    
        BASE = Path("/share/rcifdata/mgatti/mocks/runsV/")
        TARGET = "run.00100.lightcone.npy"
                            
        BASE = Path("/global/cfs/cdirs/m5099/GowerSt2/runsU")
        TARGET = "particles_100_4096.parquet"

    
        BASE = Path("/global/cfs/cdirs/m5099/GowerSt2/Lorne_runs/runsU/")
        TARGET = "run.00100.lightcone.npy"
                            
                            
        have = sorted(p for p in BASE.glob("run*/") if (p / TARGET).is_file())
        missing = sorted(set(BASE.glob("run*/")) - set(have))
    
        done = 0 
        
        runs = []

        rots = [0,1,2,3]
        noise_rels =  [0]
        delta_rots = [0]
        
        # this make sure you have at least one density real + halo catalog 
        for path in have:
            if not os.path.exists(str(path)+'/delta_b_1024_0.npy'):
                runs.append([str(path)+'/',rots,delta_rots, noise_rels])
            else:
                done += 1

        #'''
        for path in have:
            missing_any = False
            delta_exist = True
            for rot in rots:
                for noise_rel in noise_rels:
                    for delta_rot in delta_rots:
                        #if True:
                        if True:
                            label_baryonification = 'baryonified_{0}'.format(noise_rel)
                            path_maps_gower = str(path)+'/maps_Gower_baryonified_{0}_{1}_{2}_{3}.npy'.format(rot,delta_rot,noise_rel,experiments[-1])
                        if not os.path.exists(path_maps_gower):
                            if  os.path.exists(str(path)+'/delta_b_1024_{0}.npy'.format(noise_rel)):
                                runs.append([str(path)+'/',rots,delta_rots, noise_rels])
               
                                missing_any = True
                                break
                            else:
                                delta_exist = False
                    if missing_any: break
                if missing_any: break
        
        
                                
            if not missing_any:
                if delta_exist:
                    done+=1    
    	#'''
        print ('RUNSTODO: ',len(runs),'RUNS DONE: ',done)
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    
        # Start with run_count = 0, but each process handles tasks based on rank
        run_count = rank 
    
        while run_count < len(runs):
       #     try:
                path, rots, delta_rots, noise_rels = runs[run_count]
                run(path, rots, delta_rots, noise_rels)
                run_count += size
       #     except:
        #        pass
        comm.Barrier()
    







#module load python; source activate pyccl_env;  python  generate_mocks.py
#module load python; source activate pyccl_env; srun --nodes=4 --tasks-per-node=4 python  generate_mocks.py
#module load python; source activate pyccl_env; srun --nodes=4 --tasks-per-node=1 python  generate_mocks.py
