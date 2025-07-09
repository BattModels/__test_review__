import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import torch.jit
import os
import pandas as pd
import matplotlib.pyplot as plt
from .energy import sampling, convex_hull, CommonTangent, calculate_S_config_total, calculate_S_vib_total_PDOS, _H_vib_PDOS_quaderature_model, _PDOS_evaluator, _calculate_mole_of_atoms_from_PDOS, legendre_poly_recurrence, chebyshev_poly_recurrence, solve_Theta_max
from .utils import _convert_JMCA_Tdsdx_data_to_dsdx
import copy

# import multiprocessing
# num_cores = multiprocessing.cpu_count()
# torch.set_num_threads(num_cores)


# fit pure substance Cp to get pure substance base model
def fit_pure_substance_Cp(Ts = [220,240,260,280,300],
                        Cps_measured = [0,0,0,0,0],
                        n_in_PDOS_i = 6,
                        learning_rate = 0.01, 
                        total_training_epochs = 8000,
                        g_0_bounds = [10.0,10.1],
                        g_i_bounds = [-0.1,0.1],
                        loss_threshold = 0.01,
                        quaderature_points=10,
                        style = "Legendre",
                        pretrained_g_i_values = None,
                        is_normalization = False,
                        ):
    # convert data
    Ts = torch.from_numpy(Ts)
    Cps_measured = torch.from_numpy(Cps_measured)
    # initialize g_i_list 
    g_i_list = []
    if pretrained_g_i_values != None:
        # load pretrained value
        for i in range(0, n_in_PDOS_i):
            g_i_now = nn.Parameter( torch.from_numpy(np.array([pretrained_g_i_values[i]],dtype="float32")) )
            g_i_list.append(g_i_now)
    else:
        g_i_start = np.random.uniform(low=g_0_bounds[0], high=g_0_bounds[1])
        g_i_list.append(nn.Parameter( torch.from_numpy(np.array([g_i_start],dtype="float32")) )) 
        for _ in range(1, n_in_PDOS_i):
            g_i_start = np.random.uniform(low=g_i_bounds[0], high=g_i_bounds[1])
            g_i_list.append(nn.Parameter( torch.from_numpy(np.array([g_i_start],dtype="float32")) )) 

    # init optimizer
    optimizer = optim.Adam([{'params': g_i_list, 'lr': learning_rate},] )
    # init log
    with open("log_Cp",'w') as fout:
        fout.write("")
    # train
    loss = 9999.9 # init total loss
    epoch = -1
    while loss > loss_threshold and epoch < total_training_epochs:
        # clean grad info
        optimizer.zero_grad()
        # use current params to calculate predicted phase boundary
        epoch = epoch + 1
        ## init collocation loss components 
        loss_cp_collocation = 0.0
        for i in range(0, len(Ts)):
            Cp_measured_now = Cps_measured[i]
            T_now = Ts[i]
            T_now = T_now.requires_grad_()
            Theta_max = solve_Theta_max(g_i_list = g_i_list, is_x_dependent = False, style = style)
            H_vib_now = _H_vib_PDOS_quaderature_model(g_i_list = g_i_list, Theta_max=Theta_max, quaderature_points=quaderature_points, is_x_dependent = False, T = T_now, style = style)
            Cp_calculated_now = autograd.grad(outputs=H_vib_now, inputs=T_now, create_graph=True)[0]
            loss_cp_collocation = loss_cp_collocation + (Cp_calculated_now - Cp_measured_now)**2
        loss_cp_collocation = loss_cp_collocation / len(Ts)
        # make sure PDOS is positive everywhere
        _Theta_max = Theta_max.detach().numpy()[0]
        Theta_list = torch.from_numpy(np.linspace(0,_Theta_max,100).astype("float32"))
        g_omega_list = torch.zeros(len(Theta_list))
        for i in range(0, len(Theta_list)):
            Theta_now = Theta_list[i]
            g_omega_now = _PDOS_evaluator(Theta=Theta_now, Theta_max = Theta_max, g_i_list = g_i_list, is_x_dependent = False, style = style)
            g_omega_list[i] = g_omega_now
        # PDOS should be positive everywhere. Otherwise, punish it
        mask_less_than_0 = (g_omega_list <= 0).int()
        loss_PDOS_greater_than_0 = torch.sum((g_omega_list*mask_less_than_0)**2)*100000000
        ## make sure integration of PDOS gives 1, i.e. 1 mole of atoms. Otherwise, punish it
        if is_normalization:
            N_now = _calculate_mole_of_atoms_from_PDOS(g_i_list = g_i_list, Theta_max=Theta_max, quaderature_points=quaderature_points, is_x_dependent = False)
            loss_PDOS_1_mole = (N_now-1)**2 *100000000
        else:
            loss_PDOS_1_mole = 0.0
        # define loss function
        loss = loss_cp_collocation + loss_PDOS_greater_than_0 + loss_PDOS_1_mole 
        # backward
        loss.backward()
        optimizer.step()
        # print output
        output_txt = "Epoch %3d  Loss %.4f Cp_col %.4f PDOS>0 %.4f PDOS_1_mole %.4f    " %(epoch, loss, loss_cp_collocation, loss_PDOS_greater_than_0, loss_PDOS_1_mole)
        for i in range(0, len(g_i_list)):
            output_txt = output_txt + "g_%d %.8f "%(i, g_i_list[i].item())
        output_txt = output_txt + "      "
        print(output_txt)
        with open("log_Cp",'a') as fout:
            fout.write(output_txt)
            fout.write("\n")
    return g_i_list








# loss function 
def collocation_loss_all_pts(mu, x, T, phase_boundarys_fixed_point, GibbsFunction, total_params_list, style = "Legendre", quaderature_points = 20, alpha_miscibility = 1.0):
    """
    Calculate the collocation points loss for all datapoints (that way we don't need hessian loss and common tangent loss, everything is converted into collocation loss)
    mu is the measured OCV data times Farady constant
    x is the measured SOC data
    T: temperature
    phase_boundarys_fixed_point is the list of starting and end point of miscibility gap(s)
    GibbsFunction is the Gibbs free energy landscape (must be PDOS version)
    total_params_list = [params_list, S_config_params_list, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM, Theta_max_Li]
    alpha_miscibility: weight of miscibility loss (must be 1.0, deprecated)
    """
    # see if x is in any gaps
    def _is_inside_gaps(_x, _gaps_list):
        _is_inside = False
        _index = -99999
        if len(_gaps_list) == 0:
            return False, -99999
        for i in range(0, len(_gaps_list)):
            if _x >= _gaps_list[i][0] and _x <= _gaps_list[i][1]:
                _is_inside = True
                _index = i
                break
        return _is_inside, _index
    # calculate loss
    # do them in parallel 
    def worker_function(x_mu_combination, T, total_params_list, phase_boundarys_fixed_point, GibbsFunction, inside_gap_func, style = 'Legendre', quaderature_points = 20):
        x_now = x_mu_combination[0]
        mu_now = x_mu_combination[1]
        is_inside, index = inside_gap_func(x_now, phase_boundarys_fixed_point)
        if is_inside == False:
            # outside miscibility gap 
            x_now = x_now.requires_grad_()
            g_now = GibbsFunction(x_now, T, total_params_list,  style = style, quaderature_points = quaderature_points)
            mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
            return mu_pred_now
        else: 
            # inside miscibility gap
            x_alpha = phase_boundarys_fixed_point[index][0]
            x_beta = phase_boundarys_fixed_point[index][1]
            ct_pred = (GibbsFunction(x_alpha, T, total_params_list,  style = style, quaderature_points = quaderature_points) - GibbsFunction(x_beta, T, total_params_list,  style = style, quaderature_points = quaderature_points))/(x_alpha - x_beta) 
            if torch.isnan(ct_pred):
                print("Common tangent is NaN")
                x_alpha = 99999.9
                x_beta = -99999.9
            if x_alpha > x_beta:
                print("Error in phase equilibrium boundary, x_left %.4f larger than x_right %.4f. If Hessian loss is not 0, it's fine. Otherwise check code carefully!" %(x_alpha, x_beta))
                x_alpha = 99999.9
                x_beta = -99999.9
            if torch.isnan(ct_pred):
                print("Warning: skipped for loss calculation at a filling fraction x")
                ct_pred = None
            return ct_pred
    # exec in parallel 
    x_mu_list_inputs = torch.stack((x, mu), dim=1)  # just like zip them together
    futures = [torch.jit.fork(worker_function, x_mu_combination, T, total_params_list, phase_boundarys_fixed_point, GibbsFunction, _is_inside_gaps, style, quaderature_points) for x_mu_combination in x_mu_list_inputs]
    mu_with_None_list = [torch.jit.wait(f) for f in futures]
    ## calculate loss, but remember to exclude None
    loss_ = 0.0
    n_count = 0
    for i in range(0, len(mu_with_None_list)):
        mu_calculated_now = mu_with_None_list[i]
        mu_now = mu[i]
        if mu_calculated_now == None:
            pass
        else:
            loss_ = loss_ + ((mu_calculated_now-mu_now)/(8.314*T))**2 
            n_count = n_count + 1
  
    return loss_/n_count


# entropy loss calculation
def calc_loss_entropy(x_measured, dsdx_measured, T_dsdx, \
                      S_config_params_list, n_list,
                      g_ij_list_LixHM,  \
                      g_i_list_HM, Theta_max_HM, \
                      g_i_list_LiHM, Theta_max_LiHM, \
                      g_i_list_Li, Theta_max_Li, \
                      quaderature_points=20, style = "Legendre",
                      is_g_ij_collapse_to_g_i_at_end_points = True,):
    # init loss components
    loss = 0.0 # init total loss
    dsdx_calculated = torch.zeros(len(dsdx_measured))
    ## calculate dsdx for collocation loss
    def worker_function_collocation(x, T_dsdx, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM, Theta_max_Li, S_config_params_list, quaderature_points = 20, style="Legendre"):
        Theta_max_LixHM = solve_Theta_max(g_ij_matrix = g_ij_list_LixHM, is_x_dependent = True, x = x, style = style)
        x = x.requires_grad_()
        s_vib_tot = calculate_S_vib_total_PDOS(x, T_dsdx, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = quaderature_points, style=style)
        s_config_tot, _, _ = calculate_S_config_total(x, S_config_params_list, style = style)
        s_tot = s_vib_tot + s_config_tot
        dsdx_calculated_now = autograd.grad(outputs=s_tot, inputs=x, create_graph=True)[0]
        return dsdx_calculated_now
    # exec in parallel
    futures = [torch.jit.fork(worker_function_collocation, x, T_dsdx, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM, Theta_max_Li, S_config_params_list, quaderature_points, style) for x in x_measured]
    dsdx_calculated_list = [torch.jit.wait(f) for f in futures]
    for i in range(0, len(dsdx_calculated_list)):
        dsdx_calculated[i] = dsdx_calculated_list[i]*1.0

    ## calculate sconfig & its upper bound (i.e. s_ideal)
    x_calculated = np.linspace(0.0001,0.9999,100).astype("float32")
    x_calculated = torch.from_numpy(x_calculated)
    s_calculated = torch.zeros(len(x_calculated))
    s_upper_bound = torch.zeros(len(x_calculated))
    def worker_function_s_calculation(x, S_config_params_list, style = "Legendre"):
        x = x.requires_grad_()
        s_config, _, _ = calculate_S_config_total(x, S_config_params_list, style = style)
        return s_config
    # exec in parallel
    # calculate s_config
    futures = [torch.jit.fork(worker_function_s_calculation, x, S_config_params_list, style) for x in x_calculated]
    s_calculated_list = [torch.jit.wait(f) for f in futures]
    for i in range(0, len(s_calculated_list)):
        s_calculated[i] = s_calculated_list[i]*1.0
    # calculate upper bound, i.e. s_ideal
    futures = [torch.jit.fork(worker_function_s_calculation, x, [0.0], style) for x in x_calculated]
    s_upper_bound_list = [torch.jit.wait(f) for f in futures]
    for i in range(0, len(s_upper_bound_list)):
        s_upper_bound[i] = s_upper_bound_list[i]*1.0
        
    ## s_config should be larger than 0    
    mask_s_lower_bound = (s_calculated <= 0).int()
    loss_s_config_leq_0 = torch.sum((s_calculated*mask_s_lower_bound)**2)*1000000
    ## s_config should be smaller than ideal configurational entropy
    mask_s_upper_bound = (s_calculated >= s_upper_bound).int()
    loss_s_config_geq_upper_bound = torch.sum(((s_calculated-s_upper_bound)*mask_s_upper_bound)**2)*1000000
    ## minimize data loss, i.e. collocation loss
    loss_dsdx = torch.sum((dsdx_calculated-dsdx_measured)**2)

    ## constraints on g_ij:
    ## 1) make sure PDOS is positive everywhere for all Theta and x
    # strategy: sample only at some x points and Theta points
    Theta_ID_sample_list = torch.from_numpy(np.linspace(0, 10, 11).astype("float32"))
    x_sample_list = torch.from_numpy(np.linspace(0.0,1.0,11).astype("float32"))
    g_omega_list = torch.zeros(len(Theta_ID_sample_list)*len(x_sample_list))
    all_combinations_theta_x = torch.cartesian_prod(Theta_ID_sample_list, x_sample_list)
    # helper func
    def worker_function_PDOS_sampling(theta_x, g_ij_matrix, is_x_dependent = True, style = "Legendre"):
        Theta_ID_now = int(theta_x[0])
        x_now = theta_x[1]
        Theta_max = solve_Theta_max(g_ij_matrix = g_ij_list_LixHM, is_x_dependent = True, x = x_now, style = style)
        Theta_sample_list = torch.from_numpy(np.linspace(0,Theta_max.detach().numpy()[0],11).astype("float32"))
        Theta_now = Theta_sample_list[Theta_ID_now]
        g_omega_now = _PDOS_evaluator(Theta_now, x = x_now, g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, is_x_dependent = is_x_dependent, style = style)
        return g_omega_now
    # exec in parallel
    futures = [torch.jit.fork(worker_function_PDOS_sampling, theta_x, g_ij_list_LixHM, True, "Legendre") for theta_x in all_combinations_theta_x]
    g_omega_list_all = [torch.jit.wait(f) for f in futures]
    for i in range(0, len(g_omega_list_all)):
        g_omega_list[i] = g_omega_list_all[i]*1.0
    # PDOS should be positive everywhere. Otherwise, punish it
    mask_less_than_0 = (g_omega_list <= 0).int()
    loss_PDOS_greater_than_0 = torch.sum((g_omega_list*mask_less_than_0)**2)*100000000
    
    ## 2) at x=0 and 1, it collapse to fitted g_i_list_HM and g_i_list_LiHM
    if is_g_ij_collapse_to_g_i_at_end_points:
        n_i_LixHM = len(g_ij_list_LixHM)
        n_j_LixHM = len(g_ij_list_LixHM[0])
        # calculate x=0 
        g_i_x_at_0 = torch.zeros(len(g_i_list_HM))  
        x = torch.tensor([0.0])   
        for i in range(0, n_i_LixHM):  
            _t = 2*x -1
            if style == "Legendre":
                Pn_values = legendre_poly_recurrence(_t, n_j_LixHM-1) 
            elif style == "Chebyshev":
                Pn_values = chebyshev_poly_recurrence(_t, n_j_LixHM-1)
            g_i = 0.0
            for j in range(0, n_j_LixHM):
                g_i = g_i + g_ij_list_LixHM[i][j]*Pn_values[j]
            g_i_x_at_0[i] = g_i
        # calculate x=1
        g_i_x_at_1 = torch.zeros(len(g_i_list_LiHM))  
        x = torch.tensor([1.0])   
        for i in range(0, n_i_LixHM):  
            _t = 2*x -1
            if style == "Legendre":
                Pn_values = legendre_poly_recurrence(_t, n_j_LixHM-1) 
            elif style == "Chebyshev":
                Pn_values = chebyshev_poly_recurrence(_t, n_j_LixHM-1)
            g_i = 0.0
            for j in range(0, n_j_LixHM):
                g_i = g_i + g_ij_list_LixHM[i][j]*Pn_values[j]
            g_i_x_at_1[i] = g_i   
        loss_PDOS_two_side = ( torch.sum((g_i_x_at_0-g_i_list_HM)**2) + torch.sum((g_i_x_at_1-g_i_list_LiHM)**2) )*100000000
    else:
        loss_PDOS_two_side = 0.0
    
    loss = loss_s_config_leq_0 + loss_s_config_geq_upper_bound + loss_dsdx  + loss_PDOS_greater_than_0 + loss_PDOS_two_side
    
    return loss, loss_dsdx, loss_s_config_leq_0, loss_s_config_geq_upper_bound, loss_PDOS_greater_than_0, loss_PDOS_two_side, dsdx_calculated



def calc_loss_Cp_of_end_member(Ts = [220,240,260,280,300],
                        Cps_measured = [0,0,0,0,0],
                        g_ij_matrix = None,
                        quaderature_points=20,
                        style = "Legendre",
                        x = torch.tensor([0.0]),
                        ):
    # import model
    from .energy import _H_vib_PDOS_quaderature_model, _PDOS_evaluator, _calculate_mole_of_atoms_from_PDOS
    # convert data
    Ts = torch.from_numpy(Ts)
    Cps_measured = torch.from_numpy(Cps_measured)
    
    ## init collocation loss components 
    # helper func
    def worker_function_Cp_calc(T_now, g_ij_matrix, x, style = style, quaderature_points=20, is_x_dependent=True):
        Theta_max = solve_Theta_max(g_ij_matrix = g_ij_matrix, is_x_dependent = True, x = x, style = style)
        T_now = T_now.requires_grad_()
        H_vib_now = _H_vib_PDOS_quaderature_model(g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, quaderature_points=quaderature_points, is_x_dependent = True, x = x, T = T_now, style = style)
        N_now = _calculate_mole_of_atoms_from_PDOS(g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, quaderature_points=quaderature_points, is_x_dependent = True, x = x, style = style)
        H_vib_now = H_vib_now/N_now
        Cp_calculated_now = autograd.grad(outputs=H_vib_now, inputs=T_now, create_graph=True)[0]
        return Cp_calculated_now
    # exec in parallel
    loss_cp_collocation = 0.0
    futures = [torch.jit.fork(worker_function_Cp_calc, T_now, g_ij_matrix, x, style, quaderature_points, True) for T_now in Ts]
    Cp_calculated_list_all = [torch.jit.wait(f) for f in futures]
    for i in range(0, len(Cp_calculated_list_all)):
        Cp_calculated_now = Cp_calculated_list_all[i]*1.0
        Cp_measured_now = Cps_measured[i]
        loss_cp_collocation = loss_cp_collocation + (Cp_calculated_now - Cp_measured_now)**2
    
    # ## make sure integration of PDOS gives 1, i.e. 1 mole of atoms. Otherwise, punish it
    # N_now = _calculate_mole_of_atoms_from_PDOS(g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, quaderature_points=quaderature_points, is_x_dependent = True, x = x, style = style)
    # loss_PDOS_1_mole = (N_now-1)**2 *100000000
    loss_PDOS_1_mole = 0.0 # because we solved Theta_max from the constraint that N=1, we don't need to calculate this loss

    # define loss function
    loss = loss_cp_collocation + loss_PDOS_1_mole 
       
    return loss


    


# def write_ocv_functions(params_list, polynomial_style = "R-K", T = 300, outpyfile_name = "fitted_ocv_functions.py"):
#     """
#     T is temperature
#     """
#     if polynomial_style == "Legendre":
#         from .energy import GibbsFE_Legendre as GibbsFE
#     elif polynomial_style == "R-K":
#         from .energy import GibbsFE_RK as GibbsFE
#     elif polynomial_style == "Chebyshev":
#         from .energy import GibbsFE_Chebyshev as GibbsFE

#     # sample the Gibbs free energy landscape
#     print("Temperature is %.4f" %(T))
#     sample = sampling(GibbsFE, params_list, T=T, sampling_id=1, ngrid=99)
#     # give the initial guess of miscibility gap
#     phase_boundarys_init, _ = convex_hull(sample, ngrid=99) 
#     # refinement & calculate loss
#     if phase_boundarys_init != []:
#         # There is at least one phase boundary predicted 
#         phase_boundary_fixed_point = []
#         for phase_boundary_init in phase_boundarys_init:
#             common_tangent = CommonTangent(GibbsFE, params_list, T = T) # init common tangent model
#             phase_boundary_now = phase_boundary_init.requires_grad_()
#             phase_boundary_fixed_point_now = common_tangent(phase_boundary_now) 
#             phase_boundary_fixed_point.append(phase_boundary_fixed_point_now)
#     else:
#         # No boundary find.
#         phase_boundary_fixed_point = []

#     # print detected phase boundary
#     cts = []
#     if len(phase_boundary_fixed_point) > 0:
#         print("Found %d phase coexistence region(s):" %(len(phase_boundary_fixed_point)))
#         for i in range(0, len(phase_boundary_fixed_point)):
#             x_alpha = phase_boundary_fixed_point[i][0]
#             x_beta = phase_boundary_fixed_point[i][1]
#             ct_now = (GibbsFE(x_alpha, params_list, T=T) - GibbsFE(x_beta, params_list, T=T))/(x_alpha - x_beta) 
#             cts.append(ct_now)
#             print("From x=%.16f to x=%.16f, mu_coex=%.16f" %(phase_boundary_fixed_point[i][0], phase_boundary_fixed_point[i][1], ct_now))
#     else:
#         print("No phase separation region detected.")

#     # print output function in python
#     with open(outpyfile_name, "w") as fout:
#         fout.write("import numpy as np\nimport pybamm\nfrom pybamm import exp, log, tanh, constants, Parameter, ParameterValues\n#from numpy import log, exp\n#import matplotlib.pyplot as plt\n\n")
#         fout.write("def fitted_OCP(sto):\n")
#         fout.write("    _eps = 1e-7\n")
#         fout.write("    # params\n")
#         # write fitted params
#         if isinstance(params_list[0], list) == True and len(params_list) == 4:
#             # this means the first one is excess enthalpy free energy params, 
#             # the second one is excess config entropy params
#             # the thrid and fourth are param for S_vib
#             energy_params_list = params_list[0]
#             entropy_params_list = params_list[1]
#             ns_list = params_list[2] 
#             Theta_Es_list = params_list[3]
#             # excess G params
#             fout.write("    # excess enthalpy params\n")
#             fout.write("    G0 = %.6f # G0 is the pure substance gibbs free energy \n" %(energy_params_list[-1].item()))
#             for i in range(0, len(energy_params_list)-1):
#                 fout.write("    Omega%d = %.6f \n" %(i, energy_params_list[i].item()))
#             text = "    Omegas =["
#             for i in range(0, len(energy_params_list)-1):
#                 text=text+"Omega"+str(i)
#                 if i!= len(energy_params_list)-2:
#                     text=text+", "
#                 else:
#                     text=text+"]\n"
#             fout.write(text)
#             # excess configurational S params
#             fout.write("    # configurational entropy params\n")
#             for i in range(0, len(entropy_params_list)):
#                 fout.write("    omega%d = %.6f \n" %(i, entropy_params_list[i].item()))
#             text = "    omegas =["
#             for i in range(0, len(entropy_params_list)):
#                 text=text+"omega"+str(i)
#                 if i!= len(entropy_params_list)-1:
#                     text=text+", "
#                 else:
#                     text=text+"]\n"
#             fout.write(text)
#             # vibrational S param, ns
#             fout.write("    # vibrational entropy params, ns (ns[0] is place holder)\n")
#             for i in range(0, len(ns_list)):
#                 fout.write("    n%d = %.6f \n" %(i, ns_list[i]))
#             text = "    ns =["
#             for i in range(0, len(ns_list)):
#                 text=text+"n"+str(i)
#                 if i!= len(ns_list)-1:
#                     text=text+", "
#                 else:
#                     text=text+"]\n"
#             fout.write(text)
#             # vibrational S param, ThetaEs
#             fout.write("    # vibrational entropy params, ThetaEs\n")
#             # Theta_LiHM as a function of x
#             for i in range(0, len(Theta_Es_list[0])):
#                 fout.write("    ThetaE0_%d = %.6f \n" %(i, Theta_Es_list[0][i].item()))
#             text = "    ThetaE0 =["
#             for i in range(0, len(Theta_Es_list[0])):
#                 text=text+"ThetaE0_"+str(i)
#                 if i!= len(Theta_Es_list[0])-1:
#                     text=text+", "
#                 else:
#                     text=text+"]\n"
#             fout.write(text)
#             # Theta_HM and Theta_Li
#             for i in range(1, len(Theta_Es_list)):
#                 fout.write("    ThetaE%d = %.6f \n" %(i, Theta_Es_list[i].item()))
#             text = "    ThetaEs =["
#             for i in range(0, len(Theta_Es_list)):
#                 text=text+"ThetaE"+str(i)
#                 if i!= len(Theta_Es_list)-1:
#                     text=text+", "
#                 else:
#                     text=text+"]\n"
#             fout.write(text)
#         else:
#             # no entropy params, only excess enthalpy parameters & temperature independent
#             fout.write("    G0 = %.6f # G0 is the pure substance gibbs free energy \n" %(params_list[-1].item()))
#             for i in range(0, len(params_list)-1):
#                 fout.write("    Omega%d = %.6f \n" %(i, params_list[i].item()))
#             text = "    Omegas =["
#             for i in range(0, len(params_list)-1):
#                 text=text+"Omega"+str(i)
#                 if i!= len(params_list)-2:
#                     text=text+", "
#                 else:
#                     text=text+"]\n"
#             fout.write(text)
            
#         # write phase boundaries & addition part
#         if len(phase_boundary_fixed_point)>0:
#             for i in range(0, len(phase_boundary_fixed_point)):
#                 fout.write("    # phase boundary %d\n" %(i))
#                 fout.write("    x_alpha_%d = %.16f\n" %(i, phase_boundary_fixed_point[i][0]))
#                 fout.write("    x_beta_%d = %.16f\n" %(i, phase_boundary_fixed_point[i][1]))
#                 fout.write("    mu_coex_%d = %.16f\n" %(i, cts[i]))
#                 fout.write("    is_outside_miscibility_gap_%d = (sto<x_alpha_%d) + (sto>x_beta_%d)\n" %(i,i,i))
#             fout.write("    # whether is outside all gap\n")
#             text = "    is_outside_miscibility_gaps = "
#             for i in range(0, len(phase_boundary_fixed_point)):
#                 text = text + "is_outside_miscibility_gap_%d " %(i)
#                 if i!=len(phase_boundary_fixed_point)-1:
#                     text = text + "* "
#             fout.write(text)
#             fout.write("    \n")
#             fout.write("    mu_outside = G0 + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))\n" %(T))
#             if isinstance(params_list[0], list) == True and len(params_list) == 4:
#                 # this means the first one is excess enthalpy free energy params, 
#                 # the second one is excess config entropy params
#                 # the thrid and fourth are param for S_vib
#                 ## write S_vib contribution
#                 # LiHM S_vib
#                 # first sum up the temperature at composition x
#                 fout.write("    ## S_vib\n")
#                 fout.write("    # Theta_LiHM\n")
#                 if polynomial_style == "Legendre":
#                     fout.write("    _t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Pn_values = legendre_poly_recurrence(_t,len(ThetaEs[0])-1)\n")
#                     fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(_t, len(ThetaEs[0])-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                 elif polynomial_style == "Chebyshev":
#                     fout.write("    _t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Pn_values = chebyshev_poly_recurrence(_t,len(ThetaEs[0])-1)\n")
#                     fout.write("    Pn_derivatives_values = chebyshev_derivative_poly_recurrence(_t, len(ThetaEs[0])-1)  # Compute Chebyshev polynomials up to degree len(coeffs) - 1\n")
#                 fout.write("    Theta_LiHM = 0.0\n")
#                 fout.write("    Theta_LiHM_derivative = 0.0\n")
#                 fout.write("    for i in range(0, len(ThetaEs[0])):\n")
#                 fout.write("        Theta_LiHM = Theta_LiHM + ThetaEs[0][i]*Pn_values[i]\n")
#                 fout.write("        Theta_LiHM_derivative = Theta_LiHM_derivative + (-2)*ThetaEs[0][i]*Pn_derivatives_values[i]\n")
#                 fout.write("    t_Theta_LiHM = -Theta_LiHM*100/%.4f\n" %(T))
#                 fout.write("    t_Theta_LiHM_derivative = Theta_LiHM_derivative*100/%.4f  # NOTE that there is no negative sign here\n" %(T))
#                 fout.write("    mu_outside = mu_outside - %.4f*(-3)*(ns[2])*8.314*( log(1.0 - exp(t_Theta_LiHM)) + t_Theta_LiHM*1.0/(exp(-t_Theta_LiHM)-1) ) \n" %(T))
#                 fout.write("    mu_outside = mu_outside - %.4f*(-3)*(1.0*ns[1]+sto*ns[2])*8.314*( 1/(1-exp(t_Theta_LiHM))* exp(t_Theta_LiHM)*t_Theta_LiHM_derivative  - t_Theta_LiHM_derivative*1/(exp(-t_Theta_LiHM) -1)   -t_Theta_LiHM * exp(-t_Theta_LiHM)/(exp(-t_Theta_LiHM) -1)**2 *t_Theta_LiHM_derivative  ) \n" %(T))                    
#                 ## mu does not depend on HM,
#                 # fout.write("    # Theta_HM \n")
#                 # fout.write("    t = -ThetaEs[1]*100/%.4f\n" %(T))
#                 # fout.write("    mu_outside = mu_outside - As[1] * %.4f*(-3)*8.314*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))
#                 ## mu does depend on Li, because it's (-x*s_Li)
#                 fout.write("    # Theta_Li\n")
#                 fout.write("    t = -ThetaEs[2]*100/%.4f\n" %(T))
#                 fout.write("    mu_outside = mu_outside +  %.4f*(-3)*ns[2]*8.314*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))        
#                 ## write H_vib contribution
#                 fout.write("    ## H_vib\n")
#                 fout.write("    # Theta_LiHM\n")
#                 fout.write("    Theta_LiHM_real = Theta_LiHM*100\n")
#                 fout.write("    Theta_LiHM_derivative_real = Theta_LiHM_derivative*100\n")
#                 fout.write("    mu_outside = mu_outside +  3*(ns[2])*8.314*( 1/2*Theta_LiHM_real + Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f)-1) ) \n" %(T))           
#                 fout.write("    mu_outside = mu_outside +  3*(1.0*ns[1]+sto*ns[2])*8.314*( 0.5*Theta_LiHM_derivative_real + Theta_LiHM_derivative_real/(exp(Theta_LiHM_real/%.4f) -1) - Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f) -1)**2 *  exp(Theta_LiHM_real/%.4f) * Theta_LiHM_derivative_real/%.4f ) \n" %(T,T,T,T))           
#                 ## mu does not depend on HM
#                 # fout.write("    # Theta_HM \n")
#                 # fout.write("    Theta_Li_real = ThetaEs[1]*100\n")
#                 # fout.write("    mu_outside = mu_outside + As[1] *3*8.314*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))           
#                 ## mu does depend on Li, because it's (x*s_Li)
#                 fout.write("    # Theta_Li\n")
#                 fout.write("    Theta_Li_real = ThetaEs[2]*100\n")
#                 fout.write("    mu_outside = mu_outside - 3*ns[2]*8.314*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))                
#                 ## write S_config contribution
#                 if polynomial_style == "Legendre":
#                     fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Pn_values = legendre_poly_recurrence(t,len(omegas)-1)\n")
#                     fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                     fout.write("    for i in range(0, len(omegas)):\n")
#                     fout.write("        mu_outside = mu_outside + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))*(omegas[i]*Pn_values[i]) -2.0*8.314*%.4f*(sto*log(sto) + (1-sto)*log(1-sto))*(omegas[i]*Pn_derivatives_values[i]) \n" %(T, T))
#                 elif polynomial_style == "Chebyshev":
#                     fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Tn_values = chebyshev_poly_recurrence(t,len(Omegas)-1)\n")
#                     fout.write("    Tn_derivatives_values = chebyshev_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                     fout.write("    for i in range(0, len(omegas)):\n")
#                     fout.write("        mu_outside = mu_outside + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))*(omegas[i]*Tn_values[i]) -2.0*8.314*%.4f*(sto*log(sto) + (1-sto)*log(1-sto))*(omegas[i]*Tn_derivatives_values[i]) \n" %(T, T))         
#                 else:
#                     print("polynomial_style not recognized in write_ocv")
#                     exit()
#             # write excess G part 
#             if polynomial_style == "R-K":
#                 fout.write("    for i in range(0, len(Omegas)):\n")
#                 fout.write("        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))\n")
#             elif polynomial_style == "Legendre":              
#                 fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                 fout.write("    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)\n")
#                 fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                 fout.write("    for i in range(0, len(Omegas)):\n")
#                 fout.write("        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])\n")
#             elif polynomial_style == "Chebyshev":
#                 fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                 fout.write("    Tn_values = chebyshev_poly_recurrence(t,len(Omegas)-1)\n")
#                 fout.write("    Tn_derivatives_values = chebyshev_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                 fout.write("    for i in range(0, len(Omegas)):\n")
#                 fout.write("        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Tn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Tn_values[i])\n")
#             else:
#                 print("polynomial_style not recognized in write_ocv")
#                 exit()

#             text0 = "    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   "
#             text1 = ""
#             for i in range(0, len(cts)):
#                 text1 = text1 + "(1-is_outside_miscibility_gap_%d)*mu_coex_%d " %(i, i)
#                 if i != len(cts)-1:
#                     text1 = text1 + " + "
#             text = text0 + "(" + text1 + ")\n"
#             fout.write(text)
#             fout.write("    return -mu_e/96485.0\n\n\n\n")
#         else:
#             # no phase boundaries required, just mu and return -mu/F
#             fout.write("    mu = G0 + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))\n" %(T))
#             if isinstance(params_list[0], list) == True and len(params_list) == 4:
#                 # this means the first one is excess enthalpy free energy params, 
#                 # the second one is excess config entropy params
#                 # the thrid and fourth are param for S_vib
#                 ## write S_vib contribution
#                 # LiHM S_vib
#                 # first sum up the temperature at composition x
#                 fout.write("    ## S_vib\n")
#                 fout.write("    # Theta_LiHM\n")
#                 if polynomial_style == "Legendre":
#                     fout.write("    _t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Pn_values = legendre_poly_recurrence(_t,len(ThetaEs[0])-1)\n")
#                     fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(_t, len(ThetaEs[0])-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                 elif polynomial_style == "Chebyshev":
#                     fout.write("    _t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Pn_values = chebyshev_poly_recurrence(_t,len(ThetaEs[0])-1)\n")
#                     fout.write("    Pn_derivatives_values = chebyshev_derivative_poly_recurrence(_t, len(ThetaEs[0])-1)  # Compute Chebyshev polynomials up to degree len(coeffs) - 1\n")
#                 fout.write("    Theta_LiHM = 0.0\n")
#                 fout.write("    Theta_LiHM_derivative = 0.0\n")
#                 fout.write("    for i in range(0, len(ThetaEs[0])):\n")
#                 fout.write("        Theta_LiHM = Theta_LiHM + ThetaEs[0][i]*Pn_values[i]\n")
#                 fout.write("        Theta_LiHM_derivative = Theta_LiHM_derivative + (-2)*ThetaEs[0][i]*Pn_derivatives_values[i]\n")
#                 fout.write("    t_Theta_LiHM = -Theta_LiHM*100/%.4f\n" %(T))
#                 fout.write("    t_Theta_LiHM_derivative = Theta_LiHM_derivative*100/%.4f  # NOTE that there is no negative sign here\n" %(T))
#                 fout.write("    mu = mu - %.4f*(-3)*(ns[2])*8.314*( log(1.0 - exp(t_Theta_LiHM)) + t_Theta_LiHM*1.0/(exp(-t_Theta_LiHM)-1) ) \n" %(T))
#                 fout.write("    mu = mu - %.4f*(-3)*(1.0*ns[1]+sto*ns[2])*8.314*( 1/(1-exp(t_Theta_LiHM))* exp(t_Theta_LiHM)*t_Theta_LiHM_derivative  - t_Theta_LiHM_derivative*1/(exp(-t_Theta_LiHM) -1)   -t_Theta_LiHM * exp(-t_Theta_LiHM)/(exp(-t_Theta_LiHM) -1)**2 *t_Theta_LiHM_derivative  ) \n" %(T))                    
#                 ## mu does not depend on HM,
#                 # fout.write("    # Theta_HM \n")
#                 # fout.write("    t = -ThetaEs[1]*100/%.4f\n" %(T))
#                 # fout.write("    mu = mu - As[1] * %.4f*(-3)*8.314*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))
#                 ## mu does depend on Li, because it's (x*s_Li)
#                 fout.write("    # Theta_Li\n")
#                 fout.write("    t = -ThetaEs[2]*100/%.4f\n" %(T))
#                 fout.write("    mu = mu -  %.4f*(-3)*ns[2]*8.314*2*sto*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))        
#                 ## write H_vib contribution
#                 fout.write("    ## H_vib\n")
#                 fout.write("    # Theta_LiHM\n")
#                 fout.write("    Theta_LiHM_real = Theta_LiHM*100\n")
#                 fout.write("    Theta_LiHM_derivative_real = Theta_LiHM_derivative*100\n")
#                 fout.write("    mu = mu +  3*(ns[2])*8.314*( 1/2*Theta_LiHM_real + Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f)-1) ) \n" %(T))           
#                 fout.write("    mu = mu +  3*(1.0*ns[1]+sto*ns[2])*8.314*( 0.5*Theta_LiHM_derivative_real + Theta_LiHM_derivative_real/(exp(Theta_LiHM_real/%.4f) -1) - Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f) -1)**2 *  exp(Theta_LiHM_real/%.4f) * Theta_LiHM_derivative_real/%.4f ) \n" %(T,T,T,T))           
#                 ## mu does not depend on HM
#                 # fout.write("    # Theta_HM \n")
#                 # fout.write("    Theta_Li_real = ThetaEs[1]*100\n")
#                 # fout.write("    mu = mu + As[1] *3*8.314*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))           
#                 ## mu does depend on Li, because it's (x*s_Li)
#                 fout.write("    # Theta_Li\n")
#                 fout.write("    Theta_Li_real = ThetaEs[2]*100\n")
#                 fout.write("    mu = mu + 3*ns[2]*8.314*2*sto*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))                
#                 ## write S_config contribution
#                 if polynomial_style == "Legendre":
#                     # this means the first one is excess gibbs free energy params, 
#                     # the second one is excess config entropy params
#                     # the thrid one is Theta_Li param for S_vib
#                     fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Pn_values = legendre_poly_recurrence(t,len(omegas)-1)\n")
#                     fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                     fout.write("    for i in range(0, len(omegas)):\n")
#                     fout.write("        mu = mu + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))*(omegas[i]*Pn_values[i]) -2.0*8.314*%.4f*(sto*log(sto) + (1-sto)*log(1-sto))*(omegas[i]*Pn_derivatives_values[i]) \n" %(T, T))         
#                 elif polynomial_style == "Chebyshev":
#                     fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                     fout.write("    Tn_values = chebyshev_poly_recurrence(t,len(Omegas)-1)\n")
#                     fout.write("    Tn_derivatives_values = chebyshev_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                     fout.write("    for i in range(0, len(omegas)):\n")
#                     fout.write("        mu = mu + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))*(omegas[i]*Tn_values[i]) -2.0*8.314*%.4f*(sto*log(sto) + (1-sto)*log(1-sto))*(omegas[i]*Tn_derivatives_values[i]) \n" %(T, T))              
#                 else:
#                     print("polynomial_style not recognized in write_ocv")
#                     exit()
#             # excess G
#             if polynomial_style == "R-K":
#                 fout.write("    for i in range(0, len(Omegas)):\n")
#                 fout.write("        mu = mu + Omegas[i]*((1-2*sto)**(i+1) - 2*i*sto*(1-sto)*(1-2*sto)**(i-1))\n")
#             elif polynomial_style == "Legendre":              
#                 fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                 fout.write("    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)\n")
#                 fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                 fout.write("    for i in range(0, len(Omegas)):\n")
#                 fout.write("        mu = mu -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])\n")
#             elif polynomial_style == "Chebyshev":
#                 fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
#                 fout.write("    Tn_values = chebyshev_poly_recurrence(t,len(Omegas)-1)\n")
#                 fout.write("    Tn_derivatives_values = chebyshev_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
#                 fout.write("    for i in range(0, len(Omegas)):\n")
#                 fout.write("        mu = mu -2*sto*(1-sto)*(Omegas[i]*Tn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Tn_values[i])\n")
#             fout.write("    return -mu/96485.0\n\n\n\n")
            
            
#     if polynomial_style == "Legendre":
#         abs_path = os.path.abspath(__file__)[:-8]+"__legendre_derivatives.py"
#         with open(abs_path,'r') as fin:
#             lines = fin.readlines()
#         with open(outpyfile_name, "a") as fout:
#             for line in lines:
#                 fout.write(line)
#     elif polynomial_style == "Chebyshev":
#         abs_path = os.path.abspath(__file__)[:-8]+"__chebyshev_derivatives.py"
#         with open(abs_path,'r') as fin:
#             lines = fin.readlines()
#         with open(outpyfile_name, "a") as fout:
#             for line in lines:
#                 fout.write(line)
            
#     # print output function in matlab
#     if polynomial_style == "R-K" and isinstance(params_list[0], list) == False:
#         with open("fitted_ocv_functions.m", "w") as fout:
#             fout.write("function result = ocv(sto):\n")
#             fout.write("    eps = 1e-7;\n")
#             fout.write("    %% rk params\n")
#             fout.write("    G0 = %.6f; %%G0 is the pure substance gibbs free energy \n" %(params_list[-1].item()))
#             for i in range(0, len(params_list)-1):
#                 fout.write("    Omega%d = %.6f; \n" %(i, params_list[i].item()))
#             text = "    Omegas =["
#             for i in range(0, len(params_list)-1):
#                 text=text+"Omega"+str(i)
#                 if i!= len(params_list)-2:
#                     text=text+", "
#                 else:
#                     text=text+"];\n"
#             fout.write(text)
#             # write phase boundaries
#             if len(phase_boundary_fixed_point)>0:
#                 for i in range(0, len(phase_boundary_fixed_point)):
#                     fout.write("    %% phase boundary %d\n" %(i))
#                     fout.write("    x_alpha_%d = %.16f ; \n" %(i, phase_boundary_fixed_point[i][0]))
#                     fout.write("    x_beta_%d = %.16f ; \n" %(i, phase_boundary_fixed_point[i][1]))
#                     fout.write("    mu_coex_%d = %.16f ; \n" %(i, cts[i]))
#                     fout.write("    is_outside_miscibility_gap_%d = (sto<x_alpha_%d) + (sto>x_beta_%d) ; \n" %(i,i,i))
#                 fout.write("    %% whether is outside all gap\n")
#                 text = "    is_outside_miscibility_gaps = "
#                 for i in range(0, len(phase_boundary_fixed_point)):
#                     text = text + "is_outside_miscibility_gap_%d " %(i)
#                     if i!=len(phase_boundary_fixed_point)-1:
#                         text = text + "* "
#                 fout.write(text)
#                 fout.write(";    \n")
#                 fout.write("    mu_outside = G0 + 8.314*%.4f*log((sto+eps)/(1-sto+eps)) ; \n" %(T)) 
                
#                 fout.write("    for i=0:length(Omegas)-1\n")
#                 fout.write("        mu_outside = mu_outside + is_outside_miscibility_gaps * Omegas[i+1]*((1-2*sto)^(i+1) - 2*i*sto*(1-sto)*(1-2*sto)^(i-1));\n")
#                 fout.write("end\n")
                
#                 text0 = "    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   "
#                 text1 = ""
#                 for i in range(0, len(cts)):
#                     text1 = text1 + "(1-is_outside_miscibility_gap_%d)*mu_coex_%d " %(i, i)
#                     if i != len(cts)-1:
#                         text1 = text1 + " + "
#                 text = text0 + "(" + text1 + ") ;\n"
#                 fout.write(text)
#                 fout.write("    result = -mu_e/96485.0 ;")
#                 fout.write("    return;")
#                 fout.write("end  \n\n\n")
#     else:
#         print("Writing matlab ocv function only support R-K")
#     # write complete
#     print("\n\n\n\n\n Fitting Complete.\n")
#     print("Fitted OCV function written in PyBaMM function (copy and paste readay!):\n")
#     print("###################################\n")
#     with open(outpyfile_name, "r") as fin:
#         lines = fin.readlines()
#     for line in lines:
#         print(line, end='')
#     print("\n\n###################################\n")
#     print("Or check %s and fitted_ocv_functions.m (if polynomial style = R-K) for fitted thermodynamically consistent OCV model in PyBaMM & Matlab formats. " %(outpyfile_name))





def read_OCV_data(datafile_name):
    # read data1
    df = pd.read_csv(datafile_name,header=None)
    data = df.to_numpy()
    x = data[:,0]
    mu = -data[:,1]*96485 # because -mu_e- = OCV*F, -OCV*F = mu
    # convert to torch.tensor
    x = x.astype("float32")
    x = torch.from_numpy(x)
    mu = mu.astype("float32")
    mu = torch.from_numpy(mu)
    return x, mu



def _get_phase_boundaries(GibbsFE, total_params_list, T, end_points = [0,1], is_clamp = True, style = "Legendre", quaderature_points = 20, ngrid=99):
    """ 
    total_params_list: in the sequence of [enthalpy_mixing_params_list, config_entropy_params_list, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM, Theta_max_Li]
    """
    # sample the Gibbs free energy landscape
    sample = sampling(GibbsFE, total_params_list, T, end_points = end_points, sampling_id=1, is_PDOS_version_G = True, style = style, quaderature_points = quaderature_points, ngrid=ngrid)
    # give the initial guess of miscibility gap
    phase_boundarys_init, _ = convex_hull(sample) 
    # refinement & calculate loss
    if phase_boundarys_init != []:
        # There is at least one phase boundary predicted 
        phase_boundary_fixed_point = []
        for phase_boundary_init in phase_boundarys_init:
            common_tangent = CommonTangent(GibbsFE, total_params_list, T = T, is_PDOS_version_G = True, end_points= end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points) 
            phase_boundary_now = phase_boundary_init.requires_grad_()
            phase_boundary_fixed_point_now = common_tangent(phase_boundary_now) 
            phase_boundary_fixed_point.append(phase_boundary_fixed_point_now)
    else:
        # No boundary find.
        phase_boundary_fixed_point = []
    # check if there is any solved point that are not normal
    phase_boundary_fixed_point_checked = []
    for i in range(0, len(phase_boundary_fixed_point)):
        if phase_boundary_fixed_point[i][0] < phase_boundary_fixed_point[i][1] and phase_boundary_fixed_point[i][0]>=end_points[0] and phase_boundary_fixed_point[i][1] <= end_points[1]:
            phase_boundary_fixed_point_checked.append(phase_boundary_fixed_point[i])
        else:
            # if phase_boundary_fixed_point[i][0] > phase_boundary_fixed_point[i][1] and phase_boundary_fixed_point[i][1]>=end_points[0] and phase_boundary_fixed_point[i][0] <= end_points[1]:
            #     # sequence inversed
            #     phase_boundary_fixed_point_checked.append(phase_boundary_fixed_point[i])
            print("Abandoned abnormal solution ", phase_boundary_fixed_point[i])
    # print(phase_boundary_fixed_point_checked, "in _get_phase_boundaries after common tangent, AMYAO DEBUG")
    return phase_boundary_fixed_point_checked



def calc_loss_from_pd(phase_boundaries_from_phase_diagram, GibbsFE, total_params_list, end_points = [0,1], is_clamp = True, style = "Legendre", quaderature_points = 20, ngrid=99):
    """ 
    construct loss term from phase boundary points from phase diagram. Currently only supports 1 miscibility gap at each temperature (e.g. LFP)
    
    Input: 
    phase_boundaries_from_phase_diagram: phase boundary from experimental determined phase diagram, in the sequence of [[T1, [x_10,x_11]], [T2, [x_20, x_21], ...]]
    """
    # format conversion
    Ts_list = []
    phase_boundaries_pd_list = []
    for i in range(0, len(phase_boundaries_from_phase_diagram)):
        Ts_list.append(phase_boundaries_from_phase_diagram[i][0])
        phase_boundaries_pd_list.append(torch.tensor(phase_boundaries_from_phase_diagram[i][1]))
    # calculate phase boundary at each temperature
    loss = 0.0
    for i in range(0, len(Ts_list)):
        T_now = Ts_list[i]
        phase_boundary_calc_this_T = _get_phase_boundaries(GibbsFE, total_params_list, T_now, end_points = end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points, ngrid=ngrid)
        if len(phase_boundary_calc_this_T) > 0:
            for j in range(0, len(phase_boundary_calc_this_T)):
                loss = loss + torch.sum((phase_boundary_calc_this_T[j]-phase_boundaries_pd_list[i])**2)
    return loss


""" 
fit dSdx data only
"""

def fit_dSdx(datafile_dsdx_name=None,
              T_dsdx = None,
              read_in_dOCV_dT_function = _convert_JMCA_Tdsdx_data_to_dsdx,
              g_i_list_Li = None, # pretrained via fit_pure_substance_Cp
              g_i_list_HM = None, # pretrained via fit_pure_substance_Cp
              g_i_list_LiHM = None, # pretrained via fit_pure_substance_Cp
              number_of_S_config_excess_omegas=5,
              n_i_LixHM = 4,
              n_j_LixHM = 4,
              style = "Legendre",
              quaderature_points = 20,
              learning_rate = 0.01, 
              total_training_epochs = 8000,
              loss_threshold = 0.01,
              n_list = [9999.9, 6.0, 1.0],  # LixHM, HM, Li (9999.9 is placeholder)
              pretrained_value_of_S_config_excess_omegas = None,
              is_g_ij_collapse_to_g_i_at_end_points = False, 
              Cp_HM_data = None,
              T_HM_data = None,
              Cp_LiHM_data = None,
              T_LiHM_data = None,
              alpha_Cp_regularization = 1.0,
              ):
    """ 
    Fit S_config and S_vib of LixHM to dSdx data

    Inputs:
    datafile_dsdx_name: the csv file which contains OCV and SoC data, first column must be Li filling fraction (Be careful that Li filling fraction might be SoC or 1-SoC!), second column must be dS/dx (unit must be J/(K.mol)). Must not with header & index.
    T_dsdx: temperature of ds/dx data in datafile_dsdx_name
    read_in_dOCV_dT_function: the function to read dOCV/dT data and convert into x and dS/dx
    g_i_list_fitted_Li: fitted PDOS parameters for Li, pretrained via fit_pure_substance_Cp
    g_i_list_fitted_HM: fitted PDOS parameters for HM, pretrained via fit_pure_substance_Cp. Note that the length has to be the same of g_i_list_fitted_LiHM
    g_i_list_fitted_LiHM: fitted PDOS parameters for LiHM, pretrained via fit_pure_substance_Cp
    number_of_S_config_excess_omegas: number of excess configurational entropy omegas
    n_i_LixHM: number of parameters to expand PDOS of LixHM wrt Theta (note it has to be consistent with the pretrained g_i_list of HM)
    n_j_LixHM: number of parameters to expand PDOS of LixHM wrt x
    style: style of polynomials to expand excess thermo, can be "Legendre", "R-K", "Chebyshev".
    quaderature_points: how many sampling points for Gauss-Legendre quaderature evaluation
    learning_rate: learning rate for updating parameters
    total_training_epochs: total epochs for fitting
    loss_threshold: threshold of loss, below which training stops automatically
    n_list: how many moles of atoms are there in 1 mole of LixHM, HM and Li (first one is placeholder, can be anything)
    pretrained_value_of_S_config_excess_omegas: pretrained values, a list of float
    is_g_ij_collapse_to_g_i_at_end_points: whether we are constraining g_ij_list of LixHM through matching the value of g_ij to g_i at x=0 and 1, or are we training g_ij_list through Cp data of HM and LiHM
    Cp_HM_data, T_HM_data, Cp_LiHM_data, T_LiHM_data: only if when is_g_ij_collapse_to_g_i_at_end_points=True, we constrain g_ij_value with the measured Cp data of end members 
   
    Outputs:
    None, please see log_entropy for fitted params
    """
    assert n_i_LixHM == len(g_i_list_HM)
    assert len(g_i_list_HM) == len(g_i_list_LiHM)
    if pretrained_value_of_S_config_excess_omegas != None:
        assert len(pretrained_value_of_S_config_excess_omegas) == number_of_S_config_excess_omegas

    # convert pretrained value into torch.tensor
    g_i_list_Li = torch.tensor(g_i_list_Li) 
    g_i_list_HM = torch.tensor(g_i_list_HM)
    g_i_list_LiHM = torch.tensor(g_i_list_LiHM)
    
    Theta_max_Li = solve_Theta_max(g_i_list = g_i_list_Li, is_x_dependent = False, style = style)
    Theta_max_Li = Theta_max_Li.detach()
    Theta_max_HM = solve_Theta_max(g_i_list = g_i_list_HM, is_x_dependent = False, style = style)
    Theta_max_HM = Theta_max_HM.detach()
    Theta_max_LiHM = solve_Theta_max(g_i_list = g_i_list_LiHM, is_x_dependent = False, style = style)
    Theta_max_LiHM = Theta_max_LiHM.detach()

    print(Theta_max_Li, Theta_max_HM, Theta_max_LiHM)
    
    ## read file
    x_measured, dsdx_measured = read_in_dOCV_dT_function(
                                        datafile_name=datafile_dsdx_name, 
                                        T=T_dsdx)
    ## declare all params
    # omegas for excess configurational entropy
    S_config_params_list = []
    if pretrained_value_of_S_config_excess_omegas == None:
        for _ in range(0, number_of_S_config_excess_omegas):
            S_config_params_list.append(nn.Parameter( torch.from_numpy(np.array([np.random.randint(-100,100)*0.01],dtype="float32")) ) )
    else:
        for _ in range(0, number_of_S_config_excess_omegas):
            S_config_params_list.append(nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_S_config_excess_omegas[_]],dtype="float32")) ) )
    # g_ij for LixHM PDOS & vibrational entropy. 
    # Note that it is constrained by fitted g_i_list of HM and LiHM (at x=0 and 1)
    g_ij_list_LixHM = []
    i = 0
    g_i_now = [nn.Parameter(g_i_list_HM[0] )]
    for j in range(1, n_j_LixHM):
        g_i_now.append(nn.Parameter( torch.from_numpy(np.array([0.0],dtype="float32")) ))
    g_ij_list_LixHM.append(g_i_now)
    for i in range(1, n_i_LixHM):
        g_i_now = []
        for j in range(0, n_j_LixHM):
            g_ij_start = np.random.uniform(low=-0.1, high=0.1)
            g_i_now.append(nn.Parameter( torch.from_numpy(np.array([g_ij_start],dtype="float32")) ))
        g_ij_list_LixHM.append(g_i_now)
        
    # init optimizer
    params_list = []
    for item in S_config_params_list:
        params_list.append(item)
    for i in range(0, len(g_ij_list_LixHM)):
        single_list = g_ij_list_LixHM[i]
        for item in single_list:
            params_list.append(item)
    optimizer = optim.Adam(params_list, lr=learning_rate)
    # init logs
    with open("log_entropy",'w') as fout:
        fout.write("")
    os.makedirs("records_entropy", exist_ok=True)
    # train
    loss = 9999.9 # init total loss
    epoch = -1
    while loss > 0.01 and epoch < total_training_epochs:
        # clean grad info
        optimizer.zero_grad()
        # use current params to calculate predicted phase boundary
        epoch = epoch + 1
        loss = 0.0 # init total loss
        # calculate entropy loss
        loss, \
        loss_dsdx, \
        loss_s_config_leq_0, \
        loss_s_config_geq_upper_bound, \
        loss_PDOS_greater_than_0, \
        loss_PDOS_two_side, \
        dsdx_calculated\
        = calc_loss_entropy(x_measured, dsdx_measured, T_dsdx, \
                      S_config_params_list, n_list,
                      g_ij_list_LixHM, \
                      g_i_list_HM, Theta_max_HM, \
                      g_i_list_LiHM, Theta_max_LiHM, \
                      g_i_list_Li, Theta_max_Li, \
                      quaderature_points=quaderature_points, style = style,
                      is_g_ij_collapse_to_g_i_at_end_points = is_g_ij_collapse_to_g_i_at_end_points)
        if is_g_ij_collapse_to_g_i_at_end_points == False:
            # instead we constrain g_ij with measured Cp data
            loss_Cp_HM = calc_loss_Cp_of_end_member(Ts = T_HM_data,
                        Cps_measured = Cp_HM_data,
                        g_ij_matrix = g_ij_list_LixHM,
                        quaderature_points=quaderature_points,
                        style = style,
                        x = torch.tensor([0.0]),
                        )
            loss_Cp_LiHM = calc_loss_Cp_of_end_member(Ts = T_LiHM_data,
                        Cps_measured = Cp_LiHM_data,
                        g_ij_matrix = g_ij_list_LixHM,
                        quaderature_points=quaderature_points,
                        style = style,
                        x = torch.tensor([1.0]),
                        )
        else:
            loss_Cp_HM = 0.0
            loss_Cp_LiHM = 0.0
        loss = loss + loss_Cp_HM*alpha_Cp_regularization + loss_Cp_LiHM*alpha_Cp_regularization
        # backprop
        loss.backward()
        optimizer.step()
        # print output
        output_txt = "Epoch %d  Loss tot %.4f dsdx %.4f s>0 %.4f s<s_max %.4f PDOS>0 %.4f PDOS_end_points %.4f Cp_HM %.4f Cp_LiHM %.4f  " %(epoch, loss, loss_dsdx, loss_s_config_leq_0, loss_s_config_geq_upper_bound, loss_PDOS_greater_than_0, loss_PDOS_two_side, loss_Cp_HM, loss_Cp_LiHM)
        for i in range(0, len(S_config_params_list)): 
            output_txt = output_txt + "omega%d %.8f "%(i, S_config_params_list[i].item())
        for i in range(0, len(g_ij_list_LixHM)): 
            for j in range(0, len(g_ij_list_LixHM[0])):
                output_txt = output_txt + "g%d%d %.8f "%(i, j, g_ij_list_LixHM[i][j].item())   
        output_txt = output_txt + "      "
        print(output_txt)
        with open("log_entropy",'a') as fout:
            fout.write(output_txt)
            fout.write("\n")
        if epoch%100 == 0:
            # draw figures 
            _x = x_measured.numpy()
            _y = dsdx_measured.numpy()
            plt.plot(_x, _y, "b*")
            _y1 = dsdx_calculated.detach().numpy()
            plt.plot(_x, _y1, "k-")
            os.chdir("records_entropy")
            name = str(epoch)+".png"
            plt.savefig(name)
            plt.close()
            os.chdir("../")








def train_multiple_OCVs_and_dS_dx_PDOS_version(
          datafile1_name='graphite.csv', 
          T1 = 300,
          datafile2_name=None,
          T2 = None,
          datafile3_name=None,
          T3 = None,
          datafile_dsdx_name=None,
          T_dsdx = None,
          number_of_Omegas=6, 
          number_of_S_config_excess_omegas=5,
          n_i_LixHM = 4,
          n_j_LixHM = 4,
          end_points = [0.0,1.0], 
          quaderature_points = 20,
          style = "Legendre",
          learning_rate = 1000.0, 
          learning_rate_other_than_H_mix_excess = 0.01,
          epoch_optimize_params_other_than_H_mix_only_after_this_epoch = 1000,
          alpha_dsdx = 1.0/1000, # if dsdx has been pretrained, then set it as a small number
          total_training_epochs = 10000,
          loss_threshold = 0.01,
          G0_rand_range=[-10*5000,-5*5000], 
          Omegas_rand_range=[-10*100,10*100],
          records_y_lims = [0.0,0.6],
          n_list = [9999.9, 6.0, 1.0], 
          read_in_dOCV_dT_function = _convert_JMCA_Tdsdx_data_to_dsdx,
          pretrained_value_of_Omega_G0 = None,
          pretrained_value_of_S_config_excess_omegas = None,
          pretrained_value_of_g_ij_list_LixHM = None,
          pretrained_value_of_g_i_list_Li = None, 
          pretrained_value_of_g_i_list_HM = None, 
          pretrained_value_of_g_i_list_LiHM = None, 
          is_g_ij_collapse_to_g_i_at_end_points = False,
          Cp_HM_data = None,
          T_HM_data = None,
          Cp_LiHM_data = None,
          T_LiHM_data = None,
          alpha_Cp_regularization = 1.0,
          is_learn_from_phase_diagram = False,
          phase_boundaries_from_phase_diagram = [[298, [0.1, 0.9]], [300, [0.11, 0.89]], [305, [0.12, 0.88]]],
          alpha_phase_boundary = 100.0,
          ):
    """
    Fit the diffthermo OCV function with up to 3 OCV meausured at different temperatures & 1 dS/dx data

    Inputs:
    datafile1_name, datafile2_name, datafile3_name: the csv file which contains OCV and SoC data, first column must be Li filling fraction (Be careful that Li filling fraction might be SoC or 1-SoC!), second column must be OCV. Must not with header & index. 
    T1, T2, T3: temperature of OCV contained in datafile1_name, datafile2_name, datafile3_name
    datafile_dsdx_name: the csv file which contains OCV and SoC data, first column must be Li filling fraction (Be careful that Li filling fraction might be SoC or 1-SoC!), second column must be dS/dx (unit must be J/(K.mol)). Must not with header & index.
    T_dsdx: temperature of ds/dx data in datafile_dsdx_name
    number_of_Omegas: number of R-K parameters. Note that the order of R-K expansion = number_of_Omegas - 1
    number_of_S_config_excess_omegas: number of excess configurational entropy omegas
    n_i_LixHM: expansion order for LixHM PDOS (g_ij_matrix) wrt i (i.e. Theta-dependent part)
    n_j_LixHM: expansion order for LixHM PDOS (g_ij_matrix) wrt j (i.e. x-dependent part)
    end_points: end point of x-filling fraction. Usually [0,1]
    quaderature_points: how many quaderature points to be used for quaderature integration 
    style: style of polynomials to expand excess thermo, can be "Legendre", "Chebyshev".
    learning_rate: learning rate for updating parameters
    learning_rate_other_than_H_mix_excess: learning rate for other parameters
    epoch_optimize_params_other_than_H_mix_only_after_this_epoch: before this number of epochs, only H excess mix params are optimized; After this epoch, all params are optimized
    alpha_dsdx: weight of dsdx loss
    total_training_epochs: total epochs for fitting
    loss_threshold: threshold of loss, below which training stops automatically
    G0_rand_range: the range for randomly initialize G0
    Omegas_rand_range: the range for randomly initialize R-K parameters
    records_y_lims: the range for records y axis lims
    n_list: the amount of atoms in 1 mole of substance, in the sequence of LixHM, HM and Li (always 1). The first item can be arbitrary, it will be fixed later in the code
    read_in_dOCV_dT_function: the function to read dOCV/dT data and convert into x and dS/dx
    pretrained_value_of_*: pretrained values
    is_g_ij_collapse_to_g_i_at_end_points: whether we are constraining g_ij_list of LixHM through matching the value of g_ij to g_i at x=0 and 1, or are we training g_ij_list through Cp data of HM and LiHM
    Cp_HM_data, T_HM_data, Cp_LiHM_data, T_LiHM_data: only if when is_g_ij_collapse_to_g_i_at_end_points=True, we constrain g_ij_value with the measured Cp data of end members 
    alpha_Cp_regularization: strength of Cp regularization
    is_learn_from_phase_diagram: IN EXPERIMENT: construct loss function term from phase boundary from phase diagram
    phase_boundaries_from_phase_diagram: value of phase boundary from phase diagram, currently only support 1 miscibility gap for each temperature. Input format is [[T1, [x10, x11]], [T2, [x20, x21]], ...]
    alpha_phase_boundary: strength of phase diagram regularization
    
    Outputs: 
    None, please find parameter value from log
    """
    
    is_clamp = True  # must clamp for newton-ralphson solver
    
    # precheck:
    if pretrained_value_of_Omega_G0 != None:
        assert len(pretrained_value_of_Omega_G0) == number_of_Omegas + 1
    if pretrained_value_of_S_config_excess_omegas != None:
        assert len(pretrained_value_of_S_config_excess_omegas) == number_of_S_config_excess_omegas
    assert n_i_LixHM == len(pretrained_value_of_g_i_list_HM)
    assert len(pretrained_value_of_g_i_list_HM) == len(pretrained_value_of_g_i_list_LiHM)

    from .energy import GibbsFE_PDOS as GibbsFE
    
    working_dir = os.getcwd()
    os.chdir(working_dir)
    try:
        os.mkdir("records")
    except:
        pass
    with open("log",'w') as fin:
        fin.write("")

    # convert pretrained value into torch.tensor
    g_i_list_Li = torch.tensor(pretrained_value_of_g_i_list_Li) 
    g_i_list_HM = torch.tensor(pretrained_value_of_g_i_list_HM)
    g_i_list_LiHM = torch.tensor(pretrained_value_of_g_i_list_LiHM) 

    Theta_max_Li = solve_Theta_max(g_i_list = g_i_list_Li, is_x_dependent = False, style = style)
    Theta_max_Li = Theta_max_Li.detach()
    Theta_max_HM = solve_Theta_max(g_i_list = g_i_list_HM, is_x_dependent = False, style = style)
    Theta_max_HM = Theta_max_HM.detach()
    Theta_max_LiHM = solve_Theta_max(g_i_list = g_i_list_LiHM, is_x_dependent = False, style = style)
    Theta_max_LiHM = Theta_max_LiHM.detach()

    print(Theta_max_Li,Theta_max_HM,  Theta_max_LiHM)
    
    # read data
    x1, mu1 = read_OCV_data(datafile1_name)
    if datafile2_name != None:
        x2, mu2 = read_OCV_data(datafile2_name)
    if datafile3_name != None:
        x3, mu3 = read_OCV_data(datafile3_name)
    if datafile_dsdx_name != None:
        x_measured, dsdx_measured = read_in_dOCV_dT_function(
                                        datafile_name=datafile_dsdx_name, 
                                        T=T_dsdx)

    
    ## declare all params
    # excess enthalpy params (Hmix)
    params_list = []
    for _ in range(0, number_of_Omegas):
        if pretrained_value_of_Omega_G0 == None:
            Omegai_start = np.random.randint(Omegas_rand_range[0], Omegas_rand_range[1])
            params_list.append(nn.Parameter( torch.from_numpy(np.array([Omegai_start],dtype="float32")) )) 
        else:
            params_list.append(nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_Omega_G0[_]][0],dtype="float32")) )) 
    if pretrained_value_of_Omega_G0 == None:
        G0_start = np.random.randint(G0_rand_range[0], G0_rand_range[1]) # G0 is the pure substance gibbs free energy 
    else:
        G0_start = np.array([pretrained_value_of_Omega_G0[-1]],dtype="float32")[0]
    G0 = nn.Parameter( torch.from_numpy(np.array([G0_start],dtype="float32")) ) 
    params_list.append(G0)
    # omegas for excess configurational entropy (Smix)
    S_config_params_list = []
    if pretrained_value_of_S_config_excess_omegas == None:
        for _ in range(0, number_of_S_config_excess_omegas):
            S_config_params_list.append(nn.Parameter( torch.from_numpy(np.array([np.random.randint(-100,100)*0.01],dtype="float32")) ) )
    else:
        for _ in range(0, number_of_S_config_excess_omegas):
            S_config_params_list.append(nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_S_config_excess_omegas[_]],dtype="float32")) ) )
    # g_ij for LixHM PDOS & vibrational entropy.  (Hvib and Svib)
    g_ij_list_LixHM = []
    if pretrained_value_of_g_ij_list_LixHM != None:
        for i in range(0, len(pretrained_value_of_g_ij_list_LixHM)):
            g_i_now = []
            for j in range(0, len(pretrained_value_of_g_ij_list_LixHM[i])):
                g_i_now.append(nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_g_ij_list_LixHM[i][j]],dtype="float32"))  ))
            g_ij_list_LixHM.append(g_i_now)
    else:
        for i in range(0, n_i_LixHM):
            g_i_now = []
            for j in range(0, n_j_LixHM):
                g_ij_start = np.random.uniform(low=-0.1, high=0.1)
                g_i_now.append(nn.Parameter( torch.from_numpy(np.array([g_ij_start],dtype="float32")) ))
            g_ij_list_LixHM.append(g_i_now)
    ## define a total params list for GibbsFE function as input
    total_params_list = [params_list, S_config_params_list, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM, Theta_max_Li]
     
    
    # init optimizer
    params_list_other_than_H_mix_excess = []
    # load S_mix params
    for item in S_config_params_list:
        params_list_other_than_H_mix_excess.append(item)
    # load H_vib & S_vib params
    for i in range(0, len(g_ij_list_LixHM)):
        single_list = g_ij_list_LixHM[i]
        for item in single_list:
            params_list_other_than_H_mix_excess.append(item)
    # we also don't optimize g_i_list of Li and HM, they are pre-optimized
    # before a pre-set epoch, only H_mix is optimized since other parameters are pre-optimized. After that, all params are optimized
    optimizer_before_critical_epochs = optim.Adam([{'params': params_list, 'lr': learning_rate}] )
    optimizer_after_critical_epochs = optim.Adam([{'params': params_list, 'lr': learning_rate},
                                                  {'params': params_list_other_than_H_mix_excess, 'lr': learning_rate_other_than_H_mix_excess},] )
    
    
    # begin training
    loss = 9999.9 # init total loss
    epoch = -1
    while loss > loss_threshold and epoch < total_training_epochs:
        # use current params to calculate predicted phase boundary
        epoch = epoch + 1
        # clean grad info
        if epoch <= epoch_optimize_params_other_than_H_mix_only_after_this_epoch:
            optimizer_before_critical_epochs.zero_grad()
        else:
            optimizer_after_critical_epochs.zero_grad()
        # init loss components
        loss = 0.0 # init total loss
        # for datafile 1
        if datafile2_name != None and datafile3_name != None:
            # we do them parallel 
            def worker_function(GibbsFE, total_params_list, mu_x_T_list, end_points = [0,1], is_clamp = True, style = style, quaderature_points = quaderature_points):
                mu = mu_x_T_list[0]
                x = mu_x_T_list[1]
                T = mu_x_T_list[2]
                phase_boundary_fixed_point = _get_phase_boundaries(GibbsFE, total_params_list, T, end_points = end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points, ngrid=99)
                loss_collocation = collocation_loss_all_pts(mu, x, T, phase_boundary_fixed_point, GibbsFE, total_params_list, style = style, quaderature_points = quaderature_points)
                return loss_collocation

            mu_x_T_list_inputs = [ [mu1, x1, T1], [mu2, x2, T2], [mu3, x3, T3] ]
            futures = [torch.jit.fork(worker_function, GibbsFE, total_params_list, mu_x_T, end_points, is_clamp, style, quaderature_points) for mu_x_T in mu_x_T_list_inputs]
            loss_collocation_list = [torch.jit.wait(f) for f in futures]
            loss_collocation_1 = loss_collocation_list[0]
            loss_collocation_2 = loss_collocation_list[1]
            loss_collocation_3 = loss_collocation_list[2]
            loss = loss + loss_collocation_1 + loss_collocation_2 + loss_collocation_3
        else:
            phase_boundary_fixed_point = _get_phase_boundaries(GibbsFE, total_params_list, T1, end_points = end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points, ngrid=99)
            loss_collocation_1 = collocation_loss_all_pts(mu1, x1, T1, phase_boundary_fixed_point, GibbsFE, total_params_list, style = style, quaderature_points = quaderature_points)
            loss = loss + loss_collocation_1
            # datafile 2
            if datafile2_name != None:
                phase_boundary_fixed_point2 = _get_phase_boundaries(GibbsFE, total_params_list, T2, end_points = end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points, ngrid=99)
                loss_collocation_2 = collocation_loss_all_pts(mu2, x2, T2, phase_boundary_fixed_point, GibbsFE, total_params_list, style = style, quaderature_points = quaderature_points)
                loss = loss + loss_collocation_2
            else:
                loss_collocation_2 = 0
            # datafile 3
            if datafile3_name != None:
                phase_boundary_fixed_point3 = _get_phase_boundaries(GibbsFE, total_params_list, T3, end_points = end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points, ngrid=99)
                loss_collocation_3 = collocation_loss_all_pts(mu3, x3, T3, phase_boundary_fixed_point, GibbsFE, total_params_list, style = style, quaderature_points = quaderature_points)
                loss = loss + loss_collocation_3
            else:
                loss_collocation_3 = 0
        # dsdx file
        if epoch <= epoch_optimize_params_other_than_H_mix_only_after_this_epoch:
            loss_entropy = 0.0
            loss_dsdx = 0.0
        else:
            if datafile_dsdx_name != None:
                # calculate entropy loss
                loss_entropy, \
                loss_dsdx, \
                loss_s_config_leq_0, \
                loss_s_config_geq_upper_bound, \
                loss_PDOS_greater_than_0, \
                loss_PDOS_two_side, \
                dsdx_calculated\
                = calc_loss_entropy(x_measured, dsdx_measured, T_dsdx, \
                            S_config_params_list, n_list,
                            g_ij_list_LixHM, \
                            g_i_list_HM, Theta_max_HM, \
                            g_i_list_LiHM, Theta_max_LiHM, \
                            g_i_list_Li, Theta_max_Li, \
                            quaderature_points=quaderature_points, style = style,
                            is_g_ij_collapse_to_g_i_at_end_points = is_g_ij_collapse_to_g_i_at_end_points)
                loss = loss + alpha_dsdx * loss_entropy 
            else:
                loss_entropy = 0.0
                loss_dsdx = 0.0
        # constraint on g_ij_list at end points (x=0 and 1), it should be fitted to the Cp data of measured HM and LiHM
        if epoch <= epoch_optimize_params_other_than_H_mix_only_after_this_epoch:
            loss_Cp_HM = 0.0 
            loss_Cp_LiHM = 0.0
        else:
            if is_g_ij_collapse_to_g_i_at_end_points == False:
                # instead we constrain g_ij with measured Cp data
                loss_Cp_HM = calc_loss_Cp_of_end_member(Ts = T_HM_data,
                            Cps_measured = Cp_HM_data,
                            g_ij_matrix = g_ij_list_LixHM,
                            quaderature_points=quaderature_points,
                            style = style,
                            x = torch.tensor([0.0]),
                            )
                loss_Cp_LiHM = calc_loss_Cp_of_end_member(Ts = T_LiHM_data,
                            Cps_measured = Cp_LiHM_data,
                            g_ij_matrix = g_ij_list_LixHM,
                            quaderature_points=quaderature_points,
                            style = style,
                            x = torch.tensor([1.0]),
                            )
            else:
                loss_Cp_HM = 0.0
                loss_Cp_LiHM = 0.0
        loss = loss + loss_Cp_HM * alpha_Cp_regularization + loss_Cp_LiHM * alpha_Cp_regularization
        # calculate phase diagram loss
        if is_learn_from_phase_diagram == True:
            loss_pd = calc_loss_from_pd(phase_boundaries_from_phase_diagram, GibbsFE, total_params_list, end_points = [0,1], is_clamp = True, style = "Legendre", quaderature_points = 20, ngrid=99)
            loss = loss + loss_pd*alpha_phase_boundary
        else:
            loss_pd = 0.0
        # backprop & update
        loss.backward()
        if epoch <= epoch_optimize_params_other_than_H_mix_only_after_this_epoch:
            optimizer_before_critical_epochs.step()
        else:
            optimizer_after_critical_epochs.step()
        # print output
        output_txt = "Epoch %3d  Loss %.4f  OCV1 %.4f  OCV2 %.4f  OCV3 %.4f  loss_entropy %.4f (dSdx_col %.4f) Cp_HM %.4f Cp_LiHM %.4f PD %.4f    " %(epoch, loss, loss_collocation_1, loss_collocation_2, loss_collocation_3, loss_entropy, loss_dsdx, loss_Cp_HM, loss_Cp_LiHM, loss_pd)
        # H excess params
        for i in range(0, len(params_list)-1):
            output_txt = output_txt + "Omega%d %.4f "%(i, params_list[i].item())
        output_txt  = output_txt + "G0 %.4f "%(params_list[-1].item())
        # S_config_excess params
        for i in range(0, len(S_config_params_list)): 
            output_txt = output_txt + "omega%d %.8f "%(i, S_config_params_list[i].item())
        # H_vib and S_vib params
        for i in range(0, len(g_ij_list_LixHM)): 
            for j in range(0, len(g_ij_list_LixHM[0])):
                output_txt = output_txt + "g%d%d %.8f "%(i, j, g_ij_list_LixHM[i][j].item())   
        output_txt = output_txt + "      "
        print(output_txt)
        with open("log",'a') as fin:
            fin.write(output_txt)
            fin.write("\n")
        # check training for every 100 epochs
        if epoch % 100 == 0:
            phase_boundary_fixed_point = _get_phase_boundaries(GibbsFE, total_params_list, T1, end_points = end_points, is_clamp = is_clamp, style = style, quaderature_points = quaderature_points, ngrid=99)
            # # draw the fitted results, but only draw datafile 1
            mu_pred = []
            for i in range(0, len(x1)):
                x_now = x1[i]
                mu_now = mu1[i]
                x_now = x_now.requires_grad_()
                g_now = GibbsFE(x_now, T1, total_params_list, style = style, quaderature_points = quaderature_points)
                mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
                mu_pred.append(mu_pred_now.detach().numpy())
            mu_pred = np.array(mu_pred)
            SOC = x1.clone().numpy()
            # plot figure
            plt.figure(figsize=(5,4))
            # plot the one before common tangent construction
            U_pred_before_ct = mu_pred/(-96485)
            plt.plot(SOC, U_pred_before_ct, 'k--', label="Prediction Before CT Construction")
            # plot the one after common tangent construction
            mu_pred_after_ct = []
            # see if x is inside any gaps
            def _is_inside_gaps(_x, _gaps_list):
                _is_inside = False
                _index = -99999
                for i in range(0, len(_gaps_list)):
                    if _x >= _gaps_list[i][0] and _x <= _gaps_list[i][1]:
                        _is_inside = True
                        _index = i
                        break
                return _is_inside, _index
            # pred
            for i in range(0, len(x1)):
                x_now = x1[i]
                mu_now = mu1[i]
                is_inside, index = _is_inside_gaps(x_now, phase_boundary_fixed_point)
                if is_inside == False:
                    # outside miscibility gap 
                    mu_pred_after_ct.append(mu_pred[i])
                else: 
                    # inside miscibility gap
                    x_alpha = phase_boundary_fixed_point[index][0]
                    x_beta = phase_boundary_fixed_point[index][1]
                    ct_pred = (GibbsFE(x_alpha, T1, total_params_list, style = style, quaderature_points = quaderature_points) - GibbsFE(x_beta, T1, total_params_list, style = style, quaderature_points = quaderature_points))/(x_alpha - x_beta) 
                    if torch.isnan(ct_pred) == False:
                        mu_pred_after_ct.append(ct_pred.clone().detach().numpy()[0]) 
                    else:
                        mu_pred_after_ct.append(mu_pred[i])
            mu_pred_after_ct = np.array(mu_pred_after_ct)
            U_pred_after_ct = mu_pred_after_ct/(-96485)
            plt.plot(SOC, U_pred_after_ct, 'r-', label="Prediction After CT Construction")
            U_true_value = mu1.numpy()/(-96485) # plot the true value
            plt.plot(SOC, U_true_value, 'b-', label="True OCV")
            plt.xlim([0,1])
            plt.ylim(records_y_lims)
            plt.legend()
            plt.xlabel("SOC")
            plt.ylabel("OCV")
            fig_name = "At_Epoch_%d_datafile1.png" %(epoch)
            os.chdir("records")
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close()
            os.chdir("../")
    print("training complete")
    return 0