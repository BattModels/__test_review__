import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from .solver import FixedPointOperation, FixedPointOperation1D, newton_raphson # FixedPointOperationForwardPass

global  _eps
_eps = 1e-7



"""
The basic theory behind:
For an intercalation reaction Li+ + e- + HM = Li-HM (HM is host material), we have
mu_Li+ + mu_e- + mu_HM = mu_Li-HM
while  mu_e- = -N_A*e*OCV = -F*OCV (OCV is the open circut voltage), 
and mu_Li-HM can be expressed with the gibbs free energy for mixing (pure+ideal+excess), we have
-mu_e- = F*OCV = mu_Li+ + mu_HM - mu_Li-HM
i.e. 
mu_e- = -F*OCV = -mu_Li+ - mu_HM + mu_Li-HM 
We want to fit G_guess via diff thermo such that mu_guess = dG_guess/dx = mu_e- = -F*OCV
"""
"""
For OCV curves, since F*OCV = - mu_e-, the stability criteria according to 2nd law of thermo is dmu_e-/dx > 0, i.e.  d OCV / dx < 0, 
i.e. whenever d OCV /dx >0, this region is unstable.
"""

"""
We have G_Li+ + G_e- + G_HM = G_Li-HM
while G_Li-HM = x*G0 + (1-x)*G1 + R*T*(x*ln(x) + (1-x)*ln(1-x)) + L0*x*(1-x)
Therefore
G_e- = G_Li-HM - G_Li+ - G_HM = (redefine a ref state) x*G0 + (1-x)*0 + R*T*(x*ln(x) + (1-x)*ln(1-x)) + L0*x*(1-x)
"""


def GibbsFE_PDOS(x, T, params_list, style = "Legendre", quaderature_points = 20):
    """
    Expression for Delta Gibbs Free Energy of charging / discharging process
    Delta_G = H_mix + H_vib - T(S_config + S_vib)
    _____
    Input params:
    x: Li-filling fraction
    T: temperature (Kelvin)
    params_list: in the sequence of [enthalpy_mixing_params_list, config_entropy_params_list, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_HM_value, Theta_max_Li_value]
    style: style of polynomial, can be "Legendre" (default) or "Chebyshev"
    quaderature_points: number of sampled points when doing Gauss-Legendre quaderature
    """
    if style == "Legendre":
        poly_eval_function = legendre_poly_recurrence
    elif style == "Chebyshev":
        poly_eval_function = chebyshev_poly_recurrence
    else:
        print("ERROR: polynomial style %s not supported." %(style))
        exit()
    enthalpy_mixing_params_list = params_list[0]
    config_entropy_params_list = params_list[1]
    n_list = params_list[2]
    g_ij_list_LixHM = params_list[3]
    g_i_list_HM = params_list[4]
    g_i_list_Li = params_list[5]
    Theta_max_HM = params_list[6]
    Theta_max_Li = params_list[7]
    # need to solve Theta_max_LixHM from g_ij_list_LixHM
    x = torch.clamp(x, min=_eps, max=1.0-_eps)
    Theta_max_LixHM = solve_Theta_max(g_ij_matrix = g_ij_list_LixHM, is_x_dependent = True, x = x, style = style)
    # H_mix
    G0 = enthalpy_mixing_params_list[-1]
    G = x*G0 + (1-x)*0.0 
    t = 2 * x -1 # Transform x to (2x-1) for legendre expansion
    Pn_values = poly_eval_function(t, len(enthalpy_mixing_params_list)-2)  # Compute Legendre polynomials up to degree len(coeffs) - 1 # don't need to get Pn(G0)
    for i in range(0, len(enthalpy_mixing_params_list)-1):
        G = G + x*(1-x)*(enthalpy_mixing_params_list[i]*Pn_values[i])
    # S_mix (S_ideal + S_excess)
    # S_ideal
    G = G - T*(-8.314)*(x*torch.log(x)+(1-x)*torch.log(1-x)) 
    # S_excess, i.e. excess configurational entropy
    t = 2 * x -1 # Transform x to (2x-1) for legendre expansion
    Pn_values = poly_eval_function(t, len(config_entropy_params_list)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1 
    for i in range(0, len(config_entropy_params_list)):
        G = G - T*(-8.314)*(x*torch.log(x)+(1-x)*torch.log(1-x))*(config_entropy_params_list[i]*Pn_values[i])
    # S_vib, i.e. vibrational entropy
    G = G - T*calculate_S_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = quaderature_points, style=style)
    # H_vib, i.e. vibrational enthalpy corresponds to the vibrational entropy
    # we need to satisfy dH/dT = T*dS/dT
    G = G + calculate_H_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = quaderature_points, style=style)
    return G
    

def calculate_S_config_total(x, S_config_params_list, style = "Legendre"):
    """ 
    S_total = -R(x*log(x) + (1-x)*log(1-x))*(1+\sum_{i=0}^n \Omega_i P_i(1-2x))
    where P_i is i-th order Legendre/Chebyshev polynomial
    S_config_params_list = [omega0, omega1, ... omega_n], length n+1, up to nth order
    """
    S_ideal = -8.314*(x*torch.log(x) + (1-x)*torch.log(1-x) )
    S_expand = 1.0
    t = 2*x-1
    if style == "Legendre":
        Pn_values = legendre_poly_recurrence(t, len(S_config_params_list)-1) 
    elif style == "Chebyshev":
        Pn_values = chebyshev_poly_recurrence(t, len(S_config_params_list)-1)
    for i in range(0, len(S_config_params_list)):
        S_expand = S_expand + S_config_params_list[i] * Pn_values[i]
    S_total = S_ideal * S_expand
    return S_total, S_ideal, S_expand


def _PDOS_evaluator(Theta, g_ij_matrix = None, g_i_list = None, Theta_max = None, is_x_dependent = False, x = None, style = "Legendre"):
    """ 
    evaluate \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1)  value at given Theta & x
    
    PDOS expression is 
    g(Theta, x) = 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n (\sum_{j=0}^m g_ij P_j(2*x-1)) P_i(2*Theta/Theta_max-1)
    When PDOS is x-independent, it becomes
    g(Theta) = 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n g_i P_i(2*Theta/Theta_max-1)
    
    Explaination: 
    PDOS is orignally 
    g(omega) = (omega/omega_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*omega/omega_max-1), 0<omega<omega_max, 
    the (omega/omega_max)**2 prefactor ensures g(omega) \propto omega**2 at low frequencies,
    the 10**-2 *  prefactor ensures g_i is on the order of O(1)
    and the constraint to g(omega) is \int_0^omega_max g(omega) domega = 3N_A
    Now, define Theta = h_bar/kB*omega, where h_bar = h/2pi = 6.62607015*10**-34, kB = 1.380649*10**-23
    we have 
    kB/h_bar * \int_0^omega_max 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1) dh_bar/kB*omega = 3N_A
    kB/(h_bar*N_A) * \int_0^Theta_max (Theta/Theta_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1) dTheta = 3
    redefine g_i(x) = kB/(h_bar*N_A) * g_i(x), i.e. absorbing kB/(h_bar*N_A) into g_i, 
    we have the constraint to be 
    \int_0^Theta_max 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1) dTheta = 3
    ________________
    Input Parameters:
    Theta: value of frequency, scaled by h/k_B (i.e. Theta = h*omega/kB)
    g_ij_matrix: contains all params for g_ij (only if when is_x_dependent = True)
    g_i_list: contains all params for g_i (only if when is_x_dependent = False)
    Theta_max: maximum value of Theta, subject to constraint \int_0^Theta_max g(Theta) dTheta = 3. Beyond which, we assume g(Theta) is always 0
    is_x_dependent: whether the PDOS expression is Li-filling-fraction (x) dependent
    x: value of Li-filling-fraction if PDOS is x dependent
    style: polynomial style of P_j if PDOS is x dependent, can be Legendre (default) or Chebyshev
    """
    if is_x_dependent == False:
        """ e.g. Li metal anode """
        _t = 2*Theta/Theta_max -1 # scale [0, Theta_max] to [-1,1]
        if style == "Legendre":
            Pn_values = legendre_poly_recurrence(_t, len(g_i_list)-1) 
        elif style == "Chebyshev":
            Pn_values = chebyshev_poly_recurrence(_t, len(g_i_list)-1)
        # calculate g(Theta)
        g = 0.0
        # print(Theta_Li)
        for i in range(0, len(g_i_list)):
            g = g + g_i_list[i] *Pn_values[i]   
    else:
        """ LixHM """
        n_i = len(g_ij_matrix)
        n_j = len(g_ij_matrix[0])
        # first calculate the value of g_i, i.e. summing over j
        g_i_value_list = []        
        for i in range(0, n_i):  
            _t = 2*x -1
            if style == "Legendre":
                Pn_values = legendre_poly_recurrence(_t, n_j-1) 
            elif style == "Chebyshev":
                Pn_values = chebyshev_poly_recurrence(_t, n_j-1)
            g_i = 0.0
            for j in range(0, n_j):
                g_i = g_i + g_ij_matrix[i][j]*Pn_values[j]
            g_i_value_list.append(g_i)
        # now summing over i
        g = 0.0
        _t = 2*Theta/Theta_max -1
        if style == "Legendre":
            Pn_values = legendre_poly_recurrence(_t, n_i-1) 
        elif style == "Chebyshev":
            Pn_values = chebyshev_poly_recurrence(_t, n_i-1)
        # calculate g(Theta)
        g = 0.0
        for i in range(0, n_i):
            g = g + g_i_value_list[i] *Pn_values[i]   
    ## ensuring that at low freqencies, g is propotional to omega squared
    g = g *1.0
    return g




def _calculate_mole_of_atoms_from_PDOS(Theta_max = None, g_ij_matrix = None, g_i_list = None, quaderature_points=10, is_x_dependent = False, x = None, style = "Legendre"):
    """" 
    Calculate the mole of atoms given PDOS, assuming g(omega) already satisfies constraint
    Constraint is 
    \int_0^Theta_max g(Theta,x) dTheta = 3
    Therefore the mole of atoms N is given by
    N = 1/3 \int_0^\Theta_max g(Theta,x) dTheta
    i.e. N = 1/3 \int_0^Theta_max 10**-2 * (Theta/Theta_max)**2 * \sum_{i=0}^n g_i(x) P_i(2*Theta/Theta_max-1) dTheta 
    To perform G-L integration, we need to map integration interval from [0, Theta_max] to [-1,1]
    define y = 2*Theta/Theta_max-1, i.e. Theta = Theta_max*(y+1)/2
    we have 
    N = 1/3 * Theta_max/2 * 10**-2 * \int_-1^1 ((y+1)/2)**2 * \sum_{i=0}^n g_i(x) P_i(y) dy
    NOTE that we need n+2+1 quaderature points to do the exact evaluation, given the PDOS expression is purely polynomials
    
    Inputs:
    g_ij_matrix: contains all params for g_ij (only if when is_x_dependent = True)
    g_i_list: contains all params for g_i (only if when is_x_dependent = False)
    Theta_max_list: maximum value of Theta (only if when is_x_dependent = True).
    Theta_max_value: maximum value of Theta (only if when is_x_dependent = False). Beyond which, we assume g(Theta) is always 0
    is_x_dependent: whether the PDOS expression is Li-filling-fraction (x) dependent
    x: value of Li-filling-fraction if PDOS is x dependent
    quaderature_points: how many points in Gauss-Legendre quaderature
    style: polynomial style of P_j if PDOS is x dependent, can be Legendre (default) or Chebyshev
    """
    ys, weights = np.polynomial.legendre.leggauss(quaderature_points)
    ys = torch.from_numpy(ys.astype("float32"))
    weights = torch.from_numpy(weights.astype("float32"))
    N = 0.0
    for i in range(0, len(ys)):
        y_now = ys[i] 
        weight_now = weights[i]
        """ y = 2*Theta/Theta_max - 1 """
        Theta = Theta_max/2*(1+y_now)
        if is_x_dependent == False:
            """ e.g. Li metal anode """
            g_omega_now = _PDOS_evaluator(Theta, g_i_list = g_i_list, Theta_max = Theta_max, is_x_dependent = False, style = style)
        else:
            """ LixHM """
            g_omega_now = _PDOS_evaluator(Theta, g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, is_x_dependent = True, x=x, style = style)
        """ N = 1/3 * Theta_max/2 * 10**-2 *  \int_-1^1 ((y+1)/2)**2 * \sum_{i=0}^n g_i(x) P_i(y) dy """
        N = N + 1/3* Theta_max/2 * 10**-2 * weight_now *  ((y_now+1)/2)**2 *g_omega_now
    return N


def solve_Theta_max(g_ij_matrix = None, g_i_list = None, is_x_dependent = False, x = None, style = "Legendre"):
    """ 
    Given g_ij_matrix or g_i_list, solve Theta_max that satisfies constraint
    \int_0^Theta_max g(Theta,x) dTheta = 3
    Plug-in expression of g(Theta, x), we have 
    Theta_max/2 * 10**-2 * \sum_{i=0}^n g_i(x) \int_-1^1 ((y+1)/2)**2 * P_i(y) dy  = 3
    we need to evaluate \int_-1^1 ((y+1)/2)**2 * P_i(y) dy  analytically, and this can be done by decomposing ((y+1)/2)**2  into P_i s:
    ((y+1)/2)**2 = 1/6 P_2 + 1/2 P_1 + 1/3 P_0, where P_2 = 3/2 y**2 - 1/2, P_1 = y, P_0 = 1
    For Legendre polynomials, according to its orthonomality, we have
    \int_-1^1 P_n*P_m dy = 0 if n!=m,  or 2/(2*n+1) if n=m
    With this, we can re-write the integration 
    Theta_max/2 * 10**-2 * \sum_{i=0}^n g_i(x) \int_-1^1 ((y+1)/2)**2 * P_i(y) dy  = 3
    as 
    Theta_max/2 * 10**-2 * \int_-1^1 (1/6 P_2 + 1/2 P_1 + 1/3 P_0) * \sum_{i=0}^n g_i(x) P_i(y) dy  = 3
    Therefore
    Theta_max/2 * 10**-2 * \int_-1^1 (1/6*g_2(x)*P_2**2 + 1/2*g_1(x)*P_1**2 + 1/3*g_0(x)*P_0**2) dy  = 3
    We have
    \int_-1^1 P_0**2 dy = 2, \int_-1^1 P_1**2 dy = 2/3, \int_-1^1 P_2**2 dy = 2/5
    Therefore we have
    Theta_max/2 * 10**-2 *  (1/6*g_2(x)*2/5 + 1/2*g_1(x)*2/3 + 1/3*g_0(x)*2)   = 3
    i.e.
    Theta_max/2 * 10**-2 *  (1/15*g_2(x) + 1/3*g_1(x) + 2/3*g_0(x)) = 3
    i.e.  we can solve Theta_max analytically as
    Theta_max  = 600/( 1/15*g_2(x) + 1/3*g_1(x) + 2/3*g_0(x) ) 
    
    """
    if is_x_dependent == True:
        # convert g_ij_matrix into g_i_list given x
        g_i_list = []
        n_i = len(g_ij_matrix)
        n_j = len(g_ij_matrix[0])
        for i in range(0, n_i):  
            _t = 2*x-1
            if style == "Legendre":
                Pn_values = legendre_poly_recurrence(_t, n_j-1) 
            elif style == "Chebyshev":
                Pn_values = chebyshev_poly_recurrence(_t, n_j-1)
            g_i = 0.0
            for j in range(0, n_j):
                g_i = g_i + g_ij_matrix[i][j]*Pn_values[j]
            g_i_list.append(g_i)
    else:
        pass
    # Now see if g_i_list is long enough
    if len(g_i_list) == 1:
        # only have g_0 term, we say g_1 and g_2 are 0
        g_i_list.append(0.0)
        g_i_list.append(0.0)
    elif len(g_i_list) == 2:
        # only have g_0 and g_1 term, we say g_2 is 0
        g_i_list.append(0.0)
    ## Now we can do integration
    Theta_max = 600.0/( 1/15*g_i_list[2] + 1/3*g_i_list[1] + 2/3*g_i_list[0] ) 
    return Theta_max


def _H_vib_PDOS_quaderature_model(g_ij_matrix = None, g_i_list = None, Theta_max = None, quaderature_points=10, is_x_dependent = False, x = None, T = 320, style = "Legendre"):
    """" 
    calculate H_vib given the expression of g(Theta)
    H_vib = \int_0^\infty [0.5*h*omega + h*omega/(exp(h*omega/(kB*T)) -1) ] g(omega) domega
    where g(omega) is PDOS
    To use Gauss-Legendre Quad, we use Theta_max to approximate \infty, i.e. beyond Theta_max g(Theta) is always 0
    define Theta = h*omega/kB, we have
    H_vib = \int_0^\Theta_max k_B [0.5*h*omega/k_B + h*omega/k_B/(exp(h*omega/(kB*T)) -1) ] g(omega)*k_B/h dh*omega/k_B
    H_vib = \int_0^\Theta_max k_B [0.5*Theta + Theta/(exp(Theta/T) -1) ] g(omega)*k_B/h dTheta
    express PDOS using Theta, and absorb *k_B/(h*N_A) into g(omega) (i.e. scale g_ij) which is consistent with the definition of g(Theta): 
    H_vib = R * \int_0^\Theta_max [0.5*Theta + Theta/(exp(Theta/T) -1) ]* g(Theta) dTheta
    H_vib = R * \int_0^\Theta_max [0.5*Theta + Theta/(exp(Theta/T) -1) ]* 10**-2 *  (Theta/Theta_max)**2 * \sum_{i=0}^n g_i P_i(2*Theta/Theta_max-1)  dTheta
    we use Gauss-Legendre quaderature to do this integration, 
    define y = 2*Theta/Theta_max - 1, i.e. Theta = Theta_max*(y+1)/2
    we have 
    H_vib = R * Theta_max**2/2* 10**-2 *  \int_-1^1 [(y+1)/4 + (y+1)/2 * 1/(exp(Theta_max*(y+1)/(2*T)) -1) ]* ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
    The sum of g_i*Pi part of g(Theta) can be evaluated with _PDOS_evaluator
    Theta_max can be evaluated with solve_Theta_max 
    
    Input:
    g_ij_matrix: contains all params for g_ij (only if when is_x_dependent = True)
    g_i_list: contains all params for g_i (only if when is_x_dependent = False)
    Theta_max: solved Theta_max given PDOS parameters (solved outside). Beyond which, we assume g(Theta) is always 0
    is_x_dependent: whether the PDOS expression is Li-filling-fraction (x) dependent
    x: value of Li-filling-fraction if PDOS is x dependent
    T: temperature
    style: polynomial style of P_j if PDOS is x dependent, can be Legendre (default) or Chebyshev
    """
    ys, weights = np.polynomial.legendre.leggauss(quaderature_points)
    ys = torch.from_numpy(ys.astype("float32"))
    weights = torch.from_numpy(weights.astype("float32"))
    H_vib = 0.0  
    for i in range(0, len(ys)):
        y_now = ys[i] 
        weight_now = weights[i]
        Theta = Theta_max/2*(1+y_now)
        if is_x_dependent == False:
            """ e.g. Li metal anode """
            g_omega_now = _PDOS_evaluator(Theta, g_i_list = g_i_list, Theta_max = Theta_max, is_x_dependent = False, style = style)
        else:
            """ LixHM """
            g_omega_now = _PDOS_evaluator(Theta, g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, is_x_dependent = True, x=x, style = style)
        # H_vib = R * Theta_max**2/2* 10**-2 *  \int_-1^1 [(y+1)/4 + (y+1)/2 * 1/(exp(Theta_max*(y+1)/(2*T)) -1) ]* ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
        f_of_x = ((y_now+1)/4 + (y_now+1)/2 * 1/(torch.exp(Theta_max*(y_now+1)/(2*T)) -1))* ((y_now+1)/2)**2 * g_omega_now
        H_vib = H_vib + 8.314 * Theta_max**2/2* 10**-2 *  weight_now*f_of_x     
    return H_vib


def calculate_H_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = 20, style="Legendre"):
    """
    Corresponding H_vib of the reaction
    we need to satisfy dH/dT = TdS/dT, this is guaranteed by using the same set of PDOS params of LixHM, HM and Li
    
    x is the filling fraction
    T is the temperature
    n_list is how many moles of atoms are there in 1 mole of substance, the first element is wrong (should be deduced from the other two elements)
    g_ij_list_LixHM is the PDOS params for LixHM
    g_i_list_HM is the PDOS params for HM
    g_i_list_Li is the PDOS params for Li metal 
    Theta_max_LixHM, Theta_max_HM and Theta_max_Li should be solved outside and input here (Theta_max_Li and Theta_max_HM is always unchanged, so solve it at the beginning of training to save compute)
    quaderature_points: number of sampling points in Gauss-Legendre quaderatures 
    style: polynomial style, can be Legendre or Chebyshev
    """
    H_vib = 0.0 
    ## we have h_excess = h_LixHM - h_HM - h*s_Li
    ## LixHM: there is 1 mole of HM, and x mole of Li
    H_vib = H_vib + (1.0*n_list[1] + x*n_list[2])* _H_vib_PDOS_quaderature_model(x=x, T=T, g_ij_matrix = g_ij_list_LixHM, Theta_max = Theta_max_LixHM, quaderature_points=quaderature_points, is_x_dependent = True, style = style)
    ## HM: there is 1 mole of HM
    H_vib = H_vib - (1.0*n_list[1])* _H_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_HM, Theta_max = Theta_max_HM, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    ## Li: there is x mole of Li
    H_vib = H_vib - (x*n_list[2])*  _H_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_Li, Theta_max = Theta_max_Li, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    return H_vib



def _S_vib_PDOS_quaderature_model(g_ij_matrix = None, g_i_list = None, Theta_max = None, quaderature_points=10, is_x_dependent = False, x = None, T = 320, style = "Legendre"):
    """ 
    calculate S_vib given the expression of g(Theta) (define Theta = h*omega/kB)
    S_vib = k_B \int_0^\infty [Theta/T * 1/(exp(Theta/T) -1) - log(1-exp(-Theta/T)) ] g(omega) domega
    where g(omega) is PDOS
    We use Theta_max to approximate \infty, i.e. beyond Theta_max g(Theta) is always 0
    S_vib = R* \int_0^\Theta_max [Theta/T * 1/(exp(Theta/T) -1) - log(1-exp(-Theta/T)) ] g(Theta) dTheta
    S_vib = R* \int_0^\Theta_max [Theta/T * 1/(exp(Theta/T) -1) - log(1-exp(-Theta/T)) ] * 10**-2 *  (Theta/Theta_max)**2 * \sum_{i=0}^n g_i P_i(2*Theta/Theta_max-1)  dTheta
    we use Gauss-Legendre quaderature to do this integration, 
    define y = 2*Theta/Theta_max - 1, i.e. Theta = Theta_max*(y+1)/2
    we have 
    S_vib = R* Theta_max/2 * 10**-2 *  \int_-1^1 [Theta_max*(y+1)/(2*T) * 1/(exp(Theta_max*(y+1)/(2*T)) -1) - log(1-exp(-Theta_max*(y+1)/(2*T))) ] * ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
    The sum of g_i*Pi part of g(Theta) can be evaluated with _PDOS_evaluator
    Theta_max can be evaluated with solve_Theta_max 

    
    """
    ys, weights = np.polynomial.legendre.leggauss(quaderature_points)
    ys = torch.from_numpy(ys.astype("float32"))
    weights = torch.from_numpy(weights.astype("float32"))
    H_vib = 0.0  
    for i in range(0, len(ys)):
        y_now = ys[i] 
        weight_now = weights[i]
        Theta = Theta_max/2*(1+y_now)
        if is_x_dependent == False:
            """ e.g. Li metal anode """
            g_omega_now = _PDOS_evaluator(Theta, g_i_list = g_i_list, Theta_max = Theta_max, is_x_dependent = False, style = style)
        else:
            """ LixHM """
            g_omega_now = _PDOS_evaluator(Theta, g_ij_matrix = g_ij_matrix, Theta_max = Theta_max, is_x_dependent = True, x=x, style = style)
        # S_vib = R* Theta_max/2 * 10**-2 *  \int_-1^1 [Theta_max*(y+1)/(2*T) * 1/(exp(Theta_max*(y+1)/(2*T)) -1) - log(1-exp(-Theta_max*(y+1)/(2*T))) ] * ((y+1)/2)**2 * \sum_{i=0}^n g_i P_i(y)  dy
        f_of_x = (Theta_max*(y_now+1)/(2*T) * 1/(torch.exp(Theta_max*(y_now+1)/(2*T)) -1) - torch.log(1-torch.exp(-Theta_max*(y_now+1)/(2*T)))) * ((y_now+1)/2)**2 * g_omega_now
        H_vib = H_vib + 8.314 * Theta_max/2* 10**-2 *  weight_now*f_of_x     
    return H_vib



def calculate_S_vib_total_PDOS(x, T, n_list, g_ij_list_LixHM, g_i_list_HM, g_i_list_Li, Theta_max_LixHM, Theta_max_HM, Theta_max_Li, quaderature_points = 20, style="Legendre"):
    """
    Corresponding S_vib of the reaction
    we need to satisfy dH/dT = TdS/dT, this is guaranteed by using the same set of PDOS params of LixHM, HM and Li
    
    x is the filling fraction
    T is the temperature
    n_list is how many moles of atoms are there in 1 mole of substance, the first element is wrong (should be deduced from the other two elements)
    g_ij_list_LixHM is the PDOS params for LixHM
    g_i_list_HM is the PDOS params for HM
    g_i_list_Li is the PDOS params for Li metal 
    Theta_max_LixHM, Theta_max_HM and Theta_max_Li should be solved outside and input here (Theta_max_Li and Theta_max_HM is always unchanged, so solve it at the beginning of training to save compute)
    quaderature_points: number of sampling points in Gauss-Legendre quaderatures 
    style: polynomial style, can be Legendre or Chebyshev
    """
    S_vib = 0.0
    ## we have s_excess = s_LixHM - s_HM - x*s_Li
    ## LixHM: there is 1 mole of HM, and x mole of Li
    S_vib = S_vib + (1.0*n_list[1] + x*n_list[2])* _S_vib_PDOS_quaderature_model(x=x, T=T, g_ij_matrix = g_ij_list_LixHM, Theta_max = Theta_max_LixHM, quaderature_points=quaderature_points, is_x_dependent = True, style = style)
    ## HM: there is 1 mole of HM
    S_vib = S_vib - (1.0*n_list[1])* _S_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_HM, Theta_max = Theta_max_HM, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    ## Li: there is x mole of Li
    S_vib = S_vib - (x*n_list[2])*  _S_vib_PDOS_quaderature_model(T=T, g_i_list = g_i_list_Li, Theta_max = Theta_max_Li, quaderature_points=quaderature_points, is_x_dependent = False, style = style)
    return S_vib


















# Function to compute Legendre polynomials using recurrence relation
def legendre_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials up to degree n 
    using the Bonnet's recursion formula (i+1)P_(i+1)(x) = (2i+1)xP_i(x) - iP_(i-1)(x)
    and return all n functions in a list
    """
    P = [torch.ones_like(x), x]  # P_0(x) = 1, P_1(x) = x
    for i in range(1, n):
        P_i_plus_one = ((2 * i + 1) * x * P[i] - i * P[i - 1]) / (i + 1)
        P.append(P_i_plus_one)
    return P

# Function to compute Chebyshev polynomials (first kind) using recurrence relation
def chebyshev_poly_recurrence(x, n):
    """
    Compute the Chebyshev polynomials (first kind) up to degree n 
    using the recursion formula T_(n+1)(x) = 2xT_n(x) - T_(n-1)(x),
    and return all n functions in a list
    """
    T = [torch.ones_like(x), x]  # T_0(x) = 1, T_1(x) = x
    for i in range(1, n):
        T_i_plus_1 = 2*x*T[i] - T[i-1]
        T.append(T_i_plus_1)
    return T







def sampling(GibbsFunction, params_list, T, end_points, sampling_id, ngrid=99, requires_grad = False, is_PDOS_version_G = False, style = "Legendre", quaderature_points = 20):
    """
    Sampling a Gibbs free energy function (GibbsFunction)
    sampling_id is for recognition, must be a interger
    """
    x = np.concatenate((np.array([end_points[0]+_eps]),np.linspace(end_points[0]+1/(ngrid+1),end_points[1]-1/(ngrid+1),ngrid),np.array([end_points[1]-_eps]))) 
    x = torch.from_numpy(x.astype("float32"))
    x = x.requires_grad_()
    if is_PDOS_version_G == False:
        sample = torch.tensor([[x[i], GibbsFunction(x[i], params_list, T), sampling_id] for i in range(0, len(x))])
    else:
        sample = torch.tensor([[x[i], GibbsFunction(x[i], T, params_list, style = style, quaderature_points = quaderature_points), sampling_id] for i in range(0, len(x))])
    return sample



def convex_hull(sample, ngrid=99, tolerance = _eps):
    """
    Convex Hull Algorithm that provides the initial guess for common tangent
    Need NOT to be differentiable
    returning the initial guess for common tangent & corresponding phase id
    Adapted from Pinwe's Jax-Thermo with some modifications
    """
    # convex hull, starting from the furtest points at x=0 and 1 and find all pieces
    base = [[sample[0,:], sample[-1,:]]]
    current_base_length = len(base) # currently len(base) = 1
    new_base_length = 9999999
    base_working = base.copy()
    n_iter = 0
    while new_base_length != current_base_length:
        n_iter = n_iter + 1
        # save historical length of base, for comparison at the end
        current_base_length = len(base)
        # continue the convex hull pieces construction until we find all pieces
        base_working_new=base_working.copy()
        for i in range(len(base_working)):   # len(base_working) = 1 at first, but after iterations on n, the length of this list will be longer
            # the distance of sampling points to the hyperplane formed by base vector
            # sample[:,column]-h[column] calculates the x and y distance for all sample points to the base point h
            # 0:2 deletes the sampling_id
            # t[column]-h[column] is the vector along the hyperplane (line in 2D case)
            # dot product of torch.tensor([[0.0,-1.0],[1.0,0.0]]) and t[column]-h[column] calculates the normal vector of the hyperplane defined by t[column]-h[column]
            h = base_working[i][0]; t = base_working[i][1] # h is the sample point at left side, t is the sample point at right side
            _n = torch.matmul(torch.from_numpy(np.array([[0.0,-1.0],[1.0,0.0]]).astype("float32")), torch.reshape((t[0:2]-h[0:2]), (2,1)))
            # limit to those having x value between the x value of h and t
            left_id = torch.argmin(torch.abs(sample[:,0]-h[0])) + 1 # limiting the searching range within h and t
            right_id = torch.argmin(torch.abs(sample[:,0]-t[0]))
            if left_id == right_id: # it means this piece of convex hull is the shortest piece possible
                base_working_new.remove(base_working[i])
            else:
                # it means it's still possible to make this piece of convex hull shorter
                sample_current = sample[left_id:right_id, :] 
                _t = sample_current[:,0:2]-h[0:2]
                dists = torch.matmul(_t, _n).squeeze()
                if dists.shape == torch.Size([]): # in case that there is only 1 item in dists, .squeeze wil squeeze ALL dimension and make dists a 0-dim tensor
                    dists = torch.tensor([dists])
                # select those underneath the hyperplane
                outer = []
                for _ in range(0, sample_current.shape[0]):
                    if dists[_] < -_eps: 
                        outer.append(sample_current[_,:]) 
                # if there are points underneath the hyperplane, select the farthest one. If no outer points, then this set of working base is dead
                if len(outer):
                    pivot = sample_current[torch.argmin(dists)] # the furthest node below the hyperplane defined hy t[column]-h[column]
                    # after find the furthest node, we remove the current hyperplane and rebuild two new hyperplane
                    z = 0
                    while (z<=len(base)-1):
                        # i.e. finding the plane corresponding to the current working plane
                        diff = torch.max(  torch.abs(torch.cat((base[z][0], base[z][1])) - torch.cat((base_working[i][0], base_working[i][1])))  )
                        if diff < tolerance:
                            # remove this plane
                            base.pop(z) # The pop() method removes the item at the given index from the list and returns the removed item.
                        else:
                            z=z+1
                    # the furthest node below the hyperplane is picked up to build two new facets with the two corners 
                    base.append([h, pivot])
                    base.append([pivot, t])
                    # update the new working base
                    base_working_new.remove(base_working[i])
                    base_working_new.append([h, pivot])
                    base_working_new.append([pivot, t])
                else:
                    base_working_new.remove(base_working[i])
        base_working=base_working_new
        # update length of base
        new_base_length = len(base)
    # find the pieces longer than usual. If for a piece of convex hull, the length of it is longer than delta_x
    delta_x = 1.0/(ngrid+1.0) + tolerance
    miscibility_gap_x_left_and_right = []
    miscibility_gap_phase_left_and_right = []
    for i in range(0, len(base)):
        convex_hull_piece_now = base[i]
        if convex_hull_piece_now[1][0]-convex_hull_piece_now[0][0] > delta_x:
            miscibility_gap_x_left_and_right.append(torch.tensor([convex_hull_piece_now[0][0], convex_hull_piece_now[1][0]]))
            miscibility_gap_phase_left_and_right.append(torch.tensor([convex_hull_piece_now[0][2], convex_hull_piece_now[1][2]]))
    # sort the init guess of convex hull
    left_sides = torch.zeros(len(miscibility_gap_x_left_and_right))
    for i in range(0, len(miscibility_gap_x_left_and_right)):
        left_sides[i] = miscibility_gap_x_left_and_right[i][0]
    _, index =  torch.sort(left_sides)
    miscibility_gap_x_left_and_right_sorted = []
    miscibility_gap_phase_left_and_right_sorted = []
    for _ in range(0, len(index)):
        miscibility_gap_x_left_and_right_sorted.append(miscibility_gap_x_left_and_right[_])
        miscibility_gap_phase_left_and_right_sorted.append(miscibility_gap_phase_left_and_right[_])
    return miscibility_gap_x_left_and_right_sorted, miscibility_gap_phase_left_and_right_sorted    



class CommonTangent(nn.Module):
    """
    Common Tangent Approach for phase equilibrium boundary calculation
    """
    def __init__(self, G, params_list, T = 300, is_PDOS_version_G = False, end_points = [0,1], scaling_alpha = 1e-5, is_clamp = True, style = "Legendre", quaderature_points = 20, f_thres=1e-6):
        super(CommonTangent, self).__init__()
        # self.f_forward = FixedPointOperationForwardPass(G, params_list, T, is_PDOS_version_G=is_PDOS_version_G, end_points=end_points, style=style, quaderature_points = quaderature_points) # define forward operation here
        self.f = FixedPointOperation(G, params_list, T, is_PDOS_version_G=is_PDOS_version_G, end_points=end_points, scaling_alpha = scaling_alpha, style=style, quaderature_points = quaderature_points) # define backward operation here    
        self.solver = newton_raphson
        self.end_points = end_points
        self.scaling_alpha = scaling_alpha
        self.is_clamp = is_clamp
        self.f_thres = f_thres
        self.T = T
        self.G = G
        self.is_PDOS_version_G = is_PDOS_version_G
        self.style=style 
        self.quaderature_points = quaderature_points
        self.params_list = params_list
    def forward(self, x, **kwargs):
        """
        x is the initial guess provided by convex hull
        """
        if x[0]-self.end_points[0] >= _eps*2.0 and self.end_points[1]-x[1] >= _eps*2.0:
            """ 
            This means the miscibility gap does not start or end at endpoints, 
            G
            | x
            |  a    
            |      x     
            |        x          
            |         b                       x 
            |           x              x   
            |              x   x
            |__________________________________ x
            
            this is the normal situation where we can apply common tangent
            to solve the position of a and b
            """
            # Forward pass
            x_star = self.solver(self.f, x, threshold=self.f_thres, end_points = self.end_points, is_clamp=self.is_clamp) # use newton-raphson to get the fixed point
            # if torch.any(torch.isnan(x_star)) == True: # in case that the previous one doesn't work
            #     print("Fixpoint solver failed at T = %d. Use traditional approach instead" %(self.T))
            #     x_star = self.f_forward(x)
            # (Prepare for) Backward pass
            new_x_star = self.f(x_star.requires_grad_()) # go through the process again to get derivative
            # register hook, can do anything with the grad that passed in
            def backward_hook(grad):
                # we use this hook to calculate dz/dtheta, where z is the equilibrium and theta is the learnable params in the model
                if self.hook is not None:
                    self.hook.remove()
                    # torch.cuda.synchronize()   # To avoid infinite recursion
                """
                Compute the fixed point of y = yJ + grad, 
                where y is the new_grad, 
                J=J_f is the Jacobian of f at z_star, 
                grad is the input from the chain rule.
                From y = yJ + grad, we have (I-J)y = grad, so y = (I-J)^-1 grad
                """
                # # Original implementation by Shaojie Bai, DEQ https://github.com/locuslab/deq:
                # new_grad = self.solver(lambda y: autograd.grad(new_x_star, x_star, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), threshold=self.f_thres, in_backward_hood=True)
                # AM Yao: use inverse jacobian as we aren't solving a large matrix here
                I_minus_J = torch.eye(2) - autograd.functional.jacobian(self.f, x_star)
                new_grad = torch.linalg.pinv(I_minus_J)@grad
                return new_grad
            # hook registration
            self.hook = new_x_star.register_hook(backward_hook)
            # all set! return 
            return new_x_star
        else:
            if x[0]-self.end_points[0] < _eps*2.0 and self.end_points[1]-x[1] < _eps*2.0:
                """ 
                this means this whole region is concave, 
                G
                |
                |       
                |             x
                |       x           x
                |   x                       x
                | a                             b
                |
                |__________________________________ x
                In this case, just connecting a and b will give the final convex hull (instead of getting common tangent for a and b)
                just return end_points and Gibbs free energy is minimized
                """
                print("WARNING: WHOLE REGION COMPLETELY CONCAVE, consider re-initialize parameters!!!")
                # pseudo-gradient here, it's all 0
                f_now = FixedPointOperation1D(self.G, self.params_list, self.T, x_fixed_at_endpoint_ID = -99999, is_PDOS_version_G=self.is_PDOS_version_G, style=self.style, quaderature_points = self.quaderature_points)
                new_x_star = f_now.forward_0D(torch.tensor(self.end_points).requires_grad_()) # go through the process again to get derivative
                # register hook, can do anything with the grad that passed in
                def backward_hook(grad):
                    # we use this hook to calculate dz/dtheta, where z is the equilibrium and theta is the learnable params in the model
                    if self.hook is not None:
                        self.hook.remove()
                        # torch.cuda.synchronize()   # To avoid infinite recursion
                    zeros = torch.eye(2) - torch.eye(2)
                    new_grad = zeros@grad
                    return new_grad
                # hook registration
                self.hook = new_x_star.register_hook(backward_hook)
                # all set! return 
                return new_x_star
            else:
                """ 
                when one of these things are end points, we do not have common tangent condition for g curve itself, 
                instead we just find out where is the gibbs free energy minimum place as one of the phase boundary points that is not endpoints
                G
                |
                |       
                |           
                |       x           
                |   x     x                     b 
                |  a          x              x   
                |                   c
                |__________________________________ x
                In this case, connect a and c gives the convex hull, instead of finding two points on g curve that give the same derivative value
                i.e. we need to find the common tangent of the straight line ac and the G curve at point c
                i.e. the tangent of ac is the same as that of G curve at point c
                this is now a 1D solver, we need to solve xc, i.e.
                (G(xa)-G(xc))/(xa-xc) = mu(xc)  [Note that xa is fixed value]
                writing in fixed-point solving scheme:
                xc = xa - (G(xa)-G(xc))/mu(xc)
                """
                from .solver import solver_1D
                if x[0]-self.end_points[0] < _eps*2.0 and self.end_points[1]-x[1] >= _eps*2.0:
                    f_now = FixedPointOperation1D(self.G, self.params_list, self.T, x_fixed_at_endpoint_ID = 0, is_PDOS_version_G=self.is_PDOS_version_G, scaling_alpha = self.scaling_alpha, style=self.style, quaderature_points = self.quaderature_points)
                    x_star = solver_1D(f_now, x, ID_changing_one=1) # threshold cannot be too large, because our quaderature also has error!  
                else:
                    f_now = FixedPointOperation1D(self.G, self.params_list, self.T, x_fixed_at_endpoint_ID = 1, is_PDOS_version_G=self.is_PDOS_version_G, scaling_alpha = self.scaling_alpha, style=self.style, quaderature_points = self.quaderature_points) 
                    x_star = solver_1D(f_now, x, ID_changing_one=0) # threshold cannot be too large, because our quaderature also has error!
                # (Prepare for) Backward pass    
                new_x_star = f_now(x_star.requires_grad_()) # go through the process again to get derivative
                ## checking
                if x[0]-self.end_points[0] < _eps*2.0 and self.end_points[1]-x[1] >= _eps*2.0:
                    error = torch.abs(x_star[1] - new_x_star[1])
                else:
                    error = torch.abs(x_star[0] - new_x_star[0])
                if error >= 1e-2:
                    print("WARNING: 1D solution might be wrong, before fixedpoint is ", x_star, " after is ", new_x_star)
                # register hook
                def backward_hook(grad):
                    # we use this hook to calculate dz/dtheta, where z is the equilibrium and theta is the learnable params in the model
                    if self.hook is not None:
                        self.hook.remove()
                        # torch.cuda.synchronize()   # To avoid infinite recursion
                    I_minus_J = torch.eye(2) - autograd.functional.jacobian(self.f, x_star)
                    new_grad = torch.linalg.pinv(I_minus_J)@grad
                    return new_grad
                # hook registration
                self.hook = new_x_star.register_hook(backward_hook)
                # all set! return 
                return new_x_star

            

