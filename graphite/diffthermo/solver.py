import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt


global  _eps
_eps = 1e-7


def newton_raphson(func, x0, threshold=1e-6, end_points = [0,1], is_clamp = True):
    """
    x0: initial guess, with shape torch.Size([2])
    """
    error = 9999999.9
    x_now = x0.clone()
    # define g function for Newton-Raphson
    def g(x):
        return func(x) - x
    # iteration
    n_iter = -1
    while error > threshold and n_iter < 100:
        x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
        f_now = g(x_now)
        J = autograd.functional.jacobian(g, x_now)
        f_now = torch.reshape(f_now, (2,1)) 
        x_new = x_now - torch.reshape(torch.linalg.pinv(J)@f_now, (2,)) 
        # detach for memory saving
        x_new = x_new.clone().detach() # detach for memory saving
        # clamp, actually it MUST be clamped
        if is_clamp:
            x_new[0] = torch.max(torch.tensor([end_points[0]+_eps, x_new[0]]))
            x_new[0] = torch.min(torch.tensor([end_points[1]-_eps, x_new[0]]))
            x_new[1] = torch.max(torch.tensor([end_points[0]+_eps, x_new[1]]))
            x_new[1] = torch.min(torch.tensor([end_points[1]-_eps, x_new[1]])) # +- 1e-6 is for the sake of torch.log. We don't want log function to blow up at x=0!
        x_now = x_now.clone().detach() # detach for memory saving
        # calculate error
        if torch.abs(x_new[0]-x_now[0]) < torch.abs(x_new[1]-x_now[1]):
            error = torch.abs(x_new[1]-x_now[1])
        else:
            error = torch.abs(x_new[0]-x_now[0])
        # step forward
        x_now = x_new.clone()
        n_iter = n_iter + 1
    if n_iter >= 99:
        print("Warning: Max iteration in Newton-Raphson solver reached.")
    return x_now



def solver_1D(func, x0, ID_changing_one=0, threshold=1e-4, end_points=[0.0,1.0]):
    """
    For Solving the "common tangent" when one of the points are endpoint.
    In this case, we have 
    G
    |
    |       
    |           
    |       x           
    |   x     x                     b 
    |  a          x              x   
    |                   c
    |__________________________________ x
    In this case, connect a and c gives the convex hull, 
    i.e. we need to find the common tangent of the straight line ac and the G curve at point c
    i.e. the tangent of ac is the same as that of G curve at point c
    this is now a 1D solver, we need to solve xc, i.e.
    (G(xa)-G(xc))/(xa-xc) = mu(xc)  [Note that xa is fixed value]
    writing in fixed-point solving scheme:
    xc = xa - (G(xa)-G(xc))/mu(xc) 
    
    func: 
    x0: initial guess, with shape torch.Size([1])
    ID: which one is the changing one 
    """
    x_now = x0[ID_changing_one]*1.0
    x_fixed = x0[1-ID_changing_one]*1.0
    x_now = x_now.reshape(1)
    x_fixed = x_fixed.reshape(1)
    # define g function for Newton-Raphson
    def g(x):     
        func.x_fixed_point_value = x0[1-ID_changing_one]
        x_ = func.forward_for_solver(x)    
        return  x_ - x
    # iteration    
    n_iter_max = 1000
    is_reached_true_solution = False
    n_time_of_reaching_solution = 0
    while is_reached_true_solution == False: # sometimes it returns 
        # reset every intermediate variables
        n_iter = -1
        error = 9999999.9
        while error > threshold and n_iter < n_iter_max:
            x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
            f_now = g(x_now)
            J = autograd.functional.jacobian(g, x_now)
            f_now = torch.reshape(f_now, (1,1)) 
            x_new = x_now - torch.reshape(torch.linalg.pinv(J)@f_now, (1,)) 
            # clamp, if the changing point goes outside the box
            while x_new>end_points[1] or x_new<end_points[0] or torch.isnan(x_new) or torch.isinf(x_new):
                import random
                randseed = random.uniform(-1,1)
                ## random perturbation around init point
                x_new = x0[ID_changing_one] + randseed*0.05 # real solution won't be far away from the init point
                x_new = x_new.reshape(1)
                x_new = x_new.requires_grad_()
                ## random perturbation within a half space
                # if x0[ID_changing_one]>=(end_points[0]+end_points[1])/2: 
                #     x_new = torch.tensor([end_points[1] - randseed*(end_points[0]+end_points[1])/2])
                #     x_new = x_new.requires_grad_()
                # else:
                #     x_new = torch.tensor([end_points[0] + randseed*(end_points[0]+end_points[1])/2])
                #     x_new = x_new.requires_grad_()
            # calculate error
            error = torch.abs(x_new-x_now)
            x_now = x_now.clone().detach() # detach for memory saving
            # step forward
            x_now = x_new.clone()
            n_iter = n_iter + 1
        # we have reached a solution
        n_time_of_reaching_solution = n_time_of_reaching_solution + 1
        # examine whether the solved x_now is close enough to x0
        ## don't check at all
        is_reached_true_solution = True
        # # check
        # if torch.abs(x_now-x0[ID_changing_one]) <= 0.05:
        #     is_reached_true_solution = True
        #     # print("Reached final solution. Now solution is %.4f, init sol is %.4f" %(x_now,x0[ID_changing_one]))
        # else:
        #     is_reached_true_solution = False
        #     print("%d times reach solution. Re-calculate 1D solution, now solution is %.4f, but init sol is %.4f" %(n_time_of_reaching_solution, x_now,x0[ID_changing_one]))
        #     # reset every intermediate variables
        #     n_iter = -1
        #     error = 9999999.9
        #     import random
        #     randseed = random.uniform(-1,1)
        #     x_now = x0[ID_changing_one] + randseed*0.05
        #     x_now = x_now.reshape(1)
        #     x_now = x_now.requires_grad_()
        #     print("Re-initialized as %.4f" %(x_now))
    # we find true solution now
    if ID_changing_one == 1:
        return torch.cat((x_fixed, x_now))
    else:
        return torch.cat((x_now, x_fixed))


class FixedPointOperation(nn.Module):
    def __init__(self, G, params_list, T = 300, is_PDOS_version_G = False, end_points = [0,1], scaling_alpha = 1e-5, style = "Legendre", quaderature_points = 20):
        """
        The fixed point operation used in the backward pass of common tangent approach. 
        Write the forward(self, x) function in such weird way so that it is differentiable
        G is the Gibbs free energy function 
        params_list: depend on whether is_PDOS_version_G, its content varies
        """
        super(FixedPointOperation, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = torch.tensor([T])
        self.is_PDOS_version_G = is_PDOS_version_G
        assert end_points[1] > end_points[0]
        self.end_points = end_points
        self.scaling_alpha = scaling_alpha
        self.style = style
        self.quaderature_points = quaderature_points
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        x_alpha = x[0]
        x_beta = x[1]
        if self.is_PDOS_version_G == False:
            g_right = self.G(x_beta, self.params_list, self.T) 
            g_left = self.G(x_alpha, self.params_list, self.T)
        else:
            g_right = self.G(x_beta, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
            g_left = self.G(x_alpha, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
        mu_right = autograd.grad(outputs=g_right, inputs=x_beta, create_graph=True)[0]
        mu_left = autograd.grad(outputs=g_left, inputs=x_alpha, create_graph=True)[0]
        """ 
        Because  (g_right - g_left)/(x_beta-x_alpha) = mu_right = mu_left,
        we have 
        (g_right - g_left)/(x_beta-x_alpha) - mu_left = 0  ==>  x_alpha_new = x_alpha + (g_right - g_left)/(x_beta-x_alpha) - mu_left  (at fixedpoint, x_alpha_new = x_alpha)
        (g_right - g_left)/(x_beta-x_alpha) - mu_right = 0  ==>  x_beta_new = x_beta + (g_right - g_left)/(x_beta-x_alpha) - mu_right  (at fixedpoint, x_beta_new = x_beta)
        """
        x_alpha_new = x_alpha + self.scaling_alpha*((g_right - g_left)/(x_beta - x_alpha + _eps) - mu_left)
        x_beta_new = x_beta + self.scaling_alpha*((g_right - g_left)/(x_beta-x_alpha + _eps) - mu_right)
        ## old implementation
        # """ 
        # Because  (g_right - g_left)/(x_beta-x_alpha) = mu_right = mu_left,
        # we have 
        # (g_right - g_left)/(x_beta-x_alpha) = mu_right ==> x_alpha = x_beta  - (g_right - g_left)/mu_right
        # (g_right - g_left)/(x_beta-x_alpha) = mu_left  ==> x_beta  = x_alpha + (g_right - g_left)/mu_left   
        # """
        # x_alpha_new = x_beta - (g_right - g_left)/(mu_left + _eps)
        # x_beta_new = x_alpha + (g_right - g_left)/(mu_right + _eps)
        # if self.is_clamp: ## dont need to clamp here! clamped in the solver
        #     x_alpha_new = torch.clamp(x_alpha_new , min=self.end_points[0]+_eps, max=self.end_points[1]-_eps) # clamp
        #     x_beta_new = torch.clamp(x_beta_new , min=self.end_points[0]+_eps, max=self.end_points[1]-_eps) # clamp
        x_alpha_new = x_alpha_new.reshape(1)
        x_beta_new = x_beta_new.reshape(1)
        return torch.cat((x_alpha_new, x_beta_new))


# # In case that the above implementation doesn't work
# class FixedPointOperationForwardPass(nn.Module):
#     def __init__(self, G, params_list, T = 300, is_PDOS_version_G = False, end_points = [0,1], is_clamp = True, style = "Legendre", quaderature_points = 20):
#         """
#         The fixed point operation used in the forward pass of common tangent approach
#         Here we don't use the above implementation (instead we use Pinwen's implementation in Jax-TherMo) to guarantee that the solution converges to the correct places in forward pass
#         G is the Gibbs free energy function 
#         params_list: depend on whether is_PDOS_version_G, its content varies
#         """
#         super(FixedPointOperationForwardPass, self).__init__()
#         self.G = G
#         self.params_list = params_list
#         self.T = torch.tensor([T])
#         self.is_PDOS_version_G = is_PDOS_version_G
#         assert end_points[1] > end_points[0]
#         self.end_points = end_points
#         self.is_clamp = is_clamp
#         self.style = style
#         self.quaderature_points = quaderature_points
#     def forward(self, x):
#         """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
#         # x_alpha_0 = (x[0]).reshape(1)
#         # x_beta_0 = (x[1]).reshape(1)
#         x_alpha_now = x[0]
#         x_beta_now = x[1]
#         x_alpha_now = x_alpha_now.reshape(1)
#         x_beta_now = x_beta_now.reshape(1)
#         if self.is_PDOS_version_G == False:
#             g_left = self.G(x_alpha_now, self.params_list, self.T)
#             g_right = self.G(x_beta_now, self.params_list, self.T)
#         else:
#             g_right = self.G(x_beta, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
#             g_left = self.G(x_alpha, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
#         common_tangent = (g_left - g_right)/(x_alpha_now - x_beta_now)
#         dcommon_tangent = 9999999.9
#         n_iter_ct = 0
#         while dcommon_tangent>1e-4 and n_iter_ct < 300:  
#             """
#             eq1 & eq2: we want dG/dx evaluated at x1 (and x2) to be the same as common_tangent, i.e. mu(x=x1 or x2) = common_tangent
#             then applying Newton-Rapson iteration to solve f(x) = mu(x) - common_tangent, where mu(x) = dG/dx
#             Newton-Rapson iteration: x1 = x0 - f(x0)/f'(x0)
#             """  
#             if self.is_PDOS_version_G == False:
#                 def eq(x):
#                     y = self.G(x, self.params_list, self.T) - common_tangent*x
#                     return y
#             else:
#                 def eq(x):
#                     y = self.G(x, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) - common_tangent*x
#                     return y
#             # update x_alpha
#             dx = torch.tensor(999999.0)
#             n_iter_dxalpha = 0.0
#             while torch.abs(dx) > 1e-6 and n_iter_dxalpha < 300:
#                 x_alpha_now = x_alpha_now.requires_grad_()
#                 value_now = eq(x_alpha_now)
#                 f_now = autograd.grad(value_now, x_alpha_now, create_graph=True)[0]
#                 f_prime_now = autograd.grad(f_now, x_alpha_now, create_graph=True)[0]
#                 dx = -f_now/(f_prime_now)    
#                 x_alpha_now = x_alpha_now + dx
#                 x_alpha_now = x_alpha_now.clone().detach()
#                 # clamp
#                 if self.is_clamp:
#                     x_alpha_now = torch.max(torch.tensor([self.end_points[0]+_eps, x_alpha_now]))
#                     x_alpha_now = torch.min(torch.tensor([self.end_points[1]-_eps, x_alpha_now])) 
#                 x_alpha_now = x_alpha_now.reshape(1)
#                 n_iter_dxalpha = n_iter_dxalpha + 1
#             # update x_beta
#             dx = torch.tensor(999999.0)
#             n_iter_dxbeta = 0.0
#             while torch.abs(dx) > 1e-6 and n_iter_dxbeta < 300:
#                 x_beta_now = x_beta_now.requires_grad_()
#                 value_now = eq(x_beta_now)
#                 f_now = autograd.grad(value_now, x_beta_now, create_graph=True)[0]
#                 f_prime_now = autograd.grad(f_now, x_beta_now, create_graph=True)[0]
#                 dx = -f_now/(f_prime_now)
#                 x_beta_now = x_beta_now + dx
#                 x_beta_now = x_beta_now.clone().detach()
#                 # clamp
#                 if self.is_clamp:
#                     x_beta_now = torch.max(torch.tensor([self.end_points[0]+_eps, x_beta_now]))
#                     x_beta_now = torch.min(torch.tensor([self.end_points[1]-_eps, x_beta_now])) 
#                 x_beta_now = x_beta_now.reshape(1)
#                 n_iter_dxbeta = n_iter_dxbeta + 1
#             # after getting new x1 and x2, calculates the new common tangent, the same process goes on until the solution is self-consistent
#             if self.is_PDOS_version_G == False:
#                 common_tangent_new = (self.G(x_alpha_now, self.params_list, self.T) - self.G(x_beta_now, self.params_list, self.T))/(x_alpha_now - x_beta_now)
#             else:
#                 common_tangent_new = (self.G(x_alpha_now, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) - self.G(x_beta_now, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points))/(x_alpha_now - x_beta_now)
#             dcommon_tangent = torch.abs(common_tangent_new-common_tangent)
#             common_tangent = common_tangent_new.clone().detach()
#             n_iter_ct = n_iter_ct + 1
#         return torch.cat((x_alpha_now, x_beta_now))


class FixedPointOperation1D(nn.Module):
    def __init__(self, G, params_list, T = 300, x_fixed_at_endpoint_ID = 0, is_PDOS_version_G = True, scaling_alpha = 1e-5, style = "Legendre", quaderature_points = 20):
        """
        The fixed point operation used in the backward pass of common tangent approach, but only one point is changing (one of them is endpoint and thus fixed)
        Write the forward(self, x) function in such weird way so that it is differentiable
        G is the Gibbs free energy function 
        params_list: depend on whether is_PDOS_version_G, its content varies
        x_fixed_at_endpoint_ID: the index of x that is fixed
        """
        super(FixedPointOperation1D, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = T
        self.is_PDOS_version_G = is_PDOS_version_G
        self.style = style
        self.quaderature_points = quaderature_points
        self.x_fixed_at_endpoint_ID = x_fixed_at_endpoint_ID
        self.scaling_alpha = scaling_alpha
    def forward(self, x):
        """x[self.x_fixed_at_endpoint_ID] is the fixed endpoint, the other one is the one can change"""
        x_fixed_at_end_point = x[self.x_fixed_at_endpoint_ID]
        x_can_move = x[1-self.x_fixed_at_endpoint_ID]
        if self.is_PDOS_version_G == False:
            raise NotImplementedError
        else:
            g_fixed_at_end_point = self.G(x_fixed_at_end_point, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
            g_c =                  self.G(x_can_move,           self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
        mu_c = autograd.grad(outputs=g_c, inputs=x_can_move, create_graph=True)[0]
        # """ we have xc = x_fixed_at_end_point - (G(x_fixed_at_end_point)-G(xc))/mu(xc)  """
        # x_moved = x_fixed_at_end_point - (g_fixed_at_end_point-g_c)/mu_c
        """ 
        or we just have xc = xc + 1e-5 * ( (G(x_fixed_at_end_point)-G(xc))/(x_fixed_at_end_point-x_c) - mu(x_c)) 
        1e-5 is for numeric stability TEST
        """
        x_moved = x_can_move +  self.scaling_alpha * ((g_fixed_at_end_point-g_c)/(x_fixed_at_end_point-x_can_move) - mu_c )
        x_fixed = x_fixed_at_end_point * 1.0
        x_moved = x_moved.reshape(1)
        x_fixed = x_fixed.reshape(1)
        if self.x_fixed_at_endpoint_ID == 0:
            return torch.cat((x_fixed, x_moved))
        else:
            return torch.cat((x_moved, x_fixed))
    def forward_for_solver(self, x_moving_point):
        # first sef self.x_fixed_point_value outside, before using this class!
        if self.is_PDOS_version_G == False:
            raise NotImplementedError
        else:
            g_fixed_at_end_point = self.G(self.x_fixed_point_value, self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
            g_c =                  self.G(x_moving_point,           self.T, self.params_list, style=self.style, quaderature_points = self.quaderature_points) 
        mu_c = autograd.grad(outputs=g_c, inputs=x_moving_point, create_graph=True)[0]
        # """ we have xc = x_fixed_at_end_point - (G(x_fixed_at_end_point)-G(xc))/mu(xc)  """
        # x_moved = self.x_fixed_point_value - (g_fixed_at_end_point-g_c)/mu_c
        """ 
        or we just have xc = xc + 1e-5 * ( (G(x_fixed_at_end_point)-G(xc))/(x_fixed_at_end_point-x_c) - mu(x_c)) 
        1e-5 is for numeric stability TEST
        """
        x_moved = x_moving_point +  self.scaling_alpha * ((g_fixed_at_end_point-g_c)/(self.x_fixed_point_value-x_moving_point) - mu_c )
        return x_moved
    def forward_0D(self, x):
        """ reserved for the all concave case"""
        return x*1.0
    
    