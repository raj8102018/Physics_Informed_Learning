import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pde_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    
    x.requires_grad=True
    t.requires_grad=True
    
    u, alpha = model(x,t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - (alpha * u_xx)

    return residual

def pinn_loss(model: nn.Module, x_col: torch.Tensor, t_col: torch.Tensor, x_data: torch.Tensor, t_data: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:


    u_pred, alpha = model(x_data, t_data)

    mse_data = F.mse_loss(u_pred, u_true)

    residual = compute_pde_residual(model, x_col, t_col)

    mse_pde = F.mse_loss(residual, torch.zeros_like(residual))


    return mse_data + mse_pde
