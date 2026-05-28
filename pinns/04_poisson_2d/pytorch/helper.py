import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pde_residual(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    x.requires_grad=True
    y.requires_grad=True
    
    u = model(x,y)
    
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
   
    forcing_term = -2 * torch.pi * torch.pi * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    
    residual = u_xx + u_yy - forcing_term

    return residual

def pinn_loss(model: nn.Module, x_col: torch.Tensor, y_col: torch.Tensor, x_bc: torch.Tensor, y_bc: torch.Tensor, u_bc_target: torch.Tensor) -> torch.Tensor:
    
    residual = compute_pde_residual(model, x_col, y_col)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))
    
    u_bc_pred = model(x_bc, y_bc)
    loss_bc = F.mse_loss(u_bc_pred, u_bc_target)
    
    return loss_pde + loss_bc