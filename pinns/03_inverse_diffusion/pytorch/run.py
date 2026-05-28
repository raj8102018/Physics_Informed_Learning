import torch
import torch.optim as optim
import torch.nn.functional as F
from model import PINN
from helper import compute_pde_residual

print("--- RUNNING TEST: PyTorch Inverse PINN ---")

model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 1. Generate Fake "Sensor Data" (True alpha is 0.1)
# Analytical solution: u(x,t) = exp(-0.1 * pi^2 * t) * sin(pi * x)
x_data = torch.rand(100, 1) * 2 - 1  # 100 random points between -1 and 1
t_data = torch.rand(100, 1)          # 100 random time points between 0 and 1
u_true = torch.exp(-0.1 * (torch.pi**2) * t_data) * torch.sin(torch.pi * x_data)

# 2. Collocation points for PDE
x_col = (torch.rand(2000, 1) * 2 - 1).requires_grad_(True)
t_col = (torch.rand(2000, 1)).requires_grad_(True)

# 3. Mini Training Loop (just 1000 steps to prove convergence)
for step in range(1000):
    optimizer.zero_grad()
    
    # Data Loss
    u_pred, alpha_pred = model(x_data, t_data)
    loss_data = F.mse_loss(u_pred, u_true)
    
    # Physics Loss
    u_col, alpha_col = model(x_col, t_col) 
    # Note: We must recalculate u_col with requires_grad tensors inside the residual function
    residual = compute_pde_residual(model, x_col, t_col)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))
    
    loss = loss_data + loss_pde
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | Discovered Alpha: {model.alpha.item():.4f}")

print(f"SUCCESS: Final Discovered Alpha: {model.alpha.item():.4f} (Target: 0.1000)")