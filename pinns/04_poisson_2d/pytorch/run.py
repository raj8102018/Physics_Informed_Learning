import torch
import torch.optim as optim
import torch.nn.functional as F
import math
from model import PINN
from helper import compute_pde_residual

print("--- RUNNING TEST: PyTorch 2D Poisson Equation ---")

# 1. Initialize Model and Optimizer
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 2. Generate Collocation Points (Inside the square)
num_col = 2000
x_col = (torch.rand(num_col, 1) * 2 - 1).requires_grad_(True)
y_col = (torch.rand(num_col, 1) * 2 - 1).requires_grad_(True)

# 3. Generate Boundary Points (The perimeter of the square)
num_bc_per_edge = 100

# Top edge (y = 1)
x_top = torch.rand(num_bc_per_edge, 1) * 2 - 1
y_top = torch.ones(num_bc_per_edge, 1)

# Bottom edge (y = -1)
x_bot = torch.rand(num_bc_per_edge, 1) * 2 - 1
y_bot = -torch.ones(num_bc_per_edge, 1)

# Right edge (x = 1)
x_right = torch.ones(num_bc_per_edge, 1)
y_right = torch.rand(num_bc_per_edge, 1) * 2 - 1

# Left edge (x = -1)
x_left = -torch.ones(num_bc_per_edge, 1)
y_left = torch.rand(num_bc_per_edge, 1) * 2 - 1

# Combine all boundary points into a single batch
x_bc = torch.cat([x_top, x_bot, x_right, x_left], dim=0)
y_bc = torch.cat([y_top, y_bot, y_right, y_left], dim=0)

# Target for boundary is always 0 because our analytical solution is sin(pi*x)*sin(pi*y)
u_bc_target = torch.zeros_like(x_bc)

# 4. Training Loop
for step in range(2001):
    optimizer.zero_grad()

    # Physics Loss (Collocation)
    # Note: We pass x_col and y_col directly to our residual engine
    residual = compute_pde_residual(model, x_col, y_col)
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # Boundary Loss
    # We pass the concatenated boundary points directly to the model
    u_bc_pred = model(x_bc, y_bc)
    loss_bc = F.mse_loss(u_bc_pred, u_bc_target)

    # Total Loss
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step:4d} | Total: {loss.item():.5f} | PDE: {loss_pde.item():.5f} | BC: {loss_bc.item():.5f}")

print("SUCCESS: 2D Poisson Equation trained successfully.")