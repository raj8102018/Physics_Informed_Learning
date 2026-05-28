import jax
import jax.numpy as jnp
import optax
import math
from model import PINN
from helper import pde_residual_batch, u_t_batch

print("--- RUNNING TEST: JAX 1D Wave Equation ---")

# 1. Initialize Model
model = PINN()
key = jax.random.PRNGKey(42)
dummy_x, dummy_t = jnp.ones((1, 1)), jnp.ones((1, 1))
params = model.init(key, dummy_x, dummy_t)

# 2. Domain Sampling (x in [0,1], t in [0,1])
key, *subkeys = jax.random.split(key, 6)
num_col = 2000
x_col = jax.random.uniform(subkeys[0], (num_col, 1), minval=0.0, maxval=1.0)
t_col = jax.random.uniform(subkeys[1], (num_col, 1), minval=0.0, maxval=1.0)

# 3. Boundary Points (x=0 and x=1)
num_bc = 100
x_bc = jnp.concatenate([jnp.zeros((num_bc, 1)), jnp.ones((num_bc, 1))], axis=0)
t_bc = jax.random.uniform(subkeys[2], (num_bc * 2, 1), minval=0.0, maxval=1.0)
u_bc_target = jnp.zeros_like(x_bc)

# 4. Initial Condition Points (t=0)
num_ic = 100
x_ic = jax.random.uniform(subkeys[3], (num_ic, 1), minval=0.0, maxval=1.0)
t_ic = jnp.zeros((num_ic, 1))
u_ic_target = jnp.sin(math.pi * x_ic)

# 5. Define Loss and Optimizer
def loss_fn(p, xc, tc, xb, tb, ub_tgt, xi, ti, ui_tgt):
    # PDE Loss
    res = pde_residual_batch(p, xc.flatten(), tc.flatten())
    loss_pde = jnp.mean(res ** 2)
    
    # Boundary Loss (Position)
    u_b_pred = jax.vmap(lambda x, t: PINN().apply(p, jnp.array([[x]]), jnp.array([[t]])).squeeze())(xb.flatten(), tb.flatten())
    loss_bc = jnp.mean((u_b_pred - ub_tgt.flatten()) ** 2)
    
    # Initial Condition Loss 1 (Position)
    u_i_pred = jax.vmap(lambda x, t: PINN().apply(p, jnp.array([[x]]), jnp.array([[t]])).squeeze())(xi.flatten(), ti.flatten())
    loss_ic1 = jnp.mean((u_i_pred - ui_tgt.flatten()) ** 2)

    # Initial Condition Loss 2 (Velocity) 
    # JAX makes this beautiful: just evaluate the batched velocity function!
    v_i_pred = u_t_batch(p, xi.flatten(), ti.flatten())
    loss_ic2 = jnp.mean(v_i_pred ** 2) # Target is 0
    
    return loss_pde + loss_bc + loss_ic1 + loss_ic2

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step_fn(p, state, xc, tc, xb, tb, ub_tgt, xi, ti, ui_tgt):
    loss, grads = jax.value_and_grad(loss_fn)(p, xc, tc, xb, tb, ub_tgt, xi, ti, ui_tgt)
    updates, state = optimizer.update(grads, state)
    p = optax.apply_updates(p, updates)
    return p, state, loss

# 6. Training Loop
for step in range(2001):
    params, opt_state, loss = step_fn(params, opt_state, x_col, t_col, x_bc, t_bc, u_bc_target, x_ic, t_ic, u_ic_target)
    if step % 200 == 0:
        print(f"Step {step:4d} | Total Loss: {loss:.5f}")

print("SUCCESS: JAX 1D Wave Equation trained successfully.")