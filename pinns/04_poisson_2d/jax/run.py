import jax
import jax.numpy as jnp
import optax
from model import PINN
from helper import pde_residual_batch

print("--- RUNNING TEST: JAX 2D Poisson Equation ---")

# 1. Initialize Model
model = PINN()
key = jax.random.PRNGKey(0)
dummy_x, dummy_y = jnp.ones((1, 1)), jnp.ones((1, 1))
params = model.init(key, dummy_x, dummy_y)

# 2. Generate Collocation Points (Inside the square)
key, subkey1, subkey2 = jax.random.split(key, 3)
num_col = 2000
x_col = jax.random.uniform(subkey1, (num_col, 1), minval=-1.0, maxval=1.0)
y_col = jax.random.uniform(subkey2, (num_col, 1), minval=-1.0, maxval=1.0)

# 3. Generate Boundary Points (The perimeter)
num_bc = 100
key, *b_keys = jax.random.split(key, 5)

x_top = jax.random.uniform(b_keys[0], (num_bc, 1), minval=-1.0, maxval=1.0)
y_top = jnp.ones((num_bc, 1))

x_bot = jax.random.uniform(b_keys[1], (num_bc, 1), minval=-1.0, maxval=1.0)
y_bot = -jnp.ones((num_bc, 1))

x_right = jnp.ones((num_bc, 1))
y_right = jax.random.uniform(b_keys[2], (num_bc, 1), minval=-1.0, maxval=1.0)

x_left = -jnp.ones((num_bc, 1))
y_left = jax.random.uniform(b_keys[3], (num_bc, 1), minval=-1.0, maxval=1.0)

x_bc = jnp.concatenate([x_top, x_bot, x_right, x_left], axis=0)
y_bc = jnp.concatenate([y_top, y_bot, y_right, y_left], axis=0)
u_bc_target = jnp.zeros_like(x_bc)

# 4. Define Loss and Optimizer
def loss_fn(p, xc, yc, xb, yb, ub_target):
    # Physics Loss
    res = pde_residual_batch(p, xc.flatten(), yc.flatten())
    loss_pde = jnp.mean(res ** 2)
    
    # Boundary Loss
    u_b_pred = jax.vmap(lambda x, y: PINN().apply(p, jnp.array([[x]]), jnp.array([[y]])).squeeze())(xb.flatten(), yb.flatten())
    loss_bc = jnp.mean((u_b_pred - ub_target.flatten()) ** 2)
    
    return loss_pde + loss_bc

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

@jax.jit
def step_fn(p, state, xc, yc, xb, yb, ub_target):
    loss, grads = jax.value_and_grad(loss_fn)(p, xc, yc, xb, yb, ub_target)
    updates, state = optimizer.update(grads, state)
    p = optax.apply_updates(p, updates)
    return p, state, loss

# 5. Training Loop
for step in range(2001):
    params, opt_state, loss = step_fn(params, opt_state, x_col, y_col, x_bc, y_bc, u_bc_target)
    if step % 200 == 0:
        print(f"Step {step:4d} | Total Loss: {loss:.5f}")

print("SUCCESS: JAX 2D Poisson Equation trained successfully.")