import jax
import flax.linen as nn
import jax.numpy as jnp
from model import PINN

def u_fn(params, x, t):

    x = jnp.asarray(x).reshape(())
    t = jnp.asarray(t).reshape(())

    x_array = jnp.array([[x]])
    t_array = jnp.array([[t]])

    u_array  = PINN().apply(params, x_array, t_array)

    return u_array.squeeze()

u_t_fn = jax.grad(u_fn, argnums = 2)
u_x_fn = jax.grad(u_fn, argnums = 1)

u_xx_fn = jax.grad(u_x_fn, argnums = 1)
u_tt_fn = jax.grad(u_t_fn, argnums = 2)

def pde_residual_single(params, x, t):

    u_xx = u_xx_fn(params, x, t)
    u_tt = u_tt_fn(params, x, t)

    return u_tt - u_xx

pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0))

u_t_batch = jax.vmap(u_t_fn, in_axes=(None, 0, 0))