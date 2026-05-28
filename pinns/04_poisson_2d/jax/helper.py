import jax
import flax.linen as nn
import jax.numpy as jnp
from model import PINN

def u_fn(params, x, y):

    x = jnp.asarray(x).reshape(())
    y = jnp.asarray(y).reshape(())

    x_array = jnp.array([[x]])
    y_array = jnp.array([[y]])

    u_array = PINN().apply(params, x_array, y_array)

    return u_array.squeeze()

u_x_fn = jax.grad(u_fn, argnums=1)
u_y_fn = jax.grad(u_fn, argnums=2)

u_xx_fn = jax.grad(u_x_fn, argnums=1)
u_yy_fn = jax.grad(u_y_fn, argnums=2)

def pde_residual_single(params, x, y):

    u_xx = u_xx_fn(params, x, y)

    u_yy = u_yy_fn(params, x, y)

    forcing_term = -2 * jnp.pi * jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    return u_xx + u_yy - forcing_term

pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0))