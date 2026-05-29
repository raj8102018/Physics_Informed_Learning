import jax
import flax.nnx as nnx
import jax.numpy as jnp
from model import PINN

def u_fn(model, z, t):

    z = jnp.asarray(z).reshape(())
    t = jnp.asarray(t).reshape(())

    z_array = jnp.array([[z]])
    t_array = jnp.array([[t]])

    u_array = model(z_array, t_array)

    return u_array.squeeze()

u_t_fn = jax.grad(u_fn, argnums=2)
u_z_fn = jax.grad(u_fn, argnums=1)

u_zz_fn = jax.grad(u_z_fn, argnums=1)

def pde_residual_single(model, z, t):

    u_t = u_t_fn(model, z, t)
    u_zz = u_zz_fn(model, z, t)

    return u_t - u_zz

pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None,0,0))