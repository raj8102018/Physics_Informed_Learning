import jax
import flax.nnx as nnx
import jax.numpy as jnp

class PINN(nnx.Module):

    def __init__(self, rngs):

        self.linear1 = nnx.Linear(2, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 32, rngs=rngs)
        self.linear3 = nnx.Linear(32, 32, rngs=rngs)
        self.linear4 = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x, t):

        inputs = jnp.concatenate([x,t], axis=-1)

        z = self.linear1(inputs)
        z = jax.nn.tanh(z)

        z = self.linear2(z)
        z = jax.nn.tanh(z)

        z = self.linear3(z)
        z = jax.nn.tanh(z)

        return self.linear4(z)
