import deepxde as dde
import numpy as np

geom = dde.geometry.Rectangle([-1, -1], [1, 1])

def pde(x, u):

    u_xx = dde.grad.hessian(u, x, i=0, j=0)

    u_yy = dde.grad.hessian(u, x, i=1, j=1)

    forcing_term = -2 * np.pi * np.pi * dde.backend.sin(np.pi * x[:, 0:1]) * dde.backend.sin(np.pi * x[:, 1:2])

    return u_xx + u_yy - forcing_term

def bc_func(x):

    return np.zeros((len(x), 1))

bc = dde.icbc.DirichletBC(geom, bc_func, lambda _, on_boundary: on_boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=2000, num_boundary=400)

net = dde.nn.FNN([2] + [32, 32, 32] + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(iterations=2000)

print("DeepXDE successfully compiled and trained the 2D poisson PDE.")