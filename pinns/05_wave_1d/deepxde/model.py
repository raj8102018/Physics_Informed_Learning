import deepxde as dde
import numpy as np

geom = dde.geometry.Interval(0, 1)

timedomain = dde.geometry.TimeDomain(0, 1)

geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, u):

    u_tt = dde.grad.hessian(u, x, i=1, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)

    return u_tt - u_xx

def bc_func(x):

    return np.zeros((len(x),1))

def ic_func(x):

    return np.sin(np.pi * x[:,0:1])

bc = dde.icbc.DirichletBC(geomtime, bc_func, lambda _, on_boundary: on_boundary)

ic1 = dde.icbc.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

def initial_velocity(x, u, _):

    return dde.grad.jacobian(u, x, i=0, j=1)

ic2 = dde.icbc.OperatorBC(geomtime, initial_velocity, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [bc, ic1, ic2  ], num_domain=2000, num_boundary=100, num_initial=100)

net = dde.nn.FNN([2] + [32,32,32] + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(iterations=2000)

print("DeepXDE successfully compiled and trained the 1D Wave PDE.")