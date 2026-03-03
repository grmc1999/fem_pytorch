from parabolic.diffusion import linear_diffusion,control_linear_diffusion
from models.models import model_diffusion
import torch
from torch import nn
import firedrake as fd
import firedrake as fd
from firedrake import adjoint
from firedrake.ml import pytorch
from firedrake.adjoint import continue_annotation

mesh = fd.UnitSquareMesh(50,50)
num_step=20
V = fd.FunctionSpace(mesh, "CG", 1)


X = fd.SpatialCoordinate(V)
q_h = list(fd.Function(V).interpolate(
    fd.exp(-1*((X[0]-0.5)**2 + (X[1]-0.5)**2 )/0.001)
) for _ in range(num_step+2))
LD = linear_diffusion(mesh,dt = 0.01, T = 0.5)

q_h = list(fd.Function(V).interpolate(
    fd.exp(-1*((X[0]-0.5)**2 + (X[1]-0.5)**2 )/0.001)
) for _ in range(int(10/0.01)+2))

CLD = control_linear_diffusion(
    model = model_diffusion(V.dim()).double(),
    num_step = 20,
    mesh = mesh,
    dt = 0.1,
    T = 0.5
    )


train_iterations = 20
for epoch in range(train_iterations):
    composed_function_loss,p_h,q_h_ = CLD.control_f(p_h_tilde = q_h,V = V)
