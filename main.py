from parabolic.diffusion import linear_diffusion,control_linear_diffusion
from models.models import model_diffusion
import torch
from torch import nn
import firedrake as fd

mesh = fd.UnitSquareMesh(50,50)
num_step=20
V = fd.FunctionSpace(mesh, "CG", 1)

CLD = control_linear_diffusion(
    model = model_diffusion(V.dim()).double(),
    num_step = 20,
    mesh = mesh,
    dt = 0.1,
    T = 0.5
    )