import firedrake as fd
import numpy as np
import torch
from torch import optim, nn
import copy
from typing import List
from functools import reduce


class linear_wave(object):
    def __init__(self, mesh: fd.mesh.MeshGeometry, c: float = 1.0, dt: float = 1.0, T: float = 10.0):
        self.mesh = mesh

        self.T = T

        self.dt = fd.Constant(dt)
        self.c = fd.Constant(c)
        self.beta = 0.25

    def get_coordinate_functions(self ,V: fd.functionspaceimpl.WithGeometry):
        return tuple(fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))
        
    def IC_definition(self):
        return self.ic
    
    def BC_definition(self):
        return self.bc

    def PDE_definition(self,
                       u: fd.function.Function,
                       V: fd.functionspaceimpl.WithGeometry
                       ):
        
        v = fd.TestFunction(V)

        M = fd.assemble(u * v * fd.dx)
        K = fd.assemble(self.c**2 * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx)

        # Combined matrix for Newmark: A = M + β Δt² K
        A = M + (self.beta * self.dt.values()[0]**2)