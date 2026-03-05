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
        
    def IC_definition(self,V,f):
        self.ic = fd.Function(V).interpolate(f)
        return self.ic
    
    def BC_definition(self,V,g):
        self.bc = fd.Function(V).interpolate(g)
        return self.bc

    def PDE_definition(self,
                       e_n: fd.function.Function,
                       e_c: fd.function.Function,
                       e_o: fd.function.Function,
                       f: fd.function.Function,
                       V: fd.functionspaceimpl.WithGeometry
                       ):
        
        v = fd.TestFunction(V)

        #M = fd.assemble(u * v * fd.dx, bcs = [bc])
        M = fd.inner(e_n,v) * fd.dx

        # We solve:   M · ez_new  =  rhs
        # where rhs  = (2M - dt² c² K) · ez  -  M · ez_old  +  dt² · source

        rhs = 2 * (fd.inner(e_c,v)) \
            - self.dt**2 * self.c**2 * e_c \
            - fd.inner(e_o,v) \
            + self.dt**2 * f * v
        return fd.assemble(M),fd.assemble(rhs)
    
    def solve_step(self,
                    e_n: fd.function.Function,
                    e_c: fd.function.Function,
                    e_o: fd.function.Function,
                    f: fd.function.Function,
                    V: fd.functionspaceimpl.WithGeometry
                   ):
        bc = self.BC_definition(V)
        M,rhs = self.PDE_definition(e_n,e_c,e_o,f,V)
        fd.solve(M == rhs, e_n, bcs=[bc])
        return e_n

    def solve(self,
                e_n: fd.function.Function,
                e_c: fd.function.Function,
                e_o: fd.function.Function,
                f: List[fd.function.Function],
                V: fd.functionspaceimpl.WithGeometry
                ):
        bc = self.BC_definition(V, fd.Constant(0.0))
        e_o = self.IC_definition(V,fd.Function(V).interpolate(fd.Constant(0.0)))
        e_c = self.IC_definition(V,fd.Function(V).interpolate(fd.Constant(0.0)))
        e_h = [copy.deepcopy(e_o), copy.deepcopy(e_c)]
        
        num_step = int(self.T/self.dt.values())

        for nstep in range(2,num_step):
            print(f"check this value should be 0 {nstep-2}")
            M,rhs = self.PDE_definition(
                e_n = e_n,
                e_c = e_c,
                e_o = e_o,
                f = f[nstep],
                V = V
                )
            fd.solve(M == rhs, e_n, bcs=[bc])
            e_h.append(copy.deepcopy(e_n))
            e_o.assing(e_c)
            e_c.assing(e_n)
        return e_h