import firedrake as fd
import numpy as np
import torch
from torch import optim, nn
import copy
from typing import List
from functools import reduce


class linear_diffusion(object):
    def __init__(self, mesh: fd.mesh.MeshGeometry, phi: float = 0.2, c_t: float = 1.0, dt: float = 1.0, T: float = 10.0):
        self.mesh = mesh 
        self.phi = fd.Constant(phi)
        self.c_t = fd.Constant(c_t)
        self.dt = fd.Constant(dt)
        self.T = T

        self.theta = fd.Constant(1.0)

    def get_boundary_points(self):
        self.bc.nodes
        return self.bc.nodes.shape[0]

    def get_coordinate_functions(self ,V: fd.functionspaceimpl.WithGeometry):
        return tuple(fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))

    def PDE_definition(self,
                    p: fd.function.Function,
                    p_n: fd.function.Function,
                    q: fd.function.Function,
                    q_n: fd.function.Function,
                    V: fd.functionspaceimpl.WithGeometry):
        """
        Docstring for PDE_definition
        :param p: at p_{n+1}
        :param p_n: at p_n
        :param q: at q_{n+1}
        :param q_n: at q_n
        :param V: Description
        """
        v = fd.TestFunction(V)
        #IC
        p_theta = self.theta*p + (1 - self.theta)*p_n
        q_theta = self.theta*q + (1 - self.theta)*q_n

#        F = (((self.phi*self.c_t)/self.dt)*(p - p_n)*v)*fd.dx + fd.inner( fd.grad(p_theta), fd.grad(v)) * fd.dx - (q_theta)*fd.dx
        F = (((self.phi*self.c_t)/self.dt)*(p - p_n)*v)*fd.dx \
            + fd.inner( fd.grad(p_theta), fd.grad(v)) * fd.dx \
            - (q_theta*v)*fd.dx
        return F
    
    def IC_definition(self,V,p_0):
        self.ic = p_0
        return self.ic

    def BC_definition(self,V,g):
        bc = fd.DirichletBC(V, g, "on_boundary")
        return bc

    def solve(self,
                p: fd.function.Function,
                q_h: List[fd.function.Function],
                V: fd.functionspaceimpl.WithGeometry,
                ):
        bc = self.BC_definition(V, fd.Constant(0.0))
        p_n = self.IC_definition(V,fd.Function(V).interpolate(fd.Constant(0.0)))
        p_h = [copy.deepcopy(p_n)]

        #q_h = [q_h[0]] + q_h

        num_step = int(self.T/self.dt.values())

        for nstep in range(num_step-1):
            F = self.PDE_definition( p, p_n, q_h[nstep+1], q_h[nstep], V)
            fd.solve(F == 0, p, bcs = [bc])
            p_h.append(copy.deepcopy(p))
            p_n.interpolate(p)
        return p_h
    

class control_linear_diffusion(linear_diffusion):
    def __init__(self,model: nn.Module, num_step: int,**args):
        super().__init__(**args)
        self.model = model
        self.optimiser = optim.AdamW(self.model.parameters(), lr = 1e-3, eps=1e-8)
        self.num_step = num_step

    def control_problem(self, q_h: List[fd.function.Function], p_h_tilde: List[fd.function.Function], V):

        p = fd.Function(V)
        p_sol = self.solve(
            p = p,
            q_h= q_h,
            V = V
        )
        assert len(p_sol) == len(p_h_tilde), f"shapes p_sol {len(p_sol)} and p_h_tilde {len(p_h_tilde)} are not equal "

        cost = reduce(lambda a,b: a+b, list( (p_ - p_tilde)*fd.dx for p_,p_tilde in zip(p_sol,p_h_tilde)))
        return cost
    
    def control_f(self, p_h_tilde: List[fd.function.Function], V):
        """
        control problem example:
        ml model estimates source (f) as a mapping from coordinates to f
        """
        num_step = int(self.T/self.dt.values())
        dof_f = self.get_coordinate_functions(V)
        dof_f = tuple(fd.ml.pytorch.to_torch(dof_f_) for dof_f_ in dof_f)
        t_encoding = list(torch.tensor(self.dt.values()).unsqueeze(0)*step for step in range(num_step))

        f_p = self.model(*dof_f,t_encoding)

        fd.adjoint.continue_annotation()
        q_h = list(fd.Function(V) for _ in range(len(p_h_tilde)))

        print("p_h_tilde ",len(p_h_tilde))
        print("q_h ",len(q_h))
        print("f_p ",len(f_p))

        c = map(fd.adjoint.Control, q_h)
        Jhat = fd.adjoint.ReducedFunctional(
            self.control_problem(
                q_h = q_h,
                p_h_tilde = p_h_tilde,
                V = V
            ),
            c)
        G = fd.ml.pytorch.torch_operator(Jhat)
        fd.adjoint.stop_annotating()

        composed_function_loss = G(f_p)

        return composed_function_loss,f_p

