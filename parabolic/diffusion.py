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
    
    def solve_step(self,
                p: fd.function.Function,
                p_n: fd.function.Function,
                q: fd.function.Function,
                q_n: fd.function.Function,
                V: fd.functionspaceimpl.WithGeometry,
                ):
        
        bc = self.BC_definition(V, fd.Constant(0.0))
        F = self.PDE_definition( p, p_n, q, q_n, V)
        fd.solve(F == 0, p, bcs = [bc])
        return p
    

class control_linear_diffusion(linear_diffusion):
    def __init__(self,model: nn.Module, **args):
        super().__init__(**args)
        self.model = model
        self.optimiser = optim.AdamW(self.model.parameters(), lr = 1e-3, eps=1e-8)

    def control_problem(self,
                        p_n: fd.function.Function,
                        q: fd.function.Function,
                        q_n: fd.function.Function,
                        p_tilde: fd.function.Function,
                        V
                        ):

        p = fd.Function(V)
        p_sol = self.solve_step(
            p = p,
            p_n = p_n,
            q = q,
            q_n = q_n,
            V = V,
        )
        #assert len(p_sol) == len(p_h_tilde), f"shapes p_sol {len(p_sol)} and p_h_tilde {len(p_h_tilde)} are not equal "

        cost = ((p_sol - p_tilde)**2)*fd.dx
        #e(lambda a,b: a+b, list( ((p_ - p_tilde)**2)*fd.dx for p_,p_tilde in zip(p_sol,p_h_tilde)))
        return fd.assemble(cost), p_sol

    def control_f(self, p_h_tilde: List[fd.function.Function], V):
        """
        control problem example:
        ml model estimates source (f) as a mapping from coordinates to f
        """
        num_step = int(self.T/self.dt.values())
        dof_f = self.get_coordinate_functions(V)
        dof_f = tuple(fd.ml.pytorch.to_torch(dof_f_) for dof_f_ in dof_f)

        bc = self.BC_definition(V, fd.Constant(0.0))
        p_n = self.IC_definition(V,fd.Function(V).interpolate(fd.Constant(0.0)))
        p_h = [copy.deepcopy(p_n)]

        # Compile all step graph
        fd.adjoint.continue_annotation()

        G_h = []
        q_h_ = []

        composed_function_loss = 0

        for step in range(num_step-1):
           #q = q_h[step+1]
           q = fd.Function(V)
           q.dat.data[:] = self.model(*dof_f,torch.tensor(self.dt.values()).unsqueeze(0)*step).detach().numpy()
           #q_n = q_h[step]
           q_n = fd.Function(V)
           q_n.dat.data[:] = self.model(*dof_f,torch.tensor(self.dt.values()).unsqueeze(0)*(step+1)).detach().numpy()

           c = fd.adjoint.Control(q)

           cost, p_sol = self.control_problem(
                    p_n = p_h[step],
                    q = q,
                    q_n = q_n,
                    p_tilde = p_h_tilde[step+1],
                    V = V
                    )

           Jhat = fd.adjoint.ReducedFunctional(cost,c)

           p_h.append(copy.deepcopy(p_sol))

           G = fd.ml.pytorch.torch_operator(Jhat)
           G_h.append(G)
           t_encoding = torch.tensor(self.dt.values()).unsqueeze(0)*(step + 1)
           f_p = self.model(*dof_f,t_encoding)


           composed_function_loss = composed_function_loss + G_h[step](f_p)
           q_h_sol = fd.Function(V)
           q_h_sol.dat.data[:] = f_p.detach().numpy()
           q_h_.append(q_h_sol.copy(deepcopy=True))

        fd.adjoint.stop_annotating()


        return composed_function_loss,p_h,q_h_

