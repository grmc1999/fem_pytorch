import firedrake as fd
import numpy as np
import torch
from torch import optim


class linear_diffusion(object):
    def __init__(self,mesh):
        self.mesh = mesh

    def get_boundary_points(self):
        self.bc.nodes
        return self.bc.nodes.shape[0]

    def get_coordinate_functions(self,V):
        return tuple(fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))

    def PDE_definition(self,u,f,V):
        v = fd.TestFunction(V)
        F = (fd.inner(fd.grad(u),fd.grad(v)) + fd.inner(u,v) - fd.inner(f,v)) * fd.dx
        return F
    
    def BC_definition(self,V,g):
        bc = fd.DirichletBC(V, g, "on_boundary")
        return bc

    def solve(self,u,f,V):
        F = self.PDE_definition(u,f,V)
        bc = self.BC_definition(V, fd.Constant(1.0))
        fd.solve(F == 0, u, bcs = [bc])
        return u
    

class control_linear_diffusion(linear_diffusion):
    def __init__(self,model,**args)
        super().__init__(**args)
        self.model = model
        self.optimiser = optim.AdamW(self.model.parameters(), lr = 1e-3, eps=1e-8)

    def control_problem(self,u,f,V):

        u = fd.Function(V)
        u_sol = self.solve(u,f,V)

        X = fd.SpatialCoordinate(self.mesh)
        u_j = fd.Function(V)
        u_j.interpolate(0.5*fd.exp((-1*(X[0]-0.25)**2)/0.01) + 0.5*fd.exp((-1*(X[0]-0.75)**2)/0.01))
        return fd.assemble(((u_j-u_sol)**2)*fd.dx)
    
    def control_f(self, u, V,K):
    """
    control problem example:
    ml model estimates source (f) as a mapping from coordinates to f
    """
    dof_f = tuple(fd.ml.pytorch.to_torch(fd.Function(V).interpolate(dof)) for dof in fd.SpatialCoordinate(self.mesh))
    f_p = self.model(*dof_f)
    fd.adjoint.continue_annotation()
    f = fd.Function(V)
    c = fd.adjoint.Control(f)
    Jhat = fd.adjoint.ReducedFunctional(self.control_problem(f, V, K),c)
    G = fd.ml.pytorch.torch_operator(Jhat)
    fd.adjoint.stop_annotating()
    composed_function_loss = G(f_p)
    return composed_function_loss,f_p

