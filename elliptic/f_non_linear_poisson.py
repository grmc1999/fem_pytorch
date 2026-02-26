import firedrake as fd
import numpy as np
import torch
from torch import optim
from f_poisson import linear_poisson

class non_linear_poisson(linear_poisson):
  def PDE_definition(self,u,f,V,K):
    v = fd.TestFunction(V)
    k = fd.Function(K)
    x = fd.SpatialCoordinate(mesh)
    # Spatial coordinate dependent
    k.interpolate(ufl.as_tensor([[x[0]**2  , x[0]*x[1]],
                                 [x[0]*x[1], x[1]**2]]))
    # Solution space dependent
    #k.interpolate(ufl.as_tensor([[u**2-1, 0],
    #                             [0   , u**2-1]]))
    F = fd.inner(k*fd.grad(u), fd.grad(v)) * fd.dx - fd.inner(f, v) * fd.dx
    return F

  def solve(self,u,f,V,K):
    v = fd.TestFunction(V)
    F = self.PDE_definition(u,f,V,K)
    bc = self.BC_definition(V,fd.Constant(0.1))
    fd.solve(F == 0,u, bcs = [bc])
    return u

  def control_problem(self, f, V, K):
    """
    example of control problem:
    this control problem finds the source term such that the solution has shape u_j
    """
    u = fd.Function(V)
    u_sol = self.solve(u,f,V,K)
    x,y = fd.SpatialCoordinate(self.mesh)
    u_j = fd.Function(V)
    u_j.interpolate(0.5*fd.exp((-1*(x-0.25)**2)/0.01) + 0.5*fd.exp((-1*(x-0.75)**2)/0.01))
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
    return composed_function_loss