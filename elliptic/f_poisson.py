import firedrake as fd
import numpy as np
import torch


class linear_poisson(object):
    def __init__(self,mesh,model):
        self.mesh = mesh
        self.model = model

    def PDE_definition(self,u,f,V):
        v = fd.TestFunction(V)
        F = (fd.inner(fd.grad(u),fd.grad(v)) + fd.inner(u,v) - fd.inner(f,v)) * fd.dx
        return F
    
    def BC_definition(self,V,g):
        bc = fd.DirichletBC(V, g, "on_boundary")
        return bc

    def solve(self,u,f,V):

        v = fd.TestFunction(V)
        F = self.PDE_definition(u,f,V)
        bc = self.BC_definition(V, fd.Constant(1.0))
        fd.solve(F == 0, u, bcs = [bc])
        return u

    def control_problem(self, f, V):
        """
        example of control problem:
        this control problem finds the source term such that the solution has shape u_j
        """
        u = fd.Function(V)
        u_sol = self.solve(u,f,V)

        x,y = fd.SpatialCoordinate(self.mesh)
        u_j = fd.Function(V)
        u_j.interpolate(0.5*fd.exp((-1*(x-0.25)**2)/0.01) + 0.5*fd.exp((-1*(x-0.75)**2)/0.01))

    def control_f(self, u, V):
        """
        control problem example:
        ml model estimates source (f) as a mapping from coordinates to f
        """
        x,y = fd.SpatialCoordinate(self.mesh)
        fx = fd.Function(V)
        fx.interpolate(x)
        fy = fd.Function(V)
        fy.interpolate(y)

        f_x = fd.ml.pytorch.to_torch(fx)
        f_y = fd.ml.pytorch.to_torch(fy)

        f_p = self.model(x,y)

        fd.adjoint.continue_annotation()
        f = fd.Function(V)
        c = fd.adjoint.Control(f)

        Jhat = fd.adjoint.ReducedFunctional(self.control_problem(f,V),c)
        G = fd.ml.pytorch.torch.operator(Jhat)

        fd.adjoint.stop_annotation()

        composed_function_loss = G(f_p)

        return composed_function_loss

    def optimization_iteration(self):



class poisson_BC_control(linear_poisson):
    def __init__(**args):
        super().__init__(**args)

    def solve(self,u, f, g, V):

        v = fd.TestFunction(V)
        F = self.PDE_definition(u,f,V)
        bc = self.BC_definition(V, g)
        fd.solve(F == 0, u, bcs = [bc])
        return u

    def control_problem(self, g, V):
        u = fd.Function(V)
        u_sol = self.solve(u,f, g, V)

