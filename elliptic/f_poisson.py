import firedrake as fd
import numpy as np
import torch
from torch import optim


class linear_poisson(object):
    def __init__(self,mesh,model):
        self.mesh = mesh
        self.model = model
        self.optimiser = optim.AdamW(self.model.parameters(), lr = 1e-3, eps=1e-8)

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
        return fd.assemble(((u_j-u_sol)**2)*fd.dx)
        

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
        G = fd.ml.pytorch.torch_operator(Jhat)

        fd.adjoint.stop_annotating()

        composed_function_loss = G(f_p)

        return composed_function_loss

#    def optimization_iteration(self):



class poisson_BC_control(linear_poisson):
    def __init__(self,V,g,**args):
        super().__init__(**args)
        _ = self.BC_definition(V,g)


    def BC_definition(self,V,g):
        self.bc = fd.Dirichlet(V,g,"on_boundary")
        return self.bc

    def solve(self,u, f, g, V):

        v = fd.TestFunction(V)
        F = self.PDE_definition(u,f,V)
        bc = self.BC_definition(V, g)
        fd.solve(F == 0, u, bcs = [bc])
        return u

    def control_problem(self, f, g, V):
        u = fd.Function(V)
        u_sol = self.solve(u,f, g, V)
        fdofs = list( fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))
        x = fdofs[0]
        y = fdofs[1]
        u_j = fd.Function(V)
        u_j.interpolate(0.5*fd.exp((-1*(x-0.25)**2)/0.01) + 0.5*fd.exp((-1*(x-0.75)**2)/0.01))

        return fd.assemble(((u_j-u_sol)**2)*fd.dx)


    def get_boundary_points(self):
        
        self.bc.nodes
        return self.bc.nodes.shape[0]

    def get_coordinate_functions(self,V):
        return list( fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))

    def control_f(self,u,V):
        fdofs = list( fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh)) # Fx,Fy

        bcs_dof = []
        bcs_ind = self.bc.nodes
        for fdof in fdofs:
            bcs_dof.append(fdof.dat.data[bcs_ind])

        f_dof = tuple(map(lambda d:torch.tensor(d).double(),bcs_dof))
        f_p = self.model(*f_dof)
        g_p = torch.zeros(fdof.dat.data.shape).double()
        g_p[torch.tensor(bcs_ind)] = f_p

        fd.adjoint.continue_annotation()
        g = fd.Function(V)
        c = fd.adjoint.Control(g)
        Jhat = fd.adjoint.ReducedFunctional(self.control_problem(g,V),c)
        G = fd.ml.pytorch.torch_operator(Jhat)
        fd.adjoint.stop_annotating()
        composed_functional_loss = self.G(g_p)
        return composed_functional_loss,g_p


