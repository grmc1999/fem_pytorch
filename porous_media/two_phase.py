# Firedrake: 2-phase incompressible, immiscible Darcy (no capillary) in 1D/2D/3D
# IMPES / sequential:
#   1) Solve pressure (elliptic) with mobility lambda_t(Sw)
#   2) Compute total Darcy flux u_t = -K * lambda_t(Sw) * grad(p)
#   3) Update Sw with DG(0) explicit upwind:
#        phi (Sw^{n+1}-Sw^n)/dt + div( f_w(Sw^n) * u_t ) = q_w
#
# Notes:
# - Scales to 2D/3D: uses grad/div, FacetNormal, dS/ds.
# - Pressure: CG1. Saturation: DG0 (robust for advection).
# - Pressure BC: Dirichlet on "on_boundary" by default (gauge-fixing).
#   If you want injection via Neumann flux / well rates at boundary, tell me and I’ll add it cleanly.

from __future__ import annotations
from typing import List, Optional, Union
import firedrake as fd


ScalarOrExpr = Union[float, fd.Constant, fd.Function, fd.Coefficient]


class two_phase_darcy_impes(object):
    def __init__(
        self,
        mesh: fd.mesh.MeshGeometry,
        phi: float = 0.2,
        dt: float = 1.0,
        T: float = 10.0,
        # rock/permeability (scalar K; can be Function/Constant/expression)
        K: Optional[ScalarOrExpr] = None,
        # viscosities
        mu_w: float = 1.0,
        mu_o: float = 5.0,
        # time scheme: explicit Sw (keep theta=1 for pressure if you later add compressibility)
        theta: float = 1.0,
        bc_type: str = "constant",
    ):
        self.mesh = mesh
        self.phi = fd.Constant(phi)
        self.dt = fd.Constant(dt)
        self.T = float(T)
        self.theta = fd.Constant(theta)

        self.K = fd.Constant(1.0) if K is None else self._as_coeff(K)
        self.mu_w = fd.Constant(mu_w)
        self.mu_o = fd.Constant(mu_o)

        self.bc_type = bc_type
        self.bc_p: Optional[fd.DirichletBC] = None

    # -------------------------
    # Utilities
    # -------------------------
    def _as_coeff(self, val: ScalarOrExpr) -> fd.Coefficient:
        if isinstance(val, (fd.Function, fd.Constant)):
            return val
        if isinstance(val, (int, float)):
            return fd.Constant(float(val))
        return val  # UFL expression

    def get_boundary_points(self) -> int:
        if self.bc_p is None:
            raise RuntimeError("Pressure BC not defined yet. Call BC_p_definition(...) first.")
        return self.bc_p.nodes.shape[0]

    def get_coordinate_functions(self, V: fd.functionspaceimpl.WithGeometry):
        X = fd.SpatialCoordinate(self.mesh)
        return tuple(fd.Function(V, name=f"x{i}").interpolate(X[i]) for i in range(len(X)))

    # -------------------------
    # Physics: relperm, mobility, fractional flow
    # (override these methods if you want different laws / residual sats)
    # -------------------------
    def krw(self, Sw):
        return Sw**2

    def kro(self, Sw):
        So = 1.0 - Sw
        return So**2

    def lam_w(self, Sw):
        return self.krw(Sw) / self.mu_w

    def lam_o(self, Sw):
        return self.kro(Sw) / self.mu_o

    def lam_t(self, Sw):
        return self.lam_w(Sw) + self.lam_o(Sw)

    def f_w(self, Sw):
        return self.lam_w(Sw) / self.lam_t(Sw)

    # -------------------------
    # Boundary conditions
    # -------------------------
    def BC_p_definition(self, Vp: fd.functionspaceimpl.WithGeometry, g: ScalarOrExpr, where="on_boundary"):
        if self.bc_type == "constant":
            self.bc_p = fd.DirichletBC(Vp, self._as_coeff(g), where)
        elif self.bc_type == "natural":
            self.bc_p = None
            self.gN = g
        
        return self.bc_p

    def create_function_spaces(self):
        self.Vp = fd.FunctionSpace(self.mesh,"CG",1)
        self.Vs = fd.FunctionSpace(self.mesh,"DG",1)
        self.Vflux = fd.VectorFunctionSpace(self.mesh, "DG", 0, dim=self.mesh.geometric_dimension())
    # -------------------------
    # Pressure step: div( -K lam_t(Sw) grad p ) = q_t
    # Weak form:
    #   ∫ K lam_t(Sw) ∇p·∇v dx - ∫ q_t v dx = 0
    # -------------------------
    
    def PDE_pressure_definition(
        self,
        p: fd.Function,
        Sw: fd.Function,
        q_t: fd.Function,
        Vp: fd.functionspaceimpl.WithGeometry,
    ):
        v = fd.TestFunction(Vp)
        a = fd.inner(self.K * self.lam_t(Sw) * fd.grad(p), fd.grad(v)) * fd.dx
        if self.bc_type == "consttant":
            L = (q_t * v) * fd.dx
        elif self.bc_type == "natural":
            L = (q_t * v) * fd.dx + self.gN * v * fd.ds
        return a - L

    # -------------------------
    # Saturation step (DG0 explicit upwind):
    #   phi (Sw^{n+1}-Sw^n)/dt + div( f_w(Sw^n) * u_t ) = q_w
    #
    # DG flux on interior facets uses upwind by sign of (u_t · n)
    # On boundary facets, uses Sw_inj if inflow, else interior Sw.
    # -------------------------
    def PDE_saturation_definition(
        self,
        Sw_new: fd.Function,
        Sw: fd.Function,
        u_t: fd.Function,     # vector field in H(div) or projected DG vector
        q_w: fd.Function,
        Sw_inj: ScalarOrExpr,
        Vs: fd.functionspaceimpl.WithGeometry,
    ):
        w = fd.TestFunction(Vs)
        n = fd.FacetNormal(self.mesh)
        dt = self.dt

        # Mass term (DG0 gives diagonal mass matrix)
        M = (self.phi / dt) * fd.inner(Sw_new, w) * fd.dx
        rhs = (self.phi / dt) * fd.inner(Sw, w) * fd.dx + fd.inner(q_w, w) * fd.dx

        # Numerical flux for F = f_w(Sw) * u_t
        # Interior facets:
        un_int = fd.dot(fd.avg(u_t), n('+'))  # scalar
        fw_plus = self.f_w(Sw('+'))
        fw_minus = self.f_w(Sw('-'))
        fw_up = fd.conditional(un_int > 0.0, fw_plus, fw_minus)
        Fhat_int = un_int * fw_up

        rhs -= (Fhat_int * fd.jump(w)) * fd.dS

        # Boundary facets:
        un_b = fd.dot(u_t, n)  # scalar on boundary
        Sw_inj_c = self._as_coeff(Sw_inj)
        fw_inj = self.f_w(Sw_inj_c)
        fw_int = self.f_w(Sw)

        # if inflow (un_b < 0): take injected state, else take interior state
        fw_b = fd.conditional(un_b < 0.0, fw_inj, fw_int)
        Fhat_b = un_b * fw_b
        rhs -= (Fhat_b * w) * fd.ds

        return M, rhs

    # -------------------------
    # Solve full transient
    # Inputs:
    # - q_t_h: list of Functions in Vp (total source) for t0..tN
    # - q_w_h: list of Functions in Vs (water source) for t0..tN (or Vp OK if same mesh; but prefer Vs)
    # - Sw0: initial Sw (Vs)
    # - Sw_inj: inflow saturation (constant or expression)
    # Returns history of Sw (and optionally p)
    # -------------------------
    def solve(
        self,
        Vp: fd.functionspaceimpl.WithGeometry,
        Vs: fd.functionspaceimpl.WithGeometry,
        Vflux: fd.functionspaceimpl.WithGeometry,
        q_t_h: List[fd.Function],
        q_w_h: List[fd.Function],
        Sw0: Optional[fd.Function] = None,
        p_gauge: ScalarOrExpr = 0.0,
        p_bc_where: str = "on_boundary",
        Sw_inj: ScalarOrExpr = 1.0,
        store_pressure: bool = False,
        solver_parameters_p: Optional[dict] = None,
    ):
        if solver_parameters_p is None:
            solver_parameters_p = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}

        # Time steps
        dt_float = float(self.dt)
        if dt_float <= 0:
            raise ValueError("dt must be positive.")
        num_steps = int(self.T / dt_float) + 1

        if len(q_t_h) < num_steps:
            raise ValueError(f"q_t_h length ({len(q_t_h)}) < required ({num_steps}).")
        if len(q_w_h) < num_steps:
            raise ValueError(f"q_w_h length ({len(q_w_h)}) < required ({num_steps}).")

        # Unknowns
        p = fd.Function(Vp, name="p")
        Sw = fd.Function(Vs, name="Sw")
        Sw_new = fd.Function(Vs, name="Sw_new")

        # IC
        if Sw0 is None:
            Sw.interpolate(fd.Constant(0.05))
        else:
            Sw.assign(Sw0)

        # Pressure BC (gauge fixing)
        bc_p = self.BC_p_definition(Vp, p_gauge, where=p_bc_where)

        # Total flux field for transport
        # Use DG vector space for robust facet flux evaluation
        dim = self.mesh.geometric_dimension()
        #Vflux = fd.VectorFunctionSpace(self.mesh, "DG", 0, dim=dim)
        u_t = fd.Function(Vflux, name="u_t")

        Sw_hist: List[fd.Function] = [Sw.copy(deepcopy=True)]
        p_hist: List[fd.Function] = [p.copy(deepcopy=True)] if store_pressure else []

        for nstep in range(num_steps - 1):
            # ---- 1) Pressure solve at time n (using Sw^n)
            Fp = self.PDE_pressure_definition(p, Sw, q_t_h[nstep], Vp)
    #        fd.solve(Fp == 0, p, bcs=[bc_p], solver_parameters=solver_parameters_p)
            
            if self.bc_type =="constant":
                fd.solve(Fp == 0, p, bcs=[bc_p], solver_parameters=solver_parameters_p)
            elif self.bc_type =="natural":
                fd.solve(Fp == 0, p, solver_parameters=solver_parameters_p)

            # ---- 2) Total flux u_t = -K * lam_t(Sw) * grad(p)
            # Project to DG0 vector space
            u_t.project(-self.K * self.lam_t(Sw) * fd.grad(p))

            # ---- 3) Explicit saturation update with upwind flux
            M, rhs = self.PDE_saturation_definition(
                Sw_new=Sw_new,
                Sw=Sw,
                u_t=u_t,
                q_w=q_w_h[nstep],
                Sw_inj=Sw_inj,
                Vs=Vs,
            )
            fd.solve(M == rhs, Sw_new, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

            # Optional clamp for safety (common in explicit transport)
            Sw_new.dat.data[:] = Sw_new.dat.data.clip(0.0, 1.0)

            # Update
            Sw.assign(Sw_new)

            Sw_hist.append(Sw.copy(deepcopy=True))
            if store_pressure:
                p_hist.append(p.copy(deepcopy=True))

        return (Sw_hist, p_hist) if store_pressure else (Sw_hist, None)

    # Single IMPES step helper (useful for differentiable / loop-in-loop)
    def solve_step(
        self,
        p: fd.Function,
        Sw_new: fd.Function,
        Sw: fd.Function,
        q_t: fd.Function,
        q_w: fd.Function,
        Vp: fd.functionspaceimpl.WithGeometry,
        Vs: fd.functionspaceimpl.WithGeometry,
        Vflux: fd.functionspaceimpl.WithGeometry,
        p_gauge: ScalarOrExpr = 0.0,
        p_bc_where: str = "on_boundary",
        Sw_inj: ScalarOrExpr = 1.0,
        solver_parameters_p: Optional[dict] = None,
    ):
        if solver_parameters_p is None:
            solver_parameters_p = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}

        bc_p = self.BC_p_definition(Vp, p_gauge, where=p_bc_where)

        # Flux space
        dim = self.mesh.geometric_dimension()
        #Vflux = fd.VectorFunctionSpace(self.mesh, "DG", 0, dim=dim)
        u_t = fd.Function(Vflux, name="u_t")

        # Pressure
        Fp = self.PDE_pressure_definition(p, Sw, q_t, Vp)
        if self.bc_type =="constant":
            fd.solve(Fp == 0, p, bcs=[bc_p], solver_parameters=solver_parameters_p)
        elif self.bc_type =="natural":
            fd.solve(Fp == 0, p, solver_parameters=solver_parameters_p)
        # Flux
        u_t.project(-self.K * self.lam_t(Sw) * fd.grad(p))

        # Saturation
        M, rhs = self.PDE_saturation_definition(Sw_new, Sw, u_t, q_w, Sw_inj, Vs)
        fd.solve(M == rhs, Sw_new, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})
        Sw_new.dat.data[:] = Sw_new.dat.data.clip(0.0, 1.0)

        return p, Sw_new


# -------------------------
# Minimal quick test (2D)
# -------------------------
if __name__ == "__main__":
    mesh = fd.UnitSquareMesh(40, 40)
    Vp = fd.FunctionSpace(mesh, "CG", 1)
    Vs = fd.FunctionSpace(mesh, "DG", 0)

    x, y = fd.SpatialCoordinate(mesh)

    # Total source (e.g., injection/production pair inside domain)
    # Make it zero-mean to avoid incompatibility if you later switch to pure Neumann BC.
    q_inj = fd.exp(-200.0 * ((x - 0.25) ** 2 + (y - 0.5) ** 2))
    q_prd = fd.exp(-200.0 * ((x - 0.75) ** 2 + (y - 0.5) ** 2))
    q_t_expr = q_inj - q_prd

    # Water source (inject water at injector, produce proportional at producer here just as demo)
    q_w_expr = q_inj  # simplest

    T = 0.02
    dt = 5e-4
    num_steps = int(T / dt) + 1

    q_t_h = [fd.Function(Vp).interpolate(q_t_expr) for _ in range(num_steps)]
    q_w_h = [fd.Function(Vs).interpolate(q_w_expr) for _ in range(num_steps)]

    # Heterogeneous K(x,y) > 0
    K = 1.0 + 0.5 * fd.sin(2 * fd.pi * x) * fd.sin(2 * fd.pi * y)

    solver = two_phase_darcy_impes(mesh, phi=0.2, dt=dt, T=T, K=K, mu_w=1.0, mu_o=5.0)

    Sw_hist, _ = solver.solve(
        Vp=Vp,
        Vs=Vs,
        q_t_h=q_t_h,
        q_w_h=q_w_h,
        Sw0=None,
        p_gauge=0.0,          # p=0 on boundary (simple gauge)
        p_bc_where="on_boundary",
        Sw_inj=1.0,           # used only for inflow boundary facets
        store_pressure=False,
    )

    Sw_last = Sw_hist[-1]
    fd.File("two_phase_impes_2d.pvd").write(Sw_last)