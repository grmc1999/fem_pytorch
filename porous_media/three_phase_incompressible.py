# Firedrake: 3-phase immiscible incompressible Darcy flow (water–oil–gas), no capillary, no mass transfer
# Scales to 1D / 2D / 3D.
#
# PDE system:
#   u_t = -k * lambda_t(S) * ∇p
#   ∇·u_t = q_t
#   φ ∂_t S_w + ∇·( f_w(S) u_t ) = q_w
#   φ ∂_t S_g + ∇·( f_g(S) u_t ) = q_g
#   S_o = 1 - S_w - S_g
#
# with mobilities:
#   lambda_alpha = k_r_alpha(S) / mu_alpha, alpha ∈ {w,o,g}
#   lambda_t = lambda_w + lambda_o + lambda_g
#   f_alpha = lambda_alpha / lambda_t
#
# Discretization (IMPES-like):
#   - Pressure: CG1 solve each time step (using current saturations)
#   - Total flux: vector DG0 (2D/3D) or scalar DG0 (1D)
#   - Saturations Sw,Sg: DG0 explicit upwind transport (two separate updates)
#
# Boundary conditions:
#   - Pressure gauge fixing: Dirichlet p = p_dirichlet_value on p_dirichlet_tag (recommended in 2D/3D)
#   - Total inflow/outflow via Neumann term on specified tags:
#       Add ∫ (u_in[tag]) * v ds(tag) to pressure equation, where u_in[tag] is positive INTO the domain.
#   - Inflow compositions for saturations on the same (or other) boundary tags:
#       sat_inflow_tags = {tag: (Sw_inj, Sg_inj)}
#       If u_t·n < 0 (inflow), upwind state is injected; otherwise interior.

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import copy
import firedrake as fd


class three_phase_immiscible_impes(object):
    def __init__(
        self,
        mesh: fd.mesh.MeshGeometry,
        phi: float = 0.2,
        dt: float = 1.0e-3,
        T: float = 1.0,
        k_abs: float = 1.0,
        mu_w: float = 1.0,
        mu_o: float = 5.0,
        mu_g: float = 0.8,
        # Corey exponents (simple defaults)
        nw: float = 2.0,
        no: float = 2.0,
        ng: float = 2.0,
        # Numerical options
        clamp_S: bool = True,
        # Pressure BC (gauge fix)
        p_dirichlet_tag: Optional[int] = None,
        p_dirichlet_value: float = 0.0,
        # Total flux BC on boundary tags (Neumann-like), values are positive INTO domain
        inflow_flux_tags: Optional[Dict[int, float]] = None,
        # Saturation inflow composition on boundary tags: (Sw_inj, Sg_inj)
        sat_inflow_tags: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        self.mesh = mesh
        self.phi = fd.Constant(phi)
        self.dt = fd.Constant(dt)
        self.T = float(T)

        self.k_abs = fd.Constant(k_abs)
        self.mu_w = fd.Constant(mu_w)
        self.mu_o = fd.Constant(mu_o)
        self.mu_g = fd.Constant(mu_g)

        self.nw = fd.Constant(nw)
        self.no = fd.Constant(no)
        self.ng = fd.Constant(ng)

        self.clamp_S = clamp_S

        self.p_dirichlet_tag = p_dirichlet_tag
        self.p_dirichlet_value = fd.Constant(p_dirichlet_value)

        self.inflow_flux_tags = inflow_flux_tags or {}
        self.sat_inflow_tags = sat_inflow_tags or {}

        self.bc_p: Optional[fd.DirichletBC] = None

    # -------------------------
    # Utilities (match your style)
    # -------------------------
    def get_boundary_points(self):
        if self.bc_p is None:
            raise RuntimeError("Call BC_definition_pressure(...) first so self.bc_p is defined.")
        self.bc_p.nodes
        return self.bc_p.nodes.shape[0]

    def get_coordinate_functions(self, V: fd.functionspaceimpl.WithGeometry):
        return tuple(fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))

    # -------------------------
    # RelPerm / mobilities / frac flows
    # -------------------------
    def _So(self, Sw, Sg):
        return 1.0 - Sw - Sg

    def krw(self, Sw):
        # Corey-style
        return fd.max_value(Sw, 0.0) ** self.nw

    def krg(self, Sg):
        return fd.max_value(Sg, 0.0) ** self.ng

    def kro(self, Sw, Sg):
        So = fd.max_value(self._So(Sw, Sg), 0.0)
        return So ** self.no

    def lam_w(self, Sw):
        return self.krw(Sw) / self.mu_w

    def lam_g(self, Sg):
        return self.krg(Sg) / self.mu_g

    def lam_o(self, Sw, Sg):
        return self.kro(Sw, Sg) / self.mu_o

    def lam_t(self, Sw, Sg):
        return self.lam_w(Sw) + self.lam_o(Sw, Sg) + self.lam_g(Sg)

    def f_w(self, Sw, Sg):
        return self.lam_w(Sw) / self.lam_t(Sw, Sg)

    def f_g(self, Sw, Sg):
        return self.lam_g(Sg) / self.lam_t(Sw, Sg)

    def f_o(self, Sw, Sg):
        return self.lam_o(Sw, Sg) / self.lam_t(Sw, Sg)

    # -------------------------
    # IC / BC definitions
    # -------------------------
    def IC_definition(self, V, u0: fd.Function):
        self.ic = u0
        return self.ic

    def BC_definition_pressure(self, Vp):
        """
        Gauge-fixing BC: strongly recommended in 2D/3D.
        If p_dirichlet_tag is None, uses "on_boundary" (often overconstrains in 2D/3D).
        """
        if self.p_dirichlet_tag is None:
            self.bc_p = fd.DirichletBC(Vp, self.p_dirichlet_value, "on_boundary")
        else:
            self.bc_p = fd.DirichletBC(Vp, self.p_dirichlet_value, self.p_dirichlet_tag)
        return self.bc_p

    # -------------------------
    # Pressure PDE (CG)
    # -------------------------
    def PDE_definition_pressure(
        self,
        p: fd.Function,
        Sw: fd.Function,
        Sg: fd.Function,
        q_t: fd.Function,
        Vp: fd.functionspaceimpl.WithGeometry,
    ):
        """
        Weak form of:
          ∇·( -k λ_t(S) ∇p ) = q_t
        i.e.
          ∫ k λ_t ∇p·∇v dx = ∫ q_t v dx + ∑_tags ∫ u_in(tag) v ds(tag)
        where u_in(tag) is positive INTO the domain.
        """
        v = fd.TestFunction(Vp)
        lt = self.lam_t(Sw, Sg)

        F = fd.inner(self.k_abs * lt * fd.grad(p), fd.grad(v)) * fd.dx - (q_t * v) * fd.dx

        # Add Neumann-like flux contributions on specified tags (positive INTO domain)
        for tag, u_in_val in self.inflow_flux_tags.items():
            F -= (fd.Constant(float(u_in_val)) * v) * fd.ds(tag)  # move to LHS as residual

        return F

    # -------------------------
    # Compute total flux u_t
    # -------------------------
    def compute_total_flux(self, Ut: fd.Function, p: fd.Function, Sw: fd.Function, Sg: fd.Function):
        """
        u_t = -k λ_t(S) ∇p
        Ut should be:
          - DG0 scalar in 1D
          - DG0 vector in 2D/3D
        """
        lt = self.lam_t(Sw, Sg)
        Ut.project(-self.k_abs * lt * fd.grad(p))
        return Ut

    # -------------------------
    # DG0 explicit upwind transport for one saturation
    # -------------------------
    def transport_step_DG0(
        self,
        S_new: fd.Function,
        S: fd.Function,
        Sw: fd.Function,
        Sg: fd.Function,
        Ut: fd.Function,
        qS: fd.Function,
        Vs: fd.functionspaceimpl.WithGeometry,
        phase: str,
    ):
        """
        Updates either Sw or Sg:
          φ (S_new - S)/dt + ∇·( f_phase(S) u_t ) = q_phase
        with upwind numerical flux.
        """
        w = fd.TestFunction(Vs)
        n = fd.FacetNormal(self.mesh)
        dim = self.mesh.geometric_dimension()

        # Choose fractional flow function for the requested phase
        if phase == "w":
            f_phase = self.f_w(Sw, Sg)
        elif phase == "g":
            f_phase = self.f_g(Sw, Sg)
        else:
            raise ValueError("phase must be 'w' or 'g'")

        # Interior facet normal flux
        if dim == 1:
            u_n_int = fd.avg(Ut) * n('+')[0]
            u_n_b = Ut * n[0]
        else:
            u_n_int = fd.dot(fd.avg(Ut), n('+'))
            u_n_b = fd.dot(Ut, n)

        # Upwind selection on interior facets
        f_plus = f_phase('+')
        f_minus = f_phase('-')
        f_up_int = fd.conditional(u_n_int > 0.0, f_plus, f_minus)
        Fhat_int = u_n_int * f_up_int

        # Boundary fluxes:
        # For tags in sat_inflow_tags: if inflow (u_n < 0), use injected composition, else interior.
        Fhat_bdry = 0
        if self.sat_inflow_tags:
            for tag, (Sw_inj, Sg_inj) in self.sat_inflow_tags.items():
                Sw_inj_c = fd.Constant(float(Sw_inj))
                Sg_inj_c = fd.Constant(float(Sg_inj))

                if phase == "w":
                    f_inj = self.f_w(Sw_inj_c, Sg_inj_c)
                else:
                    f_inj = self.f_g(Sw_inj_c, Sg_inj_c)

                f_int = f_phase
                f_up = fd.conditional(u_n_b >= 0.0, f_int, f_inj)
                Fhat_bdry += (u_n_b * f_up) * w * fd.ds(tag)
        else:
            # If you didn't provide inflow compositions, default to interior state on all boundaries.
            Fhat_bdry += (u_n_b * f_phase) * w * fd.ds

        # DG0 mass matrix equation:
        #   ∫ φ/dt S_new w dx = ∫ φ/dt S w dx - ∫_int Fhat jump(w) dS - ∫_bdry Fhat w ds + ∫ qS w dx
        M = (self.phi / self.dt) * fd.inner(S_new, w) * fd.dx
        rhs = (self.phi / self.dt) * fd.inner(S, w) * fd.dx \
              - (Fhat_int * fd.jump(w)) * fd.dS \
              - Fhat_bdry \
              + (qS * w) * fd.dx

        F = M - rhs
        fd.solve(F == 0, S_new, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

        return S_new

    # -------------------------
    # One IMPES step
    # -------------------------
    def solve_step(
        self,
        p: fd.Function,
        Sw: fd.Function,
        Sg: fd.Function,
        q_t: fd.Function,
        q_w: fd.Function,
        q_g: fd.Function,
        Vp: fd.functionspaceimpl.WithGeometry,
        Vs: fd.functionspaceimpl.WithGeometry,
        Vflux: fd.functionspaceimpl.WithGeometry,
        Ut: Optional[fd.Function] = None,
        Sw_new: Optional[fd.Function] = None,
        Sg_new: Optional[fd.Function] = None,
    ) -> Tuple[fd.Function, fd.Function, fd.Function, fd.Function]:
        if Ut is None:
            Ut = fd.Function(Vflux, name="u_t")
        if Sw_new is None:
            Sw_new = fd.Function(Vs, name="Sw_new")
        if Sg_new is None:
            Sg_new = fd.Function(Vs, name="Sg_new")

        # 1) pressure solve
        bc = self.BC_definition_pressure(Vp)
        Fp = self.PDE_definition_pressure(p, Sw, Sg, q_t, Vp)
        fd.solve(
            Fp == 0,
            p,
            bcs=[bc],
            solver_parameters={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        )

        # 2) total flux
        self.compute_total_flux(Ut, p, Sw, Sg)

        # 3) saturation updates (explicit, using Sw^n,Sg^n in fractional flows)
        self.transport_step_DG0(Sw_new, Sw, Sw, Sg, Ut, q_w, Vs, phase="w")
        self.transport_step_DG0(Sg_new, Sg, Sw, Sg, Ut, q_g, Vs, phase="g")

        # Optional clamping and renormalization
        if self.clamp_S:
            Sw_new.dat.data[:] = Sw_new.dat.data.clip(0.0, 1.0)
            Sg_new.dat.data[:] = Sg_new.dat.data.clip(0.0, 1.0)

            # Enforce Sw + Sg <= 1 by simple rescaling if needed (keeps direction, avoids negative So)
            Sw_arr = Sw_new.dat.data
            Sg_arr = Sg_new.dat.data
            ssum = Sw_arr + Sg_arr
            mask = ssum > 1.0
            if mask.any():
                Sw_arr[mask] = Sw_arr[mask] / ssum[mask]
                Sg_arr[mask] = Sg_arr[mask] / ssum[mask]

        Sw.assign(Sw_new)
        Sg.assign(Sg_new)

        return p, Sw, Sg, Ut

    # -------------------------
    # Full time loop
    # -------------------------
    def solve(
        self,
        q_t_h: List[fd.Function],
        q_w_h: List[fd.Function],
        q_g_h: List[fd.Function],
        Vp: Optional[fd.functionspaceimpl.WithGeometry] = None,
        Vs: Optional[fd.functionspaceimpl.WithGeometry] = None,
        Vflux: Optional[fd.functionspaceimpl.WithGeometry] = None,
        p0: Optional[fd.Function] = None,
        Sw0: Optional[fd.Function] = None,
        Sg0: Optional[fd.Function] = None,
    ) -> Tuple[List[fd.Function], List[fd.Function], List[fd.Function], List[fd.Function]]:
        dim = self.mesh.geometric_dimension()

        if Vp is None:
            Vp = fd.FunctionSpace(self.mesh, "CG", 1)
        if Vs is None:
            Vs = fd.FunctionSpace(self.mesh, "DG", 0)
        if Vflux is None:
            if dim == 1:
                Vflux = fd.FunctionSpace(self.mesh, "DG", 0)  # scalar flux in 1D
            else:
                Vflux = fd.VectorFunctionSpace(self.mesh, "DG", 0)  # vector flux in 2D/3D

        p = fd.Function(Vp, name="p")
        Sw = fd.Function(Vs, name="Sw")
        Sg = fd.Function(Vs, name="Sg")

        if p0 is None:
            p0 = fd.Function(Vp).interpolate(fd.Constant(0.0))
        if Sw0 is None:
            Sw0 = fd.Function(Vs).interpolate(fd.Constant(0.05))
        if Sg0 is None:
            Sg0 = fd.Function(Vs).interpolate(fd.Constant(0.0))

        self.IC_definition(Vp, p0); p.assign(self.ic)
        self.IC_definition(Vs, Sw0); Sw.assign(self.ic)
        self.IC_definition(Vs, Sg0); Sg.assign(self.ic)

        Ut = fd.Function(Vflux, name="u_t")
        Sw_new = fd.Function(Vs, name="Sw_new")
        Sg_new = fd.Function(Vs, name="Sg_new")

        num_steps = int(self.T / float(self.dt.values()))
        if len(q_t_h) < num_steps or len(q_w_h) < num_steps or len(q_g_h) < num_steps:
            raise ValueError("q_*_h lists must have at least num_steps entries.")

        p_hist = [copy.deepcopy(p)]
        Sw_hist = [copy.deepcopy(Sw)]
        Sg_hist = [copy.deepcopy(Sg)]
        Ut_hist: List[fd.Function] = []

        for n in range(num_steps - 1):
            self.solve_step(
                p=p, Sw=Sw, Sg=Sg,
                q_t=q_t_h[n + 1],
                q_w=q_w_h[n + 1],
                q_g=q_g_h[n + 1],
                Vp=Vp, Vs=Vs, Vflux=Vflux,
                Ut=Ut, Sw_new=Sw_new, Sg_new=Sg_new
            )
            p_hist.append(copy.deepcopy(p))
            Sw_hist.append(copy.deepcopy(Sw))
            Sg_hist.append(copy.deepcopy(Sg))
            Ut_hist.append(copy.deepcopy(Ut))

        return p_hist, Sw_hist, Sg_hist, Ut_hist


# -----------------------------
# Minimal example (2D)
# -----------------------------
if __name__ == "__main__":
    # Pick mesh:
    mesh = fd.UnitSquareMesh(64, 64)
    # For UnitSquareMesh, boundary tags are typically:
    #   1: x=0, 2: x=1, 3: y=0, 4: y=1

    dt = 5e-4
    T = 0.05
    num_steps = int(T / dt)

    q_t_h = [fd.Constant(0.0) for _ in range(num_steps)]
    q_w_h = [fd.Constant(0.0) for _ in range(num_steps)]
    q_g_h = [fd.Constant(0.0) for _ in range(num_steps)]

    solver = three_phase_immiscible_impes(
        mesh=mesh,
        phi=0.2,
        dt=dt,
        T=T,
        k_abs=1.0,
        mu_w=1.0,
        mu_o=5.0,
        mu_g=0.8,
        # Gauge fix p=0 on outlet x=1 (tag 2)
        p_dirichlet_tag=2,
        p_dirichlet_value=0.0,
        # Total inflow on inlet x=0 (tag 1): positive INTO domain
        inflow_flux_tags={1: 1.0},
        # Inject mostly water at inlet
        sat_inflow_tags={1: (1.0, 0.0)},
    )

    p_hist, Sw_hist, Sg_hist, Ut_hist = solver.solve(q_t_h=q_t_h, q_w_h=q_w_h, q_g_h=q_g_h)

    # Save final state (So is implicit: 1 - Sw - Sg)
    So = fd.Function(Sw_hist[-1].function_space(), name="So")
    So.interpolate(1.0 - Sw_hist[-1] - Sg_hist[-1])

    out = fd.File("three_phase_immiscible_impes.pvd")
    out.write(p_hist[-1], Sw_hist[-1], Sg_hist[-1], So)