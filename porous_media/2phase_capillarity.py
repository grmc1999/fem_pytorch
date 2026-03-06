# Firedrake: 2-phase incompressible Darcy flow WITH capillarity (scales to 2D/3D)
#
# Goal model (no gravity):
#   u_w = -k * lambda_w(Sw) * ∇p_w
#   u_o = -k * lambda_o(Sw) * ∇p_o,      p_o = p_w + p_c(Sw)
#   u_t = u_w + u_o
#   ∇·u_t = q_t
#   φ ∂_t Sw + ∇·u_w = q_w
#
# Using p_w as reference pressure:
#   u_t = -k * lambda_t(Sw) ∇p_w - k * lambda_o(Sw) ∇p_c(Sw)
#   ∇·u_t = q_t
#
# And water flux identity:
#   u_w = f_w(Sw) u_t + k * D(Sw) ∇p_c(Sw),    D = lambda_w*lambda_o/lambda_t
#
# Discretization (IMPES-like):
#   1) Pressure: CG1, solve each step with Sw^n in mobility and capillary term
#   2) Saturation: DG1, semi-implicit:
#        - advection treated explicitly with upwind flux using Sw^n
#        - capillary diffusion treated implicitly via SIPG on p_c(Sw^{n+1})
#
# Notes for scaling:
#   - Works on IntervalMesh, UnitSquareMesh, UnitCubeMesh, or any Firedrake mesh.
#   - Boundary IDs depend on mesh generator; defaults provided for common meshes.
#   - For complex geometries, pass facet markers and customize BC handling.

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import copy
import firedrake as fd


class two_phase_capillary_impes(object):
    def __init__(
        self,
        mesh: fd.mesh.MeshGeometry,
        phi: float = 0.2,
        dt: float = 1.0e-3,
        T: float = 1.0,
        k_abs: float = 1.0,
        mu_w: float = 1.0,
        mu_o: float = 5.0,
        # capillary pressure parameters (simple smooth model by default)
        pc0: float = 1.0,
        pc_exp: float = 2.0,
        pc_eps: float = 1.0e-3,   # smoothing to avoid singularities
        # solver controls
        p_degree: int = 1,
        Sw_degree: int = 1,       # DG degree (>=1 recommended for capillary diffusion)
        penalty_CIP: float = 20.0, # SIPG penalty base factor
        clamp_S: bool = True,
        # boundary configuration
        p_dirichlet_tag: Optional[int] = None,  # if None, tries to use "on_boundary" Dirichlet for p
        p_dirichlet_value: float = 0.0,
        # inflow saturation BC via upwind numerical flux: provide (tag -> Sw_inj)
        Sw_inflow_tags: Optional[Dict[int, float]] = None,
    ):
        self.mesh = mesh
        self.phi = fd.Constant(phi)
        self.dt = fd.Constant(dt)
        self.T = float(T)

        self.k_abs = fd.Constant(k_abs)
        self.mu_w = fd.Constant(mu_w)
        self.mu_o = fd.Constant(mu_o)

        self.pc0 = fd.Constant(pc0)
        self.pc_exp = fd.Constant(pc_exp)
        self.pc_eps = fd.Constant(pc_eps)

        self.p_degree = int(p_degree)
        self.Sw_degree = int(Sw_degree)
        if self.Sw_degree < 1:
            raise ValueError("For capillary diffusion, use DG degree >= 1 (DG0 cannot represent gradients).")

        self.penalty_CIP = fd.Constant(penalty_CIP)
        self.clamp_S = clamp_S

        self.p_dirichlet_tag = p_dirichlet_tag
        self.p_dirichlet_value = fd.Constant(p_dirichlet_value)

        # inflow tags: on those facets, when u_t·n < 0 (inflow), set Sw_up = Sw_inj(tag)
        self.Sw_inflow_tags = Sw_inflow_tags or {}

        self.bc_p = None

    # -------------------------
    # Utilities
    # -------------------------
    def get_boundary_points(self):
        if self.bc_p is None:
            raise RuntimeError("Call BC_definition_pressure(...) first so self.bc_p is defined.")
        self.bc_p.nodes
        return self.bc_p.nodes.shape[0]

    def get_coordinate_functions(self, V: fd.functionspaceimpl.WithGeometry):
        return tuple(fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))

    # -------------------------
    # RelPerm / Mobilities / Fractional flow
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

    def D_cap(self, Sw):
        # D(S) = lam_w lam_o / lam_t
        return (self.lam_w(Sw) * self.lam_o(Sw)) / self.lam_t(Sw)

    # -------------------------
    # Capillary pressure model p_c(Sw)
    # -------------------------
    def p_c(self, Sw):
        """
        Smooth capillary pressure law. You can swap this for Brooks–Corey / van Genuchten, etc.
        Here: pc = pc0 * ( (1 - Sw + eps)^(-exp) - 1 )
        Smooth + monotone increasing as Sw decreases.
        """
        Se = 1.0 - Sw + self.pc_eps
        return self.pc0 * (Se**(-self.pc_exp) - 1.0)

    # -------------------------
    # IC / BC
    # -------------------------
    def IC_definition_pressure(self, Vp, p0: fd.Function):
        self.p_ic = p0
        return self.p_ic

    def IC_definition_saturation(self, Vs, Sw0: fd.Function):
        self.Sw_ic = Sw0
        return self.Sw_ic

    def BC_definition_pressure(self, Vp):
        """
        Pressure BC:
          - If p_dirichlet_tag is provided: DirichletBC(Vp, value, tag)
          - Else: DirichletBC(Vp, value, "on_boundary") (works for many meshes but may overconstrain)
        Recommendation for 2D/3D: provide a specific boundary tag for gauge fixing.
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
        p: fd.Function,     # p_w^{n+1}
        Sw: fd.Function,    # Sw^n
        q_t: fd.Function,   # q_t^{n+1}
        Vp: fd.functionspaceimpl.WithGeometry,
    ):
        """
        Derivation uses:
          u_t = -k λ_t ∇p - k λ_o ∇p_c(Sw)
          ∇·u_t = q_t
        Weak form (natural for total flux):
          ∫ k λ_t ∇p·∇v dx = ∫ q_t v dx - ∫ k λ_o ∇p_c(Sw)·∇v dx
        """
        v = fd.TestFunction(Vp)
        lt = self.lam_t(Sw)
        lo = self.lam_o(Sw)
        pc = self.p_c(Sw)

        F = fd.inner(self.k_abs * lt * fd.grad(p), fd.grad(v)) * fd.dx \
            - (q_t * v) * fd.dx \
            + fd.inner(self.k_abs * lo * fd.grad(pc), fd.grad(v)) * fd.dx
        return F

    # -------------------------
    # Saturation PDE (DG + SIPG for capillarity)
    # -------------------------
    def _upwind_fw(self, Sw, Ut, tag=None):
        """
        Upwind fractional flow on a boundary facet:
          - if inflow (Ut·n < 0): use prescribed Sw_inj(tag) if available, else use interior Sw
          - if outflow (Ut·n >= 0): use interior Sw
        """
        if tag is None or tag not in self.Sw_inflow_tags:
            Sw_inj = Sw
        else:
            Sw_inj = fd.Constant(float(self.Sw_inflow_tags[tag]))
        return Sw_inj

    def PDE_definition_saturation_semiimplicit(
        self,
        Sw_new: fd.Function,  # Sw^{n+1} (unknown)
        Sw: fd.Function,      # Sw^n (known)
        Ut: fd.Function,      # total flux at time n+1 (computed from p^{n+1}, Sw^n) stored in DG space
        q_w: fd.Function,     # q_w^{n+1}
        Vs: fd.functionspaceimpl.WithGeometry,
    ):
        """
        Semi-implicit DG scheme:
          φ (Sw_new - Sw)/dt + ∇·( f_w(Sw) Ut )  - ∇·( k D(Sw_new) ∇p_c(Sw_new) ) = q_w

        - Advection: explicit in Sw^n with upwind DG flux
        - Capillary diffusion: implicit via SIPG on p_c(Sw_new) with coefficient k D(Sw_new)

        IMPORTANT:
          This is a nonlinear solve in Sw_new because p_c(Sw_new) and D(Sw_new) are nonlinear.
        """
        w = fd.TestFunction(Vs)
        n = fd.FacetNormal(self.mesh)
        h = fd.CellDiameter(self.mesh)

        # ----------------
        # (1) Mass term
        # ----------------
        F_mass = (self.phi / self.dt) * (Sw_new - Sw) * w * fd.dx

        # ----------------
        # (2) Advection term (explicit in Sw)
        # ----------------
        # interior facets
        u_n_int = fd.avg(Ut) * fd.dot(n('+'), fd.as_vector((1.0,) * self.mesh.geometric_dimension()))
        # NOTE: above line is a trick to get a scalar from n('+'); better: take dot(Ut_vec, n)
        # but Ut is scalar "normal-to-x" in 1D; for 2D/3D we need vector flux.
        # So: we interpret Ut as a *scalar normal flux* on facets only if you provide it that way.
        # Instead, we build Ut_vec = Ut * e_x for structured cases is wrong in general.
        #
        # -> For general 2D/3D you should store Ut as a *vector* in a VectorFunctionSpace
        # and use: u_n_int = dot(avg(Ut_vec), n('+')).
        #
        # To keep this class usable in 2D/3D, we assume Ut is a VECTOR Function if dim>1.
        #
        dim = self.mesh.geometric_dimension()
        if dim == 1:
            # In 1D, Ut is scalar DG; normal is ±1; choose normal from '+'
            u_n_int = fd.avg(Ut) * n('+')[0]
            u_n_L = Ut * n[0]
        else:
            # In 2D/3D, require Ut to be a vector Function (Vector DG)
            u_n_int = fd.dot(fd.avg(Ut), n('+'))
            u_n_L = fd.dot(Ut, n)

        fw_plus = self.f_w(Sw('+'))
        fw_minus = self.f_w(Sw('-'))
        fw_up_int = fd.conditional(u_n_int > 0.0, fw_plus, fw_minus)

        F_adv = (fw_up_int * u_n_int) * fd.jump(w) * fd.dS

        # boundary advection terms:
        # Use a generic upwind boundary flux with optional inflow tags
        # For "on_boundary" ds, Firedrake doesn't expose tag inside form; so we handle by tags if provided.
        # If you pass tags, we add contributions for each tag explicitly; otherwise treat all boundary as "natural".
        F_adv_bdry = 0
        if self.Sw_inflow_tags:
            for tag, Sw_inj_val in self.Sw_inflow_tags.items():
                # inflow uses injected saturation if u_n < 0
                fw_inj = self.f_w(fd.Constant(float(Sw_inj_val)))
                fw_int = self.f_w(Sw)
                fw_up = fd.conditional(u_n_L >= 0.0, fw_int, fw_inj)
                F_adv_bdry += (fw_up * u_n_L) * w * fd.ds(tag)
        else:
            # If you didn't specify inflow tags, we do "do-nothing": take interior value always.
            # This is OK for outflow-only problems; for inflow you should set Sw_inflow_tags.
            fw_int = self.f_w(Sw)
            F_adv_bdry += (fw_int * u_n_L) * w * fd.ds

        # The sign convention in DG: add boundary flux as +∫ Fhat * w ds,
        # while interior is +∫ Fhat * jump(w) dS; we already follow that convention.
        # Bring advection to residual with +F_adv + F_adv_bdry.
        # But our PDE has +∇·(f Ut) term; in DG residual we use + flux terms.
        # So: F_total += F_adv + F_adv_bdry.

        # ----------------
        # (3) Capillary diffusion: -∇·( k D(Sw_new) ∇pc(Sw_new) )
        #     SIPG on variable pc(Sw_new)
        # ----------------
        pc_new = self.p_c(Sw_new)
        Kcap = self.k_abs * self.D_cap(Sw_new)  # coefficient

        # Consistent interior diffusion terms:
        #   ∫ Kcap ∇pc · ∇w dx
        # - ∫ avg(Kcap ∇pc)·n('+') * jump(w) dS
        # - ∫ avg(Kcap ∇w) · n('+') * jump(pc) dS   (symmetric term)
        # + penalty * ∫ (avg(Kcap)/h_avg) * jump(pc) * jump(w) dS
        #
        # Note: w is DG test for Sw; we diffuse pc(Sw_new), not Sw directly.
        # This behaves like capillary diffusion in saturation.

        h_avg = fd.avg(h)
        sigma = self.penalty_CIP * (self.Sw_degree + 1)**2

        F_cap = fd.inner(Kcap * fd.grad(pc_new), fd.grad(w)) * fd.dx \
            - fd.dot(fd.avg(Kcap * fd.grad(pc_new)), n('+')) * fd.jump(w) * fd.dS \
            - fd.dot(fd.avg(Kcap * fd.grad(w)), n('+')) * fd.jump(pc_new) * fd.dS \
            + sigma * (fd.avg(Kcap) / h_avg) * fd.jump(pc_new) * fd.jump(w) * fd.dS

        # For boundary diffusion: natural no-flux by default.
        # If you want Dirichlet on pc, add boundary SIPG terms here.

        # ----------------
        # Source term
        # ----------------
        F_src = -q_w * w * fd.dx

        # Full residual
        F = F_mass + (F_adv + F_adv_bdry) + F_cap + F_src
        return F

    # -------------------------
    # Core step routines
    # -------------------------
    def solve_pressure(
        self,
        p: fd.Function,
        Sw: fd.Function,
        q_t: fd.Function,
        Vp: fd.functionspaceimpl.WithGeometry,
        solver_parameters: Optional[dict] = None,
    ) -> fd.Function:
        bc = self.BC_definition_pressure(Vp)
        Fp = self.PDE_definition_pressure(p, Sw, q_t, Vp)

        if solver_parameters is None:
            solver_parameters = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}

        fd.solve(Fp == 0, p, bcs=[bc], solver_parameters=solver_parameters)
        return p

    def compute_total_flux(
        self,
        Ut,
        p: fd.Function,
        Sw: fd.Function,
        Vflux: fd.functionspaceimpl.WithGeometry,
    ):
        """
        u_t = -k λ_t ∇p - k λ_o ∇pc(Sw)
        For 1D, you can store Ut as scalar DG (x-component).
        For 2D/3D, store Ut as vector DG in Vflux.
        """
        lt = self.lam_t(Sw)
        lo = self.lam_o(Sw)
        pc = self.p_c(Sw)
        Ut.project(-self.k_abs * lt * fd.grad(p) - self.k_abs * lo * fd.grad(pc))
        return Ut

    def solve_saturation(
        self,
        Sw_new: fd.Function,
        Sw: fd.Function,
        Ut,
        q_w: fd.Function,
        Vs: fd.functionspaceimpl.WithGeometry,
        solver_parameters: Optional[dict] = None,
    ) -> fd.Function:
        Fs = self.PDE_definition_saturation_semiimplicit(Sw_new, Sw, Ut, q_w, Vs)

        if solver_parameters is None:
            solver_parameters = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-9,
                "snes_atol": 1e-10,
                "snes_max_it": 30,
                "ksp_type": "gmres",
                "pc_type": "ilu",
            }

        fd.solve(Fs == 0, Sw_new, solver_parameters=solver_parameters)

        if self.clamp_S:
            Sw_new.dat.data[:] = Sw_new.dat.data.clip(0.0, 1.0)
        return Sw_new

    # -------------------------
    # Public API
    # -------------------------
    def solve_step(
        self,
        p: fd.Function,
        Sw: fd.Function,
        q_t: fd.Function,
        q_w: fd.Function,
        Vp: fd.functionspaceimpl.WithGeometry,
        Vs: fd.functionspaceimpl.WithGeometry,
        Vflux: fd.functionspaceimpl.WithGeometry,
        Ut=None,
        Sw_new: Optional[fd.Function] = None,
    ) -> Tuple[fd.Function, fd.Function, object]:
        """
        One step:
          1) solve pressure with Sw^n
          2) compute vector total flux u_t
          3) solve saturation Sw^{n+1} with implicit capillarity, explicit advection
        Returns (p, Sw_updated, Ut)
        """
        if Ut is None:
            Ut = fd.Function(Vflux, name="u_t")
        if Sw_new is None:
            Sw_new = fd.Function(Vs, name="Sw_new")

        # 1) pressure
        self.solve_pressure(p, Sw, q_t, Vp)

        # 2) total flux (vector in 2D/3D)
        self.compute_total_flux(Ut, p, Sw, Vflux)

        # 3) saturation
        self.solve_saturation(Sw_new, Sw, Ut, q_w, Vs)
        Sw.assign(Sw_new)

        return p, Sw, Ut

    def solve(
        self,
        q_t_h: List[fd.Function],
        q_w_h: List[fd.Function],
        p0: Optional[fd.Function] = None,
        Sw0: Optional[fd.Function] = None,
        # optional spaces; if None, they will be created using degrees in __init__
        Vp: Optional[fd.functionspaceimpl.WithGeometry] = None,
        Vs: Optional[fd.functionspaceimpl.WithGeometry] = None,
        Vflux: Optional[fd.functionspaceimpl.WithGeometry] = None,
    ) -> Tuple[List[fd.Function], List[fd.Function], List[object]]:
        """
        Time loop returning histories (deep copies):
          p_hist, Sw_hist, Ut_hist
        """
        dim = self.mesh.geometric_dimension()

        if Vp is None:
            Vp = fd.FunctionSpace(self.mesh, "CG", self.p_degree)
        if Vs is None:
            Vs = fd.FunctionSpace(self.mesh, "DG", self.Sw_degree)
        if Vflux is None:
            # Store u_t as:
            #   - scalar DG in 1D (x-flux)
            #   - vector DG in 2D/3D
            if dim == 1:
                Vflux = fd.FunctionSpace(self.mesh, "DG", self.Sw_degree)
            else:
                Vflux = fd.VectorFunctionSpace(self.mesh, "DG", self.Sw_degree)

        p = fd.Function(Vp, name="p")
        Sw = fd.Function(Vs, name="Sw")
        Sw_new = fd.Function(Vs, name="Sw_new")
        Ut = fd.Function(Vflux, name="u_t")

        # ICs
        if p0 is None:
            p0 = fd.Function(Vp).interpolate(fd.Constant(0.0))
        if Sw0 is None:
            Sw0 = fd.Function(Vs).interpolate(fd.Constant(0.05))

        self.IC_definition_pressure(Vp, p0)
        self.IC_definition_saturation(Vs, Sw0)
        p.assign(self.p_ic)
        Sw.assign(self.Sw_ic)

        num_steps = int(self.T / float(self.dt.values()))
        if len(q_t_h) < num_steps:
            raise ValueError(f"q_t_h must have at least {num_steps} entries, got {len(q_t_h)}")
        if len(q_w_h) < num_steps:
            raise ValueError(f"q_w_h must have at least {num_steps} entries, got {len(q_w_h)}")

        p_hist = [copy.deepcopy(p)]
        Sw_hist = [copy.deepcopy(Sw)]
        Ut_hist = []

        for n in range(num_steps - 1):
            self.solve_step(
                p=p,
                Sw=Sw,
                q_t=q_t_h[n + 1],
                q_w=q_w_h[n + 1],
                Vp=Vp,
                Vs=Vs,
                Vflux=Vflux,
                Ut=Ut,
                Sw_new=Sw_new,
            )
            p_hist.append(copy.deepcopy(p))
            Sw_hist.append(copy.deepcopy(Sw))
            Ut_hist.append(copy.deepcopy(Ut))

        return p_hist, Sw_hist, Ut_hist


# -----------------------------
# Minimal usage examples
# -----------------------------
if __name__ == "__main__":
    # ---- Choose ONE mesh ----
    # 1D:
    # mesh = fd.IntervalMesh(400, 1.0)

    # 2D:
    mesh = fd.UnitSquareMesh(64, 64)

    # 3D:
    # mesh = fd.UnitCubeMesh(24, 24, 24)

    dt = 5e-4
    T = 0.05
    num_steps = int(T / dt)

    # Sources
    q_t_h = [fd.Constant(0.0) for _ in range(num_steps)]
    q_w_h = [fd.Constant(0.0) for _ in range(num_steps)]

    # In 2D/3D, you should provide a *single* boundary tag for pressure gauge fixing.
    # For UnitSquareMesh, boundary ids are typically:
    #   1: x=0, 2: x=1, 3: y=0, 4: y=1
    # We'll fix p=0 on x=1 (tag=2). Also set inflow saturation on x=0 (tag=1).
    solver = two_phase_capillary_impes(
        mesh=mesh,
        phi=0.2,
        dt=dt,
        T=T,
        k_abs=1.0,
        mu_w=1.0,
        mu_o=5.0,
        pc0=0.05,
        pc_exp=2.0,
        pc_eps=1e-3,
        penalty_CIP=30.0,
        p_dirichlet_tag=2,          # gauge fix on x=1
        p_dirichlet_value=0.0,
        Sw_inflow_tags={1: 1.0},    # inject water on x=0 when inflow
    )

    p_hist, Sw_hist, Ut_hist = solver.solve(q_t_h=q_t_h, q_w_h=q_w_h)

    # Save final state
    out = fd.File("two_phase_capillary_impes.pvd")
    out.write(p_hist[-1], Sw_hist[-1])