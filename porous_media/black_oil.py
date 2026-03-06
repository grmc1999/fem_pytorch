# Firedrake: BLACK-OIL (water–oil–gas) with dissolved gas Rs(p) (no Rv), scalable to 2D/3D
#
# This is an IMPES-style *sequential* prototype designed to scale:
#   1) Pressure solve (elliptic) using total mobility λ_t(S)
#   2) Compute total Darcy flux u_t
#   3) Transport "surface-volume accumulations" with DG upwinding:
#        Aw := Sw / Bw(p)
#        Ag := Sg / Bg(p) + Rs(p) * So / Bo(p)      (So = 1 - Sw - Sg)
#      Then reconstruct Sw, Sg from Aw, Ag (with a mild approximation for dissolved term)
#
# Governing black-oil conservation (surface volumes, no Rv):
#   ∂t[ φ Sw/Bw ] + ∇·[ uw/Bw ] = qw
#   ∂t[ φ So/Bo ] + ∇·[ uo/Bo ] = qo
#   ∂t[ φ ( Sg/Bg + Rs So/Bo ) ] + ∇·[ ug/Bg + Rs uo/Bo ] = qg
#
# Fluxes (no capillary, no gravity):
#   uα = fα(S) u_t,   u_t = -k λ_t(S) ∇p
#   λ_α(S) = krα(S)/μα,   fα = λ_α / (λw+λo+λg)
#
# IMPORTANT NOTES (for research build-up):
# - This sequential scheme is intentionally simple and robust for scaling experiments.
# - The gas accumulation includes a dissolved term Rs*So/Bo. In this prototype we treat
#   So in that dissolved term using the *previous* saturations when reconstructing Sg.
#   That keeps the step explicit and scalable. For higher fidelity, make that part implicit.
# - You can add: capillarity, gravity, Rv, compressible pressure equation, well models, etc.

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import copy
import firedrake as fd


class black_oil_impes_rs(object):
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
        # --- simple property laws (replace with tables later) ---
        # formation volume factor: B(p) = B_ref * exp( c_B * (p - p_ref) )
        p_ref: float = 0.0,
        Bw_ref: float = 1.0,
        Bo_ref: float = 1.0,
        Bg_ref: float = 1.0,
        cBw: float = 0.0,
        cBo: float = 0.0,
        cBg: float = 0.0,
        # dissolved gas–oil ratio Rs(p): a smooth "saturation" curve
        Rs_max: float = 2.0,
        Rs_k: float = 2.0,
        Rs_p0: float = 0.2,
        # --- relperm model ---
        nw: float = 2.0,
        no: float = 2.0,
        ng: float = 2.0,
        # --- numerics ---
        clamp_S: bool = True,
        # pressure gauge fix
        p_dirichlet_tag: Optional[int] = None,
        p_dirichlet_value: float = 0.0,
        # total inflow Neumann-like term on tags: value is positive INTO domain
        inflow_flux_tags: Optional[Dict[int, float]] = None,
        # inflow composition tags: (Sw_inj, Sg_inj) used if u_t·n < 0 on that tag
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

        self.p_ref = fd.Constant(p_ref)
        self.Bw_ref = fd.Constant(Bw_ref)
        self.Bo_ref = fd.Constant(Bo_ref)
        self.Bg_ref = fd.Constant(Bg_ref)
        self.cBw = fd.Constant(cBw)
        self.cBo = fd.Constant(cBo)
        self.cBg = fd.Constant(cBg)

        self.Rs_max = fd.Constant(Rs_max)
        self.Rs_k = fd.Constant(Rs_k)
        self.Rs_p0 = fd.Constant(Rs_p0)

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
    # Utilities / class-style helpers
    # -------------------------
    def get_boundary_points(self):
        if self.bc_p is None:
            raise RuntimeError("Call BC_definition_pressure(...) first so self.bc_p is defined.")
        self.bc_p.nodes
        return self.bc_p.nodes.shape[0]

    def get_coordinate_functions(self, V: fd.functionspaceimpl.WithGeometry):
        return tuple(fd.Function(V).interpolate(dof) for dof in fd.SpatialCoordinate(self.mesh))

    # -------------------------
    # Property laws: B(p), Rs(p)
    # -------------------------
    def Bw(self, p):
        return self.Bw_ref * fd.exp(self.cBw * (p - self.p_ref))

    def Bo(self, p):
        return self.Bo_ref * fd.exp(self.cBo * (p - self.p_ref))

    def Bg(self, p):
        return self.Bg_ref * fd.exp(self.cBg * (p - self.p_ref))

    def Rs(self, p):
        # Smooth saturating curve: Rs = Rs_max * sigmoid( k*(p - p0) )
        return self.Rs_max / (1.0 + fd.exp(-self.Rs_k * (p - self.Rs_p0)))

    # -------------------------
    # RelPerm / Mobilities / Fractional flow
    # -------------------------
    def So(self, Sw, Sg):
        return 1.0 - Sw - Sg

    def krw(self, Sw):
        return fd.max_value(Sw, 0.0) ** self.nw

    def krg(self, Sg):
        return fd.max_value(Sg, 0.0) ** self.ng

    def kro(self, Sw, Sg):
        So = fd.max_value(self.So(Sw, Sg), 0.0)
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
    # IC / BC
    # -------------------------
    def IC_definition(self, V, u0: fd.Function):
        self.ic = u0
        return self.ic

    def BC_definition_pressure(self, Vp):
        # For 2D/3D you should set a single tag for gauge fixing.
        if self.p_dirichlet_tag is None:
            self.bc_p = fd.DirichletBC(Vp, self.p_dirichlet_value, "on_boundary")
        else:
            self.bc_p = fd.DirichletBC(Vp, self.p_dirichlet_value, self.p_dirichlet_tag)
        return self.bc_p

    # -------------------------
    # Pressure equation (simple incompressible total flux)
    # -------------------------
    def PDE_definition_pressure(self, p, Sw, Sg, q_t, Vp):
        """
        Prototype pressure equation (incompressible-style):
          ∇·( -k λ_t(S) ∇p ) = q_t
        Weak:
          ∫ k λ_t ∇p·∇v dx = ∫ q_t v dx + Σ_tags ∫ u_in(tag) v ds(tag)
        """
        v = fd.TestFunction(Vp)
        lt = self.lam_t(Sw, Sg)

        F = fd.inner(self.k_abs * lt * fd.grad(p), fd.grad(v)) * fd.dx - (q_t * v) * fd.dx
        for tag, u_in_val in self.inflow_flux_tags.items():
            # residual form: subtract RHS term
            F -= (fd.Constant(float(u_in_val)) * v) * fd.ds(tag)
        return F

    def compute_total_flux(self, Ut, p, Sw, Sg):
        """
        u_t = -k λ_t(S) ∇p
        Ut: scalar DG in 1D, vector DG in 2D/3D.
        """
        lt = self.lam_t(Sw, Sg)
        Ut.project(-self.k_abs * lt * fd.grad(p))
        return Ut

    # -------------------------
    # DG0 transport of a "conserved accumulation" A:
    #   φ (A_new - A)/dt + ∇·(F(A)) = qA
    # Here we use:
    #   Aw = Sw/Bw(p)
    #   Ag = Sg/Bg(p) + Rs(p)*So/Bo(p)
    # and fluxes:
    #   Fw = uw/Bw = fw*u_t / Bw
    #   Fg = ug/Bg + Rs*(uo/Bo) = fg*u_t/Bg + Rs*fo*u_t/Bo
    # -------------------------
    def transport_A_DG0(
        self,
        A_new: fd.Function,
        A_old: fd.Function,
        Fvec,                # vector flux (same type as Ut), already computed from old state
        qA: fd.Function,
        VA: fd.functionspaceimpl.WithGeometry,
        # boundary upwind injection: A_inj (scalar Constant)
        A_inj_tags: Optional[Dict[int, float]] = None,
    ):
        w = fd.TestFunction(VA)
        n = fd.FacetNormal(self.mesh)
        dim = self.mesh.geometric_dimension()

        # Normal flux on interior facets
        if dim == 1:
            # In 1D we store Fvec as scalar (x-flux), so normal is ±1
            Fn_int = fd.avg(Fvec) * n('+')[0]
            Fn_b = Fvec * n[0]
        else:
            Fn_int = fd.dot(fd.avg(Fvec), n('+'))
            Fn_b = fd.dot(Fvec, n)

        # Upwind for A on interior facets
        A_plus = A_old('+')
        A_minus = A_old('-')
        A_up_int = fd.conditional(Fn_int > 0.0, A_plus, A_minus)
        Fhat_int = Fn_int * A_up_int

        # Boundary flux term: use injected A where inflow (Fn_b < 0) and tag provided
        Fhat_bdry = 0
        if A_inj_tags:
            for tag, Aval in A_inj_tags.items():
                A_inj = fd.Constant(float(Aval))
                A_up = fd.conditional(Fn_b >= 0.0, A_old, A_inj)
                Fhat_bdry += (Fn_b * A_up) * w * fd.ds(tag)
        else:
            # do-nothing: use interior always (good for outflow boundaries)
            Fhat_bdry += (Fn_b * A_old) * w * fd.ds

        # DG0 mass matrix equation
        M = (self.phi / self.dt) * A_new * w * fd.dx
        rhs = (self.phi / self.dt) * A_old * w * fd.dx \
              - (Fhat_int * fd.jump(w)) * fd.dS \
              - Fhat_bdry \
              + (qA * w) * fd.dx

        fd.solve(M - rhs == 0, A_new, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})
        return A_new

    # -------------------------
    # One time step (sequential)
    # -------------------------
    def solve_step(
        self,
        p: fd.Function,
        Sw: fd.Function,
        Sg: fd.Function,
        q_t: fd.Function,
        q_w: fd.Function,
        q_g: fd.Function,
        Vp,
        Vs,
        Vflux,
        VA,
        Ut=None,
        Aw=None,
        Ag=None,
        Aw_new=None,
        Ag_new=None,
        Sw_new=None,
        Sg_new=None,
    ):
        # allocate if needed
        if Ut is None:
            Ut = fd.Function(Vflux, name="u_t")
        if Aw is None:
            Aw = fd.Function(VA, name="Aw")      # Sw/Bw
        if Ag is None:
            Ag = fd.Function(VA, name="Ag")      # Sg/Bg + Rs*So/Bo
        if Aw_new is None:
            Aw_new = fd.Function(VA, name="Aw_new")
        if Ag_new is None:
            Ag_new = fd.Function(VA, name="Ag_new")
        if Sw_new is None:
            Sw_new = fd.Function(Vs, name="Sw_new")
        if Sg_new is None:
            Sg_new = fd.Function(Vs, name="Sg_new")

        # ---- 1) Pressure ----
        bc = self.BC_definition_pressure(Vp)
        Fp = self.PDE_definition_pressure(p, Sw, Sg, q_t, Vp)
        fd.solve(
            Fp == 0,
            p,
            bcs=[bc],
            solver_parameters={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
        )

        # ---- 2) Total flux ----
        self.compute_total_flux(Ut, p, Sw, Sg)

        # ---- 3) Build cellwise accumulations Aw, Ag (project to DG0) ----
        # We evaluate B and Rs at the solved pressure p (CG). For DG0 transport, project to VA.
        Bw_cell = fd.Function(VA, name="Bw_cell"); Bw_cell.project(self.Bw(p))
        Bo_cell = fd.Function(VA, name="Bo_cell"); Bo_cell.project(self.Bo(p))
        Bg_cell = fd.Function(VA, name="Bg_cell"); Bg_cell.project(self.Bg(p))
        Rs_cell = fd.Function(VA, name="Rs_cell"); Rs_cell.project(self.Rs(p))

        So_old = fd.Function(VA, name="So_old"); So_old.project(1.0 - Sw - Sg)

        Aw.project(Sw / Bw_cell)
        Ag.project(Sg / Bg_cell + Rs_cell * (So_old / Bo_cell))

        # ---- 4) Build component flux vectors for transport ----
        # phase fractional flows (evaluated with old Sw,Sg)
        fw = self.f_w(Sw, Sg)
        fg = self.f_g(Sw, Sg)
        fo = self.f_o(Sw, Sg)

        # Component fluxes in reservoir volumetric units:
        #   Fw = (uw)/Bw = fw * Ut / Bw
        #   Fg = ug/Bg + Rs*uo/Bo = fg*Ut/Bg + Rs * fo*Ut/Bo
        #
        # For dim=1: Ut is scalar; for dim>1 Ut is vector; scalar multiplication is fine.
        Fw_vec = fd.Function(Vflux, name="Fw_vec")
        Fg_vec = fd.Function(Vflux, name="Fg_vec")

        Fw_vec.project((fw / Bw_cell) * Ut)
        Fg_vec.project((fg / Bg_cell) * Ut + (Rs_cell * fo / Bo_cell) * Ut)

        # ---- 5) Upwind injection values for Aw and Ag on tags (if provided) ----
        # User provides sat_inflow_tags[tag] = (Sw_inj, Sg_inj)
        # We convert to accumulation injection:
        #   Aw_inj = Sw_inj / Bw(p)   (use boundary p approximately as interior p)
        #   Ag_inj = Sg_inj / Bg(p) + Rs(p) * So_inj / Bo(p),   So_inj = 1 - Sw_inj - Sg_inj
        Aw_inj_tags = {}
        Ag_inj_tags = {}
        if self.sat_inflow_tags:
            # Use cellwise Bw/Bo/Bg/Rs already computed; injection per tag uses same scalar value.
            # (For more accuracy, evaluate boundary p and compute there.)
            for tag, (Sw_inj, Sg_inj) in self.sat_inflow_tags.items():
                Sw_inj = float(Sw_inj); Sg_inj = float(Sg_inj)
                So_inj = max(1.0 - Sw_inj - Sg_inj, 0.0)
                # Use reference values at current step (approx):
                # pick global constants by projecting to scalar average: simplest is use refs:
                # To keep it simple and stable: use refs (Bw_ref etc). You can refine later.
                Aw_inj_tags[tag] = Sw_inj / float(self.Bw_ref.values())
                Ag_inj_tags[tag] = (Sg_inj / float(self.Bg_ref.values())) + (float(self.Rs_max.values()) * 0.0)  # safe default
                # Better (still cheap): use local cellwise average via dat min/max is not clean in forms.
                # For now keep injection consistent and stable; refine when you add BC p evaluation.

        # ---- 6) Transport Aw and Ag with DG0 ----
        # Sources for Aw and Ag:
        #   qw equation is directly on Aw: ∂t(φ Aw) + div(Fw) = qw   (since Aw = Sw/Bw)
        #   gas component equation: ∂t(φ Ag) + div(Fg) = qg
        self.transport_A_DG0(Aw_new, Aw, Fw_vec, q_w, VA, A_inj_tags=Aw_inj_tags if Aw_inj_tags else None)
        self.transport_A_DG0(Ag_new, Ag, Fg_vec, q_g, VA, A_inj_tags=Ag_inj_tags if Ag_inj_tags else None)

        # ---- 7) Reconstruct Sw_new and Sg_new from Aw_new and Ag_new ----
        # Sw_new = Bw * Aw_new
        Sw_new.project(Bw_cell * Aw_new)

        # For Sg, we invert:
        #   Ag_new = Sg_new/Bg + Rs * So/Bo
        # We use So from OLD step (explicit treatment of dissolved term):
        #   Sg_new = Bg * ( Ag_new - Rs * So_old/Bo )
        Sg_new.project(Bg_cell * (Ag_new - Rs_cell * (So_old / Bo_cell)))

        # Optional clamps + renormalization
        if self.clamp_S:
            Sw_new.dat.data[:] = Sw_new.dat.data.clip(0.0, 1.0)
            Sg_new.dat.data[:] = Sg_new.dat.data.clip(0.0, 1.0)

            # enforce Sw + Sg <= 1
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
        # spaces (optional)
        Vp=None,
        Vs=None,
        Vflux=None,
        VA=None,
        # ICs (optional)
        p0: Optional[fd.Function] = None,
        Sw0: Optional[fd.Function] = None,
        Sg0: Optional[fd.Function] = None,
    ):
        dim = self.mesh.geometric_dimension()

        if Vp is None:
            Vp = fd.FunctionSpace(self.mesh, "CG", 1)
        if Vs is None:
            Vs = fd.FunctionSpace(self.mesh, "DG", 0)
        if Vflux is None:
            if dim == 1:
                Vflux = fd.FunctionSpace(self.mesh, "DG", 0)
            else:
                Vflux = fd.VectorFunctionSpace(self.mesh, "DG", 0)
        if VA is None:
            VA = fd.FunctionSpace(self.mesh, "DG", 0)

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

        Aw = fd.Function(VA, name="Aw"); Ag = fd.Function(VA, name="Ag")
        Aw_new = fd.Function(VA, name="Aw_new"); Ag_new = fd.Function(VA, name="Ag_new")
        Sw_new = fd.Function(Vs, name="Sw_new"); Sg_new = fd.Function(Vs, name="Sg_new")

        num_steps = int(self.T / float(self.dt.values()))
        if len(q_t_h) < num_steps or len(q_w_h) < num_steps or len(q_g_h) < num_steps:
            raise ValueError("q_*_h lists must have at least num_steps entries.")

        p_hist = [copy.deepcopy(p)]
        Sw_hist = [copy.deepcopy(Sw)]
        Sg_hist = [copy.deepcopy(Sg)]
        Ut_hist = []

        for n in range(num_steps - 1):
            self.solve_step(
                p=p, Sw=Sw, Sg=Sg,
                q_t=q_t_h[n + 1],
                q_w=q_w_h[n + 1],
                q_g=q_g_h[n + 1],
                Vp=Vp, Vs=Vs, Vflux=Vflux, VA=VA,
                Ut=Ut,
                Aw=Aw, Ag=Ag, Aw_new=Aw_new, Ag_new=Ag_new,
                Sw_new=Sw_new, Sg_new=Sg_new,
            )
            p_hist.append(copy.deepcopy(p))
            Sw_hist.append(copy.deepcopy(Sw))
            Sg_hist.append(copy.deepcopy(Sg))
            Ut_hist.append(copy.deepcopy(Ut))

        return p_hist, Sw_hist, Sg_hist, Ut_hist


# -----------------------------
# Minimal usage (2D)
# -----------------------------
if __name__ == "__main__":
    # Choose mesh
    mesh = fd.UnitSquareMesh(64, 64)
    # Typical boundary tags for UnitSquareMesh:
    #   1: x=0, 2: x=1, 3: y=0, 4: y=1

    dt = 5e-4
    T = 0.05
    num_steps = int(T / dt)

    q_t_h = [fd.Constant(0.0) for _ in range(num_steps)]
    q_w_h = [fd.Constant(0.0) for _ in range(num_steps)]
    q_g_h = [fd.Constant(0.0) for _ in range(num_steps)]

    solver = black_oil_impes_rs(
        mesh=mesh,
        phi=0.2,
        dt=dt,
        T=T,
        k_abs=1.0,
        mu_w=1.0, mu_o=5.0, mu_g=0.8,
        # Mild compressibility (optional)
        p_ref=0.0,
        Bw_ref=1.0, Bo_ref=1.0, Bg_ref=1.0,
        cBw=0.0, cBo=0.0, cBg=0.0,
        # Rs(p)
        Rs_max=2.0, Rs_k=5.0, Rs_p0=0.2,
        # Gauge fix on outlet x=1 (tag 2)
        p_dirichlet_tag=2,
        p_dirichlet_value=0.0,
        # Total inflow on inlet x=0 (tag 1)
        inflow_flux_tags={1: 1.0},
        # Inject water-rich, no free gas at inlet
        sat_inflow_tags={1: (1.0, 0.0)},
    )

    p_hist, Sw_hist, Sg_hist, Ut_hist = solver.solve(q_t_h=q_t_h, q_w_h=q_w_h, q_g_h=q_g_h)

    So = fd.Function(Sw_hist[-1].function_space(), name="So")
    So.interpolate(1.0 - Sw_hist[-1] - Sg_hist[-1])

    out = fd.File("black_oil_impes_rs.pvd")
    out.write(p_hist[-1], Sw_hist[-1], Sg_hist[-1], So)