
import firedrake as fd
import matplotlib.pyplot as plt
L = 1.0                 # length of domain
nx = 100                # number of cells
mesh = fd.IntervalMesh(nx, L)   # 1D mesh from 0 to L

V = fd.FunctionSpace(mesh, "CG", 1)

# ---------------------------
# 2. Choose desired potential φ_desired
#    Example: φ_desired = sin(π x)   (satisfies φ=0 at both ends)
# ---------------------------
x = fd.SpatialCoordinate(mesh)
phi_desired_expr = fd.sin(fd.pi * x[0])   # x[0] is the x-coordinate in 1D

# Interpolate into a Function
phi_desired = fd.Function(V).interpolate(phi_desired_expr)

# ---------------------------
# 3. Compute charge density ρ = -ε0 * d²φ_desired/dx²
#    Use variational forms to compute the Laplacian.
# ---------------------------
epsilon0 = 1.0   # We can set ε0 = 1 for simplicity (normalised units)

# Test function
v = TestFunction(V)

# Compute the weak Laplacian of phi_desired: ∫ (∇φ_desired·∇v) dx
# Because ∫ (∇φ_desired·∇v) dx = -∫ (∇²φ_desired) v dx + boundary terms.
# We'll ignore boundary terms here (they are zero for our BCs, but we'll double-check).
a_laplace = fd.inner(fd.grad(phi_desired), fd.grad(v)) * fd.dx
L_laplace = fd.inner(fd.Constant(0.0), v) * fd.dx   # dummy rhs, we'll assemble the bilinear form

# Assemble the vector representing -∫ (∇²φ_desired) v dx = ∫ (∇φ_desired·∇v) dx
# This vector is actually the RHS of a Poisson problem with φ_desired as source.
# But we want ρ itself: ρ = -ε0 ∇²φ_desired.
# We can obtain a Function rho by solving a mass matrix problem:
# ∫ ρ v dx = -ε0 ∫ (∇²φ_desired) v dx = ε0 ∫ (∇φ_desired·∇v) dx (after integrating by parts and dropping boundary terms)
# So ρ satisfies: ∫ ρ v dx = ε0 * a_laplace(v)

# Assemble the RHS vector b where b[v] = ε0 * a_laplace
b = fd.assemble(epsilon0 * a_laplace)

# Solve for ρ using the mass matrix (i.e., L2 projection)
rho = fd.Function(V)
mass_matrix = fd.assemble(v * v * fd.dx)   # mass matrix is diagonal for CG1? Actually it's not diagonal, we need to solve.
# We'll solve using a linear solver. For CG1, mass matrix is well-conditioned.
solve(mass_matrix, rho, b, solver_parameters={'ksp_type': 'cg', 'pc_type': 'jacobi'})

# Alternatively, we could use the projection function:
# rho = project(epsilon0 * div(grad(phi_desired)), V)   # but div(grad) is not directly available; we need to compute Laplacian.
# The approach above is more explicit.

# For verification, we can compute the Laplacian directly via UFL (if desired):
# laplace_phi = div(grad(phi_desired))
# rho_expr = -epsilon0 * laplace_phi
# rho = Function(V).interpolate(rho_expr)   # This would be a direct interpolation, but UFL can handle div(grad) on a Function.

# However, the projection method is safer because it accounts for the weak form.

# ---------------------------
# 4. Solve Poisson: -∇²φ = ρ/ε0
# ---------------------------
phi = Function(V)

# Weak form: ∫ ∇φ·∇v dx = ∫ (ρ/ε0) v dx
# with Dirichlet BCs φ = φ_desired on the boundary (since we want to recover the same potential).
# The boundary conditions are already given by φ_desired at x=0 and x=L.
bc = DirichletBC(V, phi_desired, "on_boundary")

# RHS
v = TestFunction(V)
F = (inner(grad(phi), grad(v)) - (rho/epsilon0) * v) * dx

# Solve
solve(F == 0, phi, bcs=bc, solver_parameters={'ksp_type': 'cg', 'pc_type': 'gamg'})

# ---------------------------
# 5. Check error
# ---------------------------
error = assemble((phi - phi_desired)**2 * dx)
print(f"L2 error between computed and desired phi: {error:.2e}")

# Plot results
x_vals = mesh.coordinates.dat.data[:]
phi_vals = phi.dat.data[:]
phi_desired_vals = phi_desired.dat.data[:]
rho_vals = rho.dat.data[:]

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(x_vals, phi_desired_vals, 'k--', label='Desired φ')
plt.plot(x_vals, phi_vals, 'r-', label='Computed φ')
plt.xlabel('x'); plt.ylabel('φ'); plt.legend(); plt.grid(True)

plt.subplot(1,3,2)
plt.plot(x_vals, rho_vals, 'b-')
plt.xlabel('x'); plt.ylabel('ρ / ε0'); plt.grid(True)

plt.subplot(1,3,3)
plt.plot(x_vals, phi_vals - phi_desired_vals, 'g-')
plt.xlabel('x'); plt.ylabel('Error'); plt.grid(True)
plt.tight_layout()
plt.show()