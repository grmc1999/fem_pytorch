import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import firedrake as fd
from typing import List
from firedrake.petsc import PETSc

def dataframe_to_firedrake(df: pd.DataFrame, position_cols: List[str] = ['X', 'Y', 'Z'], u_col: str ='u'):
    """
    Create a Firedrake mesh and function from a DataFrame of scattered data.

    The DataFrame must contain columns for the coordinates (x, y, optionally z)
    and a column for the scalar field u. The points are used as mesh vertices,
    and a Delaunay triangulation (2D) or tetrahedralization (3D) is performed
    to create the cells. A piecewise linear continuous function space (CG1) is
    built on this mesh, and the values from `u_col` are assigned to the
    corresponding vertices. The resulting function interpolates the data exactly
    at the input points.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    x_col : str, optional
        Name of the column containing x-coordinates.
    y_col : str, optional
        Name of the column containing y-coordinates.
    z_col : str, optional
        Name of the column containing z-coordinates. If None, 2D is assumed.
    u_col : str, optional
        Name of the column containing the scalar values to assign.

    Returns
    -------
    mesh : firedrake.Mesh
        The mesh built from the input points.
    u : firedrake.Function
        A CG1 function on `mesh` with values set from `u_col`.

    """
    # Ensure required columns exist
    required = position_cols
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Extract coordinates as a numpy array
    points = df[position_cols].values.astype(np.float64)
    dim = len(position_cols)

    # Perform Delaunay triangulation to obtain simplices (cells)
    tri = Delaunay(points)
    cells = tri.simplices

    # Create the Firedrake mesh from vertices and cells
    # The vertex list is exactly the points array (order preserved)
    plex = PETSc.DMPlex().createFromCellList(
        dim,
        cells,
        points,
        comm=fd.COMM_WORLD,
    )
    plex.markBoundaryFaces("on_boundary")

    mesh = fd.Mesh(plex, reorder=False)
    breakpoint()

    # Build a piecewise linear continuous function space
    V = fd.FunctionSpace(mesh, "CG", 1)

    # Create a function and assign the values
    u = fd.Function(V)
    # The vertex values are stored in the data array of the function.
    # Because the mesh vertices correspond one-to-one with DataFrame rows,
    # we can simply copy the values.
    u.dat.data[:] = df[u_col].values.astype(np.float64)

    return mesh, V, u


class extract_info(object):
    def __init__(self,file):
        
        self.filter_trash = lambda df: pd.DataFrame(np.unique(df[list(df.keys())[3:]].values,axis=0), columns=list(df.keys())[3:])

        df_din = pd.DataFrame()

        with pd.HDFStore(file, 'r') as store:
            self.df_f=store['/features']
            self.df_r=store['/Restrictions']

        self.df_f = self.filter_trash(self.df_f)
        self.stat_keys = list( e[0] for e in list(map(lambda c: c.split("_"), list(self.df_f.columns))) if len(e) < 2)
        self.dyna_keys = list( e[0] for e in list(map(lambda c: c.split("_"), list(self.df_f.columns))) if len(e) >= 2)

        






