import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import firedrake as fd
from typing import List
from firedrake.petsc import PETSc

def dataframe_to_firedrake(df: pd.DataFrame, position_cols: List[str] = ['X', 'Y', 'Z'], u_col: str ='u'):
    """
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

    plex = PETSc.DMPlex().createFromCellList(
        dim,
        cells,
        points,
        comm=fd.COMM_WORLD,
    )
    plex.markBoundaryFaces("on_boundary")

    mesh = fd.Mesh(plex, reorder=False)

    V = fd.FunctionSpace(mesh, "CG", 1)

    u = fd.Function(V)
    u.dat.data[:] = df[u_col].values.astype(np.float64)

    return mesh, V, u


class extract_info(object):
    def __init__(self,file):
        
        self.filter_trash = lambda df: pd.DataFrame(np.unique(df[list(df.keys())[3:]].values,axis=0), columns=list(df.keys())[3:])

        self.read_h5(file)

        self.stat_keys = list( e[0] for e in list(map(lambda c: c.split("_"), list(self.df_f.columns))) if len(e) < 2)
        self.dyna_keys = list( e[0] for e in list(map(lambda c: c.split("_"), list(self.df_f.columns))) if len(e) >= 2)


    def read_h5(self,file):
        with pd.HDFStore(file, 'r') as store:
            self.df_f=store['/features']
            self.df_r=store['/Restrictions']

        self.df_f = self.filter_trash(self.df_f)

    def create_functional_space(self,position_cols: List[str] = ['X', 'Y', 'Z']):

        # Ensure required columns exist
        self.position_cols = position_cols
        required = position_cols
        for col in required:
            if col not in self.df_f.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Extract coordinates as a numpy array
        points = self.df_f[position_cols].values.astype(np.float64)
        dim = len(position_cols)

        # Perform Delaunay triangulation to obtain simplices (cells)
        self.tri = Delaunay(points)
        self.cells = self.tri.simplices

        plex = PETSc.DMPlex().createFromCellList(
            dim,
            self.cells,
            points,
            comm=fd.COMM_WORLD,
        )
        plex.markBoundaryFaces("on_boundary")

        self.mesh = fd.Mesh(plex, reorder=True)
        
        self.V = fd.FunctionSpace(self.mesh, "CG", 1)

    def set_space_values(self, u_col: str ='u', V: fd.functionspaceimpl.WithGeometry = None):
        u = fd.Function(self.V)
        u.dat.data[:] = self.df_f[u_col].values.astype(np.float64)
        u_f = fd.Function(V)
        u_f.project(u)
        return u_f
