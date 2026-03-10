from data.dataloader import extract_info
from data.dataloader import dataframe_to_firedrake
import os
import random
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc

if __name__ == '__main__':
    root = (os.path.join("data","2phase"))
    ds_list = os.listdir(os.path.join("data","2phase"))
    ds = random.choice(ds_list)
    EI = extract_info(os.path.join(root,ds)) 

    #u, V, mesh = dataframe_to_firedrake(EI.df_f,['X','Y','Z'],'FR')
    u, V, mesh = dataframe_to_firedrake(EI.df_f,['GridCentroidX','GridCentroidY'],'FR')
