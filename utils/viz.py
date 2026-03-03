from typing import List,Tuple
import matplotlib
import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf
import firedrake as fd
import numpy as np
import matplotlib.animation as animation

def set_initial_frame(fig: matplotlib.figure.Figure, ax: plt.Axes, field: List[fd.function.Function], title: str):
  all_p_data = np.concatenate([p_step.dat.data[:] for p_step in field])
  p_min_global = all_p_data.min()
  p_max_global = all_p_data.max()

  if p_min_global == p_max_global:
      p_min_global -= 1e-9
      p_max_global += 1e-9

  global_levels_p = np.linspace(p_min_global, p_max_global, 51)

  # Initialize the plot for field
  initial_contour_p = tricontourf(field[0], levels=global_levels_p, axes=ax, cmap="inferno")
  ax.set_title(f"{title} at time step 0")
  ax.set_aspect("equal")
  cbar_p = fig.colorbar(initial_contour_p, ax=ax, label=f"{title} Value")
  return global_levels_p

def redraw_frame(current_field: fd.function.Function, global_levels_p: List[np.ndarray[float]], ax: plt.Axes, i: int):
  tricontourf(current_field[i], levels=global_levels_p, axes=ax, cmap="inferno")

def animate_solution(loss: np.ndarray[float], fields: Tuple[List[fd.function.Function]], titles: List[str]):

  fig, axs = plt.subplots(1, int(len(fields)+1), figsize=(14, 6))

  axs[0].plot(loss)
  axs = axs[1:]

  global_levels_p = tuple(set_initial_frame(fig, ax, field, title) for ax, field, title in zip(axs, fields, titles))

  plt.tight_layout() # Adjust layout for both initial plots

  def animate(i):
    map(lambda ax: ax.clear(), axs)

    list(redraw_frame(current_p, global_level_p, ax, i) for current_p, global_level_p, ax in zip(fields, global_levels_p, axs))

    return []

  anim = animation.FuncAnimation(
      fig, animate, frames=len(fields[0]), interval=200, blit=False
  )
  plt.close(fig)
  return anim