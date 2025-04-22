import numpy as np
import pandas as pd
import pyvista as pv

file = "RESULTS/N_00000123_Agg_00000001.dat"
data = pd.read_csv(file, sep=r"\s+", header=None)
data = data.to_numpy()

position = data[:, :3]
radii = data[:, 3]

position -= np.mean(position, axis=0)
point_cloud = pv.PolyData(position)
point_cloud["radius"] = [2 * i for i in radii]

geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
glyphed = point_cloud.glyph(scale="radius", geom=geom, orient=False)  # type: ignore
pl = pv.Plotter(window_size=[400, 400])
pl.add_mesh(glyphed, color="white", smooth_shading=True, pbr=True)
pl.view_isometric()  # type: ignore
pl.link_views()
pl.show()
