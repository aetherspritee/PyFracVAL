"""Plotting functions using PyVista."""

import numpy as np
import pyvista as pv


def plot_particles(positions: np.ndarray, radii: np.ndarray) -> pv.Plotter:
    """Generate a PyVista plot of spheres representing particles.

    Creates a 3D visualization using sphere glyphs scaled by particle radii.

    Parameters
    ----------
    positions : np.ndarray
        Nx3 array of particle center coordinates.
    radii : np.ndarray
        N array of particle radii.

    Returns
    -------
    pyvista.Plotter
        A PyVista Plotter object configured with the particle mesh.
        Call `.show()` on the returned object to display interactively.
        Returns an empty Plotter if input `positions` is empty.
    """
    if positions.shape[0] == 0:
        print("Warning: Cannot plot empty particle data.")
        return pv.Plotter()  # Return empty plotter

    point_cloud = pv.PolyData(positions)
    # Use actual radii for scaling glyphs
    point_cloud["radius"] = 2 * radii

    # Create a sphere glyph primitive
    # Adjust resolution for performance vs quality
    geom = pv.Sphere(theta_resolution=16, phi_resolution=16)

    # Apply glyph filter
    # orient=False is usually fine if spheres are uniform color
    # tolerance can sometimes help with glyph placement issues if centers are very close
    glyphed = point_cloud.glyph(scale="radius", geom=geom, orient=False, tolerance=0.0)

    # Setup plotter
    pl = pv.Plotter(window_size=[800, 800])
    pl.add_mesh(
        glyphed,
        color="lightblue",  # Or choose another color
        smooth_shading=True,
        pbr=True,  # Use physically based rendering for better appearance
        metallic=0.3,
        roughness=0.5,
    )
    pl.view_isometric()
    pl.enable_anti_aliasing("fxaa")  # Nicer visuals
    # pl.link_views() # Usually not needed for single view
    pl.background_color = "white"  # Or black, grey etc.

    return pl
