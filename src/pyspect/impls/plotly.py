"""
Plotly-based plotting utilities and an extensible plotting interface for pyspect.

This module provides:
- update_theme: Configures a Plotly figure with light/dark 2D/3D themes, common font,
  axis styles, and background colors. The function mutates the provided figure
  (template, paper/plot background) and supplies a template_layout for consistent
  styling.
- with_figure: A decorator that ensures a Plotly figure is available, normalizes
  keyword arguments, and applies theming and layout updates. Depending on dim:
  - dim=None: no axis remapping
  - dim=2: xaxis_* and yaxis_* kwds map to layout_xaxis_* / layout_yaxis_*
  - dim=3: xaxis_*, yaxis_*, zaxis_* map to layout_scene_*; camera_* maps to
    layout_scene_camera_*
  It also collects theme_* and layout_* prefixes to feed update_theme and figure layout.

The core interface is PlotlyImpl[R] (subclass of AxesImpl), which offers:
- A general dispatcher PlotlyImpl.plot(..., method="name") that calls either
  self.plot_name or self.name on provided inputs.
- Three transformation hooks to implement in subclasses:
  - transform_to_bitmap(inp, axes) -> 2D boolean array
  - transform_to_surface(inp, axes) -> 2D float array
  - transform_to_isosurface(inp, axes) -> 3D float array

Ready-to-use plotting methods (decorated with with_figure) build meshes over axis
bounds (min/max) via numpy.meshgrid (indexing='ij'), label axes using axis_name/unit,
and add corresponding Plotly traces:
- plot_bitmap: 2D Heatmap of a boolean mask (True mapped to `value` in [zmin, zmax]).
- plot_contour: 2D contour lines from a 2D scalar field.
- plot_surface: 3D surface (z from 2D scalar field) with scene aspectmode='cube'.
- plot_isosurface: 3D isosurface at a given level from a 3D volume; caps hidden and
  surface_count=1 by default; scene aspectmode='cube'.

The nested PlotlyImpl.PLOT namespace provides spherical-to-cartesian helpers and a
set of precomputed camera.eye presets (e.g., EYE_HI_NE, EYE_MH_S, EYE_ZENITH, etc.)
to quickly position 3D views.

PlotlyImpl is designed for easy extension, adding plotting functionality to
implementations that inherit from it.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure

from .axes import Axis, AxesImpl
from ..utils import *

__all__ = [
    'PlotlyImpl',
    'with_figure',
]


# ---------- Theming & figure setup ----------

# TODO: Agent, update for argument `aspectratio`
def update_theme(name: Optional[str] = None, *,
                 aspectratio: str = "4:3",
                 fig: BaseFigure) -> Dict[str, Any]:
    """Apply layout for 2D/3D light/dark themes."""
    
    layout = dict(margin=dict(l=60, r=20, t=40, b=60))

    if name is not None:

        # Font
        font = dict(family="Roboto, Arial, sans-serif", size=14)
        layout.update(font=font)

        # Dimensions
        axes = dict(linewidth=2)
        if name[-2:] not in ("2D", "3D"):
            name += "2D"
        if name.endswith("2D"):
            layout.update(xaxis=axes, yaxis=axes)
        if name.endswith("3D"):
            layout.update(scene=dict(xaxis=axes, yaxis=axes, zaxis=axes))

        # Color Theme
        if name.startswith("Light"):
            fig.update_layout(template="plotly_white")
            layout.update(paper_bgcolor="rgba(255, 255, 255, 1)",
                          plot_bgcolor="rgba(250, 250, 250, 1)")
            font.update(color="black")
            axes.update(linecolor="rgba(0, 0, 0, 0.3)",
                        gridcolor="rgba(0, 0, 0, 0.1)",
                        zerolinecolor="rgba(0, 0, 0, 0.3)")
        if name.startswith("Dark"):
            fig.update_layout(template="plotly_dark")
            layout.update(paper_bgcolor="rgba(26, 28, 36, 1)",
                          plot_bgcolor="rgba(26, 28, 36, 1)")
            font.update(color="white")
            axes.update(linecolor="rgba(255, 255, 255, 0.3)",
                        gridcolor="rgba(255, 255, 255, 0.1)",
                        zerolinecolor="rgba(255, 255, 255, 0.3)")

    fig.update_layout(template_layout=layout)

def with_figure(f: Optional[Callable] = None, *, dim: Optional[int] = None):
    """Decorator to handle figure creation and theming for plotting methods.
    Args:
        dim: Dimensionality of the plot (2 or 3).
    """

    if f is not None and not callable(f):
        raise ValueError("with_figure decorator only accepts keyword arguments")

    if dim not in (None, 2, 3):
        raise ValueError("with_figure decorator expects dim=2 or dim=3")
    
    def decorator(f: Callable[..., BaseFigure]):
        @wraps(f)
        def wrapper(self, *args, **kwds) -> BaseFigure:
            
            flatten(kwds, inplace=True)

            # Create new figure if not provided
            if "fig" not in kwds:
                kwds["fig"] = go.Figure()

            # Merge common options into layout
            match dim:
                case None:
                    pass
                case 2:
                    for k, v in collect_prefix(kwds, "xaxis_", remove=True).items():
                        kwds.setdefault(f"layout_xaxis_{k}", v)
                    for k, v in collect_prefix(kwds, "yaxis_", remove=True).items():
                        kwds.setdefault(f"layout_yaxis_{k}", v)
                case 3:
                    for k, v in collect_prefix(kwds, "xaxis_", remove=True).items():
                        kwds.setdefault(f"layout_scene_xaxis_{k}", v)
                    for k, v in collect_prefix(kwds, "yaxis_", remove=True).items():
                        kwds.setdefault(f"layout_scene_yaxis_{k}", v)
                    for k, v in collect_prefix(kwds, "zaxis_", remove=True).items():
                        kwds.setdefault(f"layout_scene_zaxis_{k}", v)
                    for k, v in collect_prefix(kwds, "camera_", remove=True).items():
                        kwds.setdefault(f"layout_scene_camera_{k}", v)

            # Collect theme and layout options
            theme_args = collect_keys(kwds, "theme").values()
            theme_kwds = collect_prefix(kwds, "theme_", remove=True)
            layout = collect_prefix(kwds, "layout_", remove=True)

            # Call the decorated function
            fig = f(self, *args, **kwds)

            # Apply theme and layout
            update_theme(*theme_args, **theme_kwds, fig=fig)
            fig.update_layout(**layout)

            return fig
        return wrapper
    
    return decorator if f is None else decorator(f)


# ---------- Example plotting implementation ----------

Axes2D = tuple[Axis, Axis]
Axes3D = tuple[Axis, Axis, Axis]

class PlotlyImpl[R](AxesImpl):
    """
    Example plotting interface.
    Integrate these methods where pyspect emits data/sets/meshes
    and call with either an existing `fig=` or let it create one.
    """

    class PLOT:

        @staticmethod
        def sph_to_cart(r, theta, phi):
            """Spherical (deg) → cartesian dict compatible with Plotly camera.eye."""
            th = np.deg2rad(theta)
            ph = np.deg2rad(phi)
            s = np.sin(th)
            return dict(
                x=r * s * np.cos(ph),
                y=r * s * np.sin(ph),
                z=r * np.cos(th),
            )

        # Layer 1: Higher elevation (closer to the zenith)
        EYE_HI_W    = sph_to_cart(2.2, 20, -180)  # West, high up
        EYE_HI_SW   = sph_to_cart(2.5, 30, -135)  # Southwest, high up
        EYE_HI_S    = sph_to_cart(2.5, 20, -90)   # South, high up
        EYE_HI_SE   = sph_to_cart(2.5, 30, -45)   # Southeast, high up
        EYE_HI_E    = sph_to_cart(2.2, 20, 0)     # East, high up
        EYE_HI_NE   = sph_to_cart(2.5, 30, 45)    # Northeast, high up
        EYE_HI_N    = sph_to_cart(2.5, 20, 90)    # North, high up
        EYE_HI_NW   = sph_to_cart(2.5, 30, 135)   # Northwest, high up

        # Layer 2: Medium-high elevation (closer to the horizon, around 45°)
        EYE_MH_W    = sph_to_cart(2.2, 45, -180)  # West, medium height
        EYE_MH_SW   = sph_to_cart(2.5, 45, -135)  # Southwest, medium height
        EYE_MH_S    = sph_to_cart(2.5, 45, -90)   # South, medium height
        EYE_MH_SE   = sph_to_cart(2.5, 45, -45)   # Southeast, medium height
        EYE_MH_E    = sph_to_cart(2.2, 45, 0)     # East, medium height
        EYE_MH_NE   = sph_to_cart(2.5, 45, 45)    # Northeast, medium height
        EYE_MH_N    = sph_to_cart(2.5, 45, 90)    # North, medium height
        EYE_MH_NW   = sph_to_cart(2.5, 45, 135)   # Northwest, medium height

        # Layer 3: Medium-low elevation (closer to the nadir)
        EYE_ML_W    = sph_to_cart(2.2, 70, -180)  # West, low elevation
        EYE_ML_SW   = sph_to_cart(2.5, 60, -135)  # Southwest, low elevation
        EYE_ML_S    = sph_to_cart(2.5, 70, -90)   # South, low elevation
        EYE_ML_SE   = sph_to_cart(2.5, 60, -45)   # Southeast, low elevation
        EYE_ML_E    = sph_to_cart(2.2, 70, 0)     # East, low elevation
        EYE_ML_NE   = sph_to_cart(2.5, 60, 45)    # Northeast, low elevation
        EYE_ML_N    = sph_to_cart(2.5, 70, 90)    # North, low elevation
        EYE_ML_NW   = sph_to_cart(2.5, 60, 135)   # Northwest, low elevation

        # Layer 4: Low elevation (closer to the nadir)
        EYE_LO_W    = sph_to_cart(2.2, 90, -180)  # West, low elevation
        EYE_LO_SW   = sph_to_cart(2.5, 80, -135)  # Southwest, low elevation
        EYE_LO_S    = sph_to_cart(2.5, 90, -90)   # South, low elevation
        EYE_LO_SE   = sph_to_cart(2.5, 80, -45)   # Southeast, low elevation
        EYE_LO_E    = sph_to_cart(2.2, 90, 0)     # East, low elevation
        EYE_LO_NE   = sph_to_cart(2.5, 80, 45)    # Northeast, low elevation
        EYE_LO_N    = sph_to_cart(2.5, 90, 90)    # North, low elevation
        EYE_LO_NW   = sph_to_cart(2.5, 80, 135)   # Northwest, low elevation

        # Example of viewing from directly above and below
        EYE_ZENITH  = sph_to_cart(2.5, 0, 0)      # Directly above (zenith)
        EYE_NADIR   = sph_to_cart(2.5, 180, 0)    # Directly below (nadir)


    @with_figure
    def plot(self, *args: R | tuple[R, dict], method: str, fig: BaseFigure, **kwds) -> BaseFigure:
        """General plotting interface.
        
        Args:
            *args: TODO.
            method: Plotting method to use. Implementation must provide `{method}` or `plot_{method}`.
            fig: Existing figure to plot into. If not provided, a new figure is created.
            **kwds: Additional keyword arguments passed to the plotting method.

        Returns:
            fig: The figure containing the plots.
        """

        func = (getattr(self, 'plot_' + method, None) or getattr(self, method, None))
        if not callable(func):
            raise ValueError(f"Unknown plotting method '{method}'")
        
        normalize = lambda x: x if isinstance(x, tuple) else (x, {})
        for arg, kw in map(normalize, args):
            setdefaults(kw, kwds)
            func(arg, fig=fig, **kw)
        
        return fig

    def transform_to_bitmap(self, inp: R, axes: Axes2D, **kwds) -> np.ndarray:
        """Transform input data to a bitmap (2D boolean array).
        
        This is a stub implementation. Actual implementation depends on the data type R.
        
        Args:
            inp: Input data to transform.
            axes: Two axes to project onto.
            **kwds: Additional keyword arguments for the transformation.
        
        Returns:
            A 2D boolean numpy array representing the bitmap.
        """
        raise NotImplementedError("transform_to_bitmap not implemented")

    @with_figure(dim=2)
    def plot_bitmap(self, 
                    inp: R, *,
                    value: float = 0.5,
                    axes: Axes2D = (0, 1),
                    fig: BaseFigure,
                    **kwds) -> BaseFigure:
        """Plot a 2D bitmap.

        This method visualizes the input data as a 2D bitmap using a heatmap. To select the color
        for the "True" values in the bitmap, use the `value` argument. This must be within the
        range defined by `zmin` and `zmax` (arguments to go.Heatmap). `zmin` and `zmax` default to
        0 and 1, respectively.

        *Note:* Requires the `transform_to_bitmap` method to be implemented.

        Args:
            inp: Input data to plot.
            value: Value to represent "True" in the bitmap.
            axes: Two axes to project onto.
            fig: Figure to plot into. If not provided, a new figure is created.
            **kwds: Additional keyword arguments for the heatmap.

        Returns:
            fig: The figure containing the bitmap plot.
        """

        setdefaults(kwds,
                    zmin=0, zmax=1,
                    colorscale="Greens",
                    showscale=False)

        if not len(axes) == 2:
            raise ValueError("plot_bitmap expects exactly 2 axes")
        axes = tuple(self.axis(ax) for ax in axes)

        if not kwds['zmin'] <= value <= kwds['zmax']:
            raise ValueError("plot_bitmap expects value within [zmin, zmax]")

        transf_kw = collect_prefix(kwds, 'transform_', remove=True)
        Z = self.transform_to_bitmap(inp, axes=axes, **transf_kw)
        if Z.ndim != 2: raise ValueError("transform_to_bitmap must return a 2D array")

        min_bounds = kwds.pop('min_bounds', [self._min_bounds[i] for i in axes])
        max_bounds = kwds.pop('max_bounds', [self._max_bounds[i] for i in axes])
        if len(min_bounds) != 2: raise ValueError("plot_bitmap expects exactly 2 min_bounds")
        if len(max_bounds) != 2: raise ValueError("plot_bitmap expects exactly 2 max_bounds")

        X, Y = np.meshgrid(np.linspace(min_bounds[0], max_bounds[0], Z.shape[0]),
                            np.linspace(min_bounds[1], max_bounds[1], Z.shape[1]),
                            indexing='ij')

        Z = np.where(Z, value, np.nan)

        xaxis = dict(zeroline=False, showline=False, showticklabels=True)
        yaxis = xaxis.copy()

        for i, axis in enumerate((xaxis, yaxis)):
            title = self.axis_name(axes[i])
            if unit := self.axis_unit(axes[i]):
                title += f' [{unit}]'
            axis.update(range=[min_bounds[i], max_bounds[i]], title=title)

        fig.update_layout(xaxis=xaxis, yaxis=yaxis)

        return fig.add_trace(go.Heatmap(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            **kwds
        ))
    
    def transform_to_scatter(self, inp: R, axes: Axes2D, **kwds) -> np.ndarray:
        """Transform input data to scatter points (N x 2 float array).
        
        This is a stub implementation. Actual implementation depends on the data type R.
        
        Args:
            inp: Input data to transform.
            axes: Two axes to project onto.
            **kwds: Additional keyword arguments for the transformation.
        
        Returns:
            An (N, 2) float numpy array representing the scatter points.
        """
        raise NotImplementedError("transform_to_scatter not implemented")
    
    @with_figure(dim=2)
    def plot_fill(self,
                  inp: R, *,
                  axes: Axes2D = (0, 1),
                  fig: BaseFigure,
                  **kwds) -> BaseFigure:
        """Plot a filled 2D area.

        This method visualizes the input data as a filled area using a scatter plot with
        `fill='toself'`. The area is defined by the points returned by `transform_to_scatter`.

        *Note:* Requires the `transform_to_scatter` method to be implemented.

        Args:
            inp: Input data to plot.
            axes: Two axes to project onto.
            fig: Figure to plot into. If not provided, a new figure is created.
            **kwds: Additional keyword arguments for the scatter plot.

        Returns:
            fig: The figure containing the filled area plot.
        """

        setdefaults(kwds,
                    fill='toself',
                    mode='lines',
                    line=dict(width=2),
                    fillcolor='LightGreen',
                    opacity=0.5,
                    showlegend=False)

        if not len(axes) == 2:
            raise ValueError("plot_fill expects exactly 2 axes")
        axes = tuple(self.axis(ax) for ax in axes)

        transf_kw = collect_prefix(kwds, 'transform_', remove=True)
        P = self.transform_to_scatter(inp, axes=axes, **transf_kw)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError("transform_to_scatter must return an (N, 2) array")

        xaxis = dict(zeroline=False, showline=False, showticklabels=True)
        yaxis = xaxis.copy()

        for i, axis in enumerate((xaxis, yaxis)):
            title = self.axis_name(axes[i])
            if unit := self.axis_unit(axes[i]):
                title += f' [{unit}]'
            axis.update(title=title)

        fig.update_layout(xaxis=xaxis, yaxis=yaxis)

        return fig.add_trace(go.Scatter(
            x=P[:, 0],
            y=P[:, 1],
            **kwds
        ))
    
    def transform_to_surface(self, inp: R, axes: Axes2D, **kwds) -> np.ndarray:
        """Transform input data to a 3D surface (2D float array).
        
        This is a stub implementation. Actual implementation depends on the data type R.
        
        Args:
            inp: Input data to transform.
            axes: Two axes to project onto.
            **kwds: Additional keyword arguments for the transformation.
        
        Returns:
            A 2D float numpy array whose value represents the surface.
        """
        raise NotImplementedError("transform_to_surface not implemented")

    @with_figure(dim=2)
    def plot_contour(self,
                     inp: R, *,
                     axes: Axes2D = (0, 1),
                     fig: BaseFigure,
                     **kwds) -> BaseFigure:
        """Plot a 2D contour.

        This method visualizes the input data as a 2D contour using a contour plot. The contour
        levels are determined by the values in the 2D array returned by `transform_to_surface`.

        *Note:* Requires the `transform_to_surface` method to be implemented.

        Args:
            inp: Input data to plot.
            axes: Two axes to project onto.
            fig: Figure to plot into. If not provided, a new figure is created.
            **kwds: Additional keyword arguments for the contour plot.

        Returns:
            fig: The figure containing the contour plot.
        """

        setdefaults(kwds,
                    showscale=False,
                    contours_coloring='lines',
                    line=dict(width=2))
        
        if not len(axes) == 2:
            raise ValueError("plot_contour expects exactly 2 axes")
        axes = tuple(self.axis(ax) for ax in axes)

        transf_kw = collect_prefix(kwds, 'transform_', remove=True)
        Z = self.transform_to_surface(inp, axes=axes, **transf_kw)
        if Z.ndim != 2: raise ValueError("transform_to_surface must return a 2D array")

        min_bounds = kwds.pop('min_bounds', [self._min_bounds[i] for i in axes])
        max_bounds = kwds.pop('max_bounds', [self._max_bounds[i] for i in axes])
        if len(min_bounds) != 2: raise ValueError("plot_contour expects exactly 2 min_bounds")
        if len(max_bounds) != 2: raise ValueError("plot_contour expects exactly 2 max_bounds")
        
        X, Y = np.meshgrid(np.linspace(min_bounds[0], max_bounds[0], Z.shape[0]),
                           np.linspace(min_bounds[1], max_bounds[1], Z.shape[1]),
                           indexing='ij')

        xaxis, yaxis = {}, {}
        for i, axis in enumerate((xaxis, yaxis)):
            title = self.axis_name(axes[i])
            if unit := self.axis_unit(axes[i]):
                title += f' [{unit}]'
            axis.update(range=[min_bounds[i], max_bounds[i]], title=title)

        fig.update_layout(xaxis=xaxis, yaxis=yaxis)

        return fig.add_trace(go.Contour(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            **kwds
        ))

    @with_figure(dim=3)
    def plot_surface(self,
                     inp: R, *,
                     axes: Axes2D = (0, 1),
                     fig: BaseFigure,
                     **kwds) -> BaseFigure:
        """Plot a 3D surface.
        
        This method visualizes the input data as a 3D surface using a surface plot. The height of
        the surface is determined by the values in the 2D array returned by `transform_to_surface`.
        
        *Note:* Requires the `transform_to_surface` method to be implemented.
        
        Args:
            inp: Input data to plot.
            axes: Two axes to project onto.
            fig: Figure to plot into. If not provided, a new figure is created.
            **kwds: Additional keyword arguments for the surface plot.
        
        Returns:
            fig: The figure containing the surface plot.
        """

        setdefaults(kwds,
                    showscale=False)

        if not len(axes) == 2:
            raise ValueError("plot_surface expects exactly 2 axes")
        axes = tuple(self.axis(ax) for ax in axes)

        transf_kw = collect_prefix(kwds, 'transform_', remove=True)
        Z = self.transform_to_surface(inp, axes=axes, **transf_kw)
        if Z.ndim != 2: raise ValueError("transform_to_surface must return a 2D array")

        min_bounds = kwds.pop('min_bounds', [self._min_bounds[i] for i in axes])
        max_bounds = kwds.pop('max_bounds', [self._max_bounds[i] for i in axes])
        if len(min_bounds) != 2: raise ValueError("plot_bitmap expects exactly 2 min_bounds")
        if len(max_bounds) != 2: raise ValueError("plot_bitmap expects exactly 2 max_bounds")

        X, Y = np.meshgrid(np.linspace(min_bounds[0], max_bounds[0], Z.shape[0]),
                            np.linspace(min_bounds[1], max_bounds[1], Z.shape[1]),
                            indexing='ij')
        
        xaxis, yaxis = {}, {}
        for i, axis in enumerate((xaxis, yaxis)):
            title = self.axis_name(axes[i])
            if unit := self.axis_unit(axes[i]):
                title += f' [{unit}]'
            axis.update(range=[min_bounds[i], max_bounds[i]], title=title)

        fig.update_layout(scene=dict(aspectmode='cube',
                                     xaxis=xaxis,
                                     yaxis=yaxis,
                                     zaxis_title='Value'))

        return fig.add_trace(go.Surface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            **kwds
        ))
    
    def transform_to_isosurface(self, inp: R, axes: Axes3D, **kwds) -> np.ndarray:
        """Transform input data to a 3D volume (3D float array).
        
        This is a stub implementation. Actual implementation depends on the data type R.
        
        Args:
            inp: Input data to transform.
            axes: Three axes to project onto.
            **kwds: Additional keyword arguments for the transformation.
        
        Returns:
            A 3D float numpy array whose value represents the volume.
        """
        raise NotImplementedError("transform_to_isosurface not implemented")

    @with_figure(dim=3)
    def plot_isosurface(self,
                        inp: R, *,
                        level: float = 0.0, 
                        axes: Axes3D = (0, 1, 2),
                        fig: BaseFigure,
                        **kwds) -> BaseFigure:
        """Plot a 3D isosurface.

        This method visualizes the input data as a 3D isosurface using an isosurface plot. The
        isosurface is extracted at the specified `level` from the 3D volume returned by
        `transform_to_isosurface`.

        *Note:* Requires the `transform_to_isosurface` method to be implemented.

        Args:
            inp: Input data to plot.
            level: Level at which to extract the isosurface.
            axes: Three axes to project onto.
            fig: Figure to plot into. If not provided, a new figure is created.
            **kwds: Additional keyword arguments for the isosurface plot.

        Returns:
            fig: The figure containing the isosurface plot.
        """
        
        setdefaults(kwds,
                    colorscale='Greens', 
                    showscale=False,
                    isomin=level,
                    isomax=level,
                    surface_count=1,
                    caps=dict(x_show=False, y_show=False, z_show=False))

        if not len(axes) == 3:
            raise ValueError("plot_isosurface expects exactly 3 axes")
        axes = tuple(self.axis(ax) for ax in axes)
        
        transf_kw = collect_prefix(kwds, 'transform_', remove=True)
        V = self.transform_to_isosurface(inp, axes=axes, **transf_kw)
        if V.ndim != 3: raise ValueError("transform_to_isosurface must return a 3D array")

        min_bounds = kwds.pop('min_bounds', [self._min_bounds[i] for i in axes])
        max_bounds = kwds.pop('max_bounds', [self._max_bounds[i] for i in axes])
        if len(min_bounds) != 3: raise ValueError("plot_isosurface expects exactly 3 min_bounds")
        if len(max_bounds) != 3: raise ValueError("plot_isosurface expects exactly 3 max_bounds")

        X, Y, Z = np.meshgrid(np.linspace(min_bounds[0], max_bounds[0], V.shape[0]),
                              np.linspace(min_bounds[1], max_bounds[1], V.shape[1]),
                              np.linspace(min_bounds[2], max_bounds[2], V.shape[2]),
                              indexing='ij')
        
        xaxis, yaxis, zaxis = {}, {}, {}
        for i, axis in enumerate((xaxis, yaxis, zaxis)):
            title = self.axis_name(axes[i])
            if unit := self.axis_unit(axes[i]):
                title += f' [{unit}]'
            axis.update(range=[min_bounds[i], max_bounds[i]], title=title)

        fig.update_layout(scene=dict(aspectmode='cube',
                                     xaxis=xaxis,
                                     yaxis=yaxis,
                                     zaxis=zaxis))

        return fig.add_trace(go.Isosurface(
            x=X.flatten(), 
            y=Y.flatten(), 
            z=Z.flatten(),
            value=V.flatten(),
            **kwds
        ))
