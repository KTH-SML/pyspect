from functools import wraps
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
from PIL import Image


def iterwin(seq, winlen=1):
    slices = [seq[i::winlen] for i in range(winlen)]
    yield from zip(*slices)

def setdefaults(d: dict, *args, **kwds) -> None:
    """Set dictionary defaults."""
    if len(args) == 0:
        assert kwds, 'Missing arguments'
        defaults = kwds
    elif len(args) == 1:
        defaults, = args
        assert isinstance(defaults, dict), 'Single-argument form must be default dictionary'
        assert not kwds, 'Cannot supply keywords arguments with setdefault({...}) form'
    else:
        assert not kwds, 'Cannot supply keywords arguments with setdefault(key, val) form'
        assert len(args) % 2 == 0, 'Must have even number of arguments with setdefault(key, val) form'
        defaults = {key: val for key, val in iterwin(args, 2)}
    for key, val in defaults.items():
        d.setdefault(key, val)

def collect_prefix(d: dict, prefix: str):
    return {key.removeprefix(prefix): d.pop(key) 
            for key in list(d) if key.startswith(prefix)}


def sph_to_cart(r, theta, phi):
    theta *= np.pi/180
    phi *= np.pi/180
    return dict(x=r*np.sin(theta)*np.cos(phi),
                y=r*np.sin(theta)*np.sin(phi),
                z=r*np.cos(theta))

def layout_theme(theme):
    layout = dict(margin=dict(l=60, r=20, t=40, b=60))
    axis = dict(linewidth=2)
    font = dict(family="Roboto, Arial, sans-serif", size=14)
    if theme.startswith('Light'):
        layout.update(template="plotly_white",
                      paper_bgcolor='rgba(255, 255, 255, 1)',
                      plot_bgcolor='rgba(250, 250, 250, 1)')
        axis.update(linecolor='rgba(0, 0, 0, 0.3)',
                    gridcolor='rgba(0, 0, 0, 0.1)',
                    zerolinecolor='rgba(0, 0, 0, 0.3)')
        font.update(color='black')
    if theme.startswith('Dark'):
        layout.update(template="plotly_dark",
                      paper_bgcolor='rgba(26, 28, 36, 1)',
                      plot_bgcolor='rgba(26, 28, 36, 1)')
        axis.update(linecolor='rgba(255, 255, 255, 0.3)',
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    zerolinecolor='rgba(255, 255, 255, 0.3)')
        font.update(color='white')
    if theme[-2:] not in ('2D', '3D'):
        theme += '2D'
    if theme.endswith('2D'):
        layout.update(xaxis=axis, yaxis=axis)
    if theme.endswith('3D'):
        layout.update(scene=dict(xaxis=axis, yaxis=axis, zaxis=axis))
    return layout

def new_fig(**kwargs):
    fig = go.Figure()

    theme = kwargs.pop('theme', 'Light2D')
    width = kwargs.pop('width', 600)
    height = kwargs.pop('height', 500)
    
    fig.update_layout(width=width, height=height)
    fig.update_layout(layout_theme(theme))

    return fig

def auto_fig(f):
    @wraps(f)
    def wrapper(*args, fig: BaseFigure = None, fig_enabled=True, **kwargs):
        fig_kw = {k[4:]: kwargs.pop(k) for k in tuple(kwargs) if k.startswith('fig_')}
        
        # Automatically detect 2D or 3D theme
        dim = f.__name__[4:6]
        if 'fig_theme' not in kwargs:
            fig_kw.update(fig_theme=f'Light{dim}')
        elif fig_kw['theme'][-2:] not in ('3D', '2D'):
            fig_kw['theme'] += dim
        
        if fig is None:
            fig = new_fig(**fig_kw)
        
        if fig_enabled:
            return f(*args, fig=fig, **kwargs)
        
    return wrapper

@auto_fig
def plot2D_image(source, *, fig: BaseFigure, **kwargs) -> BaseFigure:
    min_bounds = kwargs.pop('min_bounds')
    max_bounds = kwargs.pop('max_bounds')

    if isinstance(source, (str, bytes, Path)):
        source = Image.open(source)

    setdefaults(kwargs,
                x=min_bounds[0], sizex=max_bounds[0] - min_bounds[0],
                y=min_bounds[1], sizey=max_bounds[1] - min_bounds[1],
                xref="x", xanchor="left", 
                yref="y", yanchor="bottom",
                sizing="stretch",
                layer="below")

    fig.add_layout_image(source=source, **kwargs)

    fig.update_xaxes(showline=False, zeroline=False)
    fig.update_yaxes(showline=False, zeroline=False)

    return fig

@auto_fig
def plot3D_image(source, *, fig: BaseFigure, **kwargs) -> BaseFigure:
    min_bounds = kwargs.pop('min_bounds')
    max_bounds = kwargs.pop('max_bounds')
    axes = kwargs.pop('axes', (1, 2, 0))

    assert len(axes) == 3, "axes must be a tuple of 3 integers"
    I, J, K = axes

    if isinstance(source, (str, bytes, Path)):
        source = Image.open(source)
    
    source = np.asarray(source.convert('L'))

    setdefaults(kwargs,
                x=np.linspace(min_bounds[I], max_bounds[I], source.shape[0]),
                y=np.linspace(min_bounds[J], max_bounds[J], source.shape[1]),
                z=0.98*min_bounds[K]*np.ones_like(source),
                showscale=False)

    fig.update_layout(
        scene=dict(
            camera=collect_prefix(kwargs, 'camera_'),
            **collect_prefix(kwargs, 'scene_'),    
        ),
        **collect_prefix(kwargs, 'layout_'),
    )

    return fig.add_trace(go.Surface(
        surfacecolor=source,
        colorscale='gray',
        **kwargs,
    ))

@auto_fig
def plot2D_bitmap(im, *, fig: BaseFigure, **kwargs) -> BaseFigure:
    setdefaults(kwargs, 
                zmin=0, zval=0.5, zmax=1,
                transpose=True,
                colorscale='Greens',
                showscale=False)

    zval        = kwargs.pop('zval')
    transpose   = kwargs.pop('transpose')    

    im = np.where(im, zval, np.nan)
    if transpose:
        im = np.transpose(im)

    fig.update_layout(
        **collect_prefix(kwargs, 'layout_'),
    )

    return fig.add_trace(go.Heatmap(z=im, **kwargs))

@auto_fig
def plot2D_levelset(vf, *, fig: BaseFigure, axes=(0,1), level=0, **kwargs) -> BaseFigure:
    assert len(axes) == 2
    I, J = axes
    
    if 'mesh' in kwargs:
        mesh = kwargs.pop('mesh')
        assert len(mesh) == 2
        
        X = mesh[I]
        Y = mesh[J]

    else:
        min_bounds = kwargs.pop('min_bounds')
        max_bounds = kwargs.pop('max_bounds')
        assert len(min_bounds) == 2
        assert len(max_bounds) == 2

        X, Y = np.meshgrid(np.linspace(min_bounds[I], max_bounds[I], vf.shape[I]), 
                           np.linspace(min_bounds[J], max_bounds[J], vf.shape[J]),
                           indexing='ij')

    setdefaults(kwargs,
                x=X.flatten(), y=Y.flatten(),
                xtitle="x [m]", ytitle="y [m]")

    xtitle = kwargs.pop('xtitle')
    ytitle = kwargs.pop('ytitle')

    fig.update_layout(
        xaxis=dict(range=[min_bounds[I], max_bounds[I]], title=xtitle),
        yaxis=dict(range=[min_bounds[J], max_bounds[J]], title=ytitle),
        **collect_prefix(kwargs, 'layout_'),
    )

    im = vf.transpose(*axes).flatten() <= level
    return plot2D_bitmap(im, fig=fig, transpose=False, **kwargs)

@auto_fig
def plot3D_valuefun(vf, *, fig: BaseFigure, axes=(0, 1), **kwargs) -> BaseFigure:
    assert len(vf.shape) == 2
    assert len(axes) == 2
    I, J = axes

    min_bounds = kwargs.pop('min_bounds')
    max_bounds = kwargs.pop('max_bounds')
    assert len(min_bounds) == 2
    assert len(max_bounds) == 2

    X, Y = np.meshgrid(np.linspace(min_bounds[I], max_bounds[I], vf.shape[I]),
                       np.linspace(min_bounds[J], max_bounds[J], vf.shape[J]),
                       indexing='ij')

    setdefaults(kwargs,
                camera_eye=EYE_ML_NE,
                showscale=False,
                x=X, y=Y,
                xtitle='x [m]', ytitle='y [m]', ztitle='Value')

    xtitle  = kwargs.pop('xtitle')
    ytitle  = kwargs.pop('ytitle')
    ztitle  = kwargs.pop('ztitle')

    fig.update_layout(
        scene=dict(
            xaxis_title=xtitle,
            yaxis_title=ytitle,
            zaxis_title=ztitle,
            aspectmode='cube',
            camera=collect_prefix(kwargs, 'camera_'),
            **collect_prefix(kwargs, 'scene_'),
        ),
        **collect_prefix(kwargs, 'layout_'),
    )

    return fig.add_trace(go.Surface(
        z=vf.transpose(*axes),
        **kwargs,
    ))

@auto_fig
def plot3D_levelset(vf, *, fig: BaseFigure, axes=(1,2,0), level=0, **kwargs) -> BaseFigure:
    assert len(vf.shape) == 3
    assert len(axes) == 3
    I, J, K = axes

    if 'mesh' in kwargs:
        mesh = kwargs.pop('mesh')
        assert len(mesh) == 3
        
        X = mesh[I]
        Y = mesh[J]
        Z = mesh[K]

    else:
        min_bounds = kwargs.pop('min_bounds')
        max_bounds = kwargs.pop('max_bounds')
        assert len(min_bounds) == 3
        assert len(max_bounds) == 3
        
        X, Y, Z = np.meshgrid(np.linspace(min_bounds[I], max_bounds[I], vf.shape[I]), 
                              np.linspace(min_bounds[J], max_bounds[J], vf.shape[J]),
                              np.linspace(min_bounds[K], max_bounds[K], vf.shape[K]),
                              indexing='ij')

    setdefaults(kwargs,
                camera_eye=EYE_ML_NE,
                colorscale='Greens', showscale=False,
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                xtitle='x [m]', ytitle='y [m]', ztitle='t [s]',
                caps=dict(x_show=False, y_show=False, z_show=False))
    
    xtitle  = kwargs.pop('xtitle')
    ytitle  = kwargs.pop('ytitle')
    ztitle  = kwargs.pop('ztitle')

    fig.update_layout(
        scene=dict(
            xaxis_title=xtitle,
            yaxis_title=ytitle,
            zaxis_title=ztitle,
            aspectmode='cube',
            camera=collect_prefix(kwargs, 'camera_'),
            **collect_prefix(kwargs, 'scene_'),
        ),
        **collect_prefix(kwargs, 'layout_'),
    )

    return fig.add_trace(go.Isosurface(
        value=vf.transpose(*axes).flatten(),
        isomin=level,
        isomax=level,
        surface_count=1,
        **kwargs
    ))

@auto_fig
def plot_levelsets(*vfs, fig: BaseFigure, plot_func=plot2D_levelset, **kwargs) -> BaseFigure:
    f = lambda x: x if isinstance(x, tuple) else (x, {})
    for vf, kw in map(f, vfs):
        setdefaults(kw, kwargs)
        plot_func(vf, fig=fig, **kw)
    return fig


# FIXME
@auto_fig
def film_levelsets(*frames, fig: BaseFigure, plot_func=plot3D_levelset, **kwargs) -> BaseFigure:
    for vfs in frames:
        frame_fig = plot_levelsets(*vfs, plot_func=plot_func, **kwargs)
        frame = go.Frame(data=frame_fig.data)
        fig.frames += (frame,)

    # Add animation buttons
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True,
                                       "transition": {"duration": 300,
                                                      "easing": "quadratic-in-out"}}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])
            ])]
        **collect_prefix(kwargs, 'layout_'),
    )

    return fig


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