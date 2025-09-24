from functools import wraps

import numpy as np
import matplotlib.pyplot as plt


def auto_ax(f):
    @wraps(f)
    def wrapper(*args, ax: plt.Axes = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        kwargs.update(ax=ax)
        return f(*args, **kwargs)
    return wrapper

@auto_ax
def plot_im(im, *, ax, transpose=True, **kwargs):
    if transpose:
        im = np.transpose(im)
    kwargs.setdefault('cmap', 'Blues')
    kwargs.setdefault('aspect', 'auto')
    return [ax.imshow(im, vmin=0, vmax=1, origin='lower', **kwargs)]

@auto_ax
def plot_set(vf, **kwargs):
    vf = np.where(vf <= 0, 0.5, np.nan)
    kwargs.setdefault('aspect', 'equal')
    return plot_im(vf, **kwargs)

@auto_ax
def plot_set_many(*vfs, **kwargs):
    out = []
    f = lambda x: x if isinstance(x, tuple) else (x, {})
    for vf, kw in map(f, vfs):
        out += plot_set(vf, **kw, **kwargs)
    return out

def new_map(*pairs, **kwargs):
    
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9*4/3, 9))
    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")
    ax.invert_yaxis()

    if 'extent' not in kwargs:
        min_bounds = kwargs.pop('min_bounds')
        max_bounds = kwargs.pop('max_bounds')
        extent=[min_bounds[0], max_bounds[0],
                min_bounds[1], max_bounds[1]]
        kwargs['extent'] = extent

    if "background" in kwargs:
        background = kwargs.pop('background')
        if not isinstance(background, np.ndarray):
            background = plt.imread(background)
        ax.imshow(background, extent=kwargs['extent'])
        
    for cmap, vf in pairs:
        kw = dict(alpha=0.9, cmap=plt.get_cmap(cmap))
        kw.update(kwargs)
        plot_set(vf, ax=ax, **kw)
    fig.tight_layout()
    return fig