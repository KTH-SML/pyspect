######################################################################
## Set Builder

from typing import TypeVar, Callable, Tuple, Dict
from .langs.base import ImplClientMeta

__all__ = (
    'SetBuilder',
    'ReferredSet',
    'AppliedSet',
    'Set',
    'ABSURD',
    'EMPTY',
    'HalfSpaceSet',
    'BoundedSet',
)

class SetBuilder:

    free: tuple[str, ...] = ()

    def __call__(self, impl, **lmap): ...

    def __repr__(self):
        cls = type(self)
        ptr = hash(self)
        return f'<{cls.__name__} at {ptr:#0{18}x}>'

class AbsurdSet(SetBuilder):
    def __call__(self, impl, **lmap):
        raise ValueError("Cannot realize the absurd set.")
ABSURD = AbsurdSet()

class Set(SetBuilder):
    def __init__(self, arg):
        self.arg = arg
    def __call__(self, impl, **lmap):
        return self.arg

class ReferredSet(SetBuilder):
    def __init__(self, name):
        self.free += (name,)
    def __call__(self, impl, **lmap):
        name, = self.free
        sb = lmap.pop(name)
        return sb(impl, **lmap)

class AppliedSet(SetBuilder):
    def __init__(self, funcname, *builders: SetBuilder):
        self.funcname = funcname
        self.builders = builders
        for builder in self.builders:
            self.free += tuple(name for name in builder.free if name not in self.free)
    def __call__(self, impl, **m):
        args = [sb(impl, **m) for sb in self.builders]
        sb = getattr(impl, self.funcname)
        return sb(*args)
    def find(self, cls):
        yield from super().find(cls)
        for sb in self.builders:
            yield from sb.find(cls)



class EmptySet(SetBuilder):
    def __call__(self, impl, **lmap):
        return impl.empty()
EMPTY = EmptySet()

class HalfSpaceSet(SetBuilder):
    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds
    def __call__(self, impl, **lmap):
        return impl.plane_cut(*self.args, **self.kwds)

class BoundedSet(SetBuilder):
    def __init__(self, **bounds):
        self.bounds = bounds
    def __call__(self, impl, **lmap):
        s = impl.complement(impl.empty())
        _bounds = [(vmin, vmax, impl.axis(name))
                   for name, (vmin, vmax) in self.bounds.items()]
        for vmin, vmax, i in _bounds:
            if vmax < vmin and impl.axis_is_periodic(i):
                upper_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                axis_range = impl.complement(impl.intersect(upper_bound, lower_bound))
            else:
                upper_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                axis_range = impl.intersect(upper_bound, lower_bound)
            s = impl.intersect(s, axis_range)
        return s
