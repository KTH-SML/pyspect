######################################################################
## Set Builder

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_protocol_intersection import ProtocolIntersection as All

from .impls.protos import *
from .impls.base import *

__all__ = (
    'SetBuilder',
    'ReferredSet',
    'AppliedSet',
    'Set',
    'ABSURD',
    'EMPTY',
    'HalfSpaceSet',
    'BoundedSet',
    'Compl',
    'Inter',
    'Union',
)


## ## ## ## ## ## ## ##
## Special Primitives

class SetBuilderMeta(ImplClientMeta, ABCMeta, type): ...

class SetBuilder(ImplClient, metaclass=SetBuilderMeta):

    free: tuple[str, ...] = ()

    @abstractmethod
    def __call__(self, impl: 'I', **m) -> 'R': ...

    def __repr__(self) -> str:
        cls = type(self)
        ptr = hash(self)
        return f'<{cls.__name__} at {ptr:#0{18}x}>'

class AbsurdSet(SetBuilder):
    
    def __call__(self, impl: 'I', **m: SetBuilder):
        raise ValueError("Cannot realize the absurd set.")

ABSURD: AbsurdSet = AbsurdSet()

class Set(SetBuilder):

    def __init__(self, arg: 'R') -> None:
        self.arg = arg

    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        return self.arg

class ReferredSet(SetBuilder):

    def __init__(self, name: str) -> None:
        self.free += (name,)

    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        name, = self.free
        sb = m.pop(name)
        return sb(impl, **m)

class AppliedSet(SetBuilder):

    def __init__(self, funcname: str, *builders: SetBuilder) -> None:
        self.funcname = funcname
        self.builders = builders
        
        _require = (funcname,)

        for builder in self.builders:
            _require += builder.__require__
            self.free += tuple(name for name in builder.free if name not in self.free)

        self.add_requirements(_require)        

    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        try:
            func = getattr(impl, self.funcname)
        except AttributeError as e:
            raise AttributeError(f'Impl {impl.__class__.__name__} does not support "{self.funcname}".') from e
        
        args = []
        for i, sb in enumerate(self.builders):
            try:
                args.append(sb(impl, **m))
            except Exception as e:
                E = type(e)
                raise E(f'When applying "{self.funcname}" on argument {i}, received: {e!s}') from e
        
        return func(*args)


## ## ## ## ## ## ## ## ## ##
## User-friendly Primitives

class EmptySet(SetBuilder):

    __require__ = ('empty',)
    
    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        return impl.empty()
    
EMPTY: EmptySet = EmptySet()

class HalfSpaceSet(SetBuilder):

    __require__ = ('plane_cut',)

    def __init__(self, normal, offset, axes, **kwds) -> None:
        self.normal = normal
        self.offset = offset
        self.axes = axes
        self.kwds = kwds
    
    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        return impl.plane_cut(normal=self.normal, 
                              offset=self.offset, 
                              axes=[impl.axis(ax) 
                                    for ax in self.axes],
                              **self.kwds)

class BoundedSet(SetBuilder):

    __require__ = ('complement', 'plane_cut', 'intersect')

    def __init__(self, **bounds: list[int]) -> None:
        self.bounds = bounds

    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        s = impl.complement(impl.empty())
        for name, (vmin, vmax) in self.bounds.items():
            i = impl.axis(name)
            if vmin is Ellipsis:
                assert vmax is not Ellipsis, f'Invalid bounds for axis {impl.axis_name(i)}, there must be either an upper or lower bound.'
                upper_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                axis_range = upper_bound
            elif vmax is Ellipsis:
                assert vmin is not Ellipsis, f'Invalid bounds for axis {impl.axis_name(i)}, there must be either an upper or lower bound.'
                lower_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                axis_range = lower_bound
            elif impl.axis_is_periodic(i) and vmax < vmin:
                upper_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                axis_range = impl.complement(impl.intersect(upper_bound, lower_bound))
            else:
                # NOTE: See similar assertion in TVHJImpl's plane_cut
                amin, amax = impl.axis_bounds(i)
                assert amin < vmin < amax, f'For dimension "{name}", {amin} < {vmin=} < {amax}. Use Ellipsis (...) to indicate subset stretching to the space boundary.'
                assert amin < vmax < amax, f'For dimension "{name}", {amin} < {vmax=} < {amax}. Use Ellipsis (...) to indicate subset stretching to the space boundary.'
                upper_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                axis_range = impl.intersect(upper_bound, lower_bound)
            s = impl.intersect(s, axis_range)
        return s

## ## ## ## ## ## ## ## ## ##
## User-friendly Operations

def Compl(*args: SetBuilder) -> SetBuilder:
    return AppliedSet('complement', *args)

def Inter(*args: SetBuilder) -> SetBuilder:
    return AppliedSet('intersect', *args)

def Union(*args: SetBuilder) -> SetBuilder:
    return AppliedSet('union', *args)
   