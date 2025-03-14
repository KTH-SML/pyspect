######################################################################
## Set Builder

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Self, Never

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
)


## ## ## ## ## ## ## ##
## Special Primitives

class SetBuilderMeta(ImplClientMeta, ABCMeta, type): ...

class SetBuilder(ImplClient, metaclass=SetBuilderMeta):

    free: tuple[str, ...] = ()

    @abstractmethod
    def __call__(self, impl: 'I', **m: Self) -> Never | 'R': ...

    def __repr__(self) -> str:
        cls = type(self)
        ptr = hash(self)
        return f'<{cls.__name__} at {ptr:#0{18}x}>'

class AbsurdSet(SetBuilder):
    
    def __call__(self, impl: 'I', **m: SetBuilder) -> Never:
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

    def __call__(self, impl: 'I', **m: SetBuilder) -> Never | 'R':
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

    def __call__(self, impl: 'I', **m: SetBuilder) -> Never | 'R':
        try:
            args = [sb(impl, **m) for sb in self.builders]
            func = getattr(impl, self.funcname)
            return func(*args)
        except Exception as e:
            E = type(e)
            raise E(f'When applying "{self.funcname}", received: {e!s}')


## ## ## ## ## ## ## ## ## ##
## User-friendly Primitives

class EmptySet[R, I: HasEmpty](SetBuilder):

    __require__ = ('empty',)
    
    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        return impl.empty()
    
EMPTY: EmptySet = EmptySet()

class HalfSpaceSet[R, I: HasPlaneCut, **P](SetBuilder):

    __require__ = ('plane_cut',)

    def __init__(self, *args: P.args, **kwds: P.kwargs) -> None:
        self.args = args
        self.kwds = kwds
    
    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
        return impl.plane_cut(*self.args, **self.kwds)

class BoundedSet(SetBuilder):

    __require__ = ('complement', 'plane_cut', 'intersect')

    def __init__(self, **bounds: list[int]) -> None:
        self.bounds = bounds

    def __call__(self, impl: 'I', **m: SetBuilder) -> 'R':
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
