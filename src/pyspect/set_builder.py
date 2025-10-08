"""Implementation-agnostic set builders.

This module defines a tiny DSL of lazy "set builders" that describe sets and
set operations without committing to a concrete representation. A SetBuilder
is realized by an `Impl[R]` (see `pyspect.impls.*`), which interprets operations
(e.g., `empty`, `complement`, `intersect`, `halfspace`).

Key ideas:
    - Builders are composable and track requirements on the target `Impl`.
    - Builders can carry named free variables (ReferredSet) resolved at realization.
    - AppliedSet defers a call to an `Impl` method by name until realization.
"""
from __future__ import annotations
from typing import Any

from .impls.base import Impl, ImplClient
from .impls.axes import Axis

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

class SetBuilder[R](ImplClient[R]):
    """Abstract base for all set builders.

    Responsibilities:
        - Be callable with an implementation `Impl[R]` to produce a concrete set `R`.
        - Track required `Impl` operations through `ImplClient`.
        - Track free variable names (see `ReferredSet`).

    Subclasses should implement `__call__`, which is called to realize the sets.
    """

    free: tuple[str, ...] = ()

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        raise NotImplementedError(f"{type(self).__name__}.__call__ is not implemented. SetBuilders must implement __call__.")

    def __repr__(self) -> str:
        """Return a compact identifier for the builder instance."""
        cls = type(self)
        ptr = hash(self)
        return f'<{cls.__name__} at {ptr:#0{18}x}>'
    
    @property
    def uid(self) -> str:
        """Stable hexadecimal id derived from the object hash."""
        # Simple way to create a unique id from a python function.
        # - hash(sb) returns the function pointer (I think)
        # - Convert to bytes to get capture full 64-bit value (incl. zeroes)
        # - Convert to hex-string
        return hash(self).to_bytes(8,"big").hex()

class AbsurdSet[R](SetBuilder[R]):
    """A builder that cannot be realized.

    Used as a sentinel for impossible constructions. Realization raises.
    """

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        raise ValueError("Cannot realize the absurd set.")

ABSURD: AbsurdSet = AbsurdSet()

class Set[R](SetBuilder[R]):
    """Wrap a concrete set value R and return it unchanged on realization."""

    def __init__(self, arg: R) -> None:
        self.arg = arg

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        return self.arg

class ReferredSet[R](SetBuilder[R]):
    """Reference a named free variable resolved from the realization mapping.

    `ReferredSet('X')(impl, X=some_builder)` realizes to `some_builder(impl, ...)`.
    This is useful in two ways:
        1. We can be lazy when constructing the call tree, i.e. we allow users to
           define which builder to use at a later stage.
        2. This essentially allow variables to exist within the call tree which avoids
           having to reconstruct an entire tree in some cases.
    """

    def __init__(self, name: str) -> None:
        self.free += (name,)

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        name, = self.free
        sb = m.pop(name)
        return sb(impl, **m)

class AppliedSet[R](SetBuilder[R]):
    """Defer a call to `Impl.<funcname>(*args)` where `args` are realized builders.

    - Accumulates required `Impl` methods from children and adds `funcname`.
    - Propagates and de-duplicates children's free variables.
    - On realization, calls child builders first, then invokes the `Impl` method.
    - Wraps child exceptions to pinpoint which argument failed.
    """

    def __init__(self, funcname: str, *builders: SetBuilder[R]) -> None:
        self.funcname = funcname
        self.builders = builders
        
        _require = (funcname,)

        for builder in self.builders:
            _require += builder.__require__
            self.free += tuple(name for name in builder.free if name not in self.free)

        self.add_requirements(_require)        

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
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

class EmptySet[R](SetBuilder[R]):
    """Builder for the empty set.
    
    Requires:
        - `Impl.empty() -> R`
    """

    __require__ = ('empty',)

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        return impl.empty()
    
EMPTY: EmptySet = EmptySet()

class HalfSpaceSet[R](SetBuilder[R]):
    """Half-space described by the normal and offset of a hyperplane.

    Note: The set is in the direction of the normal.

    Parameters:
        normal: coefficients along each axis
        offset: offsets along each axis
        axes: axis indices (or str if using `AxesImpl`) in the `Impl`'s coordinate system
        kwds: forwarded to `Impl.halfspace`

    Requires:
        - `Impl.halfspace(normal, offset, axes, ...) -> R`
    """

    __require__ = ('halfspace',)

    def __init__(
        self,
        normal: list[float],
        offset: list[float],
        axes: list[Axis],
        **kwds: Any,
    ) -> None:
        assert len(axes) == len(normal) == len(offset)
        self.normal = normal
        self.offset = offset
        self.axes = axes
        self.kwds = kwds
    
    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        return impl.halfspace(normal=self.normal, 
                              offset=self.offset, 
                              axes=[impl.axis(ax) for ax in self.axes],
                              **self.kwds)

class BoundedSet[R](SetBuilder[R]):
    """Axis-aligned box possibly unbounded on one side per axis.

    Bounds mapping: `name -> (vmin, vmax)`. Use Ellipsis to denote an open side
    (e.g., (0, ...) or (..., 1)). For periodic axes where `vmax < vmin`, the
    range wraps around.

    Example:
    ```
    # Left half-circle bounded in y.
    A = BoundedSet(y=(-0.5, 0.5), theta=(+pi/2, -pi/2))
    ```

    Requires:
        - `Impl < AxesImpl`
        - `Impl.complement(inp: R) -> R`
        - `Impl.halfspace(normal, offset, axes, ...) -> R`
        - `Impl.intersect(inp1: R, inp2: R) -> R`
    """

    __require__ = ('complement', 'halfspace', 'intersect')

    def __init__(self, **bounds: list[int]) -> None:
        self.bounds = bounds

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        s = impl.complement(impl.empty())
        for name, (vmin, vmax) in self.bounds.items():
            i = impl.axis(name)
            if vmin is Ellipsis:
                assert vmax is not Ellipsis, f'Invalid bounds for axis {impl.axis_name(i)}, there must be either an upper or lower bound.'
                upper_bound = impl.halfspace(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                axis_range = upper_bound
            elif vmax is Ellipsis:
                assert vmin is not Ellipsis, f'Invalid bounds for axis {impl.axis_name(i)}, there must be either an upper or lower bound.'
                lower_bound = impl.halfspace(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                axis_range = lower_bound
            elif impl.axis_is_periodic(i) and vmax < vmin:
                upper_bound = impl.halfspace(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                lower_bound = impl.halfspace(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                axis_range = impl.complement(impl.intersect(upper_bound, lower_bound))
            else:
                # NOTE: See similar assertion in TVHJImpl's halfspace
                amin, amax = impl.axis_bounds(i)
                assert amin < vmin < amax, f'For dimension "{name}", {amin} < {vmin=} < {amax}. Use Ellipsis (...) to indicate subset stretching to the space boundary.'
                assert amin < vmax < amax, f'For dimension "{name}", {amin} < {vmax=} < {amax}. Use Ellipsis (...) to indicate subset stretching to the space boundary.'
                upper_bound = impl.halfspace(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                lower_bound = impl.halfspace(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                axis_range = impl.intersect(upper_bound, lower_bound)
            s = impl.intersect(s, axis_range)
        return s


## ## ## ## ## ## ## ## ## ##
## User-friendly Operations

def Compl[R](*args: SetBuilder[R]) -> SetBuilder[R]:
    """Return complement of a builder via Impl.complement."""
    return AppliedSet('complement', *args)

def Inter[R](*args: SetBuilder[R]) -> SetBuilder[R]:
    """Return intersection of builders via Impl.intersect."""
    return AppliedSet('intersect', *args)

def Union[R](*args: SetBuilder[R]) -> SetBuilder[R]:
    """Return union of builders via Impl.union."""
    return AppliedSet('union', *args)
