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
from functools import reduce
from warnings import deprecated

from .impls.dev.base import Impl, ImplClient
from .impls.dev.axes import Axis

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
    """Defer a call to a function `f` where `args` are realized builders.

    The function is specified by name `funcname` and looked up on the `Impl` at 
    realization, i.e. `Impl.<funcname>(*args)`, or a direct lambda if `func` is a callable.

    - Accumulates required `Impl` methods from children and adds `funcname`.
    - Propagates and de-duplicates children's free variables.
    - On realization, calls child builders first, then invokes the `Impl` method.
    - Wraps child exceptions to pinpoint which argument failed.
    """

    def __init__(self, func: str | callable, *builders: SetBuilder[R]) -> None:
        self.func = func
        self.builders = builders

        if isinstance(func, str):        
            _require = (func,)
        elif callable(func):
            _require = ()
        else:
            raise TypeError("Expected a string or callable for 'func'")


        for builder in self.builders:
            _require += builder.__require__
            self.free += tuple(name for name in builder.free if name not in self.free)

        self.add_requirements(_require)        

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        if callable(self.func):
            func = self.func
        else:
            try:
                func = getattr(impl, self.func)
            except AttributeError as e:
                raise AttributeError(f'Impl {impl.__class__.__name__} does not support "{self.func}".') from e
        
        args = []
        for i, sb in enumerate(self.builders):
            try:
                args.append(sb(impl, **m))
            except Exception as e:
                E = type(e)
                raise E(f'When applying "{self.func}" on argument {i}, received: {e!s}') from e
        
        return func(*args)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## User-friendly Primitives & Operations

def Compl[R](*args: SetBuilder[R]) -> SetBuilder[R]:
    """Return complement of a builder via Impl.complement."""
    return AppliedSet('complement', *args)

def Inter[R](*args: SetBuilder[R]) -> SetBuilder[R]:
    """Return intersection of builders via Impl.intersect."""
    return AppliedSet('intersect', *args)

def Union[R](*args: SetBuilder[R]) -> SetBuilder[R]:
    """Return union of builders via Impl.union."""
    return AppliedSet('union', *args)

class EmptySet[R](SetBuilder[R]):
    """Builder for the empty set.
    
    Requires:
        - `Impl.empty() -> R`
    """

    __require__ = ('empty',)

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        return impl.empty()
    
EMPTY: EmptySet = EmptySet()

## ## ## ## ## ## ## ##
## Linear Primitives

class PolytopeSet[R](SetBuilder[R]):
    """Polytope / polyhedral set as intersection of finitely many halfspaces.

    The set is defined by rows of `normals` and corresponding `offsets`, i.e.
    one halfspace per row. Geometrically, this realizes

        ⋂_i { x | n_i · (x - o_i) >= 0 }

    or whatever exact halfspace convention your `Impl.halfspace(...)` uses.

    Note:
        This builder does not check boundedness. So mathematically this is
        really a general polyhedral set; whether it is a true polytope depends
        on the supplied halfspaces.

    Parameters:
        normals: one normal vector per halfspace
        offsets: one offset point/vector per halfspace
        axes: axis indices (or str if using `AxesImpl`) in the `Impl`'s coordinate system
        kwds: forwarded to each `Impl.polytope` call

    Requires:
        - `Impl.polytope(normals, offsets, axes, ...) -> R`
    """

    __require__ = ('polytope',)

    def __init__(
        self,
        normals: list[list[float]],
        offsets: list[list[float]],
        axes: list[Axis],
        **kwds: Any,
    ) -> None:
        assert len(normals) > 0, "PolytopeSet requires at least one halfspace."
        assert len(normals) == len(offsets), "PolytopeSet requires one offset per normal."
        assert all(len(n) == len(axes) for n in normals), "Each normal must have same length as axes."
        assert all(len(o) == len(axes) for o in offsets), "Each offset must have same length as axes."
        self.normals = normals
        self.offsets = offsets
        self.axes = axes
        self.kwds = kwds

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        return impl.polytope(self.normals, self.offsets, [impl.axis(ax) for ax in self.axes], **self.kwds)

    @classmethod
    def halfspace(cls, axes: list[Axis], normal: list[float], offset: list[float], **kwds: Any) -> SetBuilder[R]:
        """Convenience constructor for a single halfspace."""
        raise NotImplementedError("PolytopeSet.halfspace is not implemented yet. Use HalfSpaceSet instead.")

    @classmethod
    def slab(cls, axes: list[Axis], normal: list[float], offset: list[float], width: float, **kwds: Any) -> SetBuilder[R]:
        """Convenience constructor for a slab between two parallel hyperplanes."""
        raise NotImplementedError("PolytopeSet.slab is not implemented yet. Use Inter(HalfSpaceSet(...), HalfSpaceSet(...)) instead.")    

    @classmethod
    def box(cls, **bounds: tuple[float, float]) -> SetBuilder[R]:
        """Convenience constructor for an axis-aligned box/hyper-cube."""
        raise NotImplementedError("PolytopeSet.box is not implemented yet. Use AlignedBoxSet instead.")
    
    @classmethod
    def polygon(
        cls, 
        order: int, 
        radius: float,
        axes: list[Axis],
        center: list[float] | None = None,
        basis: tuple[list[float], list[float]] | None = None,
        **kwds: Any,
    ) -> SetBuilder[R]:
        """Convenience constructor for a regular polygon in 2D."""
        raise NotImplementedError("PolytopeSet.polygon is not implemented yet. Use Inter(HalfSpaceSet(...), ...) instead.")
    

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

class AlignedBoxSet[R](SetBuilder[R]):
    """Axis-aligned box possibly unbounded on one side per axis.

    Bounds mapping: `name -> (vmin, vmax)`. Use Ellipsis to denote an open side
    (e.g., (0, ...) or (..., 1)). For periodic axes where `vmax < vmin`, the
    range wraps around.

    Example:
    ```
    # Left half-circle bounded in y.
    A = AlignedBoxSet(y=(-0.5, 0.5), theta=(+pi/2, -pi/2))
    ```

    Requires:
        - `Impl < AxesImpl`
        - `Impl.complement(inp: R) -> R`
        - `Impl.halfspace(normal, offset, axes, ...) -> R`
        - `Impl.intersect(inp1: R, inp2: R) -> R`
    """

    __require__ = ('complement', 'halfspace', 'intersect')

    @classmethod
    def relative(cls, anchor: list[int], size: list[int], axes: list[Axis]) -> SetBuilder[R]:
        assert len(anchor) == len(size) == len(axes), "Anchor, size, and axes must have the same length."
        bounds = {impl.axis_name(ax): (a, a + s)
                  for a, s, ax in zip(anchor, size, axes)}
        return cls(**bounds)

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

@deprecated("BoundedSet is renamed to AlignedBoxSet for clarity, please use AlignedBoxSet instead of BoundedSet.")
class BoundedSet[R](AlignedBoxSet[R]): ...

## ## ## ## ## ## ## ## ##
## Quadratic Primitives

class QuadricSet[R](SetBuilder[R]):
    
    @classmethod
    def ball(cls):
        raise NotImplementedError("QuadricSet.ball is not implemented yet. Use BallSet instead.")

    @classmethod
    def ellipsoid(cls):
        raise NotImplementedError("QuadricSet.ellipsoid is not implemented yet. Use Inter(HalfSpaceSet(...), ...) instead.")
    
    @classmethod
    def cylinder(cls):
        raise NotImplementedError("QuadricSet.cylinder is not implemented yet. Use CylinderSet instead.")

class BallSet[R](SetBuilder[R]):
    """Ball in hyperspace.

    The ball is defined over a subset of axes, with Euclidean radius in that
    subspace. If the selected axes span the whole space, this is a full
    hypersphere; otherwise it is a lower-dimensional ball embedded in the space.

    Parameters:
        center: center coordinates along the selected axes
        radius: ball radius
        axes: axis indices (or str if using `AxesImpl`) spanning the ball
        kwds: forwarded to `Impl.ball`

    Example:
    ```
    # 3D ball in x,y,z subspace
    B = BallSet(center=[0.0, 0.0, 0.0], radius=2.0, axes=['x', 'y', 'z'])
    ```

    Requires:
        - `Impl.ball(center, radius, axes, ...) -> R`
    """

    __require__ = ('ball',)

    def __init__(
        self,
        center: list[float],
        radius: float,
        axes: list[Axis],
        **kwds: Any,
    ) -> None:
        assert len(axes) == len(center)
        assert radius >= 0
        self.center = center
        self.radius = radius
        self.axes = axes
        self.kwds = kwds

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        return impl.ball(
            center=self.center,
            radius=self.radius,
            axes=[impl.axis(ax) for ax in self.axes],
            **self.kwds,
        )

class CylinderSet[R](SetBuilder[R]):
    """Cylinder in hyperspace.

    The cylinder is defined over a subset of axes, with Euclidean radius in that
    subspace, and extends freely along all remaining axes.

    Parameters:
        center: center coordinates along the selected axes
        radius: cylinder radius
        vector: direction vector along which the cylinder extends
        kwds: forwarded to `Impl.cylinder`

    Example:
    ```
    # Infinite cylinder along all axes except x,y
    C = CylinderSet(center=[0.0, 0.0], radius=1.0, vector=[0.0, 1.0], axes=['x', 'y'])
    ```

    Requires:
        - `Impl.cylinder(center, radius, vector, axes, ...) -> R`
    """

    __require__ = ('cylinder',)

    def __init__(
        self,
        center: list[float],
        radius: float,
        vector: list[float],
        **kwds: Any,
    ) -> None:
        assert len(center) == len(vector)
        assert radius >= 0
        self.center = center
        self.radius = radius
        self.vector = vector
        self.kwds = kwds

    def __call__(self, impl: Impl[R], **m: SetBuilder[R]) -> R:
        return impl.cylinder(
            center=self.center,
            radius=self.radius,
            vector=self.vector,
            **self.kwds,
        )
