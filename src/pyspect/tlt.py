"""Temporal Logic Trees (TLTs).

This module wires tuple-encoded temporal logic formulas (see pyspect.logics)
to implementation-agnostic set builders (see pyspect.set_builder) and provides
a small runtime for constructing, combining, and realizing TLTs against a
concrete implementation Impl[R] (see pyspect.impls.*).

Key ideas:
    - A TLT wraps:
        - _formula: a TLExpr made of tuples and proposition names
        - _builder: a SetBuilder[R] describing the set semantics
        - _approx: an APPROXDIR flag describing the approximation direction
        - _setmap: a mapping from proposition names to SetBuilder bindings
    - Primitives are registered on the class via
      TLT.select(...) and used by __new_from_formula__ when building trees.
    - Realization checks requirements (operations and bound props) and then calls
      the builder with the selected Impl.
"""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Optional, Callable

from .impls.base import *
from .set_builder import *
from .logics import *
from .utils import *

__all__ = (
    'TLT',
    'TLTLike',
    'Identity',
    'APPROXDIR',
    'TLTDebugger',
)

class APPROXDIR(Enum):
    """Approximation direction tag for a TLT node.

    Values:
        - INVALID: semantics cannot be guaranteed (construction error)
        - UNDER: under-approximation (subset of the exact set)
        - EXACT: exact semantics
        - OVER: over-approximation (superset of the exact set)

    The enum supports + and * with either another APPROXDIR or an int with the
    conventional meaning used by the primitive combination rules.
    """
    INVALID = None
    UNDER = -1
    EXACT = 0
    OVER = +1

    def __str__(self):
        return f'{self.name}'

    def __radd__(self, other): return self.__add__(other)

    def __add__(self, other: 'APPROXDIR | int') -> 'APPROXDIR':
        """Combine approximation directions additively, propagating INVALID."""
        lhs = self.value
        rhs = other.value if isinstance(other, APPROXDIR) else other
        return APPROXDIR(None if None in (lhs, rhs) else
                         lhs + rhs)

    def __rmul__(self, other): return self.__mul__(other)

    def __mul__(self, other: 'APPROXDIR | int') -> 'APPROXDIR':
        """Combine approximation directions multiplicatively, propagating INVALID."""
        lhs = self.value
        rhs = other.value if isinstance(other, APPROXDIR) else other
        return APPROXDIR(None if None in (lhs, rhs) else
                         lhs * rhs)

class TLTDebugger:
    """Internal helper to add lightweight debugging and tracing to TLT building.

    Use TLT.debug(print=True, indent=2) to enable, and TLT.nodebug() to reset.
    Arbitrary keyword args starting with '__debug' are collected and forwarded.
    """

    __debug = {}
    __debug_defaults = {'print': False, 'indent': 0}

    @classmethod
    def nodebug(cls):
        """Reset debug options to defaults for subsequent constructions."""
        for k in list(cls.__debug):
            cls.__debug.pop(k, None) # remove to set default

    @classmethod
    def debug(cls, **kwds):
        """Set debug options.

        Supported options (with prefix '__debug_' accepted on wrapper):
            - print: bool, enable/disable printing
            - indent: int, indentation level increment per nested step

        Pass Ellipsis to unset a debug option.
        """
        cls.__debug.update(collect_prefix(kwds, '__debug_'))
        for k, v in cls.__debug.items():
            if v is Ellipsis:
                cls.__debug.pop(k, None) # remove to set default

    @classmethod
    def wrap(cls, msg: str):
        """Decorator to print debug information for TLT construction.

        - Enable with `TLT.debug(print=True)`.
        - Choose indentation size with `TLT.debug(indent=...)`.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwds):

                dbg = cls.__debug_defaults | collect_prefix(kwds, '__debug', remove=True)

                if dbg['print']:
                    print('> ' * dbg['indent'] + msg.ljust(7), kwds)
                    dbg['indent'] = dbg['indent'] + 1

                kwds |= prefix_keys(dbg, '__debug_')
                return func(*args, **kwds)
            return wrapper
        return decorator

type SetMap[R] = idict[str, Optional[SetBuilder[R]]]
type TLTLike[R] = TLExpr | TLT[R]
type TLTLikeMap[R] = dict[str, TLTLike[R]]

class TLT[R](ImplClient[R]):
    """Temporal Logic Tree node parameterized by a concrete set type R.

    A TLT encapsulates a tuple-based logic formula, an implementation-agnostic
    set builder realizing its semantics, an approximation direction, and a
    mapping from free proposition names to bound SetBuilders.

    Construction:
        - TLT.select(primitives) must be called to choose available operators.
        - TLT(arg, **where) dispatches based on the type of `arg`:
            str          -> proposition
            SetBuilder   -> constant set, assigned a unique proposition name
            tuple(TLExpr)-> formula; recursively constructs children via primitives
            TLT          -> existing tree, optionally updated with new bindings

    Realization:
        - Call .realize(impl) to obtain an R from an implementation.
        - .assert_realizable() checks missing props/ops and approximation validity.
    """

    __primitives__ = idict({}) # Null set of primitives

    @classmethod
    def select(cls, primitives):
        """Select the set of primitive operator implementations for this class."""
        cls.__primitives__ = primitives

    @classmethod
    def construct(cls, arg: TLTLike[R], **kwds: TLTLike[R]) -> TLT[R]:
        """Explicit constructor alias to apply usual dispatch rules."""
        return cls(arg, **kwds)

    def __new__(cls, arg: TLTLike[R], **kwds: TLTLike[R]) -> TLT[R]:
        """Construct a TLT by dispatching on the kind of `arg`."""
        if 'where' in kwds: kwds |= kwds.pop('where')
        return (cls.__new_from_tlt__(arg, **kwds)      if isinstance(arg, TLT) else
                cls.__new_from_prop__(arg, **kwds)     if isinstance(arg, str) else
                cls.__new_from_builder__(arg, **kwds)  if isinstance(arg, Callable) else
                cls.__new_from_formula__(arg, **kwds))

    @classmethod
    @TLTDebugger.wrap('tlt')
    def __new_from_tlt__(cls, tlt: TLT[R], **kwds: TLTLike[R]) -> TLT[R]:
        """Rebuild a TLT from an existing one, optionally rebinding free props."""
        # Create new TLT with updated information. With this, we can bind new 
        # TLT-like objects to free variables/propositions. We do not allow 
        # updates to existing propositions. First collect all updatable props.
        kwds = {prop: kwds[prop] if prop in kwds else sb
                for prop, sb in tlt._setmap.items()
                if prop in kwds or sb is not None}

        # It is best to reconstruct builders in the "top-level" TLT when there
        # is updates to the tree since we have modified how 
        return cls.__new_from_formula__(tlt._formula, **kwds)

    @classmethod
    @TLTDebugger.wrap('prop')
    def __new_from_prop__(cls, prop: str, **kwds: TLTLike[R]) -> TLT[R]:
        """Create a TLT from a proposition name, optionally binding it from kwds."""
        return (cls(kwds.pop(prop), **kwds) if prop in kwds else
                cls.__new_init__(prop, ReferredSet(prop), setmap={prop: None}))

    @classmethod
    @TLTDebugger.wrap('builder')
    def __new_from_builder__(cls, sb: SetBuilder, **kwds: TLTLike[R]) -> TLT[R]:
        """Create a TLT from a constant SetBuilder, assigning it a unique name."""
        # Assume a formula "_0" where the set exactly represent the prop "_0".
        # In reality, we define a unique ID `uid` instead of "_0". We add one
        # extra level of indirection with a ReferredSet for `uid` and 
        # letting `setmap` hold the actual set builder `sb`. This way, the root
        # TLT will hold a full formula that refers to even constant sets
        # (information is not lost as when internally binding constant sets to
        # builder functions).
        name = f'_{sb.uid}'
        formula = name # UID is a proposition
        self = cls.__new_init__(formula, ReferredSet(name), setmap={name: sb})

        self.inherit_requirements(sb)
        
        return self

    @classmethod
    @TLTDebugger.wrap('formula')
    def __new_from_formula__(cls, formula: 'TLExpr', **kwds: TLTLike[R]) -> TLT[R]:
        """Create a TLT from a TLExpr by invoking registered primitives."""
        if isinstance(formula, str):
            # If formula is a string, it is a proposition
            return cls.__new_from_prop__(formula, **kwds)
        
        # Otherwise, it is a non-trivial formula
        head, *tail = formula

        assert head in cls.__primitives__, \
            f'Unknown operator `{head}` in formula `{formula}`. ' \
            f'Available operators: {cls.__primitives__.keys()}'

        args = [cls(arg, **kwds) for arg in tail]  # make TLTs of formula args
        return cls.__primitives__[head](*args)

    @classmethod
    def __new_init__(cls, formula=..., builder=..., approx=..., setmap=..., times=...):
        """Initialize a bare TLT instance with the provided internal fields."""
        self = super(TLT, cls).__new__(cls)

        # Lock the selected language for this instance
        self.__primitives__ = self.__primitives__

        self._formula = formula if formula is not ... else '_0'
        
        # If constructed with the absurd set, then the TLT is also absurd, i.e. cannot be realized.
        self._builder = builder if builder is not ... else ABSURD
        
        self._approx = approx if approx is not ... else APPROXDIR.EXACT
        
        # Sets are associated with names using ReferredSets.
        self._setmap = setmap if setmap is not ... else idict()

        self.inherit_requirements(self._builder)

        return self

    _formula: TLExpr
    _builder: SetBuilder[R]
    _approx: APPROXDIR
    _setmap: SetMap[R]
    
    def __repr__(self) -> str:
        cls = type(self).__name__
        approx = str(self._approx)
        formula = str(self._formula)
        return f'{cls}({approx}, {formula})'

    def realize(self, impl: Impl[R], memoize: bool = False) -> R:
        """Realize this TLT into a concrete set R using `impl`.

        Raises if:
            - some proposition is unbound
            - required Impl operations are missing
            - approximation is INVALID

        memoize=True is reserved for a future optimization where realized sets
        may be cached on the node.
        """
        self.assert_realizable(impl)
        out = self._builder(impl, **self._setmap)
        if memoize:
            raise NotImplementedError() # TODO: builder = Set(out)
        return out

    def assert_realizable(self, impl: Optional[Impl[R]] = None) -> None:
        """Validate that this TLT can be realized, optionally against `impl`."""
        for name, sb in self._setmap.items():
            if sb is None: 
                raise Exception(f'Missing proposition `{name}`')
        if impl is not None and bool(missing := self.missing_ops(impl)):
            raise Exception(f'Missing from implementation: {", ".join(missing)}')
        if self._approx is APPROXDIR.INVALID:
            raise Exception('Invalid approximation. TLT operational semantics of formula does not hold.')

    def is_realizable(self, impl: Optional[Impl[R]] = None) -> bool:
        """Return True if assert_realizable would succeed."""
        try:
            self.assert_realizable(impl)
        except Exception:
            return False
        else:
            return True
    
    def iter_frml(self, formula: Optional[TLExpr] = None, **kwds):
        """Yield sub-formulas in post-order; optionally only terminals.

        Parameters:
            - formula: root to traverse (defaults to self._formula)
            - only_terminals: if True, yield only terminal sub-formulas
        """
        only_terminals = kwds.get('only_terminals', False)
        if formula is None:
            formula = self._formula
        _, *args = formula
        for arg in args:
            yield from self.iter_frml(arg, **kwds)
        if not (only_terminals and args):
            yield formula

    def iter_free(self):
        """Iterate names of free (unbound) propositions in this TLT."""
        yield from filter(lambda p: self._setmap[p] is None, self._setmap)


def Identity[R](arg: TLTLike[R]) -> TLT[R]:
    """Identity helper that constructs a TLT from any TLTLike input."""
    return TLT(arg)
