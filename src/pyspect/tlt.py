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
    - Primitives are passed to TLT(..., primitives=...) and stored on each node.
    - Realization checks requirements (operations and bound props) and then calls
      the builder with the selected Impl.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import wraps
from typing import Optional, Callable

from .impls.dev.base import *
from .set_builder import *
from .logics import *
from .utils import *

__all__ = (
    'TLT',
    'TLTLike',
    'Identity',
    'APPROXDIR',
    'TLTDiagnosticError',
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

class TLTDiagnosticError(Exception):
    """Raised when a TLT cannot be realized."""

class TLT[R](ImplClient[R]):
    """Temporal Logic Tree node parameterized by a concrete set type R.

    A TLT encapsulates a tuple-based logic formula, an implementation-agnostic
    set builder realizing its semantics, an approximation direction, and a
    mapping from free proposition names to bound SetBuilders.

    Construction:
        - TLT(arg, primitives=..., **where) dispatches based on the type of `arg`:
            str          -> proposition
            SetBuilder   -> constant set, assigned a unique proposition name
            tuple(TLExpr)-> formula; recursively constructs children via primitives
            TLT          -> existing tree, optionally updated with new bindings

    Realization:
        - Call .realize(impl) to obtain an R from an implementation.
        - Call .explain(impl) for a diagnostic report with tree visualization.
        - .assert_realizable() / .realize() raise TLTDiagnosticError on failure.
    """

    __primitives__ = idict({}) # default when primitives= is omitted

    @classmethod
    def select(cls, primitives):
        """Set class-default primitives (prefer TLT(..., primitives=...) instead)."""
        cls.__primitives__ = primitives

    @classmethod
    def _pop_primitives(cls, arg: TLTLike[R], kwds: TLTLikeMap[R]) -> idict:
        if '_primitives' in kwds:
            return kwds['_primitives']
        if 'primitives' in kwds:
            return kwds.pop('primitives')
        if isinstance(arg, TLT):
            return arg.primitives
        return cls.__primitives__

    @classmethod
    def construct(cls, arg: TLTLike[R], /, *, primitives=..., **kwds: TLTLike[R]) -> TLT[R]:
        """Explicit constructor alias to apply usual dispatch rules."""
        if primitives is not ...:
            kwds = dict(kwds, primitives=primitives)
        return cls(arg, **kwds)

    def __new__(cls, arg: TLTLike[R], **kwds: TLTLike[R]) -> TLT[R]:
        """Construct a TLT by dispatching on the kind of `arg`."""
        if 'where' in kwds:
            kwds |= kwds.pop('where')
        kwds['_primitives'] = cls._pop_primitives(arg, kwds)
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
        prim = kwds.get('_primitives', tlt.primitives)
        kwds = {prop: kwds[prop] if prop in kwds else sb
                for prop, sb in tlt._setmap.items()
                if prop in kwds or sb is not None}
        kwds['_primitives'] = prim
        return cls.__new_from_formula__(tlt._formula, **kwds)

    @classmethod
    @TLTDebugger.wrap('prop')
    def __new_from_prop__(cls, prop: str, **kwds: TLTLike[R]) -> TLT[R]:
        """Create a TLT from a proposition name, optionally binding it from kwds."""
        if prop in kwds:
            child = cls(kwds.pop(prop), **kwds)
            child._prop_name = prop
            return child
        self = cls.__new_init__(
            prop, ReferredSet(prop), setmap={prop: None},
            primitives=kwds.get('_primitives', ...),
        )
        self._prop_name = prop
        return self

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
        self = cls.__new_init__(
            formula, ReferredSet(name), setmap={name: sb},
            primitives=kwds.get('_primitives', ...),
        )

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

        prim = kwds.get('_primitives', cls.__primitives__)

        assert head in prim, \
            f'Unknown operator `{head}` in formula `{formula}`. ' \
            f'Available operators: {prim.keys()}'

        args = [cls(arg, **kwds) for arg in tail]  # make TLTs of formula args
        node = prim[head](*args)
        node.primitives = prim
        return node

    @classmethod
    def __new_init__(cls, formula=..., builder=..., approx=..., setmap=..., children=..., primitives=...):
        """Initialize a bare TLT instance with the provided internal fields."""
        self = super(TLT, cls).__new__(cls)

        # Lock the selected language for this instance
        self.primitives = primitives if primitives is not ... else cls.__primitives__

        self._formula = formula if formula is not ... else '_0'
        
        # If constructed with the absurd set, then the TLT is also absurd, i.e. cannot be realized.
        self._builder = builder if builder is not ... else ABSURD
        
        self._approx = approx if approx is not ... else APPROXDIR.EXACT
        
        # Sets are associated with names using ReferredSets.
        self._setmap = setmap if setmap is not ... else idict()
        self._children = children if children is not ... else tuple()
        self._prop_name: str | None = None

        self.inherit_requirements(self._builder)

        return self

    _formula: TLExpr
    _builder: SetBuilder[R]
    _approx: APPROXDIR
    _setmap: SetMap[R]
    _children: tuple['TLT[R]', ...]
    _prop_name: str | None

    @staticmethod
    def _child_path(parent_path: str, i: int, n: int) -> str:
        if n == 2:
            return f'{parent_path}.{"left" if i == 0 else "right"}'
        return f'{parent_path}.arg{i}'

    def _walk(self, path: str = 'root'):
        """Visit each node with a path label (root.left, root.right, …)."""
        yield self, path
        n = len(self._children)
        for i, child in enumerate(self._children):
            yield from child._walk(self._child_path(path, i, n))

    def _tree_name(self) -> str:
        """User-facing node name (proposition name when known)."""
        if self._prop_name is not None:
            return self._prop_name
        if isinstance(self._formula, str):
            if not self._formula.startswith('_'):
                return self._formula
            sb = self._setmap.get(self._formula)
            if sb is not None and hasattr(sb, 'arg'):
                return str(sb.arg)
            if sb is not None:
                return type(sb).__name__
            return 'set'
        return self._formula[0].replace('F-', '')

    def _tree_label(self) -> str:
        """Short label for tree visualization (operator or proposition/set name)."""
        if isinstance(self._formula, tuple):
            head = self._formula[0].replace('F-', '')
            if not self._children:
                return head
            args = ', '.join(c._tree_name() for c in self._children)
            return f"{head}({args})"
        name = self._tree_name()
        if isinstance(self._formula, str) and self._formula.startswith('_'):
            sb = self._setmap.get(self._formula)
            if sb is not None and not hasattr(sb, 'arg'):
                detail = type(sb).__name__
                if detail != name:
                    return detail
        return name

    def _tree_viz(self, error_paths: set[str]) -> list[str]:
        rows: list[tuple[str, str]] = []

        def right_col(node: 'TLT[R]', path: str) -> str:
            if path in error_paths:
                return '<= error here'
            if node._approx is APPROXDIR.INVALID:
                return '...'
            return node._approx.name

        def emit(node: 'TLT[R]', path: str, prefix: str, is_last: bool, is_root: bool, side: str = ''):
            if is_root:
                left = node._tree_label()
            else:
                branch = ('└─ ' if is_last else '├─ ') + side
                left = f"{prefix}{branch}{node._tree_label()}"
            rows.append((left, right_col(node, path)))
            n = len(node._children)
            ext = '' if is_root else prefix + ('    ' if is_last else '│   ')
            for i, child in enumerate(node._children):
                name = child._tree_name()
                label = child._tree_label()
                child_side = '' if label == name else f"{name}: "
                emit(child, self._child_path(path, i, n), ext, i == n - 1, False, child_side)

        emit(self, 'root', '', True, True)
        width = max(len(left) for left, _ in rows) if rows else 0
        return [f"{left.ljust(width)}  {right}" for left, right in rows]

    @staticmethod
    def _sb_leaf_message(exc: Exception) -> str:
        msg = str(exc)
        while 'received: ' in msg:
            _, msg = msg.split('received: ', 1)
        return msg.strip()

    @staticmethod
    def _sb_arg_chain(exc: Exception) -> list[int]:
        return [int(i) for i in re.findall(r'on argument (\d+)', str(exc))]

    def _localize_sb_error(
        self,
        node: 'TLT[R]',
        path: str,
        exc: Exception,
        impl: Impl[R],
        setmap: SetMap[R],
    ) -> tuple[str, str]:
        chain = self._sb_arg_chain(exc)
        if chain:
            cur, cur_path = node, path
            for i in chain:
                n = len(cur._children)
                if i >= n:
                    break
                cur = cur._children[i]
                cur_path = self._child_path(cur_path, i, n)
            return cur_path, self._sb_leaf_message(exc)

        for i, child in enumerate(node._children):
            n = len(node._children)
            child_path = self._child_path(path, i, n)
            try:
                child._builder(impl, **setmap)
            except Exception as e:
                return self._localize_sb_error(child, child_path, e, impl, setmap)
        return path, self._sb_leaf_message(exc)

    def _format_diagnostic_report(
        self,
        issues: list[str],
        error_paths: set[str],
        *,
        tree: bool = False,
    ) -> str:
        lines = ['TLT diagnostic report:', *issues]
        if tree:
            lines.extend(['', 'Tree:'])
            lines.extend(self._tree_viz(error_paths))
        return '\n'.join(lines)

    def _explain_report(self, impl: Optional[Impl[R]] = None, *, tree: bool = False, probe_sb: bool = True) -> str:
        """Build the diagnostic report string; include tree visualization when tree=True."""
        issues: list[str] = []
        error_paths: set[str] = set()

        for name in self.iter_free():
            path = next((p for node, p in self._walk() if node._formula == name), 'root')
            error_paths.add(path)
            issues.append(f"  [MISSING_PROPOSITION_BINDING] at {path}\n  Proposition '{name}' is not bound.")

        if impl is not None:
            for node, path in self._walk():
                for op in node._builder.__require__:
                    if not hasattr(impl, op):
                        head = node._formula[0] if isinstance(node._formula, tuple) else node._formula
                        error_paths.add(path)
                        issues.append(
                            f"  [MISSING_IMPL_OPERATION] at {path}\n"
                            f"  {type(impl).__name__} missing '{op}' (needs '{head}')."
                        )

        def deepest(node: 'TLT[R]', path: str) -> Optional[tuple['TLT[R]', str]]:
            if node._approx is not APPROXDIR.INVALID:
                return None
            n = len(node._children)
            for i, child in enumerate(node._children):
                if found := deepest(child, self._child_path(path, i, n)):
                    return found
            return node, path

        if bad := deepest(self, 'root'):
            node, path = bad
            error_paths.add(path)
            tags = ', '.join(c._approx.name for c in node._children)
            head = node._formula[0] if isinstance(node._formula, tuple) else node._formula
            issues.append(f"  [INVALID_APPROX_COMPOSITION] at {path}\n  '{head}' incompatible approximations ({tags}).")

        for node, path in self._walk():
            if node._builder is ABSURD:
                error_paths.add(path)
                issues.append(f"  [SET_BUILDER_ERROR] at {path}\n  Cannot realize the absurd set.")

        if probe_sb and impl is not None and not issues:
            try:
                self._builder(impl, **self._setmap)
            except Exception as e:
                sb_path, sb_msg = self._localize_sb_error(self, 'root', e, impl, self._setmap)
                error_paths.add(sb_path)
                issues.append(f"  [SET_BUILDER_ERROR] at {sb_path}\n  {sb_msg}")

        if not issues:
            return 'No issues detected.'

        return self._format_diagnostic_report(issues, error_paths, tree=tree)

    def explain(self, impl: Optional[Impl[R]] = None) -> None:
        """Print a diagnostic report (with tree) for this TLT."""
        print(self._explain_report(impl, tree=True))
    
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
            - a SetBuilder fails at realization

        memoize=True is reserved for a future optimization where realized sets
        may be cached on the node.
        """
        self.assert_realizable(impl)
        try:
            out = self._builder(impl, **self._setmap)
        except Exception as e:
            sb_path, sb_msg = self._localize_sb_error(self, 'root', e, impl, self._setmap)
            report = self._format_diagnostic_report(
                [f"  [SET_BUILDER_ERROR] at {sb_path}\n  {sb_msg}"],
                {sb_path},
            )
            raise TLTDiagnosticError(report) from e
        if memoize:
            raise NotImplementedError() # TODO: builder = Set(out)
        return out

    def assert_realizable(self, impl: Optional[Impl[R]] = None) -> None:
        """Validate TLT-level realizability; raises TLTDiagnosticError on failure."""
        report = self._explain_report(impl, tree=False, probe_sb=False)
        if report != 'No issues detected.':
            raise TLTDiagnosticError(report)

    def is_realizable(self, impl: Optional[Impl[R]] = None) -> bool:
        """Return True if this TLT can be realized against `impl`."""
        try:
            self.assert_realizable(impl)
            if impl is not None:
                self._builder(impl, **self._setmap)
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
