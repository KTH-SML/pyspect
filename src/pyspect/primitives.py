"""Primitives for temporal logic trees (TLTs).

This module defines:
- primitive: a decorator class to register TLT primitives tied to tuple-form formulas
- Derived connectives built from equivalences (e.g., Minus, Implies, Eventually)
- Different sets of primitive TLTs for propositional and temporal operators

Concepts:
- A primitive is specified by a TLExpr (operator head and argument names).
  Decorating a factory function with @primitive(("OP", "x", "y")) binds the function
  to that operator and returns a single-element set of a primitive TLT. Union of
  such sets forms the underlying grammar/logic fragment.
- A decorated factory should return (SetBuilder[R], APPROXDIR) which specify how 
  to realize the set semantics and the approximation direction the primitive contributes.
  The decorator constructs a TLT by:
  - Substituting child formulas into the operator head
  - Composing the children's set maps
  - Propagating requirements/constraints
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

from .set_builder import *
from .logics import *
from .tlt import *

from .utils.idict import idict

__all__ = (
    'primitive',
    # Propositional
    'Falsify', 'Not', 'And', 'Or', 'Minus', 'Implies',
    'PropositionalAtomic', 'Propositional',
    # Continuous LTL
    'Next', 'Until', 'Eventually',
    'ContLTLAtomic', 'ContLTL',
    # Discrete LTL
    'DiscLTLAtomic', 'DiscLTL',
    'LTL',
)

class primitive[R, **P]:
    """Decorator class for declaring TLT primitives.

    Usage:
    - Specify the operator form and argument names:
      ```
      @primitive(AND('_1', '_2'))
      def And(_1: TLTLike[R], _2: TLTLike[R]) -> tuple[SetBuilder[R], APPROXDIR]:
          ...
      ```
      The decorator registers a callable under the operator head ("AND") that,
      when invoked with TLTLike arguments, returns a constructed TLT[R]. The
      arguments to the decorated function are mapped by name from the provided
      operands by order as they appear in the formula. The formula must consist
      of only a single connective.

    Notes:
    - The decorated function receives named arguments matching the formula's
      parameter names (strings in the TLExpr tail) and should return the pair
      (SetBuilder[R], APPROXDIR) describing the primitive's semantics and its
      approximation contribution.
    """

    type PrimitiveFunc = Callable[P, TLT[R]]
    type PrimitiveFactory = Callable[P, tuple[SetBuilder[R], APPROXDIR]]

    formula: TLExpr
    head: str
    tail: tuple[TLExpr, ...]

    def __init__(self, formula: TLExpr):
        """Bind this primitive to a formula shape.

        Args:
          formula: a TLExpr operator tuple whose head is the operator name and
                   whose tail elements are unique proposition names (str)
        """
        assert isinstance(formula, tuple), "Formula must be a tuple"
        assert isinstance(formula[0], str), "Operator name of formula must be a string"
        assert all(isinstance(arg, str) for arg in formula[1:]), "Arguments of formula must be propositions (string) for primitives"
        head, *tail = formula
        assert len(tail) == len(set(tail)), "Argument names must be unique"
        self.formula = formula
        self.head = head
        self.tail = tail

    def __call__(self, func: PrimitiveFactory) -> idict[str, PrimitiveFunc]:
        """Decorator entry: wrap `func` to build a TLT node for this primitive.

        The wrapped function:
        - Validates/assembles the operator formula with the child formulas
        - Calls `func` with named arguments mapped from the provided children
        - Constructs a TLT using the returned (SetBuilder, APPROXDIR)
        - Merges children's set maps and inherits requirements

        Returns:
            idict mapping {connective: function}
        """

        @wraps(func)
        def wrapper(*args: P.args) -> TLT[R]:
            formula = (self.head, *[arg._formula if isinstance(arg, TLT) else arg for arg in args])
            builder, approx = func(**{name: arg for name, arg in zip(self.tail, args)})
            setmap = idict(sum([list(arg._setmap.items()) for arg in args if isinstance(arg, TLT)], []))
            tree = TLT.__new_init__(formula, builder, approx, setmap)
            tree.inherit_requirements(*args)
            return tree
        
        return idict({self.head: wrapper})

    def define_as(self, equiv: TLExpr) -> idict[str, Callable[..., TLT]]:
        """Define this operator via a pure logical equivalence.

        This produces a thin wrapper that expands calls to this operator into
        the provided equivalent TLExpr, without supplying a new SetBuilder.

        Constraints:
        - `equiv` must be well-formed
        - `equiv` must reference exactly the same set of proposition names as
          the original operator formula

        Returns:
            idict mapping {connective: expansion_function}
        """
        assert (x := get_malformed(equiv)) is None, f"Equivalent formula must be well-formed, not {x}"
        assert get_props(self.formula) == get_props(equiv), "Formula and equivalent must have the same set of propositions"

        func = lambda *args: TLT(equiv, **{name: arg for name, arg in zip(self.tail, args)})

        return idict({self.head: func})

## ## ## ## ## ## ## ## ## ##
## Propositional Fragments

@primitive(FALSIFY())
def Falsify[R]() -> tuple[SetBuilder[R], APPROXDIR]:
    """Bottom/falsehood (∅)."""
    return (
        EMPTY,
        APPROXDIR.EXACT,
    )

@primitive(NOT('_1'))
def Not[R](_1: TLTLike[R]) -> tuple[SetBuilder[R], APPROXDIR]:
    """Logical negation (complement)."""
    b1, a1 = _1._builder, _1._approx
    return (
        AppliedSet('complement', b1), 
        -1 * a1,
    )

@primitive(AND('_1', '_2'))
def And[R](_1: TLTLike[R], _2: TLTLike[R]) -> tuple[SetBuilder[R], APPROXDIR]:
    """Conjunction (intersection)."""
    b1, a1 = _1._builder, _1._approx
    b2, a2 = _2._builder, _2._approx
    return (
        AppliedSet('intersect', b1, b2),
        APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else
        APPROXDIR.INVALID if APPROXDIR.UNDER in (a1, a2) else
        APPROXDIR.OVER,
    )

@primitive(OR('_1', '_2'))
def Or[R](_1: TLTLike[R], _2: TLTLike[R]) -> tuple[SetBuilder[R], APPROXDIR]:
    """Disjunction (union)."""
    b1, a1 = _1._builder, _1._approx
    b2, a2 = _2._builder, _2._approx
    return (
        AppliedSet('union', b1, b2),
        APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else
        a1 if a1 == a2 else
        a2 if a1 == APPROXDIR.EXACT else
        a1 if a2 == APPROXDIR.EXACT else
        APPROXDIR.INVALID,
    )

# Only the atomic formulas for propositional logic
PropositionalAtomic = Falsify | Not | And | Or

# Difference: lhs ∧ ¬rhs
Minus = primitive(MINUS('lhs', 'rhs')).define_as(AND('lhs', NOT('rhs')))

# Implication: ¬lhs ∨ rhs
Implies = primitive(IMPLIES('lhs', 'rhs')).define_as(OR(NOT('lhs'), 'rhs'))

Propositional = PropositionalAtomic | Minus | Implies

## ## ## ## ## ## ## ## ## ##
## Temporal Logic Fragments

@primitive(NEXT('_1'))
def Next[R](_1: TLTLike[R]) -> tuple[SetBuilder[R], APPROXDIR]:
    """Next/predecessor set."""
    b1, a1 = _1._builder, _1._approx

    ao = APPROXDIR.EXACT # TODO
    assert ao != APPROXDIR.INVALID, 'Operator may never be inherently invalid'

    return (
        AppliedSet('pre', b1, Compl(EMPTY)),
        a1      if a1 == APPROXDIR.INVALID else
        ao + a1 if ao * a1 == APPROXDIR.EXACT else 
        a1      if ao == a1 else
        APPROXDIR.INVALID,
    )

@primitive(UNTIL('_1', '_2'))
def Until[R](_1: TLTLike[R], _2: TLTLike[R]) -> tuple[SetBuilder[R], APPROXDIR]:
    """Until (reachability): hold _1 until _2 is reached."""
    b1, a1 = _1._builder, _1._approx
    b2, a2 = _2._builder, _2._approx

    ao = APPROXDIR.UNDER # TODO: reach normally implement UNDER 
    assert ao != APPROXDIR.INVALID, 'Operator may never be inherently invalid'

    return (
        AppliedSet('reach', b2, b1),
        a2      if a2 == APPROXDIR.INVALID else
        ao + a2 if ao * a2 == APPROXDIR.EXACT else 
        a2      if ao == a2 else
        APPROXDIR.INVALID,
    )

# Atomic sets for continuous/discrete LTL fragments
ContLTLAtomic = PropositionalAtomic | Until
DiscLTLAtomic = ContLTLAtomic | Next

# Eventually φ ≡ (¬⊥) U φ
Eventually = primitive(EVENTUALLY('_1')).define_as(UNTIL(NOT(FALSIFY()), '_1'))

ContLTL = ContLTLAtomic | Propositional | Eventually
DiscLTL = DiscLTLAtomic | ContLTL
LTL = DiscLTL # Default to discrete-time LTL

