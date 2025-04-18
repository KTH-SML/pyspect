######################################################################
## TLT Primitives

from functools import wraps
from typing import Tuple

from .set_builder import *
from .logics import *
from .tlt import TLT, primitive, APPROXDIR

from .idict import idict

def define(formula, equiv):
    assert (x := get_malformed(formula)) is None, f"Formula must be well-formed, not {x}"
    assert all(isinstance(arg, str) for arg in formula[1:]), "Arguments of formula must be propositions (string)"
    assert (x := get_malformed(equiv)) is None, f"Equivalent formula must be well-formed, not {x}"
    assert get_props(formula) == get_props(equiv), "Formula and equivalent must have the same set of propositions"

    head, *tail = formula

    func = lambda *args: TLT(equiv, **{name: arg for name, arg in zip(tail, args)})

    return idict({head: func})

## ## ## ## ## ## ## ## ## ##
## Propositional Fragments

@primitive(FALSIFY())
def Falsify() -> Tuple[SetBuilder, APPROXDIR]:
    return (
        EMPTY,
        APPROXDIR.EXACT,
    )

@primitive(NOT('_1'))
def Not(_1: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
    b1, a1 = _1._builder, _1._approx
    return (
        AppliedSet('complement', b1), 
        -1 * a1,
    )

@primitive(AND('_1', '_2'))
def And(_1: 'TLTLike', _2: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
    b1, a1 = _1._builder, _1._approx
    b2, a2 = _2._builder, _2._approx
    return (
        AppliedSet('intersect', b1, b2),
        APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else
        APPROXDIR.INVALID if APPROXDIR.UNDER in (a1, a2) else
        APPROXDIR.OVER,
    )

@primitive(OR('_1', '_2'))
def Or(_1: 'TLTLike', _2: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
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


Propositional = Falsify | Not | And | Or

# Minus = define(
#     MINUS('lhs', 'rhs'),
#     AND('lhs', NOT('rhs')),
# )

# Implies = define(
#     IMPLIES('lhs', 'rhs'),
#     OR(NOT('lhs'), 'rhs'),
# )

## ## ## ## ## ## ## ## ## ##
## Temporal Logic Fragments

@primitive(NEXT('_1'))
def Next(_1: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
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
def Until(_1: 'TLTLike', _2: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
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

Eventually = define(EVENTUALLY('_1'), UNTIL(NOT(FALSIFY()), '_1'))

# Continuous LTL
LTLc = Propositional | Until | Eventually # | Always

# Discrete LTL
LTLd = LTLc | Next
