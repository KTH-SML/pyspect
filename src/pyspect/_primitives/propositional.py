######################################################################
## TLT Primitives

from functools import wraps

from ..set_builder import *
from ..impls.base import *
from ..tlt import *
from .base import *
from .tools import *

__all__ = (
    'Not', 'And', 'Or',
    'Falsify', 'Truthify',
    'Minus', 'Implies',
    'Propositional',
)

def primitive(formula: FormulaExpr) -> TLT:
    """Decorator to declare a primitive TLT."""
    assert isinstance(formula, tuple), "Formula must be a tuple"
    assert isinstance(formula[0], str), "Operator name of formula must be a string"
    assert all(isinstance(arg, str) for arg in formula[1:]), "Arguments of formula must be propositions (string) when declaring primitives"
    op, *args = formula
    assert len(args) == len(set(args)), "Argument names must be unique"

    def decorator(func):

        @wraps(func)
        def wrapper(**kwds):
            kwds = {arg: kwds[arg] for arg in args}
            builder, approx = func(**kwds)
            setmap = idict(sum([list(branch._setmap.items()) for branch in kwds.values()], []))
            tree = TLT.__new_init__(formula, builder, approx, setmap)
            return idict({op: TLT.__new_init__(formula=formula, b=b, a=a)})
        return wrapper
    return decorator

## ## ## ## ## ## ## ## ## ##
## Propositional Fragments

@primitive(FALSIFY())
def Falsify() -> Tuple[SetBuilder, APPROXDIR]:
    raise NotImplementedError("Falsify is not implemented yet.")
    return (
        EMPTY,
        APPROXDIR.EXACT,
    )

@primitive(NOT('_1'))
def Not(_1: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
    b, a = _1._builder, _1._approxdir
    return (
        AppliedSet('complement', b), 
        -1 * a,
    )

@primitive(AND('_1', '_2'))
def And(_1: 'TLTLike', _2: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
    b1, a1 = _1._builder, _1._approxdir
    b2, a2 = _2._builder, _2._approxdir
    return (
        AppliedSet('intersect', b1, b2),
        APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else
        APPROXDIR.INVALID if APPROXDIR.UNDER in (a1, a2) else
        APPROXDIR.OVER,
    )

@primitive(OR('_1', '_2'))
def Or(_1: 'TLTLike', _2: 'TLTLike') -> Tuple[SetBuilder, APPROXDIR]:
    b1, a1 = _1._builder, _1._approxdir
    b2, a2 = _2._builder, _2._approxdir
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

