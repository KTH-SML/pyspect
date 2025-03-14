######################################################################
## Propositional Language Fragments

from ..set_builder import *
from ..impls.base import *
from ..tlt import *
from .base import *
from .tools import *

__all__ = (
    ## Primitives
    'NOT', 'AND', 'OR',
    ## TLT Operators and Fragments
    'Not', 'And', 'Or',
    'Minus', 'Implies',
    'Propositional',
)


## ## ## ## ## ## ## ##
## NOT / Complement

NOT = declare('NOT')

class Not(NOT):

    __default__ = 'NOT'
    __require__ = ('complement',)

    @staticmethod
    def __new_NOT__(arg: TLTLike) -> TLT:
        return TLT(NOT('_1'), _1=arg)

    @staticmethod
    def __apply_NOT__(sb: SetBuilder) -> SetBuilder:
        return AppliedSet('complement', sb)
    
    @staticmethod
    def __check_NOT__(a: APPROXDIR) -> APPROXDIR:
        return -1 * a


## ## ## ## ## ## ## ## ##
## AND / Intersection

AND = declare('AND')

class And[R, I: Impl](AND):

    __default__ = 'AND'

    @staticmethod
    def __new_AND__(lhs: TLTLike, rhs: TLTLike, *args: TLTLike) -> TLT:
        if args:
            lhs, rhs = And(lhs, rhs, *args[:-1]), args[-1]
        return TLT(AND('_1', '_2'), _1=lhs, _2=rhs)

    @staticmethod
    def __apply_AND__(sb1: SetBuilder, sb2: SetBuilder) -> SetBuilder:
        x = AppliedSet('intersect', sb1, sb2)
        return x
    
    @staticmethod
    def __check_AND__(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        return (APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else 
                APPROXDIR.INVALID if APPROXDIR.UNDER in (a1, a2) else
                APPROXDIR.OVER)


## ## ## ## ## ##
## OR / Union

OR = declare('OR')

class Or(OR):

    __default__ = 'OR'
    __require__ = ('union',)

    @staticmethod
    def __new_OR__(lhs: TLTLike, rhs: TLTLike, *args: TLTLike) -> TLT:
        if args:
            lhs, rhs = Or(lhs, rhs, *args[:-1]), args[-1]
        return TLT(OR('_1', '_2'), _1=lhs, _2=rhs)

    @staticmethod
    def __apply_OR__(sb1: SetBuilder, sb2: SetBuilder) -> SetBuilder:
        return AppliedSet('union', sb1, sb2)
    
    @staticmethod
    def __check_OR__(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        return (APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else
                a1 if a1 == a2 else
                a2 if a1 == APPROXDIR.EXACT else
                a1 if a2 == APPROXDIR.EXACT else
                APPROXDIR.INVALID)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Propositional

MINUS = declare('MINUS')
Minus = define(
    MINUS('lhs', 'rhs'),
    AND('lhs', NOT('rhs')),
    Not, And,
)

IMPLIES = declare('IMPLIES')
Implies = define(
    IMPLIES('lhs', 'rhs'),
    OR('lhs', NOT('rhs')),
    Not, Or,
)

class PropositionalNoOr(
    Minus,
    Not, And, 
): ...

class Propositional(
    Minus, Implies, 
    Not, And, Or,
): ...
