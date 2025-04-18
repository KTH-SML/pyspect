######################################################################
## Temporal Logic Fragments

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

from ..set_builder import *
from ..tlt import *
from .base import *
from .tools import *

from .propositional import *
from .propositional import __all__ as __all_propositional__

__all__ = (
    ## Inherited
    *__all_propositional__,
    ## Primitives
    'UNTIL', 'ALWAYS',
    ## TLT Operators and Fragments
    'Until', 'Eventually', 'Always',
    'ContinuousLTL',
)

NEXT = declare('NEXT')

@assert_complete
class Next(NEXT):

    __default__ = 'NEXT'
    __require__ = ('pre',)

    @staticmethod
    def __new_NEXT__(arg: 'TLTLike') -> TLT:
        return TLT(NEXT('_1'), _1=arg)

    @staticmethod
    def __apply_NEXT__(sb: SetBuilder) -> SetBuilder:
        return AppliedSet('pre', sb)
    
    @staticmethod
    def __check_NEXT__(a: APPROXDIR) -> APPROXDIR:
        ao = APPROXDIR.UNDER # TODO: pre normally implement UNDER
        assert ao != APPROXDIR.INVALID, 'Operator may never be inherently invalid'
        if a == APPROXDIR.INVALID: return a
        return (ao + a if ao * a == APPROXDIR.EXACT else 
                a if ao == a else
                APPROXDIR.INVALID)

UNTIL = declare('UNTIL')

@assert_complete
class Until(UNTIL):

    __default__ = 'UNTIL'
    __require__ = ('reach',)

    @staticmethod
    def __new_UNTIL__(lhs: 'TLTLike', rhs: 'TLTLike') -> TLT:
        return TLT(UNTIL('_1', '_2'), _1=lhs, _2=rhs)

    @staticmethod
    def __apply_UNTIL__(sb1: SetBuilder, sb2: SetBuilder) -> SetBuilder:
        return AppliedSet('reach', sb2, sb1)
    
    @staticmethod
    def __check_UNTIL__(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        ao = APPROXDIR.UNDER # TODO: reach normally implement UNDER 
        assert ao != APPROXDIR.INVALID, 'Operator may never be inherently invalid'
        if a2 == APPROXDIR.INVALID: return a2
        return (ao + a2 if ao * a2 == APPROXDIR.EXACT else 
                a2 if ao == a2 else
                APPROXDIR.INVALID)

EVENTUALLY = declare('EVENTUALLY')
Eventually = define(EVENTUALLY('_1'),
                    UNTIL(TRUTHIFY('_1'), '_1'),
                    UNTIL, TRUTHIFY)

ALWAYS = declare('ALWAYS')

@assert_complete
class Always(ALWAYS):
    
    __default__ = 'ALWAYS'
    __require__ = ('rci',)

    @staticmethod
    def __new_ALWAYS__(arg: 'TLTLike') -> TLT:
        return TLT(ALWAYS('_1'), _1=arg)

    @staticmethod
    def __apply_ALWAYS__(sb: SetBuilder) -> SetBuilder:
        return AppliedSet('rci', sb)
    
    @staticmethod
    def __check_ALWAYS__(a: APPROXDIR) -> APPROXDIR:
        ao = APPROXDIR.UNDER # TODO: rci normally implement UNDER 
        assert ao != APPROXDIR.INVALID, 'Operator may never be inherently invalid'
        if a == APPROXDIR.INVALID: return a
        return (ao + a if ao * a == APPROXDIR.EXACT else 
                a if ao == a else
                APPROXDIR.INVALID)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## LTL Fragments

@assert_complete
class ContinuousLTL(
    Propositional, 
    Until, Eventually, Always,
): ...
