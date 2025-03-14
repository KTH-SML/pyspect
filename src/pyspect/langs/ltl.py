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
    'Until', 'Always',
    'ContinuousLTL',
)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Reach / Avoid

UNTIL = declare('UNTIL')

class Until(UNTIL):

    __default__ = 'UNTIL'

    @staticmethod
    def __new_UNTIL__(lhs: TLTLike, rhs: TLTLike) -> TLT:
        return TLT(UNTIL('_1', '_2'), _1=lhs, _2=rhs)

    @staticmethod
    def __apply_UNTIL__(sb1: SetBuilder, sb2: SetBuilder) -> SetBuilder:
        return AppliedSet('reach', sb2, sb1)
    
    @staticmethod
    def __check_UNTIL__(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        a0 = APPROXDIR.UNDER # reach should normally (always?) implement UNDER
        return (APPROXDIR.INVALID if a0 != a2 else a2)

ALWAYS = declare('ALWAYS')

class Always(Not, ALWAYS):
    
    __default__ = 'ALWAYS'

    @staticmethod
    def __new_ALWAYS__(arg: TLTLike) -> TLT:
        return TLT(ALWAYS('_1'), _1=arg)

    @staticmethod
    def __apply_ALWAYS__(sb: SetBuilder) -> SetBuilder:
        return AppliedSet('complement', AppliedSet('avoid', AppliedSet('complement', sb)))
    
    @staticmethod
    def __check_ALWAYS__(a: APPROXDIR) -> APPROXDIR:
        a0 = APPROXDIR.UNDER
        return (APPROXDIR.INVALID if a != a0 else a)



## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ContinuousLTL

class ContinuousLTL(
    Until,
    Always, 
    Propositional, 
): ...
