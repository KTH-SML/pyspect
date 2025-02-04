from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

from ..set_builder import *
from ..tlt import *
from .base import *

from .propositional import *
from .propositional import __all__ as __all_propositional__

__all__ = (
    ## Inherited
    *__all_propositional__,
    ## Primitives
    'UNTIL', 'ALWAYS',
    ## Derivatives
    'ReachAvoid', 'ContinuousLTL',
    ## TLT Operators
    'Until', 'Always',
)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Reach / Avoid

UNTIL = Language.declare('UNTIL')
ALWAYS = Language.declare('ALWAYS')

def Until(lhs: TLTLike, rhs: TLTLike) -> TLT:
    with UNTIL.In(TLT):
        return TLT.construct(UNTIL('_1', '_2'), _1=lhs, _2=rhs)

def Always(arg: TLTLike) -> TLT:
    with ALWAYS.In(TLT):
        return TLT.construct(ALWAYS('_1'), _1=arg)

class ReachAvoid(Complement, UNTIL, ALWAYS):

    R = TypeVar('R')
    class Impl(ABC, Generic[R]):
    
        R = TypeVar('R')
        @abstractmethod
        def reach(self, goal: R, constraint: Optional[R]) -> R: ...
        
        R = TypeVar('R')
        @abstractmethod
        def avoid(self, goal: R, constraint: Optional[R]) -> R: ...

    @staticmethod
    def _apply__UNTIL(sb1: SetBuilder, sb2: SetBuilder) -> SetBuilder:
        return AppliedSet('reach', sb2, sb1)
    
    @staticmethod
    def _check__UNTIL(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        a0 = APPROXDIR.UNDER # reach should normally (always?) implement UNDER
        return (APPROXDIR.INVALID if a0 != a2 else a2)
    
    @staticmethod
    def _apply__ALWAYS(sb: SetBuilder) -> SetBuilder:
        return AppliedSet('complement', AppliedSet('avoid', AppliedSet('complement', sb)))
    
    @staticmethod
    def _check__ALWAYS(a: APPROXDIR) -> APPROXDIR:
        a0 = APPROXDIR.UNDER
        return (APPROXDIR.INVALID if a != a0 else a)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ContinuousLTL

class ContinuousLTL(
    Propositional, 
    ReachAvoid,
): ...

