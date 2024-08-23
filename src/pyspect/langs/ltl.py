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

class ReachAvoid(UNTIL, ALWAYS):

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
        return lambda impl, **m: impl.reach(sb2(impl, **m), sb1(impl, **m))
    
    @staticmethod
    def _check__UNTIL(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        a0 = APPROXDIR.UNDER # reach should normally (always?) implement UNDER
        return (APPROXDIR.INVALID if a0 != a2 else a2)
    
    @staticmethod
    def _apply__ALWAYS(sb: SetBuilder) -> SetBuilder:
        return lambda impl, **m: impl.complement(
            impl.avoid(
                impl.complement(sb(impl, **m))
            )
        )
    
    @staticmethod
    def _check__ALWAYS(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        a0 = APPROXDIR.UNDER # reach should normally (always?) implement UNDER
        return (APPROXDIR.INVALID if a0 != a2 or a1 != a2 else a2)
    
def Until(lhs: TLTLike, rhs: TLTLike) -> TLT:
    return TLT.construct(UNTIL('_1', '_2'), _1=lhs, _2=rhs)

def Always(arg: TLTLike) -> TLT:
    return TLT.construct(ALWAYS('_1'), _1=arg)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ContinuousLTL

class ContinuousLTL(
    Propositional, 
    ReachAvoid,
): ...

