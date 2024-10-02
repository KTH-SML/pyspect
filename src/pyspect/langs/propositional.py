from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from ..set_builder import *
from ..tlt import *
from .base import *

__all__ = (
    ## Primitives
    'NOT', 'AND', 'OR',
    ## Derivatives
    'Complement', 'Intersection', 'Union',
    'Propositional',
    ## TLT Operators
    'Not', 'And', 'Or',
    'Minus', 'Implies',
)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Complement

NOT = Language.declare('NOT')

def Not(arg: TLTLike) -> TLT:
    with NOT.In(TLT):
        return TLT.construct(NOT('_1'), _1=arg)

class Complement(NOT):

    R = TypeVar('R')
    class Impl(ABC, Generic[R]):
    
        R = TypeVar('R')
        @abstractmethod
        def complement(self, arg: R) -> R: ...

    @staticmethod
    def _apply__NOT(sb: SetBuilder) -> SetBuilder:
        return lambda impl, **m: impl.complement(sb(impl, **m))
    
    @staticmethod
    def _check__NOT(a: APPROXDIR) -> APPROXDIR:
        return -1 * a


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Intersection

AND = Language.declare('AND')

def And(lhs: TLTLike, rhs: TLTLike, *args: TLTLike) -> TLT:
    if args:
        lhs, rhs = And(lhs, rhs, *args[:-1]), args[-1]
    with AND.In(TLT):
        return TLT.construct(AND('_1', '_2'), _1=lhs, _2=rhs)

class Intersection(AND):

    R = TypeVar('R')
    class Impl(ABC, Generic[R]):
        
        R = TypeVar('R')
        @abstractmethod
        def intersect(self, lhs: R, rhs: R) -> R: ...

    @staticmethod
    def _apply__AND(sb1: SetBuilder, sb2: SetBuilder) -> SetBuilder:
        return lambda impl, **m: impl.intersect(sb1(impl, **m), sb2(impl, **m))
    
    @staticmethod
    def _check__AND(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        return (APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else 
                APPROXDIR.INVALID if APPROXDIR.UNDER in (a1, a2) else
                APPROXDIR.OVER)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Union

OR = Language.declare('OR')

def Or(lhs: TLTLike, rhs: TLTLike, *args: TLTLike) -> TLT:
    if args:
        lhs, rhs = Or(lhs, rhs, *args[:-1]), args[-1]
    with OR.In(TLT):
        return TLT.construct(OR('_1', '_2'), _1=lhs, _2=rhs)

class Union(OR):

    R = TypeVar('R')
    class Impl(ABC, Generic[R]):

        R = TypeVar('R')
        @abstractmethod
        def union(self, lhs: R, rhs: R) -> R: ...

    @staticmethod
    def _apply__OR(sb1: SetBuilder, sb2: SetBuilder) -> SetBuilder:
        return lambda impl, **m: impl.union(sb1(impl, **m), sb2(impl, **m))
    
    @staticmethod
    def _check__OR(a1: APPROXDIR, a2: APPROXDIR) -> APPROXDIR:
        return (APPROXDIR.INVALID if APPROXDIR.INVALID in (a1, a2) else
                a1 if a1 == a2 else
                a2 if a1 == APPROXDIR.EXACT else
                a1 if a2 == APPROXDIR.EXACT else
                APPROXDIR.INVALID)


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Propositional

def Minus(lhs: TLTLike, rhs: TLTLike) -> TLT:
    return And(lhs, Not(rhs))

def Implies(lhs: TLTLike, rhs: TLTLike) -> TLT:
    return Or(lhs, Not(rhs))

class Propositional(
    Complement,
    Intersection,
    Union,
): ...
