from abc import ABC, abstractmethod
from typing import Protocol, Optional

class HasAxisStuff(Protocol):
    
    @property
    @abstractmethod    
    def ndim(self) -> int: ...

    @abstractmethod    
    def axis_is_periodic(self, i: int) -> bool: ...

    @abstractmethod    
    def axis(self, name: str) -> int: ...


class HasEmpty(Protocol):

    @abstractmethod    
    def empty(self): ...

class HasPlaneCut(Protocol):

    @abstractmethod    
    def plane_cut(self, *args, **kwds): ...

class HasComplement(Protocol):

    @abstractmethod    
    def complement(self, _1: 'R') -> 'R': ...

class HasIntersect(Protocol):

    @abstractmethod    
    def intersect(self, _1: 'R', _2: 'R') -> 'R': ...

class HasUnion(Protocol):

    @abstractmethod    
    def union(self, _1: 'R', _2: 'R') -> 'R': ...

class HasReach(Protocol):

    @abstractmethod
    def reach(self, target: 'R', constraint: Optional['R']) -> 'R': ...
    
class HasAvoid(Protocol):

    @abstractmethod
    def avoid(self, target: 'R', constraint: Optional['R']) -> 'R': ...
