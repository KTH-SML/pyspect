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


class HasEmpty[R](Protocol):

    @abstractmethod    
    def empty(self) -> R: ...

class HasPlaneCut[R, **P](Protocol):

    @abstractmethod    
    def plane_cut(self, *args: P.args, **kwds: P.kwargs) -> R: ...

class HasComplement[R](Protocol):

    @abstractmethod    
    def complement(self, _1: R) -> R: ...

class HasIntersect[R](Protocol):

    @abstractmethod    
    def intersect(self, _1: R, _2: R) -> R: ...

class HasUnion[R](Protocol):

    @abstractmethod    
    def union(self, _1: R, _2: R) -> R: ...

class HasReach[R](Protocol):

    @abstractmethod
    def reach(self, target: R, constraint: Optional[R]) -> R: ...
    
class HasAvoid[R](Protocol):

    @abstractmethod
    def avoid(self, target: R, constraint: Optional[R]) -> R: ...
