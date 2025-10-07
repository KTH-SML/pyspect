"""
Base interfaces and metaclasses for implementation plug-ins.

This module defines:
- Impl: Marker base for concrete implementation backends.
- ImplClientMeta: Metaclass that aggregates and propagates required operation names.
- ImplClient: Mixin for objects using implementations to declare/query required operations.
"""

from __future__ import annotations

from typing import ClassVar, Sequence, Any

__all__ = [
    'Impl',
    'ImplClientMeta',
    'ImplClient',
]


class Impl[R]:
    """Marker base class for concrete implementation backends."""


class ImplClientMeta(type):
    """Metaclass that aggregates required operation names across inheritance.

    Classes using this metaclass can declare a tuple[str, ...] in __require__.
    During class creation, this metaclass unions all __require__ tuples found
    in the base classes (that also use ImplClientMeta) with the one declared
    on the subclass, storing the result back into __require__.
    """

    __require__: ClassVar[tuple[str, ...]]

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> None:
        _require = namespace.setdefault('__require__', ())

        require = set(_require)
        for base in bases:
            if isinstance(base, ImplClientMeta):
                require = require.union(base.__require__)
        
        namespace['__require__'] = tuple(require)

        return super().__new__(mcs, name, bases, namespace)                


class ImplClient[R](metaclass=ImplClientMeta):
    """Client-side mixin for declaring and checking required operations.

    Clients refer to objects that use an implementation backend. ImplClient
    ensures to list the names of operations they need the implementation to
    provide via the class attribute __require__. This mixin offers methods to:
    - report which operations are missing on a given Impl
    - test if an implementation satisfies all requirements
    - extend or inherit requirement sets dynamically
    """

    @classmethod
    def missing_ops(cls, impl: Impl[R]) -> list[str]:
        """Return the names of required operations not found on impl."""
        return [
            field
            for field in cls.__require__
            if not hasattr(impl, field)
        ]
    
    @classmethod
    def is_supported(cls, impl: Impl[R]) -> bool:
        """Return True if impl provides all operations listed in __require__."""
        return not cls.missing_ops(impl)

    def add_requirements(self, funcnames: Sequence[str]) -> None:
        """Add function names to the set of required operations."""
        _require = set(self.__require__)
        _require = _require.union(funcnames)
        self.__require__ = tuple(_require)

    def inherit_requirements(self, *others: ImplClient[R]) -> None:
        """Merge requirement sets from other ImplClient-like classes."""
        for other in others:
            self.add_requirements(other.__require__)
