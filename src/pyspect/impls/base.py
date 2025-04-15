
from typing import ClassVar, Optional, Union, Tuple, Set, Dict, Any

class Impl: ...

class ImplClientMeta(type):

    __require__: ClassVar[tuple[str, ...]]

    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]) -> None:

        _require = namespace.setdefault('__require__', ())

        require = set(_require)
        for base in bases:
            if isinstance(base, ImplClientMeta):
                require = require.union(base.__require__)
        
        namespace['__require__'] = tuple(require)

        return super().__new__(mcs, name, bases, namespace)                

    def missing_ops(cls, impl: Impl) -> list[str]:
        return [
            field
            for field in cls.__require__
            if not hasattr(impl, field)
        ]

    def is_supported(cls, impl: Impl) -> bool:
        return not cls.missing_ops(impl)

class ImplClient(metaclass=ImplClientMeta):

    def add_requirements(cls, funcnames):
        _require = set(cls.__require__)
        _require = _require.union(funcnames)
        cls.__require__ = tuple(_require)
