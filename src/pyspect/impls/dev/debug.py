
from io import TextIOWrapper
from contextlib import contextmanager
from functools import wraps

from .base import Impl

__all__ = ['DebugImpl']

class DebugImpl[R](Impl[R]):
    """Debug implementation that does nothing but the sets at each call.

    This implementation is primarily useful to trace the flow of set operations.
    In particular, it can be inherited from to create a debugging implementation 
    with custom printing behavior. For example:
    
    ```python
    R = ... # Some set representation type

    class MyImpl(Impl[R]):
        ...

    class MyDebugger(DebugImpl[R], MyImpl[R]):
        def debug_repr(self, inp: R) -> str:
            return repr(inp) # or some custom string representation

    impl = MyDebugger(...)

    tlt.realize(impl) # This will print the input set at each call
    ```

    It is recommended that custom implementations override the `debug_repr` method
    to provide meaningful representations of the input sets. These should be concise
    to avoid overwhelming output during debugging sessions.
    """

    __debug_file__: TextIOWrapper | None = None
    __debug_cprint__: bool = True
    __debug_indent__: int = 0

    def debug_repr(self, inp: R) -> str:
        """Return a string representation of the input set for debugging.

        This method can be overridden in subclasses to provide custom
        representations of the input set.
        
        Args:
            inp (R): The input set to represent.
        
        Returns:
            str: A string representation of the input set.
        """
        return f'<set {type(inp).__name__}>'

    def _debug_print(self, method: str, *inps: R, out: R) -> None:
        prefix = ''
        if self.__debug_indent__ > 0:
            prefix = '  ' * (self.__debug_indent__-1) + '|-- '
        
        str_inp = ', '.join(self.debug_repr(inp) for inp in inps)
        out_inp = self.debug_repr(out)
        line = f'{prefix}{method}({str_inp}) -> {out_inp}'
        
        if self.__debug_cprint__:
            print(line)
        if f := self.__debug_file__:
            f.write(line)

    @classmethod
    @wraps(open)
    @contextmanager
    def to_file(cls, *args, consoleprint: bool = False, **kwds):
        # switch consoleprint and cls.__debug_cprint__
        consoleprint, cls.__debug_cprint__ = cls.__debug_cprint__, consoleprint
        
        cls.__debug_file__ = open(*args, **kwds)
        try:
            yield
        finally:
            cls.__debug_cprint__ = consoleprint
            cls.__debug_file__.close()
            cls.__debug_file__ = None

    def empty(self) -> R:
        out = super().empty()
        self._debug_print('empty', out=out)
        return out
    
    def complement(self, inp: R) -> R:
        out = super().complement(inp)
        self._debug_print('complement', inp, out=out)
        return out
    
    def intersect(self, *inps: R) -> R:
        out = super().intersect(*inps)
        self._debug_print('intersect', *inps, out=out)
        return out
    
    def union(self, *inps: R) -> R:
        out = super().union(*inps)
        self._debug_print('union', *inps, out=out)
        return out
    
    def reach(self, target, constraints=None):
        out = super().reach(target, constraints)
        self._debug_print('reach', target, constraints, out=out)
        return out
