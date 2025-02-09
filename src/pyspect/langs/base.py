from abc import ABCMeta
from typing import ClassVar, Self, Optional, Union, Tuple, Set, Dict, Any
from contextlib import contextmanager

__all__ = (
    'Expr',
    'canonicalize',
    'LanguageFragmentMeta',
    'Void',
)

BiOp = Tuple[str, 'Expr', 'Expr'] 
UnOp = Tuple[str, 'Expr']
Term = Tuple[str] | str
Expr = Union[BiOp, UnOp, Term]

def canonicalize(expr: Expr) -> Expr:
    if isinstance(expr, str): return (expr,)
    assert isinstance(expr, tuple), 'Invalid expression'
    assert 1 <= len(expr) <= 3, 'Invalid expression'
    head, *tail = expr
    return (head, *map(canonicalize, tail))

class LanguageFragmentMeta(ABCMeta, type):

    # Cannot be set. Indicates if the fragment is a single primitive operator
    __isprimitive__: ClassVar[bool]

    # Cannot be set. For complex fragments, this contains all necessary primitive operators
    __primitives__: ClassVar[tuple[str, ...]]
    
    # Can be set. Default behavior for __new__ (either creates a formula or TLT)
    __default__: ClassVar[Optional[str]]
    
    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]):
    
        ##  Parameters ##

        _isprimitive = namespace.setdefault('__isprimitive__', False)
        _primitives = namespace.setdefault('__primitives__', ())
        _default = namespace.setdefault('__default__', None)

        ## Check Required Primitives ##
        primitives = set(_primitives)
        if _isprimitive:
            primitives.add(name)
        else:
            for base in bases:
                if isinstance(base, LanguageFragmentMeta):
                    primitives = primitives.union(base.__primitives__)

        # Assign primitives to the newly constructed language class
        namespace['__primitives__'] = tuple(primitives)

        return super().__new__(mcs, name, bases, namespace)                

    def __repr__(cls) -> str:
        return f"<language '{cls.__name__}'>"
    
    def __call__(cls, *args, **kwds) -> Expr:

        # NOTE: This overrides the ability to instantiate objects,
        #       but this is not necessary for the language classes
        #       which we only use for type checking purposes.

        if cls.__isprimitive__:
            # Creates a formula based on the primitive
            return (cls.__name__, *args)
        if cls.__default__ is not None:
            name = f'__new_{cls.__default__}__'
            __new__ = getattr(cls, name, None)
            if __new__ is None:
                raise TypeError(f'Default is set to {cls.__default__}, but {name} does not exist.')
            return __new__(*args, **kwds)
        else:
            # Use default behavior (instantiating the object)
            return super().__call__(*args, **kwds)
    
    def is_complete(cls) -> bool:
        return not bool(cls.__abstractmethods__)

    def is_modelling(cls, formula: Expr) -> bool:
        if isinstance(formula, str): return True
        match formula:
            case (prop,):
                return True
            case (op, rhs): 
                return (False if op not in cls.__primitives__ else 
                        cls.is_modelling(rhs))
            case (op, lhs, rhs):
                return (False if op not in cls.__primitives__ else 
                        cls.is_modelling(lhs) and cls.is_modelling(rhs))


# The Void fragment is a singleton for a trivial language that 
# puts no restriction on the implementation. It has only one 
# purpose, to be the by default selected language of TLTs.
Void = LanguageFragmentMeta('Void', (), {})
