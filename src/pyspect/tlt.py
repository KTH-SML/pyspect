######################################################################
## Temporal Logic Trees

from enum import Enum
from functools import wraps
from typing import Optional, TypeVar, Generic, Dict, Union, Callable

from .idict import *
from .impls.base import *
from .set_builder import *
from .logics import TLExpr

__all__ = (
    'TLT',
    'primitive',
    'Identity',
    # 'TLTLike',
    'APPROXDIR',
)

def builder_uid(sb: SetBuilder):
    # Simple way to create a unique id from a python function.
    # - hash(sb) returns the function pointer (I think)
    # - Convert to bytes to get capture full 64-bit value (incl. zeroes)
    # - Convert to hex-string
    return hash(sb).to_bytes(8,"big").hex()

class APPROXDIR(Enum):
    INVALID = None
    UNDER = -1
    EXACT = 0
    OVER = +1

    def __str__(self):
        return f'{self.name}'

    def __radd__(self, other): return self.__add__(other)

    def __add__(self, other: 'APPROXDIR | int') -> 'APPROXDIR':
        lhs = self.value
        rhs = other.value if isinstance(other, APPROXDIR) else other
        return APPROXDIR(None if None in (lhs, rhs) else
                         lhs + rhs)

    def __rmul__(self, other): return self.__mul__(other)

    def __mul__(self, other: 'APPROXDIR | int') -> 'APPROXDIR':
        lhs = self.value
        rhs = other.value if isinstance(other, APPROXDIR) else other
        return APPROXDIR(None if None in (lhs, rhs) else
                         lhs * rhs)


# type SetMap = idict[str, Optional[SetBuilder]]

# type TLTLike = Union[TLExpr, SetBuilder, 'TLT']

# type TLTLikeMap = Dict[str, TLTLike]

def tree_debugger(msg: str):
    """Decorator to print debug information for TLT construction.

    - Enable with `TLT.debug(print=True)`.
    - Choose indentation size with `TLT.debug(indent=...)`.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwds):

            __debug = {k: kwds.pop(k) 
                       for k in list(kwds)
                       if k.startswith('__debug')}
            
            _print: bool = __debug.get('__debug_print', False)
            _indent: int = __debug.get('__debug_indent', 0)
            
            if _print:
                print('> ' * _indent + msg.ljust(7), kwds)
                __debug['__debug_indent'] = _indent + 1

            return func(*args, **(kwds | __debug))
        return wrapper
    return decorator

class TLT(ImplClient):

    __debug = {}

    @classmethod
    def nodebug(cls):
        for k in list(cls.__debug):
            cls.__debug.pop(k, None) # remove to set default

    @classmethod
    def debug(cls, **kwds):
        kwds.setdefault('print', True)
        kwds.setdefault('indent', 0)
        for k, v in kwds.items():
            k = f'__debug_{k}'
            if v is Ellipsis:
                cls.__debug.pop(k, None) # remove to set default
            else:
                cls.__debug[k] = v

    __primitives__ = idict({}) # Null set of primitives

    @classmethod
    def select(cls, primitives):
        cls.__primitives__ = primitives

    @classmethod
    def construct(cls, arg: 'TLTLike', **kwds: 'TLTLike') -> 'TLT':
        return cls(arg, **kwds)

    def __new__(cls, arg: 'TLTLike', **kwds: 'TLTLike') -> 'TLT':
        if cls.__debug: kwds.update(cls.__debug)
        return (cls.__new_from_tlt__(arg, **kwds)      if isinstance(arg, TLT) else
                cls.__new_from_prop__(arg, **kwds)     if isinstance(arg, str) else
                cls.__new_from_builder__(arg, **kwds)  if isinstance(arg, Callable) else
                cls.__new_from_formula__(arg, **kwds))

    @classmethod
    @tree_debugger('tlt')
    def __new_from_tlt__(cls, tlt: 'TLT', **kwds: 'TLTLikeMap') -> 'TLT':

        # Create new TLT with updated information. With this, we can bind new 
        # TLT-like objects to free variables/propositions. We do not allow 
        # updates to existing propositions. First collect all updatable props.
        kwds = {prop: kwds[prop] if prop in kwds else sb
                for prop, sb in tlt._setmap.items()
                if prop in kwds or sb is not None}

        # It is best to reconstruct builders in the "top-level" TLT when there
        # is updates to the tree since we have modified how 
        return cls.__new_from_formula__(tlt._formula, **kwds)

    @classmethod
    @tree_debugger('prop')
    def __new_from_prop__(cls, prop: str, **kwds: 'TLTLikeMap') -> 'TLT':
        return (cls(kwds.pop(prop), **kwds) if prop in kwds else
                cls.__new_init__(prop, ReferredSet(prop), setmap={prop: None}))

    @classmethod
    @tree_debugger('builder')
    def __new_from_builder__(cls, sb: SetBuilder, **kwds: 'TLTLikeMap') -> 'TLT':
        # Assume a formula "_0" where the set exactly represent the prop "_0".
        # In reality, we define a unique ID `uid` instead of "_0". We add one
        # extra level of indirection with a ReferredSet for `uid` and 
        # letting `setmap` hold the actual set builder `sb`. This way, the root
        # TLT will hold a full formula that refers to even constant sets
        # (information is not lost as when internally binding constant sets to
        # builder functions).
        uid = '_' + builder_uid(sb)
        formula = uid # UID is a proposition
        self = cls.__new_init__(formula, ReferredSet(uid), setmap={uid: sb})
        
        self.inherit_requirements(sb)
        
        return self

    @classmethod
    @tree_debugger('formula')
    def __new_from_formula__(cls, formula: 'TLExpr', **kwds: 'TLTLikeMap') -> 'TLT':

        if isinstance(formula, str):
            # If formula is a string, it is a proposition
            return cls.__new_from_prop__(formula, **kwds)
        
        # Otherwise, it is a non-trivial formula
        head, *tail = formula

        assert head in cls.__primitives__, \
            f'Unknown operator `{head}` in formula `{formula}`. ' \
            f'Available operators: {cls.__primitives__.keys()}'

        args = [cls(arg, **kwds) for arg in tail]  # make TLTs of formula args
        setmaps = [list(arg._setmap.items()) for arg in args] # collect all set references
        
        return cls.__primitives__[head](*args)

    @classmethod
    def __new_init__(cls, formula=..., builder=..., approx=..., setmap=..., times=...):
        self = super(TLT, cls).__new__(cls)

        # Lock the selected language for this instance
        self.__primitives__ = self.__primitives__

        self._formula = formula if formula is not ... else '_0'
        
        # If constructed with the absurd set, then the TLT is also absurd, i.e. cannot be realized.
        self._builder = builder if builder is not ... else ABSURD
        
        self._approx = approx if approx is not ... else APPROXDIR.EXACT
        
        # Sets are associated with names using ReferredSets.
        self._setmap = setmap if setmap is not ... else idict()

        self.inherit_requirements(self._builder)

        return self

    _formula: 'TLExpr'
    _builder: SetBuilder
    _approx: APPROXDIR
    _setmap: 'SetMap'
    
    def __repr__(self) -> str:
        cls = type(self).__name__
        approx = str(self._approx)
        formula = str(self._formula)
        return f'{cls}({approx}, {formula})'

    def where(self, **kwds: 'TLTLike') -> 'TLT':
        return TLT(self, **kwds)

    def realize(self, impl: 'I', memoize=False) -> 'R':
        self.assert_realizable(impl)
        out = self._builder(impl, **self._setmap)
        if memoize:
            raise NotImplementedError() # TODO: builder = Set(out)
        return out

    def assert_realizable(self, impl: Optional['I'] = None) -> None:
        for name, sb in self._setmap.items():
            if sb is None: 
                raise Exception(f'Missing proposition `{name}`')
        if impl is not None and bool(missing := self.missing_ops(impl)):
            raise Exception(f'Missing from implementation: {", ".join(missing)}')
        if self._approx is APPROXDIR.INVALID:
            raise Exception('Invalid approximation. TLT operational semantics of formula does not hold.')

    def is_realizable(self, impl: Optional['I'] = None) -> bool:
        try:
            self.assert_realizable(impl)
        except Exception:
            return False
        else:
            return True
    
    def iter_frml(self, formula: Optional['TLExpr'] = None, **kwds):
        only_terminals = kwds.get('only_terminals', False)
        if formula is None:
            formula = self._formula
        _, *args = formula
        for arg in args:
            yield from self.iter_frml(arg, **kwds)
        if not (only_terminals and args):
            yield formula

    def iter_free(self):
        yield from filter(lambda p: self._setmap[p] is None, self._setmap)


def Identity(arg: 'TLTLike') -> TLT:
    return TLT(arg)


def primitive(formula: TLExpr) -> TLT:
    """Decorator to declare a primitive TLT."""
    assert isinstance(formula, tuple), "Formula must be a tuple"
    assert isinstance(formula[0], str), "Operator name of formula must be a string"
    assert all(isinstance(arg, str) for arg in formula[1:]), "Arguments of formula must be propositions (string) when declaring primitives"
    head, *tail = formula
    assert len(tail) == len(set(tail)), "Argument names must be unique"

    def decorator(func):

        @wraps(func)
        def wrapper(*args):
            formula = (head, *[arg._formula for arg in args])
            builder, approx = func(**{name: arg for name, arg in zip(tail, args)})
            setmap = idict(sum([list(arg._setmap.items()) for arg in args], []))
            tree = TLT.__new_init__(formula, builder, approx, setmap)
            tree.inherit_requirements(*args)
            return tree
        return idict({head: wrapper})
    return decorator
