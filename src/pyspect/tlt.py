######################################################################
## Temporal Logic Trees

from enum import Enum
from typing import Optional, TypeVar, Generic, Dict, Union, Callable

from .idict import *
from .impls.base import *
from .langs.base import *
from .set_builder import *

__all__ = (
    'TLT',
    'TLTLike',
    'APPROXDIR',
)

TLTFormula = Expr

def builder_uid(sb: SetBuilder):
    # Simple way to create a unique id from a python function.
    # - hash(sb) returns the function pointer (I think)
    # - Convert to bytes to get capture full 64-bit value (incl. zeroes)
    # - Convert to hex-string
    return hash(sb).to_bytes(8,"big").hex()

def replace_prop(formula: TLTFormula, prop: str, expr: Expr):
    head, *tail = formula
    if tail:
        # formula is an operator expression
        # => go down in arguments to replace prop
        return (head, *map(lambda arg: replace_prop(arg, prop, expr), tail))
    else:
        # formula is a terminal
        # => if terminal == prop, replace with expr
        return expr if head == prop else formula

class APPROXDIR(Enum):
    INVALID = None
    UNDER = -1
    EXACT = 0
    OVER = +1

    def __str__(self):
        return f'{self.name}'

    def __radd__(self, other): return self.__add__(other)

    def __add__(self, other: Union['APPROXDIR', int]) -> 'APPROXDIR':
        lhs = self.value
        rhs = other.value if isinstance(other, APPROXDIR) else other
        return APPROXDIR(None if None in (lhs, rhs) else
                         lhs + rhs)

    def __rmul__(self, other): return self.__mul__(other)

    def __mul__(self, other: Union['APPROXDIR', int]) -> 'APPROXDIR':
        lhs = self.value
        rhs = other.value if isinstance(other, APPROXDIR) else other
        return APPROXDIR(None if None in (lhs, rhs) else
                         lhs * rhs)


type SetMap = idict[str, Optional[SetBuilder]]

type TLTLike = Union[TLTFormula, SetBuilder, 'TLT']

type TLTLikeMap = Dict[str, TLTLike]


class TLT(ImplClient):

    __debug = {}

    @classmethod
    def debug(cls, **kwds):
        for k, v in kwds.items():
            k = f'__debug_{k}'
            if v is Ellipsis:
                cls.__debug.pop(k, None) # remove to set default
            else:
                cls.__debug[k] = v

    __language__ = Void

    @classmethod
    def select(cls, lang):
        # We confirm that all abstracts are implemented
        assert lang.is_complete(), \
            f'{lang.__name__} is not complete, missing: {", ".join(lang.__abstractmethods__)}'
        cls.__language__ = lang

    @classmethod
    def construct(cls, arg: TLTLike, **kwds: TLTLike) -> 'TLT':
        return cls(arg, **kwds)

    def __new__(cls, arg: TLTLike, **kwds: TLTLike) -> 'TLT':
        if cls.__debug: kwds.update(cls.__debug)
        return (cls.__new_from_tlt__(arg, **kwds)      if isinstance(arg, TLT) else
                cls.__new_from_prop__(arg, **kwds)     if isinstance(arg, str) else
                cls.__new_from_builder__(arg, **kwds)  if isinstance(arg, Callable) else
                cls.__new_from_formula__(arg, **kwds))

    @classmethod
    def __new_from_tlt__(cls, tlt: 'TLT', **kwds: TLTLikeMap) -> 'TLT':
        # Hacky debug helpers
        __debug: dict = {k: kwds.pop(k) for k in list(kwds) if k.startswith('__debug')}
        __debug_print: bool = __debug.get('__debug_print', False)
        __debug_indent: int = __debug.get('__debug_indent', 0)
        if __debug_print:
            print('> ' * __debug_indent + 'tlt'.ljust(7), kwds)
            __debug['__debug_indent'] = __debug_indent + 1

        # Create new TLT with updated information. With this, we can bind new 
        # TLT-like objects to free variables/propositions. We do not allow 
        # updates to existing propositions. First collect all updatable props.
        kwds = {prop: kwds[prop] if prop in kwds else sb
                for prop, sb in tlt._setmap.items()
                if prop in kwds or sb is not None}

        # It is best to reconstruct builders in the "top-level" TLT when there
        # is updates to the tree since we have modified how 
        return cls.__new_from_formula__(tlt._formula, **kwds, **__debug)

    @classmethod
    def __new_from_prop__(cls, prop: str, **kwds: TLTLikeMap) -> 'TLT':
        # Hacky debug helpers
        __debug: dict = {k: kwds.pop(k) for k in list(kwds) if k.startswith('__debug')}
        __debug_print: bool = __debug.get('__debug_print', False)
        __debug_indent: int = __debug.get('__debug_indent', 0)
        if __debug_print: 
            print('> ' * __debug_indent + 'prop'.ljust(7), kwds)
            __debug['__debug_indent'] = __debug_indent + 1

        formula = (prop,) # Propositions are always terminals
        return (cls(kwds.pop(prop), **kwds, **__debug) if prop in kwds else
                cls.__new_init__(formula, ReferredSet(prop), setmap={prop: None}))

    @classmethod
    def __new_from_builder__(cls, sb: SetBuilder, **kwds: TLTLikeMap) -> 'TLT':
        # Hacky debug helpers
        __debug: dict = {k: kwds.pop(k) for k in list(kwds) if k.startswith('__debug')}
        __debug_print: bool = __debug.get('__debug_print', False)
        __debug_indent: int = __debug.get('__debug_indent', 0)
        if __debug_print:
            print('> ' * __debug_indent + 'builder'.ljust(7), kwds)
            __debug['__debug_indent'] = __debug_indent + 1

        # Assume a formula "_0" where the set exactly represent the prop "_0".
        # In reality, we define a unique ID `uid` instead of "_0". We add one
        # extra level of indirection with a ReferredSet for `uid` and 
        # letting `setmap` hold the actual set builder `sb`. This way, the root
        # TLT will hold a full formula that refers to even constant sets
        # (information is not lost as when internally binding constant sets to
        # builder functions).
        uid = '_' + builder_uid(sb)
        formula = (uid,)
        self = cls.__new_init__(formula, ReferredSet(uid), setmap={uid: sb})
        
        self.add_requirements(sb.__require__)
        
        return self

    @classmethod
    def __new_from_formula__(cls, formula: TLTFormula, **kwds: TLTLikeMap) -> 'TLT':
        # Hacky debug helpers
        __debug: dict = {k: kwds.pop(k) for k in list(kwds) if k.startswith('__debug')}
        __debug_print: bool = __debug.get('__debug_print', False)
        __debug_indent: int = __debug.get('__debug_indent', 0)
        if __debug_print:
            print('> ' * __debug_indent + 'formula'.ljust(7), kwds)
            __debug['__debug_indent'] = __debug_indent + 1

        head, *tail = formula
        if tail: # Operator: head = op, tail = (arg1, ...)
            args = [cls(arg, **kwds, **__debug) for arg in tail]  # make TLTs of formula args
            
            try:
                apply = getattr(cls.__language__, f'__apply_{head}__') # `apply` creates a builder for op
                check = getattr(cls.__language__, f'__check_{head}__') # get approx check of op from lang
            except AttributeError as e:
                print(formula)
                raise e
            
            setmaps = [list(arg._setmap.items()) for arg in args] # collect all set references 
            
            self = cls.__new_init__(
                (head, *[arg._formula for arg in args]),
                 apply(*[arg._builder for arg in args]),
                 check(*[arg._approx for arg in args]),
                 idict(sum(setmaps, [])),
            )
            
            self.add_requirements([req
                                   for arg in args
                                   for req in arg.__require__])

        else: # Terminal: head = prop, tail = ()
            self = cls.__new_from_prop__(head, **kwds, **__debug)

        return self

    @classmethod
    def __new_init__(cls, formula=..., builder=..., approx=..., setmap=...):
        self = super(TLT, cls).__new__(cls)

        # Lock the selected language for this instance
        self.__language__ = self.__language__

        self._formula = formula if formula is not ... else '_0'
        
        # If constructed with the absurd set, then the TLT is also absurd, __debug_indent.e. cannot be realized.
        self._builder = builder if builder is not ... else ABSURD
        
        self._approx = approx if approx is not ... else APPROXDIR.EXACT
        
        # Sets are associated with names using ReferredSets.
        self._setmap = setmap if setmap is not ... else idict()

        self.add_requirements(self._builder.__require__)

        return self

    _formula: TLTFormula
    _builder: SetBuilder
    _approx: APPROXDIR
    _setmap: SetMap

    def __repr__(self) -> str:
        cls = type(self).__name__
        lang = self.__language__.__name__
        approx = str(self._approx)
        formula = str(self._formula)
        return f'{cls}<{lang}>({approx}, {formula})'

    def where(self, **kwds: TLTLike) -> 'TLT':
        return TLT(self, **kwds)

    def realize(self, impl: 'I', memoize=False) -> 'R':
        self.assert_realizable()
        out = self._builder(impl, **self._setmap)
        if memoize:
            raise NotImplementedError() # TODO: builder = Set(out)
        return out

    def assert_realizable(self, impl: Optional['I'] = None) -> None:
        for name, sb in self._setmap.items():
            if sb is None: 
                raise Exception(f'Missing proposition `{name}`')
        if impl is not None:
            lang_missing = self.__language__.missing_ops(impl)
            if missing:
                raise Exception(f'Missing from implementation: {", ".join(missing)}')

    def is_realizable(self, impl: Optional['I'] = None) -> bool:
        try:
            self.assert_realizable(impl)
        except Exception:
            return False
        else:
            return True
    
    def iter_frml(self, formula: Optional[TLTFormula] = None, **kwds):
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


def Identity(arg: TLTLike) -> TLT:
    return TLT(arg)
