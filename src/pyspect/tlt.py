######################################################################
## Temporal Logic Trees

from enum import Enum
from typing import Optional, TypeVar, Generic, Dict, Union, Callable

from .langs.base import *
from .idict import *
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



TLTLike = Union[TLTFormula,SetBuilder,'TLT']

TLTLikeMap = Dict[str, 'TLTLike']

LabellingMap = idict[str, Optional[SetBuilder]]
   

I, R = TypeVar('I'), TypeVar('R')
class TLT(LanguageUser, Generic[I, R]):

    __debug = {}

    @classmethod
    def debug(cls, **kwds):
        for k, v in kwds.items():
            k = f'__debug_{k}'
            if v is Ellipsis:
                cls.__debug.pop(k, None) # remove to set default
            else:
                cls.__debug[k] = v

    @classmethod
    def construct(cls, arg: TLTLike, **kwds: TLTLike) -> 'TLT[I, R]':
        if cls.__debug: kwds.update(cls.__debug)
        return (cls._construct_from_tlt(arg, **kwds)      if isinstance(arg, TLT) else
                cls._construct_from_prop(arg, **kwds)     if isinstance(arg, str) else
                cls._construct_from_builder(arg, **kwds)  if isinstance(arg, Callable) else
                cls._construct_from_formula(arg, **kwds))

    @classmethod
    def _construct_from_tlt(cls, tlt: 'TLT[I, R]', **kwds: TLTLikeMap) -> 'TLT[I, R]':
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
                for prop, sb in tlt._lmap.items()
                if prop in kwds or sb is not None}

        # It is best to reconstruct builders in the "top-level" TLT when there
        # is updates to the tree since we have modified how 
        return cls._construct_from_formula(tlt._formula, **kwds, **__debug)

    @classmethod
    def _construct_from_prop(cls, prop: str, **kwds: TLTLikeMap) -> 'TLT[I, R]':
        # Hacky debug helpers
        __debug: dict = {k: kwds.pop(k) for k in list(kwds) if k.startswith('__debug')}
        __debug_print: bool = __debug.get('__debug_print', False)
        __debug_indent: int = __debug.get('__debug_indent', 0)
        if __debug_print: 
            print('> ' * __debug_indent + 'prop'.ljust(7), kwds)
            __debug['__debug_indent'] = __debug_indent + 1

        formula = (prop,) # Propositions are always terminals
        return (cls.construct(kwds.pop(prop), **kwds, **__debug) if prop in kwds else 
                cls(formula, ReferredSet(prop), lmap={prop: None}))

    @classmethod
    def _construct_from_builder(cls, sb: SetBuilder, **kwds: TLTLikeMap) -> 'TLT[I, R]':
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
        # letting `lmap` hold the actual set builder `sb`. This way, the root
        # TLT will hold a full formula that refers to even constant sets
        # (information is not lost as when internally binding constant sets to
        # builder functions).
        uid = '_' + builder_uid(sb)
        formula = (uid,)
        return cls(formula, ReferredSet(uid), lmap={uid: sb})

    @classmethod
    def _construct_from_formula(cls, formula: TLTFormula, **kwds: TLTLikeMap) -> 'TLT[I, R]':
        # Hacky debug helpers
        __debug: dict = {k: kwds.pop(k) for k in list(kwds) if k.startswith('__debug')}
        __debug_print: bool = __debug.get('__debug_print', False)
        __debug_indent: int = __debug.get('__debug_indent', 0)
        if __debug_print:
            print('> ' * __debug_indent + 'formula'.ljust(7), kwds)
            __debug['__debug_indent'] = __debug_indent + 1

        head, *tail = formula
        if tail: # Operator: head = op, tail = (arg1, ...)
            args = [cls.construct(arg, **kwds, **__debug) for arg in tail]     # make TLTs of formula args
            try:
                apply = getattr(cls.__language__, f'_apply__{head}')    # `apply` creates a builder for op
                check = getattr(cls.__language__, f'_check__{head}')    # get approx check of op from lang
            except AttributeError as e:
                print(formula)
                raise e
            lmaps = [list(arg._lmap.items()) for arg in args]       # collect all labels 
            return cls((head, *[arg._formula for arg in args]),
                       apply(*[arg._builder for arg in args]),
                       check(*[arg._approx for arg in args]),
                       idict(sum(lmaps, [])))
        else: # Terminal: head = prop, tail = ()
            return cls._construct_from_prop(head, **kwds, **__debug)

    _formula: TLTFormula
    _builder: SetBuilder
    _approx: APPROXDIR
    _lmap: LabellingMap

    def __init__(self, formula=..., builder=..., approx=..., lmap=...):
        self._formula = formula if formula is not ... else '_0'
        
        # If constructed with the absurd set, then the TLT is also absurd, __debug_indent.e. cannot be realized.
        self._builder = builder if builder is not ... else ABSURD
        
        self._approx = approx if approx is not ... else APPROXDIR.EXACT
        
        # Sets are associated with names using ReferredSets.
        self._lmap = lmap if lmap is not ... else idict()

    def __repr__(self) -> str:
        cls = type(self).__name__
        lang = self.__language__.__name__
        approx = str(self._approx)
        formula = str(self._formula)
        return f'{cls}<{lang}>({approx}, {formula})'

    def realize(self, impl: I, memoize=False) -> R:
        assert isinstance(impl, self.__language__.Impl), \
            f'Implementation must inherit from the selected langauge implementation {self.__language__.__name__}.Impl'
        assert self.is_realizable(), \
            f'Cannot realize TLT, missing {list(self.iter_free())}'
        out = self._builder(impl, **self._lmap)
        if memoize:
            raise NotImplementedError() # TODO: builder = Set(out)
        return out

    def is_realizable(self) -> bool:
        for sb in self._lmap.values():
            if sb is None: return False
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
        yield from filter(lambda p: self._lmap[p] is None, self._lmap)


def Identity(arg: TLTLike) -> TLT:
    return TLT.construct(arg)
