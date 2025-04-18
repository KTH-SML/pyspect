from typing import ClassVar, Optional, Union, Tuple, Set, Dict, Any

from .set_builder import SetBuilder

__all__ = (
    'get_malformed',
    'get_props',
    'replace_prop',
    # Propositional
    'NOT', 'AND', 'OR', 'TRUTHIFY', 'FALSIFY',
    # Temporal Logic
    'NEXT', 'UNTIL', 'EVENTUALLY', 'ALWAYS',
)

TLProp      = str | SetBuilder
TLOpNullary = Tuple[str]
TLOpUnary   = Tuple[str, 'TLExpr']
TLOpBinary  = Tuple[str, 'TLExpr', 'TLExpr'] 
TLExpr      = Union[TLProp, TLOpNullary, TLOpUnary, TLOpBinary]

def get_malformed(formula: TLExpr) -> Optional[TLExpr]:
    if isinstance(formula, str): 
        return None if formula else formula # OK unless empty string
    elif isinstance(formula, SetBuilder):
        return None # OK, constant
    elif isinstance(formula, tuple) and len(formula) <= 3:
        head, *tail = formula
        if not isinstance(head, str): return formula # Malformed
        for arg in tail:
            if x := get_malformed(arg):
                return x # Malformed
        return None # OK
    else:
        return formula # Malformed

def get_props(formula: TLExpr) -> set[str]:
    if isinstance(formula, str):
        return {formula}
    elif isinstance(formula, SetBuilder):
        return set()
    elif isinstance(formula, tuple):
        head, *tail = formula
        props = set()
        for arg in tail:
            props |= get_props(arg)
        return props
    else:
        raise TypeError(f"Unknown formula type: {type(formula)}")

def replace_prop(formula: TLExpr, prop: str, expr: TLExpr):
    head, *tail = formula
    if tail:
        # formula is an operator expression
        # => go down in arguments to replace prop
        return (head, *map(lambda arg: replace_prop(arg, prop, expr), tail))
    else:
        # formula is a terminal
        # => if terminal == prop, replace with expr
        return expr if head == prop else formula

class LogicFragment(type):

    __narg__: Optional[int]
    __connectives__: ClassVar[tuple[str, ...]]

    def __new__(mcs, name: str, bases: tuple, namespace: dict):

        _narg = namespace.setdefault('__narg__', None)
        _connectives = set(namespace.setdefault('__connectives__', ()))

        for base in bases:
            if isinstance(base, LogicFragment):
                _connectives |= set(base.__connectives__)
            else:
                raise TypeError(f"Base class {base} is not a LogicFragment")
        
        namespace['__narg__'] = _narg
        namespace['__connectives__'] = tuple(_connectives)

        return super().__new__(mcs, name, bases, namespace)

    def __repr__(cls) -> str:
        return f"<language '{cls.__name__}'>"

    def __call__(cls, *args) -> TLExpr:
        
        # NOTE: This overrides the ability to instantiate objects,
        #       but this is not necessary for the language classes
        #       which we only use for type checking purposes.

        if len(cls.__connectives__) != 1:
            raise TypeError(f"Can only instantiate a single connective, but {L} were found in {cls.__name__}.")

        if len(args) != cls.__narg__:
            raise TypeError(f"Expected {cls.__narg__} arguments, but got {len(args)}.")

        for arg in args:
            if not isinstance(arg, (str, SetBuilder, tuple)):
                raise TypeError(f"Argument {arg} is not a valid formula.")

        # Creates a formula based on the primitive
        return (cls.__name__, *args)

    def __or__(cls, *args) -> 'LogicFragment':
        # Combine connectives using bitwise OR
        return LogicFragment('UnnamedLanguageFragment', args, {})

    def is_complete(cls, prims: set) -> bool:
        # Check if all connectives are present in the set of primitives
        return all(conn in prims for conn in cls.__connectives__)

def declare(name: str, narg: int) -> LogicFragment:
    return LogicFragment(name, (), {
        '__narg__': narg,
        '__connectives__': (name,),
    })

######################################################################
## Logic Fragment: Propositional

FALSIFY = declare('FALSIFY', 0)

TRUTHIFY = declare('TRUTHIFY', 0)

NOT = declare('NOT', 1)

AND = declare('AND', 2)

OR = declare('OR', 2)

MINUS = declare('MINUS', 2)

IMPLIES = declare('IMPLIES', 2)

PROPOSITIONAL = NOT | AND | OR | TRUTHIFY | FALSIFY

######################################################################
## Logic Fragment: Temporal Logic

UNTIL = declare('UNTIL', 2)

EVENTUALLY = declare('EVENTUALLY', 1)

ALWAYS = declare('ALWAYS', 1)

NEXT = declare('NEXT', 1)

# Continuous LTL
LTLc = PROPOSITIONAL | UNTIL | EVENTUALLY | ALWAYS

# Discrete LTL
LTLd = LTLc | NEXT
