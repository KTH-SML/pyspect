"""Logic fragments and typed tuple-based formulas.

This module defines:
    - TLExpr: a lightweight AST for temporal/propositional logic as typed tuples
    - Utilities to validate/inspect/transform formulas
    - LogicFragment meta-class to declare connectives and compose language fragments

Design:
    - A proposition is either a string name or a SetBuilder constant.
    - Operator applications are tuples of the form:
        - (op,)                  for nullary operators
        - (op, arg)              for unary operators
        - (op, lhs, rhs)         for binary operators
    - Operators are declared via `declare(name, narg)` which produces a
    class-like fragment whose `__call__` constructs the tuple form.
    - Fragments can be composed with `|` to create a combined language.
"""
from __future__ import annotations
from typing import ClassVar, Optional

from .set_builder import SetBuilder

__all__ = (
    'get_malformed',
    'get_props',
    'replace_prop',
    'TLProp', 'TLExpr',
    # Propositional
    'NOT', 'AND', 'OR', 'TRUTHIFY', 'FALSIFY',
    'MINUS', 'IMPLIES',
    # Temporal Logic
    'NEXT', 'UNTIL', 'EVENTUALLY', 'ALWAYS',
)

# Type aliases for the tuple-shaped logic AST
type TLProp      = str | SetBuilder
type TLOpNullary = tuple[str]
type TLOpUnary   = tuple[str, TLExpr]
type TLOpBinary  = tuple[str, TLExpr, TLExpr]
type TLExpr      = TLProp | TLOpNullary | TLOpUnary | TLOpBinary

def get_malformed(formula: TLExpr) -> Optional[TLExpr]:
    """Return the first malformed subexpression, or None if well-formed.

    Parameters:
        - formula: TLExpr to check

    Returns:
        - None if the expression is well-formed; otherwise the first offending subexpression

    Well-formedness rules:
        - Propositions: non-empty str, or any SetBuilder instance
        - Operator tuples: length 1..3, head must be str, and all arguments recursively valid

    """
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
    """Collect proposition names appearing in a formula.

    - String terminals are treated as proposition names and included.
    - SetBuilder terminals are constants and contribute no proposition names.
    - Operator tuples are traversed recursively.

    Parameters:
    - formula: TLExpr to inspect

    Returns:
    - Set of unique proposition names found in `formula`
    """
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

def replace_prop(formula: TLExpr, prop: str, expr: TLExpr) -> TLExpr:
    """Replace all occurrences of proposition `prop` with `expr`.

    Parameters:
        - formula: operator tuple (TLOpNullary/Unary/Binary)
        - prop: proposition name to replace
        - expr: replacement sub-expression

    Returns:
        - New TLExpr with replacements applied

    Notes:
        - This function expects `formula` to be an operator tuple (not a raw str or SetBuilder).
          It descends recursively and replaces terminals that match `prop`.
        - For nullary or terminal positions, if the terminal equals `prop`, it is replaced by `expr`.
    """
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
    """Metaclass for declaring and composing logic fragments (connectives).

    Instances of classes produced by this metaclass are not meant to be
    instantiated. Instead, calling the class constructs a tuple-based TLExpr
    for the associated connective (see __call__).

    Attributes:
        - __narg__: required arity for the connective (None for a composed fragment)
        - __connectives__: tuple of connective names included in this fragment

    Composition:
        - Fragments can be combined with `|` to form a new fragment whose
          `__connectives__` is the union of the operands.
    """

    __narg__: Optional[int]
    __connectives__: ClassVar[tuple[str, ...]]

    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        """Create a new fragment type, merging connectives from base fragments."""
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
        """Construct a TLExpr tuple for this connective.

        Constraints:
            - This only works for a fragment with exactly one connective.
            - Number of arguments must match the declared arity.
            - Arguments must themselves be valid TLExpr terminals or tuples.

        Returns:
            - (op,) | (op, arg) | (op, lhs, rhs), where op == cls.__name__
        """
        
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
        """Return a fragment that is the union of this fragment and others."""
        # Combine connectives using bitwise OR
        return LogicFragment('UnnamedLanguageFragment', args, {})

    def is_complete(cls, prims: set) -> bool:
        """Check that all connectives in this fragment exist in `prims`."""
        # Check if all connectives are present in the set of primitives
        return all(conn in prims for conn in cls.__connectives__)

def declare(name: str, narg: int) -> LogicFragment:
    """Declare a new connective fragment with given name and arity.

    Parameters:
        - name: operator name used as the tuple head in TLExpr
        - narg: arity (0, 1, or 2)

    Returns:
        - A LogicFragment-derived class representing the connective
    """
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

# PROPOSITIONAL = NOT | AND | OR | TRUTHIFY | FALSIFY

######################################################################
## Logic Fragment: (Future- and Past-) LTL

# NOTE: Future-time operators are default

FUNTIL = declare('F-UNTIL', 2)
PUNTIL = declare('P-UNTIL', 2)
UNTIL  = FUNTIL

FEVENTUALLY = declare('F-EVENTUALLY', 1)
PEVENTUALLY = declare('P-EVENTUALLY', 1)
EVENTUALLY  = FEVENTUALLY

FALWAYS = declare('F-ALWAYS', 1)
PALWAYS = declare('P-ALWAYS', 1)
ALWAYS  = FALWAYS

FNEXT = declare('F-NEXT', 1)
PNEXT = declare('P-NEXT', 1)
NEXT  = FNEXT

######################################################################
## Logic Fragment: CTL* (TODO)

######################################################################
## Logic Fragment: STL (TODO)
