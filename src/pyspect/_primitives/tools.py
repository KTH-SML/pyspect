from .base import *
from ..tlt import TLT
from abc import abstractmethod

__all__ = (
    'declare',
    'define',
    'assert_complete',
)


def declare(name: str) -> PrimitiveSetMeta:
    return PrimitiveSetMeta(name, (), {
        '__isprimitive__': True,    # direct declaration is only for primitives
        f'__apply_{name}__': abstractmethod(lambda: None),
        f'__check_{name}__': abstractmethod(lambda: None),
    })


def define(primitive: Expr, formula: Expr, *depends: PrimitiveSetMeta) -> PrimitiveSetMeta:
    primitive = canonicalize(primitive)
    formula = canonicalize(formula)
    head, *tail = primitive
    assert all(len(arg) == 1 for arg in tail), "Primitive must be an expression ('OP', args...)" 
    p_name, p_args = head, [arg[0] for arg in tail] # (op, argname0, argname1)
    assert len(set(p_args)) == len(p_args), "Argument names must be unique"

    # Dynamically evaluate equivalent formula
    def recurse(getfunc, formula, *args):
        # getfunc(op: str) -> Callable:. Produces the method corresponding to op.
        # formula: Expr. Formula we're dynamically evaluating
        # args: T. Typically SetBuilder/APPROXDIR, args for the class method. 
        # return: T. Result of args applied on getattr(cls, name-of-method) according to formula. 
        match formula:
            case (prop,):
                i = p_args.index(prop) # i = 0 for argname0, i = 1 for argname1
                assert i != -1, f'Unreachable. For {primitive=}, found "{prop}" in {formula=}.'
                return args[i]
            case (op, rhs):
                func = getfunc(op)
                return func(recurse(getfunc, rhs, *args))
            case (op, lhs, rhs):
                func = getfunc(op)
                return func(recurse(getfunc, lhs, *args),
                            recurse(getfunc, rhs, *args))

    @classmethod
    def new(cls, *args):
        kwds = {argn: args[i] for i, argn in enumerate(p_args)}
        #    = {argname0: args[0], argname[1]: args[1]}
        return TLT(primitive, **kwds)
        # getfunc = lambda op: getattr(cls, f'__new_{op}__')
        # return recurse(getfunc, formula, *args)
    
    @classmethod
    def apply(cls, *args):
        getfunc = lambda op: getattr(cls, f'__apply_{op}__')
        return recurse(getfunc, formula, *args)

    @classmethod
    def check(cls, *args):
        getfunc = lambda op: getattr(cls, f'__check_{op}__')
        return recurse(getfunc, formula, *args)

    cls = PrimitiveSetMeta(p_name.capitalize(), depends, {
        '__default__': p_name,
        f'__new_{p_name}__': new,
        f'__apply_{p_name}__': apply,
        f'__check_{p_name}__': check,
    })

    assert cls.is_modelling(formula), 'Formula contains unsupported fragments.'
    return cls

def assert_complete(cls):
    assert cls.is_complete(), (
        f"Language '{cls.__name__}' is not complete. "
        f"Please implement {cls.__abstractmethods__}."
    )
    return cls