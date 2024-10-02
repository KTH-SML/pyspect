from abc import ABCMeta, abstractmethod
from typing import Union, Tuple, Set, Dict, Any
from contextlib import contextmanager

__all__ = (
    'Expr',
    'Void',
    'Language',
    'LanguageUser',
)

class ImplClientMeta(type):

    Impl: type

    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]) -> 'Language':

        ## Dynamic Construction of Inherited Implementations ## 
        
        impls = []

        # Check if current namespace has an Impl
        if 'Impl' in namespace:
            impls.append(namespace.pop('Impl'))
        
        # Collect all Impls from base classes
        for base in bases:
            Impl = getattr(base, 'Impl', None)
            if Impl is not None:
                impls.append(Impl)

        # Construct the new Impl
        namespace['Impl'] = type(f'Impl{name}', tuple(impls), {
            '__module__': '<dynamic>',
        })

        try:
            return super().__new__(mcs, name, bases, namespace)
        except TypeError as e:
            print(impls)
            for impl in impls:
                print(impl.__qualname__, dir(impl))
            raise e

BiOp = Tuple[str, 'Expr', 'Expr'] 
UnOp = Tuple[str, 'Expr']
Term = Tuple[str]
Expr = Union[BiOp, UnOp, Term]

class Language(ABCMeta, ImplClientMeta, type):

    __declared__: bool
    __fragments__: Set[str]

    @classmethod
    def declare(mcs, name: str) -> 'Language':
        return mcs(name, (), {
            '_declared': True, # indicates direct declaration
            f'_apply__{name}': abstractmethod(lambda self: None),
            f'_check__{name}': abstractmethod(lambda self: None),
        })
    
    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]) -> 'Language':
    
        ##  Mark Direct Declaration ##

        namespace['__declared__'] = namespace.pop('_declared', False)

        ## Check Supported Language Fragments ##

        apply_funcs = []
        check_funcs = []
        
        # Collect all `_apply__*` and `_check__*` methods
        for member in namespace.keys():
            if member.startswith('_apply__'):
                apply_funcs.append(member[len('_apply__'):])
            elif member.startswith('_check__'):
                check_funcs.append(member[len('_check__'):])
        
        # To be recognized as a language fragment, both
        # methods need to exist. For example, "A" is a
        # language fragment iff the language class has 
        # both `_apply__A` and `_check__A`.
        fragments = set(apply_funcs) & set(check_funcs)

        # Collect inherited fragments
        # NOTE: For now, we only inherit fragments from 
        #       language types.
        for base in bases:
            if isinstance(base, Language):
                fragments |= base.__fragments__

        # Assign fragments to the newly constructed language class
        namespace['__fragments__'] = fragments

        ## Return Language Fragment Type ##

        return super().__new__(mcs, name, bases, namespace)                

    def __repr__(cls) -> str:
        return f"<language '{cls.__name__}'>"
    
    def __call__(cls, *args, **kwds) -> Expr:
        if cls.__declared__:
            ## Easy Formula Creation ##
            
            # NOTE: This overrides the ability to instantiate objects,
            #       but this is not necessary for the language classes
            #       which we only use for type checking purposes.

            return (cls.__name__, *args)
        else:
            return super().__call__(*args, **kwds)
    
    def is_complete(cls) -> bool:
        return not bool(cls.__abstractmethods__)

    @contextmanager    
    def In(cls, usr: 'LanguageUser'):
        cls_name = cls.__name__
        usr_name = usr.__language__.__name__
        assert issubclass(usr.__language__, cls), \
            f'Selected language "{usr_name}" does not support {cls_name}'
        yield

# The Void language is a singleton for a trivial language that 
# puts no restriction on the implementation. It has only one 
# purpose, to be the by default selected language of TLTs.
Void = Language('Void', (), {})

class LanguageUser:

    __language__: Language = Void

    @classmethod
    def select(cls, lang: Language):
        # We confirm that all abstracts are implemented
        assert lang.is_complete(), \
            f'{lang.__name__} is not complete, missing: {", ".join(lang.__abstractmethods__)}'
        cls.__language__ = lang
