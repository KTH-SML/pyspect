from math import pi

from pyspect import *
from pyspect.langs.ltl import *

TLT.select(ContinuousLTL)

class StrImpl: ...

x = EMPTY
print(x.__require__)
x = AppliedSet('complement', x)
print(x.__require__)
y = EMPTY
t = And(x, y)
print(type(t), t._builder.__require__)