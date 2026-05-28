# Get Started

Follow these steps to install **pyspect** and run your first spec.

## 0. Prerequisites

- Python **>= 3.12**

## 1. Installation

```bash
pip install pyspect

# With optional implementation
pip install pyspect[hj_reachability]
```

## 2. Write your first spec

```python
from pyspect.logics import *
from pyspect.tlt import TLT, ContLTL
from pyspect.set_builder import BoundedSet, Union, HalfSpaceSet
from pyspect.impls.hj_reachability import TVHJImpl

# Choose fragment
TLT.select(ContLTL)

# Write spec: avoid D, stay in corridor, reach goal
phi = UNTIL(AND(NOT('D'), 'corridor'), 'goal')

# Bind propositions later
tlt = TLT(phi, where={
    'D': Union(BoundedSet(...), BoundedSet(...)),
    'corridor': BoundedSet(...),
    'goal': HalfSpaceSet(...),
})

# Realize on a backend
impl = TVHJImpl(...)
Phi = tlt.realize(impl)
print("Satisfaction set:", Phi)
```

## 3. Next Steps

- **[Set Builders Guide](tutorials/set_builder_guide.md)** — describe boxes, half-spaces, and simple shapes (start here before temporal logic)
- [Set Builders reference](reference/set_builder.md) — full API
- Example notebooks in `examples/` (3D figures: `explore_3d_setbuilders.ipynb`)
