# Get Started

Follow these steps to install **pyspect** and run your first spec.

## 1. Installation

```bash
pip install pyspect

# With Hamilton–Jacobi backend (optional)
pip install pyspect[hj_reachability]

# From source
python -m venv .venv && source .venv/bin/activate
pip install -e ".[hj_reachability]"
```

## 2. Write your first spec

```python
from pyspect.logics import *
from pyspect.tlt import TLT, ContLTL
from pyspect.sets import BoundedSet, Union, HalfSpaceSet
from pyspect.impls.hj_reachability import HJImpl

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
impl = HJImpl(...)
Phi = tlt.realize(impl)
print("Satisfaction set:", Phi)
```

## 3. Next Steps

<!-- * Check the [User Guide](user-guide/) for details on **logic fragments**, **set-builders**, and **backend interfaces**. -->
Explore example notebooks in `examples/`.
