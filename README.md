# pyspect

**Compile temporal-logic specs into reachability programs.**

pyspect lets you write specifications once, then realize them against interchangeable backends (e.g., Hamilton–Jacobi level sets or Hybrid Zonotopes) via **Temporal Logic Trees (TLTs)**. The toolbox performs interface + approximation checks so your output set remains sound.

> **TL;DR**: You express *what* to verify in logic; implementations decide *how* to compute the sets. pyspect bridges the two with TLTs and small, pluggable operator primitives. See paper *"pyspect: An Extensible Toolbox for Automatic Construction of Temporal Logic Trees via Reachability Analysis."*

## Why pyspect

- **Decouple** logic, set semantics, and numerics: write specs once, compare multiple reachability methods side-by-side.
- **Method-agnostic** and easily extensible: currently supports HJ (level sets) and HZ-based (hybrid zonotopes) reachability within the same interface.
- **Correct by construction**: static checks for **approximation direction** (over/under) and backend capability before any heavy computation, avoiding invalid evaluation of the spec.

## Getting started

[Read our docs here!](https://kth-sml.github.io/pyspect)

### Installation

```bash

pip install pyspect

# Example to install with implementation-specific dependencies (Optional)
pip install pyspect[hj_reachability]

# From source
python -m venv .venv && source .venv/bin/activate
pip install -e ".[hj_reachability]"
```

### Program

```python
from pyspect.logics import *
from pyspect.tlt import TLT, ContLTL
from pyspect.impls.hj_reachability import HJImpl

# 1) Pick primitives/fragment
TLT.select(ContLTL) # Continuous-time LTL

# 2) Write the spec once
phi = UNTIL(AND(NOT('D'), 'corridor'), 'goal')   # task: avoid D, stay in corridor, then reach goal

# 3) Bind propositions later via set builders
tlt = TLT(phi, where={
    'D': Union(BoundedSet(...), BoundedSet(...)),
    'corridor': BoundedSet(...),
    'goal': HalfSpaceSet(...),
})

# 4) Realize on a backend
impl = HJImpl(...)      # Each implementation can have their own settings
Phi = tlt.realize(impl) # The satisfaction set in the backend’s representation
```

## Cite

If you use pyspect in academic work, please cite:

> TBA
