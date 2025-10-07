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


## Paper Examples

For paper examples, checkout branches:

- [`cdc24`](https://github.com/KTH-SML/pyspect/tree/cdc24/examples): Intersection & Parking
- [`cdc25`](https://github.com/KTH-SML/pyspect/tree/cdc25/examples): Double Integrator with HJ & HZ


<!--

## Core ideas

### I. Logic as a tiny AST
1. Write formulas (e.g. LTL) as lightweight, typed tuples: 
   `('AND', a, b)`, `('NOT', a)`, `('UNTIL', phi, psi)`, etc.
2. Symbols (propositions) are bound later via a mapping `M: AP -> SetBuilder`.

### II. Temporal Logic Trees (TLTs)
> A **TLT** mirrors formula structure with set/reachability nodes, verifying temporal logic using reachability.

1. `TLT.select(Q)` chooses a set of primitives `Q` matching the temporal logic fragment. **Key:** the primitives operationalize the fragment (how we evaluate).
2. `TLT(spec).realize(impl)` constructs and executes a reachability program verifying `spec`.

### III. Implementations

...

### IV. Set builders (lazy, backend-agnostic)
1. **SetBuilder** objects (`B: Impl -> R`) describe sets *implicitly* and are evaluated only when realized by an implementation (“dependency injection”).
2. pyspect provides common combinators: `Union`, `Inter`, `Compl`, plus constructors like `BoundedSet(…)`. (Backends/Implementations supply the actual set operations.)

-->
