---
title: Home
summary: Compile temporal-logic specs into reachability programs.
---

pyspect lets you write a specification **once** and realize it on interchangeable backends through
**Temporal Logic Trees (TLTs)**. It performs **interface + approximation checks** before any heavy
computation so your resulting satisfaction set remains **sound**.

## Why pyspect

- **Decouple** logic, set semantics and numerics → write specs once, compare reachability methods side-by-side.  
- **Method-agnostic & extensible** → ships with two ready-made backends for very different reachability methods (Hamilton-Jacobi and Hybrid Zonotopes).
- **Correct-by-construction** → static checks for **approximation direction** (over/under) and backend capability up-front.

## Quick Peek

```python
T = BoundedSet(x=(-50,  +50))
TASK = ALWAYS(T)

TLT.select(ContLTL)
objective = TLT(TASK, where={'goal': GOAL})

# vvv Implementation-specific vvv

from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.systems.hj_reachability import *

impl = TVHJImpl(
    dict(cls=Bicycle4D,
         wheelbase=2.7,
         min_accel=-MAX_ACCEL,
         max_accel=+MAX_ACCEL,
         min_steer=-MAX_STEER,
         max_steer=+MAX_STEER), 
    AXES,
)

out = objective.realize(impl)

impl.plot(out, axes=('x', 'y', 't')).show()
```

> For installation and full examples → **[Get Started](get_started.md)**

## Cite

```bibtex
@inproceedings{11311974,
	title = {pyspect: An Extensible Toolbox for Automatic Construction of Temporal Logic Trees via Reachability Analysis},
	issn = {2576-2370},
	url = {https://ieeexplore.ieee.org/abstract/document/11311974},
	doi = {10.1109/CDC57313.2025.11311974},
	shorttitle = {pyspect},
	pages = {6911--6918},
	booktitle = {2025 {IEEE} 64th Conference on Decision and Control ({CDC})},
	author = {Arfvidsson, Kaj Munhoz and Hadjiloizou, Loizos and Jiang, Frank J. and Johansson, Karl H. and Mårtensson, Jonas},
	date = {2025-12},
}
```
