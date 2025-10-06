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
from pyspect.logics import *
from pyspect.tlt import TLT, ContLTL

TLT.select(ContLTL)
phi = UNTIL(AND(NOT('D'), 'corridor'), 'goal')
```

> For installation and full examples → **[Get Started](get_started.md)**

## Cite

```bibtex
TBD
```
