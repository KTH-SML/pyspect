# Hybrid Zonotopes backend

In pyspect, a temporal-logic specification is compiled into a **Temporal Logic
Tree (TLT)** and **realized** by a backend. `ZonoOptImpl` is the backend for
**discrete-time linear reachability** with **hybrid zonotopes**, using the
external [`zonoopt`](https://github.com/kasnerz/zonoopt) library. Sets are
propagated as zonotopes over a fixed horizon instead of gridding the state
space.

The logic layer is **backend-agnostic**. The same spec can be realized here or
on [Hamilton–Jacobi](hj_reachability.md) level sets for comparison.

## When to use it

**Linear discrete-time** dynamics, moderate dimension, fast set propagation
without a grid. Less suited to nonlinear continuous-time systems.

## Getting started

Install after the [base package](../tutorials/get_started.ipynb):

```bash
pip install pyspect[zonoopt]
```

Provide state bounds via `axes`, control bounds as a zonotope `U`, and set
`time_horizon` / `time_step` at construction. See
[CDC '25](../papers/cdc25.ipynb) for HJ vs HZ comparisons on the same specs.

## Reference

::: pyspect.impls.zonoopt
