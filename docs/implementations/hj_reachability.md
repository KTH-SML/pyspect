# Hamilton-Jacobi backend

In pyspect, a temporal-logic specification is compiled into a **Temporal Logic
Tree (TLT)** and **realized** by a backend. `TVHJImpl` is the backend for
**Hamilton–Jacobi reachability**: it maintains a value function on a state grid
and updates it through reach/avoid/pre operators, using the external
[`hj_reachability`](https://github.com/HJReachability/hj_reachability) library.

The logic layer — formulas, set builders, approximation checks — is
**backend-agnostic**. The same spec can be realized on HJ or on
[Hybrid Zonotopes](zonoOpt.md) to compare methods side-by-side.

## When to use it

Continuous-time dynamics, possibly **nonlinear** or with **adversarial** inputs.
Grid-based **level sets**; computational cost grows quickly with state
dimension.

## Getting started

Install after the [base package](../tutorials/get_started.ipynb):

```bash
pip install pyspect[hj_reachability]
```

Pass dynamics as `dict(cls=..., **params)` with `cls` from
`pyspect.systems.hj_reachability`. Grid axes must list **`t` first**, then
spatial dimensions (`bounds`, `points`). See
[Getting Started](../tutorials/get_started.ipynb),
[Creating Sets](../tutorials/set_builders.ipynb), and
[CDC '25](../papers/cdc25.ipynb).

## Reference

::: pyspect.impls.hj_reachability
