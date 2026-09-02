# Symbolic integer signals

The symbolic backend computes exact reachable sets for a finite tuple of
non-negative integer-valued signals. Signals are continuous-time and
piecewise constant, while reachability follows the ordered sequence of event
states. No physical duration is assigned to a transition: `NEXT` means the
next transition state, and a self-loop represents waiting.

Install the optional dependency with:

```bash
pip install pyspect[symbolic]
```

## Example

```python
import portion as P

from pyspect import AND, EVENTUALLY, ExactDiscLTL, TLT
from pyspect.impls.symbolic import (
    AllDifferent,
    OneVariableTransition,
    SymbolicSet,
    SymbolicSystem,
)

system = SymbolicSystem(
    variables=("j1", "j2"),
    invariant=AllDifferent(),
    transition=OneVariableTransition(),
)

tlt = TLT(
    EVENTUALLY(AND("goal", "safe")),
    primitives=ExactDiscLTL,
    where={
        "goal": SymbolicSet(j1=P.singleton(2)),
        "safe": SymbolicSet(j2=P.closedopen(1, 5)),
    },
)

reachable = tlt.realize(system)
```

Every `portion` interval is interpreted at integer points and restricted to
`NN0`. For example, `P.closedopen(1, 5)` denotes `{1, 2, 3, 4}`. Variables not
mentioned by a `SymbolicSet` range over all non-negative integers, subject to
the global `AllDifferent` invariant.

## Set and transition semantics

`StateSet` represents a normalized finite union of symbolic regions. The
system provides exact union, intersection, complement relative to the
admissible state space, difference, membership, and subset tests.

`OneVariableTransition` permits at most one signal to change in a transition,
including a self-loop. `UnrestrictedTransition` permits every transition
between admissible states. Both are existential models: a predecessor is
included when at least one legal successor is in the target.

Use `system.pre(target)` for one transition, `system.pre_k(target, steps)` for
a fixed number of transitions, and `system.reach(target, constraints)` for the
least predecessor fixed point. `ExactDiscLTL` maps `NEXT`, `UNTIL`, and
`EVENTUALLY` to these exact operations.

Quantitative time, universal predecessors, other relational invariants, and
bounded temporal operators are not part of this backend.

## Reference

::: pyspect.impls.symbolic
