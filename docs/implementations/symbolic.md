# Symbolic integer signals

The symbolic backend computes exact reachable sets for a finite tuple of
non-negative integer-valued signals. Signals are continuous-time and
piecewise constant. By default each transition takes an arbitrary positive
duration; specifying `time_step` fixes every transition's duration. Untimed
reachability still follows the ordered sequence of event states: `NEXT` means
the next transition state, and a self-loop represents waiting.

Install the optional dependency with:

```bash
pip install pyspect[symbolic]
```

## Example

```python
import portion as P

from pyspect import AND, EVENTUALLY, ExactDiscLTL, TLT
from pyspect.impls.symbolic import (
    AllDifferentExceptZero,
    OneVariableTransition,
    SymbolicSet,
    SymbolicSystem,
)

system = SymbolicSystem(
    variables=("j1", "j2"),
    invariant=AllDifferentExceptZero(),
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
the global invariant. `AllDifferentExceptZero()` models charging allocations:
positive job IDs must be distinct, but any number of charging points may be
unallocated (`0`). Use `AllDifferent()` when zeros must also be distinct.

## Plotting exact lattice views

Plot the fixed-point growth of an `UNTIL` or `EVENTUALLY` result against the
synthetic reach-iteration axis `k`:

```python
target = system.states(j1=2, j2=P.closedopen(1, 5))

fig = system.plot(
    (reachable, {"name": "reachable", "color": "#4C78A8"}),
    (target, {"name": "target", "color": "#F58518"}),
    axes=("k", "j1"),
    layout_title="Exact symbolic reachability",
)
fig.show()
```

Here `k=0` is the target and later columns are the cumulative approximants of
the existing `system.reach()` fixed-point calculation. It is computational
provenance, not physical time or a change to the temporal-logic semantics.
Reach history is attached only to a result returned directly by `reach()`,
including a TLT whose outer operation is `UNTIL` or `EVENTUALLY`. Boolean,
`NEXT`, `pre()`, and `pre_k()` results do not carry reach history and therefore
cannot be plotted against `k`. Static layers plotted alongside a traced layer
are repeated over its iterations, and a shorter history repeats its fixed
point through the longest displayed history.

Plotting two state axes works as well:

```python
system.plot(reachable, axes=("j1", "j2"))
```

Finite bounds are inferred with one cell of padding. An unbounded or empty axis
uses `0..max(10, largest finite boundary + 1)`. Supply a partial or complete
`window` to override the inferred inclusive bounds:

```python
system.plot(reachable, axes=("j1", "j2"), window={"j2": (0, 20)})
```

Every displayed integer cell is classified exactly. Gray crosses mark states
excluded by the chosen invariant, while triangles at an edge indicate that a layer
also contains states beyond that edge.

Hidden variables are existentially projected by default. Fix any subset with
`select`, or require a complete slice by setting `project=False`:

```python
system3.plot(result, axes=("j1", "j2"), select={"j3": 4})
system3.plot(
    result,
    axes=("j1", "j2"),
    select={"j3": 4},
    project=False,
)
```

The string `"k"` is reserved for the synthetic plotting axis. If a system has
a state signal named `k`, refer to that signal by its numeric index in `axes`,
`window`, and `select`.

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

## Physical-time queries

The timed operations accept **state predicates** as `StateSet` arguments:

```python
both_free = system.states(j1=0, j2=0)
occupied = ~system.states(j1=0)

# Numeric windows are relative and half-open by default.
system.reach_timed(both_free, 3)                # eventually in [0, 3)
system.reach_timed(both_free, 3, closed=True)   # eventually in [0, 3]
system.pre_timed(both_free, 3)                 # free at exactly time 3

# Portion intervals are absolute, clipped to the future of the anchor.
system.reach_timed(
    both_free, P.closedopen(5, 8), occupied, anchor=3,
)  # occupied from 3 until a both-free witness in [5, 8)
system.invariant_timed(both_free, P.closedopen(5, 8), anchor=6)
# Some trajectory keeps both free throughout [6, 8).
```

Omitting the window means the entire future. The represented time universe
is all non-negative physical time; queries do not currently accept a separate
represented-time mask. Interval unions are supported. `closed` controls only
numeric duration windows; absolute intervals retain their own endpoints.
An empty window yields the empty set for reachability and the universe for
invariance. In particular, bare duration `0` is empty, while `0, closed=True`
checks the predicate at the anchor.

With the default `time_step=None`, transition durations are independently
chosen positive real values, without a positive minimum. Trajectories have
only finitely many changes on every bounded time interval. Any finite path
can therefore fit into any positive duration, so a positive deadline alone
does not bound its number of transitions. Constraints still hold during all
waiting before the witness, even before the lower edge of an absolute window.

For fixed timing, supply a positive `time_step`:

```python
fixed = SymbolicSystem(
    ("j1", "j2"),
    invariant=AllDifferentExceptZero(),
    transition=OneVariableTransition(),
    time_step=0.5,
)
goal = fixed.states(j1=0, j2=0)
assert (1, 2) not in fixed.reach_timed(goal, 1)
assert (1, 2) in fixed.reach_timed(goal, 1, closed=True)
```

Each timed query starts a fresh holding period at its anchor `a`. State `x_k`
holds on `[a + k*time_step, a + (k+1)*time_step)`, including the left edge.
Thus exact-time queries inspect holding periods, not only event times. An
until witness at an event edge need not satisfy the prefix constraint; one
strictly inside a holding period must also satisfy that constraint, since
the same state holds immediately before the witness. Decimal float spellings
are used for timing arithmetic (`0.3` is three `0.1` steps); `Fraction` inputs
can express exact rational times.

`invariant_timed` computes existential invariance directly. Both supported
transition relations allow self-loops, so a trajectory can stay in a predicate
forever after reaching it. It is not computed by complementing existential
reachability of the complement.

These APIs do not add interval syntax, selection masks, or general
single-trajectory temporal composition to `ExactDiscLTL`. Its `EXACT` label
describes set operations and does not establish exact existential trajectory
satisfaction for arbitrary temporal negations or conjunctions. Timed queries
do not preserve a residual clock phase across calls; modelling an ongoing
fixed schedule at arbitrary anchors would require that phase in the state.
Bounded variable dwell times, universal predecessors, and other relational
invariants are not supported.

## Reference

::: pyspect.impls.symbolic
