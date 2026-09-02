"""Exact symbolic reachability for non-negative integer-valued signals.

The backend represents sets of states as finite unions of Cartesian regions.
Each region is implicitly intersected with one global ``AllDifferent``
invariant.  One-dimensional domains are supplied as :mod:`portion` intervals,
but are interpreted as sets of integer points in ``NN0``.

``SymbolicSystem`` implements the operations consumed by a temporal-logic
tree, so it can be passed directly to :meth:`pyspect.tlt.TLT.realize`.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Mapping

import portion as P

from .dev.base import Impl
from ..set_builder import SetBuilder

__all__ = (
    "AllDifferent",
    "OneVariableTransition",
    "StateSet",
    "SymbolicSet",
    "SymbolicSystem",
    "UnrestrictedTransition",
)


type Domain = P.Interval


def _integer_bound(bound: Any) -> int:
    if not isinstance(bound, Integral) or isinstance(bound, bool):
        raise TypeError(f"Symbolic domain bounds must be integers, got {bound!r}.")
    return int(bound)


def _closed_interval(lower: int, upper: int | Any) -> Domain:
    if upper == P.inf:
        return P.closedopen(lower, P.inf)
    return P.closed(lower, upper)


def _canonical_domain(domain: Domain | Integral) -> Domain:
    """Interpret a portion interval as a canonical subset of ``NN0``."""
    if isinstance(domain, Integral) and not isinstance(domain, bool):
        domain = P.singleton(int(domain))
    if not isinstance(domain, P.Interval):
        raise TypeError(
            "Symbolic domains must be portion.Interval instances or integer values, "
            f"got {type(domain).__name__}."
        )

    integer_ranges: list[tuple[int, int | Any]] = []
    for atom in domain:
        if atom.lower == P.inf or atom.upper == -P.inf:
            continue

        if atom.lower == -P.inf:
            lower = 0
        else:
            lower = _integer_bound(atom.lower)
            if atom.left == P.OPEN:
                lower += 1
            lower = max(0, lower)

        if atom.upper == P.inf:
            upper = P.inf
        else:
            upper = _integer_bound(atom.upper)
            if atom.right == P.OPEN:
                upper -= 1

        if upper != P.inf and lower > upper:
            continue
        integer_ranges.append((lower, upper))

    if not integer_ranges:
        return P.empty()

    integer_ranges.sort(key=lambda bounds: bounds[0])
    merged: list[list[int | Any]] = []
    for lower, upper in integer_ranges:
        if not merged:
            merged.append([lower, upper])
            continue

        previous = merged[-1]
        previous_upper = previous[1]
        if previous_upper == P.inf or lower <= previous_upper + 1:
            if previous_upper != P.inf and (upper == P.inf or upper > previous_upper):
                previous[1] = upper
        else:
            merged.append([lower, upper])

    out = P.empty()
    for lower, upper in merged:
        out |= _closed_interval(int(lower), upper)
    return out


_NN0 = _canonical_domain(P.closedopen(0, P.inf))


def _domain_intersection(lhs: Domain, rhs: Domain) -> Domain:
    return _canonical_domain(lhs & rhs)


def _domain_difference(lhs: Domain, rhs: Domain) -> Domain:
    return _canonical_domain(lhs - rhs)


def _domain_subset(lhs: Domain, rhs: Domain) -> bool:
    return _domain_difference(lhs, rhs).empty


def _domain_values(domain: Domain, limit: int) -> list[int]:
    """Return at most ``limit`` integer values from a canonical domain."""
    values: list[int] = []
    for atom in domain:
        lower = int(atom.lower)
        if atom.upper == P.inf:
            stop = lower + (limit - len(values))
        else:
            stop = min(int(atom.upper) + 1, lower + (limit - len(values)))
        values.extend(range(lower, stop))
        if len(values) == limit:
            break
    return values


@dataclass(frozen=True, slots=True)
class AllDifferent:
    """Invariant requiring all variables in a system to have distinct values."""

    def is_feasible(self, domains: tuple[Domain, ...]) -> bool:
        """Test feasibility using a bounded bipartite matching problem."""
        count = len(domains)
        candidates = [_domain_values(domain, count) for domain in domains]
        if any(not values for values in candidates):
            return False

        matched: dict[int, int] = {}

        def augment(variable: int, seen: set[int]) -> bool:
            for value in candidates[variable]:
                if value in seen:
                    continue
                seen.add(value)
                if value not in matched or augment(matched[value], seen):
                    matched[value] = variable
                    return True
            return False

        order = sorted(range(count), key=lambda variable: len(candidates[variable]))
        return all(augment(variable, set()) for variable in order)


@dataclass(frozen=True, slots=True)
class _Region:
    domains: tuple[Domain, ...]


class StateSet:
    """A normalized finite union of symbolic state regions."""

    __slots__ = ("_system", "_regions")

    def __init__(self, system: SymbolicSystem, regions: Iterable[_Region] = ()) -> None:
        self._system = system
        self._regions = system._normalize_regions(regions)

    @property
    def system(self) -> SymbolicSystem:
        """The symbolic system whose admissible state space contains this set."""
        return self._system

    @property
    def regions(self) -> tuple[dict[str, Domain], ...]:
        """Return the normalized regions as variable-to-domain mappings."""
        return tuple(
            dict(zip(self._system.variables, region.domains))
            for region in self._regions
        )

    @property
    def is_empty(self) -> bool:
        return not self._regions

    def is_subset(self, other: StateSet) -> bool:
        return self._system.is_subset(self, other)

    def __contains__(self, state: object) -> bool:
        if not isinstance(state, (tuple, list)) or len(state) != len(self._system.variables):
            return False
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or value < 0
            for value in state
        ):
            return False
        values = tuple(int(value) for value in state)
        if len(values) != len(set(values)):
            return False
        return any(
            all(value in domain for value, domain in zip(values, region.domains))
            for region in self._regions
        )

    def __bool__(self) -> bool:
        return not self.is_empty

    def __and__(self, other: StateSet) -> StateSet:
        return self._system.intersect(self, other)

    def __or__(self, other: StateSet) -> StateSet:
        return self._system.union(self, other)

    def __invert__(self) -> StateSet:
        return self._system.complement(self)

    def __sub__(self, other: StateSet) -> StateSet:
        return self._system.difference(self, other)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StateSet)
            and self._system is other._system
            and self.is_subset(other)
            and other.is_subset(self)
        )

    def __repr__(self) -> str:
        if self.is_empty:
            return "StateSet()"
        return f"StateSet({self.regions!r})"


class SymbolicSet(SetBuilder[StateSet]):
    """Lazy builder for an atomic symbolic set.

    Unspecified variables range over all of ``NN0``.  Domains may be supplied
    using a mapping, keyword arguments, or both.
    """

    __require__ = ("states",)

    def __init__(
        self,
        domains: Mapping[str, Domain | Integral] | None = None,
        **kwdomains: Domain | Integral,
    ) -> None:
        merged = dict(domains or {})
        overlap = merged.keys() & kwdomains.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate symbolic domains for: {names}.")
        merged.update(kwdomains)
        self.domains = merged

    def __call__(self, impl: Impl[StateSet], **m: SetBuilder[StateSet]) -> StateSet:
        return impl.states(self.domains)


@dataclass(frozen=True, slots=True)
class OneVariableTransition:
    """Transition relation in which at most one variable changes."""

    def pre(self, system: SymbolicSystem, target: StateSet) -> StateSet:
        count = len(system.variables)
        regions: list[_Region] = []
        for target_region in target._regions:
            for changed in range(count):
                successor_values = _domain_values(target_region.domains[changed], count)
                if len(successor_values) == count:
                    domains = list(target_region.domains)
                    domains[changed] = _NN0
                    regions.append(_Region(tuple(domains)))
                    continue

                for successor_value in successor_values:
                    singleton = P.singleton(successor_value)
                    domains = [
                        _NN0 if index == changed else _domain_difference(domain, singleton)
                        for index, domain in enumerate(target_region.domains)
                    ]
                    regions.append(_Region(tuple(domains)))
        return StateSet(system, regions)


@dataclass(frozen=True, slots=True)
class UnrestrictedTransition:
    """Transition relation connecting every pair of admissible states."""

    def pre(self, system: SymbolicSystem, target: StateSet) -> StateSet:
        return system.empty() if target.is_empty else system.universe()


class SymbolicSystem(Impl[StateSet]):
    """Exact TLT backend for tuples of non-negative integer signals."""

    def __init__(
        self,
        variables: Iterable[str],
        *,
        invariant: AllDifferent,
        transition: OneVariableTransition | UnrestrictedTransition,
    ) -> None:
        self.variables = tuple(variables)
        if not self.variables:
            raise ValueError("A symbolic system needs at least one variable.")
        if any(not isinstance(variable, str) or not variable for variable in self.variables):
            raise TypeError("Symbolic variable names must be non-empty strings.")
        if len(self.variables) != len(set(self.variables)):
            raise ValueError("Symbolic variable names must be unique.")
        if not isinstance(invariant, AllDifferent):
            raise TypeError("The symbolic backend currently supports only AllDifferent().")
        if not isinstance(transition, (OneVariableTransition, UnrestrictedTransition)):
            raise TypeError(
                "transition must be OneVariableTransition() or UnrestrictedTransition()."
            )
        self.invariant = invariant
        self.transition = transition

    def _check_sets(self, *sets: StateSet) -> None:
        for state_set in sets:
            if not isinstance(state_set, StateSet):
                raise TypeError(f"Expected StateSet, got {type(state_set).__name__}.")
            if state_set.system is not self:
                raise ValueError("Cannot combine StateSets from different SymbolicSystems.")

    @staticmethod
    def _region_subset(lhs: _Region, rhs: _Region) -> bool:
        return all(
            _domain_subset(lhs_domain, rhs_domain)
            for lhs_domain, rhs_domain in zip(lhs.domains, rhs.domains)
        )

    def _normalize_regions(self, regions: Iterable[_Region]) -> tuple[_Region, ...]:
        kept: list[_Region] = []
        for region in regions:
            if len(region.domains) != len(self.variables):
                raise ValueError("Region dimensionality does not match the symbolic system.")
            normalized = _Region(tuple(_canonical_domain(domain) for domain in region.domains))
            if not self.invariant.is_feasible(normalized.domains):
                continue
            if any(self._region_subset(normalized, existing) for existing in kept):
                continue
            kept = [
                existing
                for existing in kept
                if not self._region_subset(existing, normalized)
            ]
            kept.append(normalized)
        return tuple(kept)

    def states(
        self,
        domains: Mapping[str, Domain | Integral] | None = None,
        **kwdomains: Domain | Integral,
    ) -> StateSet:
        """Construct a symbolic region; unspecified variables range over ``NN0``."""
        supplied = dict(domains or {})
        overlap = supplied.keys() & kwdomains.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate symbolic domains for: {names}.")
        supplied.update(kwdomains)
        unknown = supplied.keys() - set(self.variables)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown symbolic variables: {names}.")
        region = _Region(tuple(
            _canonical_domain(supplied.get(variable, _NN0))
            for variable in self.variables
        ))
        return StateSet(self, (region,))

    def empty(self) -> StateSet:
        return StateSet(self)

    def universe(self) -> StateSet:
        return self.states()

    def intersect(self, lhs: StateSet, rhs: StateSet) -> StateSet:
        self._check_sets(lhs, rhs)
        return StateSet(self, (
            _Region(tuple(
                _domain_intersection(lhs_domain, rhs_domain)
                for lhs_domain, rhs_domain in zip(lhs_region.domains, rhs_region.domains)
            ))
            for lhs_region in lhs._regions
            for rhs_region in rhs._regions
        ))

    def union(self, lhs: StateSet, rhs: StateSet) -> StateSet:
        self._check_sets(lhs, rhs)
        return StateSet(self, (*lhs._regions, *rhs._regions))

    def complement(self, state_set: StateSet) -> StateSet:
        self._check_sets(state_set)
        result = self.universe()
        for region in state_set._regions:
            complement_regions = []
            for index, domain in enumerate(region.domains):
                complement = _domain_difference(_NN0, domain)
                if complement.empty:
                    continue
                domains = [_NN0] * len(self.variables)
                domains[index] = complement
                complement_regions.append(_Region(tuple(domains)))
            result = self.intersect(result, StateSet(self, complement_regions))
        return result

    def difference(self, lhs: StateSet, rhs: StateSet) -> StateSet:
        self._check_sets(lhs, rhs)
        return self.intersect(lhs, self.complement(rhs))

    def is_subset(self, lhs: StateSet, rhs: StateSet) -> bool:
        self._check_sets(lhs, rhs)
        return self.difference(lhs, rhs).is_empty

    def pre(self, target: StateSet, constraints: StateSet | None = None) -> StateSet:
        """Return existential one-transition predecessors of ``target``."""
        self._check_sets(target)
        out = self.transition.pre(self, target)
        if constraints is not None:
            self._check_sets(constraints)
            out = self.intersect(out, constraints)
        return out

    def pre_k(
        self,
        target: StateSet,
        steps: int,
        constraints: StateSet | None = None,
    ) -> StateSet:
        """Apply :meth:`pre` exactly ``steps`` times."""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")
        self._check_sets(target)
        if constraints is not None:
            self._check_sets(constraints)
        out = target
        for _ in range(steps):
            out = self.pre(out, constraints)
        return out

    def reach(self, target: StateSet, constraints: StateSet | None = None) -> StateSet:
        """Return the constrained existential predecessor fixed point."""
        self._check_sets(target)
        allowed = self.universe() if constraints is None else constraints
        self._check_sets(allowed)

        current = target
        while True:
            candidate = self.union(
                target,
                self.intersect(allowed, self.transition.pre(self, current)),
            )
            if candidate.is_subset(current):
                return candidate
            current = candidate
