from __future__ import annotations

import itertools
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import portion as P

from pyspect.logics import AND, EVENTUALLY, NEXT, NOT, UNTIL
from pyspect.primitives import DiscLTL, ExactDiscLTL
from pyspect.tlt import APPROXDIR, TLT
from pyspect.impls.symbolic import (
    AllDifferent,
    OneVariableTransition,
    SymbolicSet,
    SymbolicSystem,
    UnrestrictedTransition,
)


def states_in_box(size: int, dimensions: int = 2):
    return list(itertools.product(range(size), repeat=dimensions))


class SymbolicTestCase(unittest.TestCase):
    def setUp(self):
        self.system = SymbolicSystem(
            ("j1", "j2"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )

    def assert_membership(self, state_set, expected, *, size=4):
        for state in states_in_box(size, len(self.system.variables)):
            self.assertEqual(state in state_set, state in expected, state)


class IntegerDomainTests(SymbolicTestCase):
    def test_open_closed_and_negative_bounds_are_integer_sets(self):
        state_set = self.system.states(j1=P.openclosed(-P.inf, 2))
        self.assertIn((0, 3), state_set)
        self.assertIn((1, 3), state_set)
        self.assertIn((2, 3), state_set)
        self.assertNotIn((3, 4), state_set)

        half_open = self.system.states(j1=P.closedopen(1, 5))
        self.assertIn((1, 0), half_open)
        self.assertIn((4, 0), half_open)
        self.assertNotIn((5, 0), half_open)

    def test_disjoint_adjacent_integer_intervals_are_canonicalized(self):
        domain = P.singleton(0) | P.singleton(1) | P.closed(2, 3)
        state_set = self.system.states(j1=domain)
        self.assertEqual(state_set.regions[0]["j1"], P.closed(0, 3))

    def test_empty_and_invalid_domains(self):
        self.assertTrue(self.system.states(j1=P.closed(-4, -1)).is_empty)
        with self.assertRaises(TypeError):
            self.system.states(j1=P.closed(0.5, 2))
        with self.assertRaises(TypeError):
            self.system.states(j1=True)


class InvariantAndAlgebraTests(SymbolicTestCase):
    def test_all_different_detects_collisions_and_hall_violations(self):
        self.assertTrue(self.system.states(j1=2, j2=2).is_empty)

        system3 = SymbolicSystem(
            ("j1", "j2", "j3"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )
        two_values = P.closed(0, 1)
        self.assertTrue(system3.states(j1=two_values, j2=two_values, j3=two_values).is_empty)
        self.assertFalse(system3.universe().is_empty)

    def test_set_algebra_matches_bounded_oracle(self):
        lhs = self.system.states(j1=P.closed(0, 1))
        rhs = self.system.states(j2=P.closed(1, 2))
        box = set(states_in_box(4))
        admissible = {state for state in box if state[0] != state[1]}
        lhs_oracle = {state for state in admissible if state[0] in {0, 1}}
        rhs_oracle = {state for state in admissible if state[1] in {1, 2}}

        self.assert_membership(lhs & rhs, lhs_oracle & rhs_oracle)
        self.assert_membership(lhs | rhs, lhs_oracle | rhs_oracle)
        self.assert_membership(~lhs, admissible - lhs_oracle)
        self.assert_membership(lhs - rhs, lhs_oracle - rhs_oracle)
        self.assertTrue((lhs & rhs).is_subset(lhs))
        self.assertFalse(lhs.is_subset(lhs & rhs))

    def test_union_does_not_merge_regions_componentwise(self):
        diagonal = self.system.states(j1=0, j2=1) | self.system.states(j1=1, j2=0)
        self.assertIn((0, 1), diagonal)
        self.assertIn((1, 0), diagonal)
        self.assertNotIn((0, 2), diagonal)
        self.assertEqual(len(diagonal.regions), 2)

    def test_sets_from_different_systems_are_rejected(self):
        other = SymbolicSystem(
            ("j1", "j2"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )
        with self.assertRaises(ValueError):
            self.system.union(self.system.universe(), other.universe())


class TransitionTests(SymbolicTestCase):
    def test_one_variable_predecessor_matches_bounded_graph(self):
        target = self.system.states(j1=P.closed(0, 1), j2=P.closed(2, 3))
        concrete = [state for state in states_in_box(4) if state[0] != state[1]]
        expected = {
            state
            for state in concrete
            if any(
                successor in target
                and sum(a != b for a, b in zip(state, successor)) <= 1
                for successor in concrete
            )
        }
        self.assert_membership(self.system.pre(target), expected)

    def test_predecessor_projection_preserves_all_different(self):
        # The target contains only (1, 2). Merely retaining j2 in {1, 2}
        # while changing j1 would incorrectly admit (2, 1).
        target = self.system.states(j1=1, j2=P.closed(1, 2))
        predecessor = self.system.pre(target)
        self.assertIn((1, 2), predecessor)  # wait/self-loop
        self.assertIn((0, 2), predecessor)
        self.assertNotIn((2, 1), predecessor)

    def test_unrestricted_transition(self):
        system = SymbolicSystem(
            ("j1", "j2"),
            invariant=AllDifferent(),
            transition=UnrestrictedTransition(),
        )
        self.assertEqual(system.pre(system.states(j1=0)), system.universe())
        self.assertTrue(system.pre(system.empty()).is_empty)

    def test_k_step_and_fixed_point_reachability(self):
        target = self.system.states(j1=0, j2=1)
        self.assertEqual(self.system.pre_k(target, 0), target)
        self.assertNotIn((2, 3), self.system.pre_k(target, 1))
        self.assertIn((2, 3), self.system.pre_k(target, 2))
        self.assertNotIn((1, 0), self.system.pre_k(target, 2))
        self.assertIn((1, 0), self.system.pre_k(target, 3))
        self.assertEqual(self.system.reach(target), self.system.universe())

    def test_constrained_reachability(self):
        target = self.system.states(j1=0, j2=1)
        allowed = self.system.states(j1=2)
        reached = self.system.reach(target, allowed)
        self.assertIn((0, 1), reached)
        self.assertIn((2, 3), reached)
        self.assertNotIn((3, 2), reached)
        self.assertEqual(reached, target | allowed)


class TLTIntegrationTests(SymbolicTestCase):
    def test_exact_eventually_with_boolean_target(self):
        tree = TLT(
            EVENTUALLY(AND("goal", "safe")),
            primitives=ExactDiscLTL,
            where={
                "goal": SymbolicSet(j1=0),
                "safe": SymbolicSet(j2=P.closed(1, 2)),
            },
        )
        self.assertEqual(tree._approx, APPROXDIR.EXACT)
        self.assertEqual(tree.realize(self.system), self.system.universe())

    def test_next_until_and_negation(self):
        goal = SymbolicSet(j1=0, j2=1)
        safe = SymbolicSet(j1=P.closed(0, 2))

        next_tree = TLT(NEXT("goal"), primitives=ExactDiscLTL, where={"goal": goal})
        self.assertEqual(
            next_tree.realize(self.system),
            self.system.pre(self.system.states(j1=0, j2=1)),
        )

        until_tree = TLT(
            UNTIL("safe", "goal"),
            primitives=ExactDiscLTL,
            where={"safe": safe, "goal": goal},
        )
        self.assertEqual(until_tree._approx, APPROXDIR.EXACT)
        self.assertEqual(
            until_tree.realize(self.system),
            self.system.reach(
                self.system.states(j1=0, j2=1),
                self.system.states(j1=P.closed(0, 2)),
            ),
        )

        not_tree = TLT(NOT("goal"), primitives=ExactDiscLTL, where={"goal": goal})
        self.assertEqual(not_tree.realize(self.system), ~self.system.states(j1=0, j2=1))

    def test_default_eventually_remains_conservative(self):
        tree = TLT(
            EVENTUALLY("goal"),
            primitives=DiscLTL,
            where={"goal": SymbolicSet(j1=0)},
        )
        self.assertEqual(tree._approx, APPROXDIR.UNDER)


class OptionalDependencyTests(unittest.TestCase):
    def test_base_package_import_does_not_import_portion(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        script = textwrap.dedent(
            """
            import sys

            class BlockPortion:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "portion" or fullname.startswith("portion."):
                        raise ModuleNotFoundError("portion deliberately blocked")
                    return None

            sys.meta_path.insert(0, BlockPortion())
            import pyspect
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source_root)
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
