from fractions import Fraction
import itertools
import unittest

import portion as P

from pyspect.impls.symbolic import (
    AllDifferent, AllDifferentExceptZero, OneVariableTransition,
    SymbolicSystem, UnrestrictedTransition,
)


def system(step=None, invariant=None, transition=None):
    return SymbolicSystem(
        ('j1', 'j2'), invariant=invariant or AllDifferentExceptZero(),
        transition=transition or OneVariableTransition(), time_step=step,
    )


class ChargingInvariantTests(unittest.TestCase):
    def test_zero_is_reusable_but_positive_ids_are_not(self):
        s = system()
        self.assertIn((0, 0), s.universe())
        self.assertFalse(s.states(j1=0, j2=0).is_empty)
        self.assertNotIn((1, 1), s.universe())
        self.assertTrue(s.states(j1=1, j2=1).is_empty)
        self.assertNotIn((0, 0), system(invariant=AllDifferent()).universe())
        s3 = SymbolicSystem(('a', 'b', 'c'), invariant=AllDifferentExceptZero(),
                            transition=OneVariableTransition())
        self.assertIn((0, 0, 0), s3.states(a=P.closed(0, 1), b=0, c=0))
        self.assertTrue(s3.states(a=P.closed(1, 2), b=P.closed(1, 2),
                                  c=P.closed(1, 2)).is_empty)

    def test_algebra_and_predecessors_against_finite_graph(self):
        s = system()
        box = list(itertools.product(range(3), repeat=2))
        admissible = [x for x in box if x[0] == 0 or x[1] == 0 or x[0] != x[1]]
        # Includes a target with two zeros and targets with competing IDs.
        for domains in itertools.product((0, 1, P.closed(0, 1), P.closed(1, 2)), repeat=2):
            target = s.states(j1=domains[0], j2=domains[1])
            pre = s.pre(target)
            complement = ~target
            for x in box:
                expected = x in admissible and any(
                    y in target and sum(a != b for a, b in zip(x, y)) <= 1
                    for y in admissible
                )
                self.assertEqual(x in pre, expected, (domains, x))
                self.assertEqual(x in complement, x in admissible and x not in target)

    def test_plot_marks_positive_collisions_but_not_repeated_zeros(self):
        s = system()
        fig = s.plot(s.universe(), window={'j1': (0, 2), 'j2': (0, 2)})
        invalid = next(t for t in fig.data if t.meta['pyspect_kind'] == 'invariant')
        points = set(zip(invalid.x, invalid.y))
        self.assertNotIn((0, 0), points)
        self.assertIn((1, 1), points)
        self.assertIn('AllDifferentExceptZero', invalid.hovertemplate)


class TimedReachabilityTests(unittest.TestCase):
    def test_fixed_step_endpoints_and_decimal_arithmetic(self):
        s = system(.1)
        target = s.states(j1=0, j2=0)
        self.assertNotIn((1, 2), s.reach_timed(target, .2))
        self.assertIn((1, 2), s.reach_timed(target, .2, closed=True))
        self.assertIn((1, 2), s.pre_timed(target, .2))
        self.assertNotIn((1, 2), s.pre_timed(target, .199))
        # A swap under strict distinctness needs a third, temporary value.
        strict = system(.1, AllDifferent())
        swap = strict.states(j1=0, j2=1)
        self.assertNotIn((1, 0), strict.reach_timed(swap, .3))
        self.assertIn((1, 0), strict.reach_timed(swap, .3, closed=True))

    def test_zero_and_empty_windows_in_both_timing_models(self):
        for step in (None, .5):
            s = system(step)
            b = s.states(j1=0)
            self.assertEqual(s.pre_timed(b, 0), b)
            self.assertTrue(s.reach_timed(b, 0).is_empty)
            self.assertEqual(s.reach_timed(b, 0, closed=True), b)
            self.assertEqual(s.invariant_timed(b, 0), s.universe())
            self.assertEqual(s.invariant_timed(b, 0, closed=True), b)
            self.assertTrue(s.reach_timed(b, P.closedopen(5, 8), anchor=8).is_empty)
            self.assertEqual(s.invariant_timed(b, P.empty()), s.universe())

    def test_prefix_includes_gap_before_absolute_window(self):
        for step in (None, 1):
            s = system(step)
            b = s.states(j1=0, j2=0)
            a = s.states(j1=1)
            out = s.reach_timed(b, P.closedopen(5, 8), a, anchor=3)
            self.assertIn((1, 2), out)
            self.assertNotIn((2, 1), out)
            self.assertNotIn((0, 0), out)  # B outside A cannot wait for time 5.
            self.assertIn((0, 0), s.reach_timed(b, 0, a, closed=True))

    def test_witness_inside_holding_period_requires_constraint(self):
        s = system(1)
        b = s.states(j1=0, j2=0)
        a = s.states(j1=1)
        self.assertIn((1, 0), s.pre_timed(b, 1, a))
        self.assertTrue(s.pre_timed(b, 1.5, a).is_empty)
        self.assertTrue(s.reach_timed(b, P.open(1, 2), a).is_empty)
        self.assertIn((1, 0), s.reach_timed(b, P.openclosed(1, 2), a))
        self.assertEqual(s.pre_timed(b, .5), b)

    def test_variable_durations_allow_arbitrarily_fast_finite_paths(self):
        s = system()
        b = s.states(j1=0, j2=0)
        self.assertEqual(s.reach_timed(b, 1e-12), s.reach(b))
        self.assertEqual(s.pre_timed(b, 1e-12), s.reach(b))
        self.assertEqual(s.reach_timed(b), s.reach(b))
        self.assertEqual(s.reach_timed(b, P.open(0, 1), s.empty()), s.empty())

    def test_invariance_is_existential_and_observes_holding_periods(self):
        for step in (None, 1):
            s = system(step)
            b = s.states(j1=0, j2=0)
            self.assertEqual(s.invariant_timed(b), b)
            self.assertEqual(s.invariant_timed(b, P.open(0, 1)), b)
            self.assertEqual(s.invariant_timed(b, P.closed(2, 3)), s.universe())
            self.assertTrue(s.invariant_timed(s.empty(), 1).is_empty)
        s = system(1)
        b = s.states(j1=0, j2=0)
        self.assertEqual(s.invariant_timed(b, P.open(.5, 1)), b)
        self.assertEqual(s.invariant_timed(b, P.open(1, 2)), s.pre(b))

    def test_fixed_timing_against_enumerated_piecewise_constant_paths(self):
        # All candidates are exact quarter-times. These cover event edges,
        # interior witnesses and open boundaries of the tested windows.
        windows = [P.closed(0, 0), P.open(0, .5), P.closedopen(.5, 1),
                   P.openclosed(.5, 1), P.singleton(.75), P.closed(0, 1),
                   P.singleton(.25) | P.singleton(1)]
        for invariant in (AllDifferent(), AllDifferentExceptZero()):
            for transition in (OneVariableTransition(), UnrestrictedTransition()):
                s = system(Fraction(1, 2), invariant, transition)
                nodes = [x for x in itertools.product(range(3), repeat=2) if x in s.universe()]
                b = s.states(j1=0, j2=P.closed(0, 1))
                a = s.states(j1=P.closed(1, 2), j2=P.closed(0, 2))
                def edge(x, y):
                    return isinstance(transition, UnrestrictedTransition) or sum(
                        u != v for u, v in zip(x, y)) <= 1
                for window in windows:
                    actual = s.reach_timed(b, window, a)
                    for x in nodes:
                        expected = False
                        for y, z in itertools.product(nodes, repeat=2):
                            if not edge(x, y) or not edge(y, z):
                                continue
                            path = (x, y, z)
                            for tick in range(5):
                                t = Fraction(tick, 4)
                                cell = tick // 2
                                prefix_cells = cell + (tick % 2 != 0)
                                if t in window and path[cell] in b and all(
                                    state in a for state in path[:prefix_cells]
                                ):
                                    expected = True
                        self.assertEqual(x in actual, expected, (invariant, transition, window, x))

    def test_unbounded_windows_converge_and_anchor_clips_history(self):
        s = system(.5)
        b = s.states(j1=0, j2=0)
        a = s.states(j1=1)
        self.assertEqual(s.reach_timed(b, constraints=a), s.reach(b, a))
        self.assertEqual(s.reach_timed(b, P.closedopen(5, P.inf), a, anchor=6),
                         s.reach(b, a))
        self.assertEqual(s.reach_timed(b, P.closed(1000000, 1000001)), s.universe())

    def test_invalid_timing_parameters(self):
        for step in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                system(step)
        s = system()
        b = s.universe()
        for duration in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                s.pre_timed(b, duration)
        with self.assertRaises(TypeError):
            s.reach_timed(b, True)
        with self.assertRaises(ValueError):
            s.reach_timed(b, P.closed(-1, 2))
        with self.assertRaises(ValueError):
            s.reach_timed(b, anchor=-1)
        with self.assertRaises(ValueError):
            s.reach_timed(system().universe(), 1)


if __name__ == '__main__':
    unittest.main()
