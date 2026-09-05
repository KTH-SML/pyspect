from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import patch
import unittest

import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
import portion as P

from pyspect import AND, NEXT, ExactDiscLTL, TLT, UNTIL
from pyspect.impls.symbolic import (
    AllDifferent,
    OneVariableTransition,
    SymbolicSet,
    SymbolicSystem,
)


def traces_of_kind(fig: BaseFigure, kind: str):
    return [
        trace
        for trace in fig.data
        if trace.meta is not None and trace.meta.get("pyspect_kind") == kind
    ]


class SymbolicLatticePlotTests(unittest.TestCase):
    def setUp(self):
        self.system = SymbolicSystem(
            ("j1", "j2"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )

    def reach_example(self):
        return TLT(
            UNTIL("safe", "goal"),
            primitives=ExactDiscLTL,
            where={
                "safe": SymbolicSet(j1=P.closed(0, 1)),
                "goal": SymbolicSet(j1=2, j2=1),
            },
        ).realize(self.system)

    def test_reach_history_k_axis_and_automatic_bounds(self):
        winning = self.reach_example()
        expected = self.system.states(j1=P.closed(0, 1)) | self.system.states(
            j1=2, j2=1
        )

        self.assertEqual(winning, expected)
        self.assertNotIn("history", repr(winning))
        self.assertEqual(len(winning._reach_history), 5)
        self.assertTrue(all(
            earlier != later
            for earlier, later in zip(
                winning._reach_history, winning._reach_history[1:]
            )
        ))

        fig = self.system.plot(winning, axes=("k", "j1"), show_overflow=False)
        layer, = traces_of_kind(fig, "layer")
        self.assertEqual(tuple(layer.x), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(layer.y), (0, 1, 2, 3))
        self.assertEqual(tuple(layer.z[0]), (None, 1, 1, 1, 1))
        self.assertEqual(tuple(layer.z[1]), (None, None, None, 1, 1))
        self.assertEqual(tuple(layer.z[2]), (1, 1, 1, 1, 1))
        self.assertIn("k (reach iteration)", fig.layout.annotations[-1].text)

        transposed = self.system.plot(
            winning,
            axes=("j1", "k"),
            show_overflow=False,
        )
        transposed_layer, = traces_of_kind(transposed, "layer")
        self.assertEqual(tuple(transposed_layer.x), (0, 1, 2, 3))
        self.assertEqual(tuple(transposed_layer.y), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(transposed_layer.z[0]), (None, None, 1, None))
        self.assertEqual(tuple(transposed_layer.z[4]), (1, 1, 1, None))

    def test_automatic_and_partial_windows_and_default_projection(self):
        finite = self.system.states(j1=P.closed(2, 4), j2=P.closed(6, 7))
        finite_fig = self.system.plot(finite, show_overflow=False)
        finite_layer, = traces_of_kind(finite_fig, "layer")
        self.assertEqual(tuple(finite_layer.x), (1, 2, 3, 4, 5))
        self.assertEqual(tuple(finite_layer.y), (5, 6, 7, 8))

        unbounded_fig = self.system.plot(
            self.system.universe(),
            window={"j1": (3, 4)},
            show_overflow=False,
        )
        unbounded_layer, = traces_of_kind(unbounded_fig, "layer")
        self.assertEqual(tuple(unbounded_layer.x), (3, 4))
        self.assertEqual(tuple(unbounded_layer.y), tuple(range(11)))

        far_unbounded = self.system.plot(
            self.system.states(j1=P.closedopen(20, P.inf), j2=0),
            show_overflow=False,
        )
        far_unbounded_layer, = traces_of_kind(far_unbounded, "layer")
        self.assertEqual(tuple(far_unbounded_layer.x), tuple(range(22)))

        empty_fig = self.system.plot(self.system.empty(), show_overflow=False)
        empty_layer, = traces_of_kind(empty_fig, "layer")
        self.assertEqual(tuple(empty_layer.x), tuple(range(11)))
        self.assertEqual(tuple(empty_layer.y), tuple(range(11)))

        system3 = SymbolicSystem(
            ("j1", "j2", "j3"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )
        projected = system3.plot(
            system3.states(j3=P.closed(0, 1)),
            axes=("j1", "j2"),
            window={"j1": (0, 2), "j2": (0, 2)},
            show_invariant=False,
            show_overflow=False,
        )
        projected_mask = traces_of_kind(projected, "layer")[0].z
        self.assertIsNone(projected_mask[1][0])
        self.assertEqual(projected_mask[2][0], 1)

    def test_multiple_histories_and_static_layers_use_longest_horizon(self):
        winning = self.reach_example()
        stable = self.system.reach(self.system.universe())
        static = self.system.states(j1=3)
        self.assertEqual(len(stable._reach_history), 1)

        fig = self.system.plot(
            (winning, {"name": "winning"}),
            (stable, {"name": "stable"}),
            (static, {"name": "static"}),
            axes=("k", "j1"),
            window={"j1": (0, 3)},
            show_overflow=False,
        )
        winning_layer, stable_layer, static_layer = traces_of_kind(fig, "layer")
        self.assertEqual(tuple(winning_layer.x), (0, 1, 2, 3, 4))
        self.assertTrue(all(value == 1 for value in stable_layer.z[0]))
        self.assertTrue(all(value == 1 for value in stable_layer.z[3]))
        self.assertTrue(all(value == 1 for value in static_layer.z[3]))
        self.assertTrue(all(value is None for value in static_layer.z[0]))

    def test_exact_state_and_clipped_k_overflow(self):
        winning = self.reach_example()
        fig = self.system.plot(
            (winning, {"name": "winning"}),
            axes=("k", "j1"),
            window={"k": (2, 3), "j1": (0, 1)},
        )

        overflow, = traces_of_kind(fig, "overflow")
        markers = set(zip(overflow.x, overflow.y, overflow.marker.symbol))
        self.assertEqual(markers, {
            (1.62, 0, "triangle-left"),
            (3.38, 0, "triangle-right"),
            (3.38, 1, "triangle-right"),
            (2, 1.38, "triangle-up"),
            (3, 1.38, "triangle-up"),
        })
        self.assertTrue(any("reach iteration k" in text for text in overflow.text))

    def test_history_is_not_propagated_by_set_or_predecessor_operations(self):
        winning = self.reach_example()
        tlt_results = (
            TLT(
                "goal",
                primitives=ExactDiscLTL,
                where={"goal": SymbolicSet(j1=0)},
            ).realize(self.system),
            TLT(
                AND("safe", "goal"),
                primitives=ExactDiscLTL,
                where={
                    "safe": SymbolicSet(j1=P.closed(0, 1)),
                    "goal": SymbolicSet(j2=1),
                },
            ).realize(self.system),
            TLT(
                NEXT("goal"),
                primitives=ExactDiscLTL,
                where={"goal": SymbolicSet(j1=0)},
            ).realize(self.system),
        )
        derived = tlt_results + (
            winning | self.system.empty(),
            winning & self.system.universe(),
            winning - self.system.empty(),
            ~winning,
            self.system.pre(winning),
            self.system.pre_k(winning, 0),
        )
        self.assertIsNotNone(winning._reach_history)
        self.assertTrue(all(state_set._reach_history is None for state_set in derived))
        for state_set in derived:
            with self.subTest(state_set=state_set):
                with self.assertRaisesRegex(
                    ValueError, "no layer carries reach history"
                ):
                    self.system.plot(state_set, axes=("k", "j1"))

    def test_signal_named_k_uses_numeric_index(self):
        system = SymbolicSystem(
            ("k", "j1"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )
        reached = system.reach(system.states({"k": 0, "j1": 1}))

        state_axes = system.plot(
            reached,
            axes=(0, 1),
            window={0: (0, 1), 1: (0, 1)},
            show_overflow=False,
        )
        self.assertEqual(state_axes.layout.xaxis.title.text, "k")

        synthetic_and_state = system.plot(
            reached,
            axes=("k", 0),
            select={1: 1},
            window={0: (0, 2)},
            show_overflow=False,
        )
        self.assertEqual(
            synthetic_and_state.layout.xaxis.title.text,
            "k (reach iteration)",
        )
        self.assertEqual(synthetic_and_state.layout.yaxis.title.text, "k")
        with self.assertRaisesRegex(ValueError, "synthetic 'k'"):
            system.plot(reached, axes=("k", 1), select={"k": 0})

    def test_layer_membership_orientation_styles_and_invariant(self):
        goal = self.system.states(j1=2, j2=1)
        winning = self.system.states(j1=P.closed(0, 1)) | goal

        fig = self.system.plot(
            (winning, {"name": "winning", "color": "#123456", "opacity": 0.6}),
            (goal, {"name": "goal", "color": "orange"}),
            window={"j1": (0, 3), "j2": (0, 3)},
            show_overflow=False,
        )

        self.assertIsInstance(fig, BaseFigure)
        layers = traces_of_kind(fig, "layer")
        self.assertEqual([trace.name for trace in layers], ["winning", "goal"])
        self.assertEqual(layers[0].opacity, 0.6)
        self.assertEqual(layers[0].colorscale[0][1], "#123456")

        winning_mask = layers[0].z
        goal_mask = layers[1].z
        self.assertIsNone(winning_mask[0][0])  # (0, 0) violates AllDifferent.
        self.assertEqual(winning_mask[0][1], 1)  # (1, 0) is in the stripe.
        self.assertEqual(winning_mask[3][0], 1)  # Rows are j2, columns are j1.
        self.assertEqual(winning_mask[1][2], 1)  # The isolated goal.
        self.assertIsNone(winning_mask[3][2])
        self.assertEqual(goal_mask[1][2], 1)
        self.assertIn("j1=%{x}", layers[0].hovertemplate)
        self.assertIn("winning", layers[0].hovertemplate)

        invariant, = traces_of_kind(fig, "invariant")
        invalid = set(zip(invariant.x, invariant.y))
        self.assertEqual(invalid, {(0, 0), (1, 1), (2, 2), (3, 3)})
        self.assertIn("AllDifferent", invariant.hovertemplate)

    def test_disjoint_domains_and_empty_sets(self):
        disjoint = self.system.states(
            j1=P.singleton(0) | P.singleton(2),
        )
        fig = self.system.plot(
            (disjoint, {"name": "disjoint"}),
            (self.system.empty(), {"name": "empty"}),
            window={"j1": (0, 2), "j2": (0, 2)},
            show_invariant=False,
            show_overflow=False,
        )
        disjoint_mask, empty_mask = [trace.z for trace in traces_of_kind(fig, "layer")]
        self.assertEqual(disjoint_mask[1][0], 1)
        self.assertIsNone(disjoint_mask[1][1])
        self.assertEqual(disjoint_mask[1][2], 1)
        self.assertTrue(all(value is None for row in empty_mask for value in row))

    def test_exact_overflow_markers(self):
        goal = self.system.states(j1=2, j2=1)
        winning = self.system.states(j1=P.closed(0, 1)) | goal
        fig = self.system.plot(
            (winning, {"name": "winning"}),
            (goal, {"name": "goal"}),
            window={"j1": (0, 3), "j2": (0, 3)},
        )

        overflow = traces_of_kind(fig, "overflow")
        self.assertEqual(len(overflow), 1)
        self.assertEqual(overflow[0].meta["layer"], "winning")
        markers = set(zip(overflow[0].x, overflow[0].y, overflow[0].marker.symbol))
        self.assertEqual(
            markers,
            {(0, 3.38, "triangle-up"), (1, 3.38, "triangle-up")},
        )
        self.assertIn("Triangles indicate states beyond an edge", fig.layout.annotations[-1].text)

    def test_slice_and_exact_existential_projection(self):
        system = SymbolicSystem(
            ("j1", "j2", "j3"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )

        sliced = system.plot(
            system.universe(),
            axes=("j1", "j2"),
            window={"j1": (0, 2), "j2": (0, 2)},
            select={"j3": 2},
            show_overflow=False,
        )
        sliced_mask = traces_of_kind(sliced, "layer")[0].z
        self.assertEqual(sliced_mask[1][0], 1)
        self.assertIsNone(sliced_mask[0][2])  # j1 collides with selected j3.
        self.assertIsNone(sliced_mask[2][0])  # j2 collides with selected j3.
        invalid = set(zip(
            traces_of_kind(sliced, "invariant")[0].x,
            traces_of_kind(sliced, "invariant")[0].y,
        ))
        self.assertIn((2, 0), invalid)
        self.assertIn((0, 2), invalid)

        target = system.states(j3=P.closed(0, 1))
        projected = system.plot(
            target,
            axes=("j1", "j2"),
            window={"j1": (0, 2), "j2": (0, 2)},
            project=True,
            show_invariant=False,
            show_overflow=False,
        )
        projected_mask = traces_of_kind(projected, "layer")[0].z
        self.assertIsNone(projected_mask[1][0])  # No distinct j3 remains in {0, 1}.
        self.assertEqual(projected_mask[2][0], 1)  # j3=1 is a completion.
        self.assertEqual(projected_mask[2][1], 1)  # j3=0 is a completion.

    def test_existing_figure_layout_options_and_no_show_side_effect(self):
        fig = go.Figure(go.Scatter(x=[0], y=[0], name="existing"))
        fig.update_layout(title="kept until explicitly replaced")

        with patch.object(BaseFigure, "show", side_effect=AssertionError("show called")):
            returned = self.system.plot(
                self.system.states(j1=0),
                window={"j1": (0, 2), "j2": (0, 2)},
                fig=fig,
                show_overflow=False,
                layout_title="symbolic result",
                xaxis_tickangle=45,
            )

        self.assertIs(returned, fig)
        self.assertEqual(fig.data[0].name, "existing")
        self.assertEqual(fig.layout.title.text, "symbolic result")
        self.assertEqual(fig.layout.xaxis.tickangle, 45)
        self.assertIn("Viewport:", fig.layout.annotations[-1].text)

    def test_plotting_does_not_require_undeclared_numpy_dependency(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        script = textwrap.dedent(
            """
            import sys

            class BlockNumpy:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "numpy" or fullname.startswith("numpy."):
                        raise ModuleNotFoundError("numpy deliberately blocked")
                    return None

            sys.meta_path.insert(0, BlockNumpy())

            from pyspect.impls.symbolic import (
                AllDifferent, OneVariableTransition, SymbolicSystem,
            )

            system = SymbolicSystem(
                ("j1", "j2"),
                invariant=AllDifferent(),
                transition=OneVariableTransition(),
            )
            figure = system.plot(
                system.universe(),
                window={"j1": (0, 1), "j2": (0, 1)},
            )
            assert figure.data
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

    def test_validation(self):
        states = self.system.universe()
        base = {"window": {"j1": (0, 2), "j2": (0, 2)}}
        cases = (
            (ValueError, lambda: self.system.plot(**base)),
            (ValueError, lambda: self.system.plot(states, method="bitmap", **base)),
            (ValueError, lambda: self.system.plot(states, axes=("j1", "j1"), **base)),
            (ValueError, lambda: self.system.plot(states, axes=("j1", "missing"), **base)),
            (ValueError, lambda: self.system.plot(
                states,
                window={0: (0, 2), "j1": (0, 2), "j2": (0, 2)},
            )),
            (ValueError, lambda: self.system.plot(
                states,
                window={"j1": (-1, 2), "j2": (0, 2)},
            )),
            (TypeError, lambda: self.system.plot(
                states,
                window={"j1": (0.0, 2), "j2": (0, 2)},
            )),
            (ValueError, lambda: self.system.plot(states, max_cells=8, **base)),
            (ValueError, lambda: self.system.plot(
                (states, {"unknown": True}),
                **base,
            )),
            (TypeError, lambda: self.system.plot(states, fig=object(), **base)),
            (ValueError, lambda: self.system.plot(states, axes=("k", "j1"))),
        )
        for expected, call in cases:
            with self.subTest(call=call):
                with self.assertRaises(expected):
                    call()

        other = SymbolicSystem(
            ("j1", "j2"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )
        with self.assertRaises(ValueError):
            self.system.plot(other.universe(), **base)

        system3 = SymbolicSystem(
            ("j1", "j2", "j3"),
            invariant=AllDifferent(),
            transition=OneVariableTransition(),
        )
        window3 = {"j1": (0, 2), "j2": (0, 2)}
        with self.assertRaises(ValueError):
            system3.plot(system3.universe(), window=window3, project=False)
        with self.assertRaises(ValueError):
            system3.plot(
                system3.universe(),
                window=window3,
                select={"j1": 0},
                project=True,
            )

        reached = self.reach_example()
        with self.assertRaises(ValueError):
            self.system.plot(
                reached,
                axes=("k", "j1"),
                window={"k": (0, 5)},
            )
        with self.assertRaises(ValueError):
            self.system.plot(
                reached,
                axes=("k", "j1"),
                window={"j2": (0, 2)},
            )
        with self.assertRaises(ValueError):
            self.system.plot(
                reached,
                axes=("k", "j1"),
                max_cells=19,
                show_overflow=False,
            )


if __name__ == "__main__":
    unittest.main()
