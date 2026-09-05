"""Plotly lattice visualization for exact symbolic integer state sets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
import math
from numbers import Integral, Real
from typing import Any

import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure
import portion as P

from ..impls.symbolic import StateSet, SymbolicSystem
from .plotly import update_theme

__all__ = ("plot_lattice",)


_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)
_LAYER_STYLE_KEYS = frozenset(("name", "color", "opacity"))
_K_AXIS = "k"
type _Axis = int | str


@dataclass(frozen=True, slots=True)
class _Layer:
    states: StateSet
    name: str
    color: str
    opacity: float


def _axis_index(system: SymbolicSystem, axis: int | str) -> int:
    if isinstance(axis, Integral) and not isinstance(axis, bool):
        index = int(axis)
        if index < 0:
            index += len(system.variables)
        if not 0 <= index < len(system.variables):
            raise ValueError(f"Unknown symbolic axis index: {axis}.")
        return index
    if isinstance(axis, str):
        try:
            return system.variables.index(axis)
        except ValueError as error:
            raise ValueError(f"Unknown symbolic axis: {axis!r}.") from error
    raise TypeError(f"Symbolic axes must be names or integers, got {axis!r}.")


def _plot_axis(system: SymbolicSystem, axis: int | str) -> _Axis:
    if axis == _K_AXIS and isinstance(axis, str):
        return _K_AXIS
    return _axis_index(system, axis)


def _axis_name(system: SymbolicSystem, axis: _Axis) -> str:
    return _K_AXIS if axis == _K_AXIS else system.variables[axis]


def _normalize_axes(
    system: SymbolicSystem,
    axes: tuple[int | str, int | str],
) -> tuple[_Axis, _Axis]:
    if not isinstance(axes, (tuple, list)) or len(axes) != 2:
        raise ValueError("lattice plots require exactly two axes.")
    resolved = tuple(_plot_axis(system, axis) for axis in axes)
    if resolved[0] == resolved[1]:
        raise ValueError("lattice plot axes must be distinct.")
    if _K_AXIS in resolved and not any(isinstance(axis, int) for axis in resolved):
        raise ValueError("A reach-iteration axis must be paired with a state axis.")
    return resolved


def _state_sets_for_bounds(layer: _Layer) -> tuple[StateSet, ...]:
    history = layer.states._reach_history
    return (*history, layer.states) if history is not None else (layer.states,)


def _infer_state_bounds(layers: tuple[_Layer, ...], axis: int) -> tuple[int, int]:
    minimum: int | None = None
    largest_finite: int | None = None
    unbounded = False

    for layer in layers:
        for state_set in _state_sets_for_bounds(layer):
            for region in state_set._regions:
                for atom in region.domains[axis]:
                    lower = int(atom.lower)
                    minimum = lower if minimum is None else min(minimum, lower)
                    largest_finite = (
                        lower
                        if largest_finite is None
                        else max(largest_finite, lower)
                    )
                    if atom.upper == P.inf:
                        unbounded = True
                    else:
                        upper = int(atom.upper)
                        largest_finite = max(largest_finite, upper)

    if unbounded or minimum is None or largest_finite is None:
        upper = max(10, (largest_finite or 0) + 1)
        return 0, upper
    return max(0, minimum - 1), largest_finite + 1


def _normalize_window(
    system: SymbolicSystem,
    axes: tuple[_Axis, _Axis],
    window: Mapping[int | str, tuple[int, int]] | None,
    layers: tuple[_Layer, ...],
    horizon: int | None,
) -> dict[_Axis, tuple[int, int]]:
    if window is None:
        window = {}
    if not isinstance(window, Mapping):
        raise TypeError("window must map plotted axes to inclusive bounds.")

    resolved: dict[_Axis, tuple[int, int]] = {}
    for axis, bounds in window.items():
        index = _plot_axis(system, axis)
        name = _axis_name(system, index)
        if index not in axes:
            raise ValueError(f"Window axis {name!r} is not plotted.")
        if index in resolved:
            raise ValueError(f"Duplicate window bounds for axis {name!r}.")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Window bounds for {name!r} must contain two integers."
            )
        lower, upper = bounds
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in (lower, upper)
        ):
            raise TypeError(
                f"Window bounds for {name!r} must be integers."
            )
        lower, upper = int(lower), int(upper)
        if lower < 0 or upper < lower:
            raise ValueError(
                f"Window bounds for {name!r} must satisfy "
                "0 <= lower <= upper."
            )
        if index == _K_AXIS and (horizon is None or upper > horizon):
            raise ValueError(
                f"Window bounds for 'k' must lie within the recorded reach "
                f"history 0..{horizon}."
            )
        resolved[index] = (lower, upper)

    for axis in axes:
        if axis in resolved:
            continue
        if axis == _K_AXIS:
            assert horizon is not None
            resolved[axis] = (0, horizon)
        else:
            resolved[axis] = _infer_state_bounds(layers, axis)
    return resolved


def _normalize_selection(
    system: SymbolicSystem,
    axes: tuple[_Axis, _Axis],
    select: Mapping[int | str, int] | None,
    project: bool,
) -> dict[int, int]:
    if not isinstance(project, bool):
        raise TypeError("project must be a boolean.")
    if select is None:
        select = {}
    if not isinstance(select, Mapping):
        raise TypeError("select must map hidden axes to non-negative integers.")

    resolved: dict[int, int] = {}
    for axis, value in select.items():
        index = _plot_axis(system, axis)
        if index == _K_AXIS:
            raise ValueError("select cannot fix the synthetic 'k' axis.")
        if index in axes:
            raise ValueError(
                f"select cannot fix plotted axis {system.variables[index]!r}."
            )
        if index in resolved:
            raise ValueError(
                f"Duplicate selection for axis {system.variables[index]!r}."
            )
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"Selection for {system.variables[index]!r} must be a "
                "non-negative integer."
            )
        resolved[index] = int(value)

    visible = {axis for axis in axes if isinstance(axis, int)}
    hidden = set(range(len(system.variables))) - visible
    if not project and set(resolved) != hidden:
        missing = ", ".join(
            repr(system.variables[index]) for index in sorted(hidden - set(resolved))
        )
        raise ValueError(
            "All hidden axes must be fixed when project=False; "
            f"missing selections for: {missing}."
        )
    return resolved


def _normalize_layers(
    system: SymbolicSystem,
    layers: tuple[StateSet | tuple[StateSet, Mapping[str, Any]], ...],
) -> tuple[_Layer, ...]:
    if not layers:
        raise ValueError("plot requires at least one StateSet layer.")

    normalized: list[_Layer] = []
    for index, value in enumerate(layers):
        if isinstance(value, StateSet):
            states, style = value, {}
        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[1], Mapping)
        ):
            states, raw_style = value
            style = dict(raw_style)
        else:
            raise TypeError(
                "Each layer must be a StateSet or a (StateSet, style mapping) pair."
            )

        system._check_sets(states)
        unknown = set(style) - _LAYER_STYLE_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown lattice layer style options: {names}.")

        name = style.get("name", f"StateSet {index + 1}")
        color = style.get("color", _COLORS[index % len(_COLORS)])
        opacity = style.get("opacity", 0.85)
        if not isinstance(name, str) or not name:
            raise ValueError("Layer name must be a non-empty string.")
        if not isinstance(color, str) or not color:
            raise ValueError("Layer color must be a non-empty Plotly color string.")
        if (
            not isinstance(opacity, Real)
            or isinstance(opacity, bool)
            or not 0 <= opacity <= 1
        ):
            raise ValueError("Layer opacity must be a number between 0 and 1.")
        normalized.append(_Layer(states, name, color, float(opacity)))
    return tuple(normalized)


def _pop_prefixed(options: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key.removeprefix(prefix): options.pop(key)
        for key in list(options)
        if key.startswith(prefix)
    }


def _mapping_option(options: dict[str, Any], name: str) -> dict[str, Any]:
    value = options.pop(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping when supplied.")
    return dict(value)


def _parse_layout_options(
    layout_options: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    options = dict(layout_options)
    theme = options.pop("theme", None)
    if theme is not None and not isinstance(theme, str):
        raise TypeError("theme must be a string or None.")

    theme_options = _pop_prefixed(options, "theme_")
    unknown_theme = set(theme_options) - {"aspectratio"}
    if unknown_theme:
        names = ", ".join(sorted(unknown_theme))
        raise ValueError(f"Unknown theme options: {names}.")

    layout = _mapping_option(options, "layout")
    layout.update(_pop_prefixed(options, "layout_"))
    xaxis = _mapping_option(options, "xaxis")
    xaxis.update(_pop_prefixed(options, "xaxis_"))
    yaxis = _mapping_option(options, "yaxis")
    yaxis.update(_pop_prefixed(options, "yaxis_"))
    if options:
        names = ", ".join(sorted(options))
        raise TypeError(f"Unexpected lattice plot options: {names}.")
    return theme, theme_options, layout, xaxis, yaxis


def _tick_step(lower: int, upper: int) -> int:
    return max(1, math.ceil((upper - lower + 1) / 20))


def _reach_horizon(layers: tuple[_Layer, ...], axes: tuple[_Axis, _Axis]) -> int | None:
    if _K_AXIS not in axes:
        return None
    histories = tuple(
        layer.states._reach_history
        for layer in layers
        if layer.states._reach_history is not None
    )
    if not histories:
        raise ValueError(
            "Cannot plot against 'k': no layer carries reach history. "
            "Use a StateSet returned directly by SymbolicSystem.reach() or by "
            "a TLT whose outer operation is UNTIL or EVENTUALLY."
        )
    return max(len(history) - 1 for history in histories)


def _layer_states_at(layer: _Layer, iteration: int) -> StateSet:
    history = layer.states._reach_history
    if history is None:
        return layer.states
    return history[min(iteration, len(history) - 1)]


def _has_overflow(
    system: SymbolicSystem,
    layer: _Layer,
    tail_axis: _Axis,
    tail: P.Interval | range,
    fixed_axis: _Axis,
    coordinate: int,
    selected_domains: dict[int, P.Interval],
) -> bool:
    restrictions = dict(selected_domains)
    if tail_axis == _K_AXIS:
        assert isinstance(tail, range) and isinstance(fixed_axis, int)
        restrictions[fixed_axis] = P.singleton(coordinate)
        return any(
            system._has_feasible_completion(
                _layer_states_at(layer, iteration), restrictions
            )
            for iteration in tail
        )

    assert isinstance(tail_axis, int) and isinstance(tail, P.Interval)
    restrictions[tail_axis] = tail
    if fixed_axis == _K_AXIS:
        states = _layer_states_at(layer, coordinate)
    else:
        assert isinstance(fixed_axis, int)
        restrictions[fixed_axis] = P.singleton(coordinate)
        states = layer.states
    return system._has_feasible_completion(states, restrictions)


def _overflow_trace(
    system: SymbolicSystem,
    layer: _Layer,
    axes: tuple[_Axis, _Axis],
    bounds: dict[_Axis, tuple[int, int]],
    selected_domains: dict[int, P.Interval],
    x_values: list[int],
    y_values: list[int],
    horizon: int | None,
) -> go.Scatter | None:
    x_axis, y_axis = axes
    x_min, x_max = bounds[x_axis]
    y_min, y_max = bounds[y_axis]
    xs: list[float] = []
    ys: list[float] = []
    symbols: list[str] = []
    labels: list[str] = []

    checks: list[
        tuple[_Axis, P.Interval | range, list[int], _Axis, str, str]
    ] = []
    for axis, lower, upper, coordinates, fixed_axis, direction in (
        (x_axis, x_min, x_max, y_values, y_axis, "right"),
        (y_axis, y_min, y_max, x_values, x_axis, "up"),
    ):
        name = _axis_name(system, axis)
        if axis == _K_AXIS:
            assert horizon is not None
            if upper < horizon:
                checks.append((
                    axis,
                    range(upper + 1, horizon + 1),
                    coordinates,
                    fixed_axis,
                    direction,
                    f"reach iteration {name} &gt; {upper}",
                ))
        else:
            checks.append((
                axis,
                P.closedopen(upper + 1, P.inf),
                coordinates,
                fixed_axis,
                direction,
                f"{name} &gt; {upper}",
            ))

    for axis, lower, upper, coordinates, fixed_axis, direction in (
        (x_axis, x_min, x_max, y_values, y_axis, "left"),
        (y_axis, y_min, y_max, x_values, x_axis, "down"),
    ):
        if lower == 0:
            continue
        name = _axis_name(system, axis)
        tail: P.Interval | range = (
            range(0, lower)
            if axis == _K_AXIS
            else P.closed(0, lower - 1)
        )
        description = (
            f"reach iteration {name} &lt; {lower}"
            if axis == _K_AXIS
            else f"{name} &lt; {lower}"
        )
        checks.append((
            axis,
            tail,
            coordinates,
            fixed_axis,
            direction,
            description,
        ))

    for tail_axis, tail, coordinates, fixed_axis, direction, description in checks:
        for coordinate in coordinates:
            if not _has_overflow(
                system,
                layer,
                tail_axis,
                tail,
                fixed_axis,
                coordinate,
                selected_domains,
            ):
                continue
            match direction:
                case "right":
                    x, y, symbol = x_max + 0.38, coordinate, "triangle-right"
                case "left":
                    x, y, symbol = x_min - 0.38, coordinate, "triangle-left"
                case "up":
                    x, y, symbol = coordinate, y_max + 0.38, "triangle-up"
                case "down":
                    x, y, symbol = coordinate, y_min - 0.38, "triangle-down"
            xs.append(x)
            ys.append(y)
            symbols.append(symbol)
            labels.append(
                f"{escape(layer.name)} has states with {description}"
            )

    if not xs:
        return None
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker={
            "symbol": symbols,
            "color": layer.color,
            "size": 11,
            "line": {"color": "white", "width": 1},
        },
        name=f"{layer.name} outside viewport",
        legendgroup=layer.name,
        showlegend=False,
        text=labels,
        hovertemplate="%{text}<extra></extra>",
        cliponaxis=False,
        meta={"pyspect_kind": "overflow", "layer": layer.name},
    )


def plot_lattice(
    system: SymbolicSystem,
    *layers: StateSet | tuple[StateSet, Mapping[str, Any]],
    window: Mapping[int | str, tuple[int, int]] | None = None,
    method: str = "lattice",
    axes: tuple[int | str, int | str] = (0, 1),
    select: Mapping[int | str, int] | None = None,
    project: bool = True,
    show_invariant: bool = True,
    show_overflow: bool = True,
    max_cells: int = 50_000,
    fig: BaseFigure | None = None,
    **layout_options: Any,
) -> BaseFigure:
    """Render exact symbolic membership on a bounded integer lattice."""
    if method != "lattice":
        raise ValueError(f"Unknown symbolic plotting method {method!r}.")
    if not isinstance(show_invariant, bool) or not isinstance(show_overflow, bool):
        raise TypeError("show_invariant and show_overflow must be booleans.")
    if (
        not isinstance(max_cells, Integral)
        or isinstance(max_cells, bool)
        or max_cells <= 0
    ):
        raise ValueError("max_cells must be a positive integer.")

    normalized_layers = _normalize_layers(system, layers)
    axes = _normalize_axes(system, axes)
    horizon = _reach_horizon(normalized_layers, axes)
    bounds = _normalize_window(system, axes, window, normalized_layers, horizon)
    selected = _normalize_selection(system, axes, select, project)
    theme, theme_options, layout, xaxis_options, yaxis_options = (
        _parse_layout_options(layout_options)
    )

    x_axis, y_axis = axes
    x_min, x_max = bounds[x_axis]
    y_min, y_max = bounds[y_axis]
    x_values = list(range(x_min, x_max + 1))
    y_values = list(range(y_min, y_max + 1))
    cell_count = len(x_values) * len(y_values)
    if cell_count > max_cells:
        raise ValueError(
            f"Lattice window contains {cell_count:,} cells, exceeding "
            f"max_cells={int(max_cells):,}. Use a smaller window or raise max_cells."
        )

    if fig is None:
        figure: BaseFigure = go.Figure()
        new_figure = True
    elif isinstance(fig, BaseFigure):
        figure = fig
        new_figure = False
    else:
        raise TypeError("fig must be a Plotly BaseFigure or None.")

    selected_domains = {
        index: P.singleton(value) for index, value in selected.items()
    }
    masks: list[list[list[int | None]]] = [
        [[None for _ in x_values] for _ in y_values]
        for _ in normalized_layers
    ]
    invalid_x: list[int] = []
    invalid_y: list[int] = []
    x_domains = {value: P.singleton(value) for value in x_values}
    y_domains = {value: P.singleton(value) for value in y_values}

    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            restrictions = dict(selected_domains)
            if isinstance(x_axis, int):
                restrictions[x_axis] = x_domains[x_value]
            if isinstance(y_axis, int):
                restrictions[y_axis] = y_domains[y_value]
            admissible = system._has_feasible_completion(None, restrictions)
            if not admissible:
                invalid_x.append(x_value)
                invalid_y.append(y_value)
                continue
            iteration = (
                x_value if x_axis == _K_AXIS
                else y_value if y_axis == _K_AXIS
                else None
            )
            for layer_index, layer in enumerate(normalized_layers):
                states = (
                    layer.states
                    if iteration is None
                    else _layer_states_at(layer, iteration)
                )
                if system._has_feasible_completion(states, restrictions):
                    masks[layer_index][y_index][x_index] = 1

    for layer, mask in zip(normalized_layers, masks):
        figure.add_trace(go.Heatmap(
            x=x_values,
            y=y_values,
            z=mask,
            zmin=0,
            zmax=1,
            colorscale=((0, layer.color), (1, layer.color)),
            showscale=False,
            showlegend=True,
            name=layer.name,
            legendgroup=layer.name,
            opacity=layer.opacity,
            xgap=1,
            ygap=1,
            hoverongaps=False,
            hovertemplate=(
                f"{escape(_axis_name(system, x_axis))}=%{{x}}<br>"
                f"{escape(_axis_name(system, y_axis))}=%{{y}}"
                f"<extra>{escape(layer.name)}</extra>"
            ),
            meta={"pyspect_kind": "layer", "layer": layer.name},
        ))

    if show_overflow:
        for layer in normalized_layers:
            trace = _overflow_trace(
                system,
                layer,
                axes,
                bounds,
                selected_domains,
                x_values,
                y_values,
                horizon,
            )
            if trace is not None:
                figure.add_trace(trace)

    if show_invariant and invalid_x:
        figure.add_trace(go.Scatter(
            x=invalid_x,
            y=invalid_y,
            mode="markers",
            marker={"symbol": "x", "color": "#777777", "size": 9},
            name="inadmissible",
            showlegend=True,
            hovertemplate=(
                f"{escape(_axis_name(system, x_axis))}=%{{x}}<br>"
                f"{escape(_axis_name(system, y_axis))}=%{{y}}<br>"
                f"violates {type(system.invariant).__name__}<extra>inadmissible</extra>"
            ),
            meta={"pyspect_kind": "invariant"},
        ))

    if new_figure or theme is not None:
        update_theme(theme, **theme_options, fig=figure)

    xaxis = {
        "title": (
            "k (reach iteration)"
            if x_axis == _K_AXIS
            else system.variables[x_axis]
        ),
        "range": [x_min - 0.5, x_max + 0.5],
        "tickmode": "linear",
        "tick0": x_min,
        "dtick": _tick_step(x_min, x_max),
        "zeroline": False,
        "constrain": "domain",
    }
    yaxis = {
        "title": (
            "k (reach iteration)"
            if y_axis == _K_AXIS
            else system.variables[y_axis]
        ),
        "range": [y_min - 0.5, y_max + 0.5],
        "tickmode": "linear",
        "tick0": y_min,
        "dtick": _tick_step(y_min, y_max),
        "zeroline": False,
        "scaleanchor": "x",
        "scaleratio": 1,
        "constrain": "domain",
    }
    xaxis.update(xaxis_options)
    yaxis.update(yaxis_options)
    figure.update_layout(
        xaxis=xaxis,
        yaxis=yaxis,
        hovermode="closest",
        margin={"b": 90},
    )

    x_note = (
        "k (reach iteration)" if x_axis == _K_AXIS else system.variables[x_axis]
    )
    y_note = (
        "k (reach iteration)" if y_axis == _K_AXIS else system.variables[y_axis]
    )
    note = (
        f"Viewport: {escape(x_note)} ∈ [{x_min}, {x_max}], "
        f"{escape(y_note)} ∈ [{y_min}, {y_max}]."
    )
    if show_overflow:
        note += " Triangles indicate states beyond an edge."
    figure.add_annotation(
        text=note,
        x=0,
        y=-0.17,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font={"size": 11},
    )
    if layout:
        figure.update_layout(**layout)
    return figure
