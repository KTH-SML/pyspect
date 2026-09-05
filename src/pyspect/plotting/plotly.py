"""Plotly theme helpers that do not require numerical plotting dependencies."""

from __future__ import annotations

from plotly.basedatatypes import BaseFigure

__all__ = ("update_theme",)


def update_theme(
    name: str | None = None,
    *,
    aspectratio: str = "4:3",
    fig: BaseFigure,
) -> None:
    """Apply the shared pyspect light or dark Plotly theme to ``fig``."""
    # Kept for compatibility with the existing plotting API.
    del aspectratio

    layout: dict = {"margin": {"l": 60, "r": 20, "t": 40, "b": 60}}
    if name is not None:
        font = {"family": "Roboto, Arial, sans-serif", "size": 14}
        layout["font"] = font
        axes: dict = {"linewidth": 2}
        if not name.endswith(("2D", "3D")):
            name += "2D"
        if name.endswith("2D"):
            layout.update(xaxis=axes, yaxis=axes)
        if name.endswith("3D"):
            layout.update(scene={"xaxis": axes, "yaxis": axes, "zaxis": axes})

        if name.startswith("Light"):
            fig.update_layout(template="plotly_white")
            layout.update(
                paper_bgcolor="rgba(255, 255, 255, 1)",
                plot_bgcolor="rgba(250, 250, 250, 1)",
            )
            font["color"] = "black"
            axes.update(
                linecolor="rgba(0, 0, 0, 0.3)",
                gridcolor="rgba(0, 0, 0, 0.1)",
                zerolinecolor="rgba(0, 0, 0, 0.3)",
            )
        if name.startswith("Dark"):
            fig.update_layout(template="plotly_dark")
            layout.update(
                paper_bgcolor="rgba(26, 28, 36, 1)",
                plot_bgcolor="rgba(26, 28, 36, 1)",
            )
            font["color"] = "white"
            axes.update(
                linecolor="rgba(255, 255, 255, 0.3)",
                gridcolor="rgba(255, 255, 255, 0.1)",
                zerolinecolor="rgba(255, 255, 255, 0.3)",
            )

    fig.update_layout(template_layout=layout)
