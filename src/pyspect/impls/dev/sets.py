from __future__ import annotations

import math
from typing import Any

from .axes import Axis
from .base import Impl

__all__ = [
    'LinearSetImpl',
    'QuadraticSetImpl',
]


class LinearSetImpl[R](Impl[R]):
    """Linear set primitives: intersections of half-spaces / polyhedra."""

    def polytope(self, normals, offsets, axes=None, **kwds: Any) -> R:
        raise NotImplementedError("LinearSetImpl.polytope is not implemented")

    def halfspace(self, axes: list[Axis], normal: list[float], offset: list[float], **kwds: Any) -> R:
        """Single half-space; default impl delegates to ``polytope``."""
        return self.polytope(normals=[normal], offsets=[offset], axes=axes, **kwds)

    def slab(self, normal, offset, width, axes=None, **kwds: Any) -> R:
        """Band between two parallel hyperplanes; delegates to ``polytope``."""
        assert width > 0, "slab requires positive width."
        n = [float(k) for k in normal]
        o = [float(m) for m in offset]
        norm = math.hypot(*n)
        assert norm > 0, "slab requires a non-zero normal."
        o_far = [oi + width * ni / norm for oi, ni in zip(o, n)]
        return self.polytope(
            normals=[n, [-ni for ni in n]],
            offsets=[o, o_far],
            axes=axes,
            **kwds,
        )

    def box(self, bounds: dict[str, tuple[float, float]], axes=None, **kwds: Any) -> R:
        """Closed axis-aligned hyper-rectangle; delegates to ``polytope``."""
        assert bounds, "box requires at least one axis bound."
        names = list(bounds.keys())
        normals: list[list[float]] = []
        offsets: list[list[float]] = []
        nd = len(names)
        for i, name in enumerate(names):
            vmin, vmax = bounds[name]
            if vmin is Ellipsis or vmax is Ellipsis:
                raise ValueError(
                    "box only supports closed finite bounds; "
                    "use AlignedBoxSet for unbounded sides."
                )
            lo = [0.0] * nd
            hi = [0.0] * nd
            lo[i] = float(vmin)
            hi[i] = float(vmax)
            n_lo = [0.0] * nd
            n_hi = [0.0] * nd
            n_lo[i] = 1.0
            n_hi[i] = -1.0
            normals.extend([n_lo, n_hi])
            offsets.extend([lo, hi])
        return self.polytope(normals=normals, offsets=offsets, axes=axes, **kwds)

    def from_vertices(self, vertices, axes=None, **kwds: Any) -> R:
        """Convex 2D polygon from CCW vertices; delegates to ``polytope``."""
        assert axes is not None and len(axes) == 2, "from_vertices expects exactly two axes."
        assert len(vertices) >= 3, "from_vertices requires at least 3 vertices."
        verts = [[float(v[i]) for i in range(2)] for v in vertices]
        normals: list[list[float]] = []
        offsets: list[list[float]] = []
        for k in range(len(verts)):
            v0 = verts[k]
            v1 = verts[(k + 1) % len(verts)]
            dx = v1[0] - v0[0]
            dy = v1[1] - v0[1]
            normals.append([-dy, dx])
            offsets.append(v0)
        return self.polytope(normals=normals, offsets=offsets, axes=axes, **kwds)

    def polygon(self, order, radius, axes=None, center=None, basis=None, **kwds: Any) -> R:
        """Regular 2D polygon; delegates to ``from_vertices``."""
        assert order >= 3, "polygon requires order >= 3."
        assert radius > 0, "polygon requires positive radius."
        assert axes is not None and len(axes) == 2, "polygon expects exactly two axes."
        center = [0.0, 0.0] if center is None else [float(c) for c in center]
        assert len(center) == 2, "polygon center must have length 2."
        if basis is None:
            b1, b2 = [1.0, 0.0], [0.0, 1.0]
        else:
            b1, b2 = basis
            b1, b2 = [float(v) for v in b1], [float(v) for v in b2]
            assert len(b1) == len(b2) == 2, "polygon basis vectors must have length 2."
        vertices = [
            [
                center[i] + radius * (math.cos(2 * math.pi * k / order) * b1[i]
                                      + math.sin(2 * math.pi * k / order) * b2[i])
                for i in range(2)
            ]
            for k in range(order)
        ]
        return self.from_vertices(vertices, axes=axes, **kwds)


class QuadraticSetImpl[R](Impl[R]):
    """Quadratic set primitives: balls, cylinders, ellipsoids."""

    def quadratic(self, coefficients, axes=None, **kwds: Any) -> R:
        raise NotImplementedError("QuadraticSetImpl.quadratic is not implemented")

    def ball(self, center, radius, axes=None, **kwds: Any) -> R:
        raise NotImplementedError("QuadraticSetImpl.ball is not implemented")

    def cylinder(self, center, radius, vector, axes=None, **kwds: Any) -> R:
        raise NotImplementedError("QuadraticSetImpl.cylinder is not implemented")

    def ellipsoid(self, center, radii, axes=None, **kwds: Any) -> R:
        raise NotImplementedError("QuadraticSetImpl.ellipsoid is not implemented")
